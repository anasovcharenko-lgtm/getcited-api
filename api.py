from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class AuditRequest(BaseModel):
    brand: str
    competitors: list[str] = []
    description: str = ""

async def ask_gemini(prompt: str) -> str:
    try:
        response = gemini_client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt
        )
        return response.text
    except Exception as e:
        print(f"Gemini error: {e}")
        return ""

async def ask_openai(prompt: str) -> str:
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI error: {e}")
        return ""

async def enrich_brand(brand: str, description: str = "") -> dict:
    context = f'Brand: "{brand}"'
    if description:
        context += f'\nUser description: "{description}"'
    response = await ask_openai(f"""{context}

What category/industry is this brand in? Answer in JSON format only:
{{
  "category": "short category name (e.g. project management software, CRM, note-taking app)",
  "known": true or false
}}""")
    try:
        import json
        clean = response.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except:
        return {"category": description or brand + " category", "known": False}

async def generate_prompts(brand: str, category: str) -> list[str]:
    response = await ask_openai(f"""Generate exactly 5 realistic search prompts that a potential buyer would type into ChatGPT when looking for a product in this category: "{category}".

Rules:
- Write prompts as real buyers would ask them
- Mix awareness, comparison, and decision-intent prompts
- Do NOT include the brand name "{brand}" in the prompts
- Return ONLY the 5 prompts, one per line, no numbering, no explanation""")

    prompts = [p.strip() for p in response.strip().split("\n") if p.strip()]
    return prompts[:5] if len(prompts) >= 5 else prompts

@app.post("/check-brand")
async def check_brand(request: dict):
    brand = request.get("brand", "")
    info = await enrich_brand(brand)
    return {"known": info.get("known", False), "category": info.get("category", "")}

@app.post("/audit")
async def run_audit(request: AuditRequest):
    brand = request.brand
    competitors = request.competitors
    description = request.description

    brand_info = await enrich_brand(brand, description)
    category = brand_info.get("category", brand + " category")

    prompts = await generate_prompts(brand, category)

    results = []
    for prompt in prompts:
        gemini_answer = await ask_gemini(prompt)
        openai_answer = await ask_openai(prompt)
        results.append({
            "prompt": prompt,
            "_gemini_raw": gemini_answer,
            "_chatgpt_raw": openai_answer,
            "gemini": {
                "mentioned": brand.lower() in gemini_answer.lower(),
                "competitors_found": [c for c in competitors if c.lower() in gemini_answer.lower()]
            },
            "chatgpt": {
                "mentioned": brand.lower() in openai_answer.lower(),
                "competitors_found": [c for c in competitors if c.lower() in openai_answer.lower()]
            },
        })

    gemini_mentions = sum(1 for r in results if r["gemini"]["mentioned"])
    openai_mentions = sum(1 for r in results if r["chatgpt"]["mentioned"])
    total_mentions = gemini_mentions + openai_mentions
    total_checks = len(prompts) * 2
    visibility_score = int(total_mentions / total_checks * 100) if total_checks > 0 else 0

    all_brands = [brand] + competitors
    competitor_stats = []
    for b in all_brands:
        gemini_count = sum(1 for r in results if b.lower() in r["_gemini_raw"].lower())
        chatgpt_count = sum(1 for r in results if b.lower() in r["_chatgpt_raw"].lower())
        total = gemini_count + chatgpt_count
        competitor_stats.append({
            "name": b,
            "is_your_brand": b == brand,
            "gemini_mentions": gemini_count,
            "chatgpt_mentions": chatgpt_count,
            "total_mentions": total,
            "mention_rate": round(total / total_checks * 100, 1) if total_checks > 0 else 0
        })
    competitor_stats.sort(key=lambda x: x["total_mentions"], reverse=True)
    for i, stat in enumerate(competitor_stats):
        stat["rank"] = i + 1

    summary = f"Brand: {brand}, Category: {category}, Score: {visibility_score}%, Competitors: {[s['name'] + ': ' + str(s['mention_rate']) + '%' for s in competitor_stats]}"

    rec_response = await ask_openai(f"""You are an AI visibility consultant.
Give exactly 3 specific recommendations to improve {brand}'s visibility in AI models.
Category: {category}
Format each as:
PRIORITY: [High/Medium/Low]
ACTION: [specific action]
WHY: [one sentence]
EFFORT: [Easy/Medium/Hard]
Data: {summary}""")

    for r in results:
        r.pop("_gemini_raw", None)
        r.pop("_chatgpt_raw", None)

    return {
        "brand": brand,
        "category": category,
        "visibility_score": visibility_score,
        "gemini_score": gemini_mentions,
        "chatgpt_score": openai_mentions,
        "total_prompts": len(prompts),
        "results": results,
        "competitor_ranking": competitor_stats,
        "recommendations": rec_response,
    }

@app.get("/")
def root():
    return {"status": "GetCited API is running"}
