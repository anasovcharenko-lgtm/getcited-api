from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from openai import AsyncOpenAI
import os
import re
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

def extract_urls(text: str) -> list[str]:
    pattern = r'https?://[^\s\)\]\,\"\'<>]+'
    urls = re.findall(pattern, text)
    cleaned = []
    for url in urls:
        url = url.rstrip('.,;:')
        if len(url) > 10:
            cleaned.append(url)
    return cleaned

def extract_domain(url: str) -> str:
    try:
        domain = re.sub(r'https?://', '', url)
        domain = domain.split('/')[0]
        domain = re.sub(r'^www\.', '', domain)
        return domain
    except Exception:
        return url

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
{{"category": "short category name", "known": true or false}}""")
    try:
        import json
        clean = response.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except Exception:
        return {"category": description or brand + " category", "known": False}

async def generate_prompts(brand: str, category: str) -> list[str]:
    response = await ask_openai(f"""Generate exactly 10 realistic search prompts that a potential buyer would type into ChatGPT when looking for a product in this category: "{category}".
Rules:
- Write prompts as real buyers would ask them
- Mix awareness, comparison, and decision-intent prompts
- Focus on DECISION-INTENT: "best X for Y", "X alternatives", "which X should I use"
- Do NOT include the brand name "{brand}" in the prompts
- Return ONLY the 10 prompts, one per line, no numbering, no explanation""")
    prompts = [p.strip() for p in response.strip().split("\n") if p.strip()]
    return prompts[:10] if len(prompts) >= 10 else prompts

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
    all_urls = {}

    for prompt in prompts:
        gemini_answer = await ask_gemini(prompt)
        openai_answer = await ask_openai(prompt)

        for url in extract_urls(gemini_answer):
            domain = extract_domain(url)
            if domain not in all_urls:
                all_urls[domain] = {"url": url, "domain": domain, "gemini_count": 0, "chatgpt_count": 0, "total": 0}
            all_urls[domain]["gemini_count"] += 1
            all_urls[domain]["total"] += 1

        for url in extract_urls(openai_answer):
            domain = extract_domain(url)
            if domain not in all_urls:
                all_urls[domain] = {"url": url, "domain": domain, "gemini_count": 0, "chatgpt_count": 0, "total": 0}
            all_urls[domain]["chatgpt_count"] += 1
            all_urls[domain]["total"] += 1

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
    total_checks = len(prompts) * 2
    visibility_score = int((gemini_mentions + openai_mentions) / total_checks * 100) if total_checks > 0 else 0

    all_brands = [brand] + competitors
    competitor_stats = []
    for b in all_brands:
        g = sum(1 for r in results if b.lower() in r["_gemini_raw"].lower())
        c = sum(1 for r in results if b.lower() in r["_chatgpt_raw"].lower())
        competitor_stats.append({
            "name": b, "is_your_brand": b == brand,
            "gemini_mentions": g, "chatgpt_mentions": c,
            "total_mentions": g + c,
            "mention_rate": round((g + c) / total_checks * 100, 1) if total_checks > 0 else 0
        })
    competitor_stats.sort(key=lambda x: x["total_mentions"], reverse=True)
    for i, stat in enumerate(competitor_stats):
        stat["rank"] = i + 1

    citations = sorted(all_urls.values(), key=lambda x: x["total"], reverse=True)[:10]
    summary = f"Brand: {brand}, Category: {category}, Score: {visibility_score}%"

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
        "citations": citations,
        "recommendations": rec_response,
    }

@app.get("/")
def root():
    return {"status": "GetCited API is running"}
