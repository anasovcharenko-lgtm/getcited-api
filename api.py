from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from openai import AsyncOpenAI
import os
import re
import asyncio
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

def extract_brand_from_url(url: str) -> str:
    domain = re.sub(r'https?://', '', url)
    domain = domain.split('/')[0]
    domain = re.sub(r'^www\.', '', domain)
    parts = domain.split('.')
    name = parts[0]
    for prefix in ['try', 'use', 'app', 'my']:
        if name.lower().startswith(prefix) and len(name) > len(prefix) + 2:
            name = name[len(prefix):]
            break
    return name.capitalize()

async def get_site_context(url: str) -> str:
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
            text = resp.text[:3000]
            title_match = re.search(r'<title[^>]*>(.*?)</title>', text, re.IGNORECASE | re.DOTALL)
            desc_match = re.search(r'content="([^"]{20,200})"', text)
            result = []
            if title_match:
                result.append("Title: " + title_match.group(1).strip())
            if desc_match:
                result.append("Description: " + desc_match.group(1).strip())
            return " | ".join(result)
    except Exception as e:
        print(f"Scrape error: {e}")
        return ""

async def ask_gemini(prompt: str) -> str:
    try:
        response = gemini_client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt
        )
        print(f"Gemini OK: {len(response.text)} chars, brand check: {clean_brand.lower() in response.text.lower()}")
        return response.text
    except Exception as e:
        print(f"Gemini error: {e}")
        return "GEMINI_BLOCKED"
        return ""

async def ask_openai(prompt: str) -> str:
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-search-preview",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI error: {e}")
        return ""

async def enrich_brand(brand: str, description: str = "") -> dict:
    clean_brand = brand
    if brand.startswith('http'):
        clean_brand = extract_brand_from_url(brand)

    if description:
        context = "Brand: " + clean_brand + "\nDescription: " + description
    elif brand.startswith('http'):
        context = "What company or product is at this website: " + brand + "\nBrand name extracted: " + clean_brand + "\n\nSearch your knowledge to identify what this company does. What is their specific product category?"
    else:
        context = "Brand: " + clean_brand

    response = await ask_openai(context + """

Answer in JSON format only - be very specific about the category:
{"category": "specific category (e.g. AI brand visibility tracking, project management software, email marketing platform)", "known": true or false, "clean_name": "the brand name as commonly known"}""")

    try:
        import json
        clean = response.strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(clean)
        data['original_brand'] = brand
        data['clean_brand'] = data.get('clean_name', clean_brand)
        return data
    except Exception:
        return {"category": description or clean_brand + " category", "known": False, "original_brand": brand, "clean_brand": clean_brand}

async def generate_prompts(brand: str, category: str) -> list[str]:
    response = await ask_openai("Generate exactly 10 realistic prompts that someone would type into ChatGPT when searching for or comparing tools in this category: " + category + """

Rules:
- These are BUYER prompts - people who want to find or compare tools
- Mix: best X for Y, X vs Y, alternatives to competitor, how to track X, which tool for X
- Be SPECIFIC to the category
- Do NOT include the brand name in the prompts
- Return ONLY the 10 prompts, one per line, no numbering, no explanation""")
    prompts = [p.strip().capitalize() for p in response.strip().split("\n") if p.strip()]
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
    clean_brand = brand_info.get("clean_brand", brand)
    prompts = await generate_prompts(clean_brand, category)

    async def run_prompt(prompt):
        gemini_answer, openai_answer = await asyncio.gather(
            ask_gemini(prompt),
            ask_openai(prompt)
        )
        return prompt, gemini_answer, openai_answer

    prompt_results = await asyncio.gather(*[run_prompt(p) for p in prompts])

    results = []
    all_urls = {}

    for prompt, gemini_answer, openai_answer in prompt_results:
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
                "mentioned": (" " + clean_brand.lower() + " ") in (" " + gemini_answer.lower() + " ") or (clean_brand.lower() + ".") in gemini_answer.lower(),
                "competitors_found": [c for c in competitors if c.lower() in gemini_answer.lower()]
            },
            "chatgpt": {
                "mentioned": (" " + clean_brand.lower() + " ") in (" " + openai_answer.lower() + " ") or (clean_brand.lower() + ".") in openai_answer.lower(),
                "competitors_found": [c for c in competitors if c.lower() in openai_answer.lower()]
            },
        })

    gemini_mentions = sum(1 for r in results if r["gemini"]["mentioned"])
    openai_mentions = sum(1 for r in results if r["chatgpt"]["mentioned"])
    total_checks = len(prompts) * 2
    visibility_score = int((gemini_mentions + openai_mentions) / total_checks * 100) if total_checks > 0 else 0

    def get_search_name(b):
        if b.startswith('http'):
            return extract_brand_from_url(b)
        return b

    all_brands = [clean_brand] + competitors
    competitor_stats = []
    for b in all_brands:
        search_b = get_search_name(b)
        g = sum(1 for r in results if search_b.lower() in r["_gemini_raw"].lower())
        c = sum(1 for r in results if search_b.lower() in r["_chatgpt_raw"].lower())
        competitor_stats.append({
            "name": search_b,
            "is_your_brand": b == brand or b == clean_brand,
            "gemini_mentions": g,
            "chatgpt_mentions": c,
            "total_mentions": g + c,
            "mention_rate": round((g + c) / total_checks * 100, 1) if total_checks > 0 else 0
        })
    competitor_stats.sort(key=lambda x: x["total_mentions"], reverse=True)
    for i, stat in enumerate(competitor_stats):
        stat["rank"] = i + 1

    citations = sorted(all_urls.values(), key=lambda x: x["total"], reverse=True)[:10]
    summary = "Brand: " + clean_brand + ", Category: " + category + ", Score: " + str(visibility_score) + "%"

    rec_response = await ask_openai("You are an AI visibility consultant. Give exactly 3 specific recommendations to improve " + clean_brand + " visibility in AI models. Category: " + category + """ Format each as:
PRIORITY: [High/Medium/Low]
ACTION: [specific action]
WHY: [one sentence]
EFFORT: [Easy/Medium/Hard]
Data: """ + summary)

    for r in results:
        r.pop("_gemini_raw", None)
        r.pop("_chatgpt_raw", None)

    return {
        "brand": clean_brand,
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
