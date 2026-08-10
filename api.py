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


async def get_site_context(url: str) -> str:
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
            text = resp.text[:3000]
            import re
            title = re.search(r'<title[^>]*>(.*?)</title>', text, re.IGNORECASE | re.DOTALL)
            desc = re.search(r'<meta[^>]*name=["']description["'][^>]*content=["'](.*?)["']', text, re.IGNORECASE)
            og_desc = re.search(r'<meta[^>]*property=["']og:description["'][^>]*content=["'](.*?)["']', text, re.IGNORECASE)
            result = []
            if title: result.append(f"Title: {title.group(1).strip()}")
            if desc: result.append(f"Description: {desc.group(1).strip()}")
            elif og_desc: result.append(f"Description: {og_desc.group(1).strip()}")
            return " | ".join(result)
    except Exception as e:
        print(f"Scrape error: {e}")
        return ""

async def enrich_brand(brand: str, description: str = "") -> dict:
    clean_brand = brand
    if brand.startswith('http'):
        import re
        domain = re.sub(r'https?://', '', brand)
        domain = domain.split('/')[0]
        domain = re.sub(r'^www\.', '', domain)
        parts = domain.split('.')
        clean_brand = parts[0]
        for prefix in ['try', 'get', 'use', 'app', 'my']:
            if clean_brand.startswith(prefix) and len(clean_brand) > len(prefix) + 2:
                clean_brand = clean_brand[len(prefix):]
                break
        clean_brand = clean_brand.capitalize()

    site_context = ""
    if brand.startswith('http'):
        site_context = await get_site_context(brand)
    context = f'Brand: "{clean_brand}"'
    if brand.startswith('http'):
        context += f'\nWebsite URL: "{brand}"'
    if site_context:
        context += f'\nWebsite content: "{site_context}"'
    if description:
        context += f'\nUser description: "{description}"'
    response = await ask_openai(f"""{context}

What is this brand and what category/industry is it in?
Important: use the website URL and full context to identify the correct brand and category.
Be specific - not just "software" but "AI visibility tracking tool" or "project management software".

Answer in JSON format only:
{{"category": "specific category name", "known": true or false, "clean_name": "the common brand name people use"}}""")
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
    response = await ask_openai(f"""You are helping audit brand visibility in AI search results.

Generate exactly 10 realistic prompts that someone would type into ChatGPT or Google when SEARCHING FOR or COMPARING tools in this category: "{category}".

Rules:
- These are BUYER prompts - people who want to find or compare tools
- Mix: "best X for Y", "X vs Y", "alternatives to [competitor]", "how to track X", "which tool for X"
- Be SPECIFIC to the category - not generic marketing questions
- Do NOT include "{brand}" in the prompts
- Return ONLY the 10 prompts, one per line, no numbering, no explanation, no quotes

Example for "AI visibility tracking tool":
Best tool to track brand mentions in ChatGPT
How to monitor if AI recommends my brand
AI search visibility software comparison
Track brand visibility across multiple AI models
Alternatives to Profound for AI brand monitoring
How often does ChatGPT mention my company
Best GEO monitoring tool for marketers
Which software tracks brand recommendations in AI
AI overview tracking for marketing teams
Monitor brand citations in generative AI""")
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
                "mentioned": clean_brand.lower() in gemini_answer.lower(),
                "competitors_found": [c for c in competitors if c.lower() in gemini_answer.lower()]
            },
            "chatgpt": {
                "mentioned": clean_brand.lower() in openai_answer.lower(),
                "competitors_found": [c for c in competitors if c.lower() in openai_answer.lower()]
            },
        })

    gemini_mentions = sum(1 for r in results if r["gemini"]["mentioned"])
    openai_mentions = sum(1 for r in results if r["chatgpt"]["mentioned"])
    total_checks = len(prompts) * 2
    visibility_score = int((gemini_mentions + openai_mentions) / total_checks * 100) if total_checks > 0 else 0

    all_brands = [clean_brand] + competitors
    competitor_stats = []
    for b in all_brands:
        search_b = b
        if b.startswith('http'):
            import re
            d = re.sub(r'https?://', '', b).split('/')[0]
            d = re.sub(r'^www\.', '', d)
            parts = d.split('.')
            search_b = parts[0].capitalize()
            for prefix in ['try', 'get', 'use', 'app', 'my']:
                if search_b.lower().startswith(prefix) and len(search_b) > len(prefix) + 2:
                    search_b = search_b[len(prefix):].capitalize()
                    break
        g = sum(1 for r in results if search_b.lower() in r["_gemini_raw"].lower())
        c = sum(1 for r in results if search_b.lower() in r["_chatgpt_raw"].lower())
        competitor_stats.append({
            "name": search_b if b.startswith('http') else b, "is_your_brand": b == clean_brand or b == brand,
            "gemini_mentions": g, "chatgpt_mentions": c,
            "total_mentions": g + c,
            "mention_rate": round((g + c) / total_checks * 100, 1) if total_checks > 0 else 0
        })
    competitor_stats.sort(key=lambda x: x["total_mentions"], reverse=True)
    for i, stat in enumerate(competitor_stats):
        stat["rank"] = i + 1

    citations = sorted(all_urls.values(), key=lambda x: x["total"], reverse=True)[:10]
    summary = f"Brand: {clean_brand}, Category: {category}, Score: {visibility_score}%"

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
