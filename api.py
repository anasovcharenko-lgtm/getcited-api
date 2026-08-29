from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from openai import AsyncOpenAI
import os
import re
import asyncio
from datetime import datetime, timezone
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
    website: str = ""

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

def name_to_slug(name: str) -> str:
    """Best-effort slug for matching a brand/competitor name against a domain
    when we don't have their real website (e.g. 'Notion HQ' -> 'notionhq')."""
    return re.sub(r'[^a-z0-9]', '', name.lower())

def url_matches_name(url: str, domain_hint: str, name: str) -> bool:
    """True if url's domain is (or looks like) the given site.
    domain_hint: a known domain (from a real website URL), checked first — reliable.
    name: fallback slug match against the domain — best-effort, used only if no domain_hint."""
    domain = extract_domain(url).lower()
    if domain_hint:
        domain_hint = domain_hint.lower().lstrip('www.')
        return domain == domain_hint or domain.endswith('.' + domain_hint) or domain_hint.endswith('.' + domain)
    slug = name_to_slug(name)
    return bool(slug) and slug in re.sub(r'[^a-z0-9]', '', domain)

def trim_answer(text: str, limit: int = 3000) -> str:
    """Keep the real AI answer for display, but cap it so a multi-prompt audit
    doesn't balloon the payload. Cuts on a line boundary because these answers
    are markdown - slicing mid-table leaves a broken table on screen."""
    if not text:
        return ""
    text = re.split(r'\n\nSources: ', text)[0].strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    nl = cut.rfind("\n")
    if nl > limit * 0.5:
        cut = cut[:nl]
    return cut.rstrip() + "\n\n…"

def unique_domains(urls: list[str]) -> list[str]:
    seen = []
    for u in urls:
        d = extract_domain(u)
        if d and d not in seen:
            seen.append(d)
    return seen[:6]

def find_mention_sentence(text: str, brand: str) -> str:
    """Grab one short sentence/clause mentioning the brand, for a 'what AI says' preview."""
    if not text:
        return ""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    for s in sentences:
        if brand.lower() in s.lower():
            s = s.strip()
            return s if len(s) <= 240 else s[:237] + "..."
    return ""

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

MODEL_ERRORS: dict[str, str] = {}

# gpt-4o-search-preview was shut down on 2026-07-23. Web search now runs through
# the Responses API web_search tool on a standard model. Override with env vars
# if these names change again.
OPENAI_SEARCH_MODEL = os.getenv("OPENAI_SEARCH_MODEL", "gpt-5.6-luna")
OPENAI_SEARCH_FALLBACKS = ["gpt-5.6-terra", "gpt-5.6", "gpt-4.1"]
OPENAI_UTILITY_MODEL = os.getenv("OPENAI_UTILITY_MODEL", "gpt-5.6-luna")
OPENAI_UTILITY_FALLBACKS = ["gpt-5-nano", "gpt-4.1-mini"]
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
PROMPT_COUNT = int(os.getenv("PROMPT_COUNT", "6"))
MAX_ANSWER_TOKENS = int(os.getenv("MAX_ANSWER_TOKENS", "2500"))
GEMINI_ENABLED = os.getenv("GEMINI_ENABLED", "false").lower() == "true"

async def ask_gemini(prompt: str) -> str:
    if not GEMINI_ENABLED:
        MODEL_ERRORS["gemini"] = "disabled"
        return ""
    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={"tools": [{"google_search": {}}]}
        )
        text = response.text or ""
        print(f"Gemini OK: {len(text)} chars")
        return text
    except Exception as e:
        print(f"Gemini error: {e}")
        MODEL_ERRORS["gemini"] = str(e)[:300]
        return ""

def _collect_response_urls(response) -> list[str]:
    """Pull cited URLs out of a Responses API result. These annotations are far
    more reliable than regexing URLs out of prose, and they include sources the
    model consulted but didn't render as a visible link."""
    urls: list[str] = []
    try:
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                for ann in getattr(content, "annotations", []) or []:
                    url = getattr(ann, "url", None)
                    if url:
                        urls.append(url)
    except Exception as e:
        print(f"Annotation parse warning: {e}")
    return urls

async def ask_openai(prompt: str, use_search: bool = False) -> str:
    """use_search=True runs the audit prompt with live web search (what a real
    user's ChatGPT query does). Utility calls (category lookup, prompt
    generation, recommendations) don't need search, so they skip it."""
    if not use_search:
        last_error = None
        for model in [OPENAI_UTILITY_MODEL] + OPENAI_UTILITY_FALLBACKS:
            try:
                response = await openai_client.responses.create(
                    model=model,
                    input=prompt,
                    max_output_tokens=MAX_ANSWER_TOKENS,
                )
                return response.output_text or ""
            except Exception as e:
                last_error = e
                print(f"OpenAI utility error on {model}: {e}")
                continue
        MODEL_ERRORS["chatgpt"] = str(last_error)[:300]
        return ""

    last_error = None
    for model in [OPENAI_SEARCH_MODEL] + OPENAI_SEARCH_FALLBACKS:
        try:
            response = await openai_client.responses.create(
                model=model,
                tools=[{"type": "web_search"}],
                input=prompt,
                max_output_tokens=MAX_ANSWER_TOKENS,
            )
            text = response.output_text or ""
            # Append cited URLs so downstream link extraction sees every source,
            # not just the ones the model happened to inline in the prose.
            urls = _collect_response_urls(response)
            if urls:
                text += "\n\nSources: " + " ".join(dict.fromkeys(urls))
            print(f"OpenAI OK ({model}): {len(text)} chars, {len(urls)} cited urls")
            MODEL_ERRORS.pop("chatgpt", None)
            return text
        except Exception as e:
            last_error = e
            print(f"OpenAI error on {model}: {e}")
            continue

    MODEL_ERRORS["chatgpt"] = str(last_error)[:300]
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

Identify the exact product category buyers would search for. Be narrow and concrete:
name the market this product actually competes in, not a broader adjacent one.
For example "AI search visibility tracking (GEO)" is a different market from
"social media monitoring" - do not substitute one for the other.

Answer in JSON only, no other text:
{"category": "narrow buyer-facing category, 2-5 words", "known": true or false, "clean_name": "the brand name as commonly known"}""")

    try:
        import json
        clean = response.strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(clean)
        data['original_brand'] = brand
        data['clean_brand'] = data.get('clean_name', clean_brand)
        return data
    except Exception:
        return {"category": description or clean_brand + " category", "known": False, "original_brand": brand, "clean_brand": clean_brand}

PROMPT_CACHE: dict[str, list[str]] = {}

def _is_junk_prompt(p: str) -> bool:
    low = p.lower()
    if re.search(r'\b(x vs\.? y|\[.*?\]|<.*?>|tool a|brand a)\b', low):
        return True
    if len(p.split()) > 12 or len(p.split()) < 2:
        return True
    return False

async def generate_prompts(brand: str, category: str) -> list[str]:
    cache_key = category.strip().lower()
    if cache_key in PROMPT_CACHE:
        print(f"Prompt cache hit: {cache_key}")
        return PROMPT_CACHE[cache_key]
    response = await ask_openai("Generate exactly " + str(PROMPT_COUNT) + " realistic prompts that someone would type into ChatGPT when searching for or comparing tools in this category: " + category + """

Rules:
- These are BUYER prompts - people who want to find or compare tools
- Mix: best X for Y, X vs Y, alternatives to competitor, how to track X, which tool for X
- Be SPECIFIC to the category
- Do NOT include the brand name in the prompts
- Return ONLY the prompts, one per line, no numbering, no explanation""")
    prompts = [p.strip().capitalize() for p in response.strip().split("\n") if p.strip()][:PROMPT_COUNT]
    if prompts:
        PROMPT_CACHE[cache_key] = prompts
    return prompts

@app.post("/check-brand")
async def check_brand(request: dict):
    brand = request.get("brand", "")
    info = await enrich_brand(brand)
    return {"known": info.get("known", False), "category": info.get("category", "")}

AUDIT_CACHE: dict = {}
AUDIT_CACHE_TTL = int(os.getenv("AUDIT_CACHE_TTL", "3600"))

@app.post("/audit")
async def run_audit(request: AuditRequest):
    brand = request.brand
    competitors = request.competitors
    description = request.description
    website = request.website.strip()

    cache_key = "|".join([brand.strip().lower(), ",".join(sorted(c.lower() for c in competitors)), website.lower()])
    cached = AUDIT_CACHE.get(cache_key)
    if cached and (asyncio.get_event_loop().time() - cached[0]) < AUDIT_CACHE_TTL:
        print(f"Audit cache hit: {cache_key}")
        return {**cached[1], "from_cache": True}

    MODEL_ERRORS.clear()

    brand_info = await enrich_brand(brand, description)
    category = brand_info.get("category", brand + " category")
    clean_brand = brand_info.get("clean_brand", brand)
    prompts = await generate_prompts(clean_brand, category)

    # Figure out the brand's own domain if we can — this is what lets us tell
    # "mentioned with a link to your site" (mention) apart from "named with no
    # link" (citation). Prefer an explicit website; fall back to the brand
    # input if it was a URL; otherwise we simply won't have a reliable domain
    # and mentions_with_link will be 0 for this brand until one is provided.
    if website:
        brand_domain = extract_domain(website if website.startswith('http') else 'https://' + website)
    elif brand.startswith('http'):
        brand_domain = extract_domain(brand)
    else:
        brand_domain = ""

    async def run_prompt(prompt):
        gemini_answer, openai_answer = await asyncio.gather(
            ask_gemini(prompt),
            ask_openai(prompt, use_search=True)
        )
        return prompt, gemini_answer, openai_answer

    prompt_results = await asyncio.gather(*[run_prompt(p) for p in prompts])

    results = []
    all_urls = {}
    sample_quote = ""

    def process_model(prompt, answer, competitors):
        nonlocal all_urls
        answer_lower = answer.lower()
        name_mentioned = (" " + clean_brand.lower() + " ") in (" " + answer_lower + " ") or (clean_brand.lower() + ".") in answer_lower
        urls = extract_urls(answer)
        has_own_link = any(url_matches_name(u, brand_domain, clean_brand) for u in urls)

        competitors_with_link = []
        competitors_without_link = []
        for c in competitors:
            if c.lower() not in answer_lower:
                continue
            if any(url_matches_name(u, "", c) for u in urls):
                competitors_with_link.append(c)
            else:
                competitors_without_link.append(c)

        for u in urls:
            domain = extract_domain(u)
            if domain not in all_urls:
                all_urls[domain] = {"url": u, "domain": domain, "gemini_count": 0, "chatgpt_count": 0, "total": 0, "prompt": prompt}

        return {
            "name_mentioned": name_mentioned,
            "has_own_link": has_own_link,
            "urls": urls,
            "competitors_with_link": competitors_with_link,
            "competitors_without_link": competitors_without_link,
        }

    for prompt, gemini_answer, openai_answer in prompt_results:
        g = process_model(prompt, gemini_answer, competitors)
        o = process_model(prompt, openai_answer, competitors)

        for u in g["urls"]:
            domain = extract_domain(u)
            all_urls[domain]["gemini_count"] += 1
            all_urls[domain]["total"] += 1
        for u in o["urls"]:
            domain = extract_domain(u)
            all_urls[domain]["chatgpt_count"] += 1
            all_urls[domain]["total"] += 1

        g_mentioned = g["name_mentioned"]
        o_mentioned = o["name_mentioned"]
        g_with_link = g_mentioned and g["has_own_link"]
        o_with_link = o_mentioned and o["has_own_link"]

        if not sample_quote and (g_mentioned or o_mentioned):
            sample_quote = find_mention_sentence(gemini_answer, clean_brand) or find_mention_sentence(openai_answer, clean_brand)

        results.append({
            "prompt": prompt,
            "_gemini_raw": gemini_answer,
            "_chatgpt_raw": openai_answer,
            "gemini": {
                "mentioned": g_mentioned,
                "mentioned_with_link": g_with_link,
                "mentioned_without_link": g_mentioned and not g_with_link,
                "competitors_found": list(set(g["competitors_with_link"] + g["competitors_without_link"])),
                "competitors_with_link": g["competitors_with_link"],
                "competitors_without_link": g["competitors_without_link"],
                "answer": trim_answer(gemini_answer),
                "cited_domains": unique_domains(g["urls"]),
            },
            "chatgpt": {
                "mentioned": o_mentioned,
                "mentioned_with_link": o_with_link,
                "mentioned_without_link": o_mentioned and not o_with_link,
                "competitors_found": list(set(o["competitors_with_link"] + o["competitors_without_link"])),
                "competitors_with_link": o["competitors_with_link"],
                "competitors_without_link": o["competitors_without_link"],
                "answer": trim_answer(openai_answer),
                "cited_domains": unique_domains(o["urls"]),
            },
        })

    gemini_mentions = sum(1 for r in results if r["gemini"]["mentioned"])
    openai_mentions = sum(1 for r in results if r["chatgpt"]["mentioned"])
    active_models = 1 + (1 if GEMINI_ENABLED else 0)
    total_checks = len(prompts) * active_models
    visibility_score = int((gemini_mentions + openai_mentions) / total_checks * 100) if total_checks > 0 else 0

    mentions_with_link_count = sum(1 for r in results for m in ("gemini", "chatgpt") if r[m]["mentioned_with_link"])
    mentions_without_link_count = sum(1 for r in results for m in ("gemini", "chatgpt") if r[m]["mentioned_without_link"])

    def get_search_name(b):
        if b.startswith('http'):
            return extract_brand_from_url(b)
        return b

    all_brands = [clean_brand] + competitors
    competitor_stats = []
    for b in all_brands:
        search_b = get_search_name(b)
        is_you = b == brand or b == clean_brand
        g = sum(1 for r in results if search_b.lower() in r["_gemini_raw"].lower())
        c = sum(1 for r in results if search_b.lower() in r["_chatgpt_raw"].lower())
        if is_you:
            with_link = mentions_with_link_count
            without_link = mentions_without_link_count
        else:
            with_link = sum(1 for r in results for m in ("gemini", "chatgpt") if search_b in r[m]["competitors_with_link"])
            without_link = sum(1 for r in results for m in ("gemini", "chatgpt") if search_b in r[m]["competitors_without_link"])
        competitor_stats.append({
            "name": search_b,
            "is_your_brand": is_you,
            "gemini_mentions": g,
            "chatgpt_mentions": c,
            "total_mentions": g + c,
            "mention_rate": round((g + c) / total_checks * 100, 1) if total_checks > 0 else 0,
            "mentions_with_link": with_link,
            "mentions_without_link": without_link,
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

    payload = {
        "brand": clean_brand,
        "category": category,
        "brand_domain": brand_domain,
        "run_at": datetime.now(timezone.utc).isoformat(),
        "models_used": {
            "chatgpt": OPENAI_SEARCH_MODEL,
            "gemini": GEMINI_MODEL,
        },
        "model_status": {
            "gemini": {"ok": "gemini" not in MODEL_ERRORS, "enabled": GEMINI_ENABLED, "error": MODEL_ERRORS.get("gemini")},
            "chatgpt": {"ok": "chatgpt" not in MODEL_ERRORS, "error": MODEL_ERRORS.get("chatgpt")},
        },
        "visibility_score": visibility_score,
        "gemini_score": gemini_mentions,
        "chatgpt_score": openai_mentions,
        "total_prompts": len(prompts),
        "mentions_score": mentions_with_link_count,
        "citations_score": mentions_without_link_count,
        "results": results,
        "competitor_ranking": competitor_stats,
        "citations": citations,
        "sample_quote": sample_quote,
        "recommendations": rec_response,
        "debug": {
            "category": category,
            "clean_brand": clean_brand,
            "prompts_generated": len(prompts),
            "prompts": prompts,
            "competitors_in": competitors,
            "answer_lengths": [len(r["chatgpt"].get("answer") or "") for r in results],
            "model_errors": dict(MODEL_ERRORS),
        },
    }

    got_answers = any(r["chatgpt"].get("answer") or r["gemini"].get("answer") for r in results)
    if "chatgpt" not in MODEL_ERRORS and got_answers:
        AUDIT_CACHE[cache_key] = (asyncio.get_event_loop().time(), payload)

    return payload

@app.post("/clear-cache")
async def clear_cache():
    n = len(AUDIT_CACHE) + len(PROMPT_CACHE)
    AUDIT_CACHE.clear()
    PROMPT_CACHE.clear()
    return {"cleared": n}

@app.get("/debug-prompt")
async def debug_prompt(q: str = "best AI visibility tracking tools"):
    try:
        response = await openai_client.responses.create(
            model=OPENAI_SEARCH_MODEL,
            tools=[{"type": "web_search"}],
            input=q,
            max_output_tokens=MAX_ANSWER_TOKENS,
        )
        text = response.output_text or ""
        usage = getattr(response, "usage", None)
        rt = None
        if usage is not None:
            d = getattr(usage, "output_tokens_details", None)
            rt = getattr(d, "reasoning_tokens", None) if d else None
        return {
            "model": OPENAI_SEARCH_MODEL,
            "max_output_tokens": MAX_ANSWER_TOKENS,
            "status": getattr(response, "status", None),
            "incomplete_reason": getattr(getattr(response, "incomplete_details", None), "reason", None),
            "text_chars": len(text),
            "text_preview": text[:300],
            "output_tokens": getattr(usage, "output_tokens", None) if usage else None,
            "reasoning_tokens": rt,
        }
    except Exception as e:
        return {"error": str(e)[:500]}

@app.get("/health-models")
async def health_models():
    """Quick check of which AI models actually respond. Use this to tell a real
    'low visibility' result apart from 'the model never answered'."""
    MODEL_ERRORS.clear()
    probe = "What is the best project management software? Name two tools."
    gemini_text, openai_text = await asyncio.gather(
        ask_gemini(probe),
        ask_openai(probe, use_search=True),
    )
    return {
        "gemini": {"ok": bool(gemini_text), "chars": len(gemini_text), "model": GEMINI_MODEL, "error": MODEL_ERRORS.get("gemini")},
        "chatgpt": {"ok": bool(openai_text), "chars": len(openai_text), "model": OPENAI_SEARCH_MODEL, "urls_found": len(extract_urls(openai_text)), "error": MODEL_ERRORS.get("chatgpt")},
    }

@app.get("/")
def root():
    return {"status": "GetCited API is running"}
