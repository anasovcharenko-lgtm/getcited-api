from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from openai import AsyncOpenAI
import os
import time
import re
import asyncio
from datetime import datetime, timezone
from dotenv import load_dotenv
from html.parser import HTMLParser
from urllib.parse import urlparse, parse_qs
import httpx
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

PROMPT_CACHE: dict[str, list[dict]] = {}

# What the audit asks, and in what proportion. Only 'brand' style queries were
# being generated before, which is why the audit kept surfacing vendor docs
# instead of independent write-ups.
PROMPT_MIX = [("problem", 0.5), ("comparison", 0.34), ("brand", 0.16)]

PROMPT_TYPE_GUIDE = {
    "problem": (
        "Someone with the problem who does not know this category of tool exists. "
        "They describe the symptom, not the solution.\n"
        "Examples: why does my brand not show up in chatgpt / "
        "how do i know what ai says about my company / "
        "my competitors appear in ai answers and i don't"
    ),
    "comparison": (
        "Someone who knows the category and is choosing between options.\n"
        "Examples: best ai visibility tracking tools / "
        "cheapest way to monitor brand mentions in ai / "
        "ai visibility tools compared"
    ),
    "brand": (
        "Someone who already names a specific product in this category.\n"
        "Examples: profound alternatives / is otterly worth it / "
        "peec ai vs profound"
    ),
}


def split_counts(total: int) -> dict:
    """Turn the mix into whole numbers that add up to `total` exactly."""
    counts = {}
    assigned = 0
    for i, (name, share) in enumerate(PROMPT_MIX):
        if i == len(PROMPT_MIX) - 1:
            counts[name] = total - assigned      # last one absorbs the rounding
        else:
            n = max(1, round(total * share)) if total >= len(PROMPT_MIX) else 0
            counts[name] = n
            assigned += n
    if counts[PROMPT_MIX[-1][0]] < 0:
        counts[PROMPT_MIX[-1][0]] = 0
    return counts


def _is_junk_prompt(p: str) -> bool:
    low = p.lower()
    if re.search(r'\b(x vs\.? y|\[.*?\]|<.*?>|tool a|brand a)\b', low):
        return True
    if len(p.split()) > 12 or len(p.split()) < 2:
        return True
    return False


def _build_generation_prompt(brand: str, category: str, counts: dict) -> str:
    blocks = []
    for name, n in counts.items():
        if n <= 0:
            continue
        blocks.append(f"{n} of type {name.upper()}:\n{PROMPT_TYPE_GUIDE[name]}")
    return (
        f"Category: {category}\n\n"
        f"Write search queries real people type into ChatGPT about this category.\n\n"
        + "\n\n".join(blocks) +
        "\n\nRules:\n"
        "- 3 to 10 words each. Plain lowercase, how people actually type.\n"
        "- NEVER use placeholders like 'X vs Y' or brackets.\n"
        f"- Do not mention {brand}.\n"
        "- Stay inside the stated category, do not drift to adjacent markets.\n\n"
        "Output format, one per line, nothing else:\n"
        "PROBLEM: the query\n"
        "COMPARISON: the query\n"
        "BRAND: the query"
    )


def _parse_generated(response: str, counts: dict) -> list[dict]:
    """Read the labelled lines back. Unlabelled lines are dropped: they are
    stray prose, not queries the model intended to produce."""
    out = []
    per_type = {k: 0 for k in counts}
    for raw in response.strip().split("\n"):
        line = raw.strip().lstrip("-*\u2022 ").strip()
        if not line:
            continue
        ptype, _, text = line.partition(":")
        ptype = ptype.strip().lower()
        if ptype not in counts:
            continue
        text = text.strip().strip('"\'')
        if not text or _is_junk_prompt(text):
            continue
        if per_type.get(ptype, 0) >= counts.get(ptype, 0):
            continue                              # this bucket is already full
        per_type[ptype] += 1
        out.append({"text": text[0].upper() + text[1:], "type": ptype})
    return out


def _fallback_prompts(category: str, total: int) -> list[dict]:
    """Used when generation returns nothing usable. Covers all three types
    rather than only comparison queries."""
    base = [
        {"text": "Why is my brand not showing up in AI answers", "type": "problem"},
        {"text": "How do I know what AI says about my company", "type": "problem"},
        {"text": "How to get mentioned in ChatGPT answers", "type": "problem"},
        {"text": f"Best {category} tools", "type": "comparison"},
        {"text": f"{category} compared", "type": "comparison"},
        {"text": f"Cheapest {category} tool", "type": "comparison"},
    ]
    return base[:total]


async def generate_prompts(brand: str, category: str) -> list[dict]:
    cache_key = category.strip().lower()
    if cache_key in PROMPT_CACHE:
        print(f"Prompt cache hit: {cache_key}")
        return PROMPT_CACHE[cache_key]

    counts = split_counts(PROMPT_COUNT)
    response = await ask_openai(_build_generation_prompt(brand, category, counts))
    prompts = _parse_generated(response, counts)

    if len(prompts) < 2:
        print("  prompt generation produced too little, using fallback")
        prompts = _fallback_prompts(category, PROMPT_COUNT)

    by_type = {}
    for p in prompts:
        by_type[p["type"]] = by_type.get(p["type"], 0) + 1
    print(f"  prompts by type: {by_type}")

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
    prompt_specs = await generate_prompts(clean_brand, category)
    prompts = [p["text"] for p in prompt_specs]
    prompt_type_by_text = {p["text"]: p["type"] for p in prompt_specs}

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
            "prompt_type": prompt_type_by_text.get(prompt, "comparison"),
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
            "urls": _collect_response_urls(response),
        }
    except Exception as e:
        return {"error": str(e)[:500]}
# ─────────────────────────────────────────────────────────────
# Source gap: do the pages the models cite actually mention you?
# ─────────────────────────────────────────────────────────────

SOURCE_CHECK_CONCURRENCY = 6
SOURCE_CHECK_TIMEOUT = 12
SOURCE_BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")


class _PageText(HTMLParser):
    """Readable text only. Scripts and styles would produce false positives
    when searching a page for a brand name."""

    SKIP = {"script", "style", "noscript", "svg"}

    def __init__(self):
        super().__init__()
        self.parts = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            t = data.strip()
            if t:
                self.parts.append(t)

    def text(self):
        return " ".join(self.parts)


def _reason_for_status(code: int) -> str:
    if code == 403:
        return "the site blocked us (403) - bot protection such as Cloudflare"
    if code == 404:
        return "page no longer exists (404)"
    if code in (401, 402):
        return "needs a login or paid subscription"
    if code == 429:
        return "the site rate-limited us (429) - try again later"
    if 500 <= code < 600:
        return f"the site returned a server error ({code})"
    return f"unexpected response ({code})"


async def _fetch_page_text(client, url: str) -> tuple[str, str]:
    """Return (text, reason). Exactly one of them is non-empty.

    The reason is written for whoever reads the report, not for a log file:
    'we could not check this' is only useful if it also says why.
    """
    try:
        r = await client.get(url, follow_redirects=True, timeout=SOURCE_CHECK_TIMEOUT,
                             headers={"User-Agent": SOURCE_BROWSER_UA})
    except httpx.TimeoutException:
        return "", f"took longer than {SOURCE_CHECK_TIMEOUT}s to respond"
    except httpx.ConnectError:
        return "", "could not connect - domain may be dead or blocking us"
    except Exception as exc:
        return "", f"fetch failed ({type(exc).__name__})"

    if r.status_code != 200:
        return "", _reason_for_status(r.status_code)

    ctype = r.headers.get("content-type", "").split(";")[0].strip()
    if ctype and "html" not in ctype:
        return "", f"not a web page (content type: {ctype})"

    parser = _PageText()
    try:
        parser.feed(r.text)
    except Exception:
        return "", "page markup could not be parsed"

    text = parser.text()
    if len(text) < 400:
        return "", (f"only {len(text)} characters of text - the content is most likely "
                    "rendered by JavaScript, which we cannot read")
    return text, ""

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
REDDIT_COMMENT_LIMIT = int(os.getenv("REDDIT_COMMENT_LIMIT", "60"))


def _is_reddit(url: str) -> bool:
    return urlparse(url).netloc.lower().endswith("reddit.com")


def _is_youtube(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return host.endswith("youtube.com") or host.endswith("youtu.be")


def _youtube_video_id(url: str) -> str:
    p = urlparse(url)
    if p.netloc.lower().endswith("youtu.be"):
        return p.path.lstrip("/").split("/")[0]
    if "/shorts/" in p.path or "/embed/" in p.path:
        return p.path.rstrip("/").split("/")[-1]
    return (parse_qs(p.query).get("v") or [""])[0]


def _walk_reddit_comments(node, out: list, budget: list) -> None:
    if budget[0] <= 0 or not isinstance(node, dict):
        return
    data = node.get("data", {})
    body = data.get("body")
    if isinstance(body, str) and body.strip():
        out.append(body)
        budget[0] -= 1
    replies = data.get("replies")
    if isinstance(replies, dict):
        for child in replies.get("data", {}).get("children", []):
            _walk_reddit_comments(child, out, budget)
    for child in data.get("children", []) or []:
        _walk_reddit_comments(child, out, budget)

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")

# Tokens last an hour. Cached so a 20-URL check authenticates once, not 20 times.
_reddit_token = {"value": "", "expires": 0.0}


async def _reddit_token_get(client) -> tuple[str, str]:
    """Return (token, reason).

    An empty token with an empty reason means we are running anonymously on
    purpose - no credentials configured yet - rather than failing.
    """
    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
        return "", ""

    if _reddit_token["value"] and time.time() < _reddit_token["expires"]:
        return _reddit_token["value"], ""

    try:
        r = await client.post(
            "https://www.reddit.com/api/v1/access_token",
            data={"grant_type": "client_credentials"},
            auth=(REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET),
            headers={"User-Agent": "getcited/1.0 (source gap checker)"},
            timeout=SOURCE_CHECK_TIMEOUT,
        )
    except Exception as exc:
        return "", f"reddit auth failed ({type(exc).__name__})"

    if r.status_code != 200:
        return "", f"reddit auth returned {r.status_code} - check the client id and secret"

    data = r.json()
    token = data.get("access_token", "")
    if not token:
        return "", "reddit auth returned no token"

    # Refresh a minute early so no request fires on a just-expired token.
    _reddit_token["value"] = token
    _reddit_token["expires"] = time.time() + int(data.get("expires_in", 3600)) - 60
    return token, ""


def _reddit_json_url(url: str, authed: bool) -> str:
    """Authenticated requests must go to oauth.reddit.com. Anonymous ones have
    a slightly better chance on old.reddit.com."""
    clean = url.split("?")[0].rstrip("/")
    if authed:
        for host in ("://www.reddit.com", "://old.reddit.com", "://reddit.com"):
            clean = clean.replace(host, "://oauth.reddit.com")
    else:
        clean = clean.replace("://www.reddit.com", "://old.reddit.com")
        clean = clean.replace("://reddit.com", "://old.reddit.com")
    return clean + ".json"
async def _fetch_reddit(client, url: str) -> tuple[str, str, str]:
    token, auth_reason = await _reddit_token_get(client)
    if auth_reason:
        return "", "", auth_reason

    headers = {"User-Agent": "getcited/1.0 (source gap checker)"}
    if token:
        headers["Authorization"] = f"bearer {token}"

    try:
        r = await client.get(_reddit_json_url(url, bool(token)),
                             timeout=SOURCE_CHECK_TIMEOUT,
                             headers=headers, follow_redirects=True)
    except Exception as exc:
        return "", "", f"reddit fetch failed ({type(exc).__name__})"
    if r.status_code == 403 and not token:
        return "", "", ("reddit blocked the request - add REDDIT_CLIENT_ID and "
                        "REDDIT_CLIENT_SECRET in Railway to read it")
    if r.status_code != 200:
        return "", "", f"reddit returned {r.status_code}"
    try:
        data = r.json()
    except Exception:
        return "", "", "reddit did not return JSON (link may not be a thread)"
    if not isinstance(data, list) or not data:
        return "", "", "reddit response had no thread data"

    post_parts = []
    for child in data[0].get("data", {}).get("children", []):
        d = child.get("data", {})
        for field in ("title", "selftext"):
            v = d.get(field)
            if isinstance(v, str) and v.strip():
                post_parts.append(v)

    comments = []
    if len(data) > 1:
        _walk_reddit_comments(data[1], comments, [REDDIT_COMMENT_LIMIT])

    return " ".join(post_parts), " ".join(comments), ""


async def _fetch_youtube(client, url: str) -> tuple[str, str]:
    if not YOUTUBE_API_KEY:
        return "", "no YouTube API key configured"
    vid = _youtube_video_id(url)
    if not vid:
        return "", "could not read a video id from this link"
    try:
        r = await client.get("https://www.googleapis.com/youtube/v3/videos",
                             params={"part": "snippet", "id": vid, "key": YOUTUBE_API_KEY},
                             timeout=SOURCE_CHECK_TIMEOUT)
    except Exception as exc:
        return "", f"YouTube API request failed ({type(exc).__name__})"
    if r.status_code == 403:
        return "", "YouTube API refused the key (quota or restrictions)"
    if r.status_code != 200:
        return "", f"YouTube API returned {r.status_code}"
    items = r.json().get("items", [])
    if not items:
        return "", "video not found or private"
    sn = items[0].get("snippet", {})
    return f"{sn.get('title','')} {sn.get('description','')}".strip(), ""
def _name_in_text(text: str, name: str) -> bool:
    """Word-boundary match, so 'Peec' does not fire inside 'Peecock'."""
    if not name:
        return False
    pattern = r"(?<![A-Za-z0-9])" + re.escape(name) + r"(?![A-Za-z0-9])"
    return re.search(pattern, text, re.IGNORECASE) is not None


class SourceGapRequest(BaseModel):
    urls: list[str]
    brand: str
    competitors: list[str] = []
    brand_domain: str = ""


@app.post("/source-gap")
async def source_gap(request: SourceGapRequest):
    urls = [u for u in dict.fromkeys(request.urls) if u.startswith("http")][:25]
    if not urls:
        return {"results": [], "summary": {"absent": 0, "present": 0, "unreadable": 0}}

    sem = asyncio.Semaphore(SOURCE_CHECK_CONCURRENCY)

    async with httpx.AsyncClient() as client:
        async def one(url: str) -> dict:
            comment_text = ""
            async with sem:
                if _is_reddit(url):
                    text, comment_text, reason = await _fetch_reddit(client, url)
                elif _is_youtube(url):
                    text, reason = await _fetch_youtube(client, url)
                else:
                    text, reason = await _fetch_page_text(client, url)

            domain = re.sub(r"^www\.", "", re.sub(r"https?://", "", url).split("/")[0])
            base = {"url": url, "domain": domain}

            if reason:
                return {**base, "status": "unreadable", "reason": reason,
                        "competitors_on_page": []}

            found = [c for c in request.competitors if _name_in_text(text, c)]
            you_here = _name_in_text(text, request.brand)
            if not you_here and request.brand_domain:
                you_here = request.brand_domain.lower() in text.lower()

            return {**base,
                    "status": "present" if you_here else "absent",
                    "reason": "",
                    "competitors_on_page": found}

        results = await asyncio.gather(*[one(u) for u in urls])

    # Genuine gaps first, then pages where you appear but the model ignored you,
    # then the ones we could not read at all.
    order = {"absent": 0, "present": 1, "unreadable": 2}
    results = sorted(results, key=lambda r: (order[r["status"]],
                                             -len(r["competitors_on_page"])))

    summary = {k: sum(1 for r in results if r["status"] == k)
               for k in ("absent", "present", "unreadable")}
    return {"results": results, "summary": summary}
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
