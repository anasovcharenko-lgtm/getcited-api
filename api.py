from fastapi import FastAPI, Depends, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dataclasses import dataclass
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


# ─────────────────────────────────────────────────────────────────────────────
# Who is calling
#
# Every endpoint below spends money: model calls, or fetching arbitrary pages.
# None of them asked who was calling. The API URL sits in plain text in the
# site's JavaScript bundle, CORS does not apply to anything that is not a
# browser, and the one-audit limit was counted in the browser — so the limit was
# a request, not a rule, and the spend had no ceiling at all.
#
# A caller is identified by the access token Supabase already issues to anyone
# who signs in. The token is checked against Supabase rather than verified here,
# which costs one fast request and means there is no signing secret to keep: the
# two values below are the public URL and publishable key, the same pair that
# ships in the frontend.
#
# REQUIRE_AUTH exists so this can be deployed without a broken window. The API
# and the site deploy separately, and turning the check on before the site sends
# tokens would 401 every audit, including yours. Ship both, then set it to true.
# ─────────────────────────────────────────────────────────────────────────────

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://djnvqoautxjyecfjuwhd.supabase.co").rstrip("/")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "sb_publishable_kBgEq9htfHu_1_2Eon_3nQ_ZbNzIlqo")
REQUIRE_AUTH = os.getenv("REQUIRE_AUTH", "false").lower() == "true"
FREE_AUDIT_LIMIT = int(os.getenv("FREE_AUDIT_LIMIT", "1"))
ADMIN_EMAILS = {e.strip().lower()
                for e in os.getenv("ADMIN_EMAILS",
                                   "anas.ovcharenko@gmail.com,anaovcharenko11@gmail.com").split(",")
                if e.strip()}


@dataclass
class Caller:
    id: str
    email: str
    token: str

    @property
    def is_admin(self) -> bool:
        return self.email.lower() in ADMIN_EMAILS


async def _supabase_user(token: str) -> dict | None:
    """Ask Supabase who this token belongs to. None means it is not valid."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                f"{SUPABASE_URL}/auth/v1/user",
                headers={"Authorization": f"Bearer {token}", "apikey": SUPABASE_ANON_KEY},
            )
    except Exception as e:
        print(f"auth: could not reach Supabase: {e}")
        if REQUIRE_AUTH:
            # Unreachable is not the same as invalid, and with the check on, a
            # caller must not be let through on a network blip — this is a spend
            # control, so it fails closed.
            raise HTTPException(status_code=503, detail="Could not verify sign-in. Try again.")
        return None
    if r.status_code != 200:
        return None
    body = r.json()
    return body if body.get("id") else None


async def caller(authorization: str | None = Header(default=None)) -> Caller | None:
    """The signed-in user, or None when the check is off and none was established.

    REQUIRE_AUTH means one thing and only one: whether a request can be refused.
    With it off nothing is ever blocked, including when Supabase cannot be
    reached or the token is rejected — otherwise the flag meant to make the
    deploy window safe would itself be able to break the site during it.
    """
    token = ""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()

    if not token:
        if REQUIRE_AUTH:
            raise HTTPException(status_code=401, detail="Sign in to run this.")
        print("auth: anonymous call (REQUIRE_AUTH is off)")
        return None

    user = await _supabase_user(token)
    if not user:
        if REQUIRE_AUTH:
            # Wrong or expired either way; the client's move is the same.
            raise HTTPException(status_code=401,
                                detail="Your sign-in has expired. Sign in again.")
        print("auth: token present but not accepted (REQUIRE_AUTH is off)")
        return None
    return Caller(id=user["id"], email=user.get("email") or "", token=token)


# Keyword lookups are not free: two DataForSEO endpoints per call, about ten
# cents, billed whoever asked and however many phrases were in it. Until that is
# part of a plan somebody pays for, only the owner can spend it.
#
# One variable opens it later without a code change.
KEYWORDS_ADMIN_ONLY = os.getenv("KEYWORDS_ADMIN_ONLY", "true").lower() == "true"


async def admin_only(who: Caller | None = Depends(caller)) -> Caller | None:
    """Owner-only access to a paid endpoint.

    This deliberately fails closed where `caller` fails open. REQUIRE_AUTH is
    about whether a deploy window can refuse anybody, so with it off `caller`
    returns None rather than blocking — which is right for an audit and wrong
    here. If turning off a flag meant for a deploy window also opened a metered
    vendor endpoint to the internet, that flag would be a billing incident
    waiting to happen.

    So: no caller, or a caller who is not an admin, is refused either way.
    """
    if not KEYWORDS_ADMIN_ONLY:
        return who
    if who is None or not who.is_admin:
        raise HTTPException(status_code=403,
                            detail="Keyword research is not open yet.")
    return who


async def audits_run_by(who: Caller) -> int:
    """How many audits this account has already run.

    Counted here rather than in the browser, which could simply be told not to
    ask. Read with the caller's own token, so the row-level policy on `audits`
    does the scoping and no service key is involved.
    """
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(
            f"{SUPABASE_URL}/rest/v1/audits",
            params={"select": "id", "user_id": f"eq.{who.id}"},
            headers={"Authorization": f"Bearer {who.token}", "apikey": SUPABASE_ANON_KEY,
                     "Prefer": "count=exact", "Range": "0-0"},
        )
    if r.status_code >= 400:
        print(f"quota: count failed {r.status_code}: {r.text[:200]}")
        # Failing open here would make the limit optional for anyone who can
        # make this query fail.
        raise HTTPException(status_code=503, detail="Could not check your quota. Try again.")
    total = (r.headers.get("content-range") or "").split("/")[-1]
    return int(total) if total.isdigit() else 0


async def within_quota(who: Caller | None = Depends(caller)) -> Caller | None:
    """For the endpoint that starts a new audit. Admins are exempt."""
    if who is None or who.is_admin:
        return who
    used = await audits_run_by(who)
    if used >= FREE_AUDIT_LIMIT:
        raise HTTPException(
            status_code=402,
            detail=f"Free trial used ({used}/{FREE_AUDIT_LIMIT}). Upgrade to run more audits.")
    return who

class Competitor(BaseModel):
    name: str
    website: str = ""


class AuditRequest(BaseModel):
    brand: str
    # Legacy: names only. Kept so older clients keep working.
    competitors: list[str] = []
    # Preferred: name plus site. The site is what makes link matching exact
    # instead of guessing a domain from the name.
    competitor_list: list[Competitor] = []
    description: str = ""
    # Language is separate from country on purpose: a UK-targeted brand can
    # have a Russian-speaking audience, and one field cannot express that.
    language: str = ""
    # When present, these run INSTEAD of generated ones. Choosing "write my own"
    # means measuring exactly these, so topping them up would defeat it.
    custom_prompts: list[str] = []
    # These run IN ADDITION to whatever else the audit runs. Tracked prompts
    # arrive here: the point of tracking a query is to watch it alongside the
    # rest, not to replace the rest with it.
    extra_prompts: list[str] = []
    # Extending an audit re-uses the audit path for the added prompts only, and
    # the recommendations it would write for two prompts are both wrong and
    # paid for. The caller keeps the ones it already has.
    skip_recommendations: bool = False
    # The market being sold INTO, not where the company sits. A US company
    # targeting the UK should be measured on UK results, in English.
    country: str = "US"
    website: str = ""
    # How well known the brand is: "large", "mid" or "small". It decides how the
    # queries split between broad and long-tail, because a mid-size brand cannot
    # place in a broad comparison however good its site is. Blank falls back to
    # DEFAULT_BRAND_SIZE, so an older client that omits it keeps working.
    brand_size: str = ""
    # Which models to run for THIS audit, e.g. ["chatgpt"]. Empty means every
    # model switched on globally, so a client that does not send the field keeps
    # its old behaviour. This narrows what runs; it never enables anything,
    # because enabling a model is a billing decision, not a per-audit one.
    models: list[str] = []

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

_SEPARATORS = " \t\n\r-_./\\"


def _strip_separators(text: str) -> tuple[str, list[int]]:
    """Return the text without separators, plus a map back to original indexes.

    Keeping the map is what lets us drop separators for matching and still test
    word boundaries against the real text.
    """
    out, index_map = [], []
    for i, ch in enumerate(text):
        if ch not in _SEPARATORS:
            out.append(ch.lower())
            index_map.append(i)
    return "".join(out), index_map


def brand_in_text(text: str, name: str) -> bool:
    """Does this brand appear in this text, however either side spells it?

    Models write the same brand as "Set Loyalty", "SetLoyalty" or "Set-Loyalty"
    depending on the sentence, and users type it either way in the form. Matching
    the literal string reported zero visibility for brands that were in fact
    named. Separators are ignored on both sides; word boundaries are still
    checked against the original text so "Peec" does not match "Peecock".
    """
    if not name or not text:
        return False
    flat_name, _ = _strip_separators(name)
    if not flat_name:
        return False
    flat_text, index_map = _strip_separators(text)

    pos = flat_text.find(flat_name)
    while pos != -1:
        start_i = index_map[pos]
        end_i = index_map[pos + len(flat_name) - 1]
        before = text[start_i - 1] if start_i > 0 else ""
        after = text[end_i + 1] if end_i + 1 < len(text) else ""
        if not before.isalnum() and not after.isalnum():
            return True
        pos = flat_text.find(flat_name, pos + 1)
    return False


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

def join_list(items: list[str], conj: str = "or") -> str:
    """"title, H1 or meta description". The conjunction is the caller's to choose,
    because these lists sit inside sentences of opposite polarity: "not in the
    title, H1 or meta description" needs "or", while "missing from the H1 and the
    meta description" needs "and"."""
    if len(items) <= 1:
        return items[0] if items else ""
    return ", ".join(items[:-1]) + f" {conj} " + items[-1]


def split_sources(text: str) -> tuple[str, list[str]]:
    """The URLs the model actually cited, separated from the prose.

    We append them ourselves (search_openai) so that link extraction sees every
    source and not only the ones the model inlined. They used to be dropped on the
    way to the screen, which left the one question a reader has - why was someone
    else named and not me - answerable only by the CITED chips, which say nothing
    about where the answer came from. Kept out of the answer body deliberately:
    `discovered_brands` matches brand names against the answer text, and a bare
    URL containing a brand's domain would otherwise read as a mention."""
    if not text:
        return "", []
    parts = re.split(r'\n\nSources: ', text, maxsplit=1)
    body = parts[0].strip()
    urls = parts[1].split() if len(parts) > 1 else []
    return body, list(dict.fromkeys(u for u in urls if u.startswith("http")))


def trim_answer(text: str, limit: int = 3000) -> str:
    """Keep the real AI answer for display, but cap it so a multi-prompt audit
    doesn't balloon the payload. Cuts on a line boundary because these answers
    are markdown - slicing mid-table leaves a broken table on screen."""
    if not text:
        return ""
    text, _ = split_sources(text)
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

async def fetch_site_text(url: str) -> tuple[str, str]:
    """Return (text, reason). Exactly one is non-empty.

    The old version read only <title> and a meta tag, which for a small brand is
    often just the brand name again - enough to identify nothing. Reading the
    page body is what actually reveals what the company sells.
    """
    if not url:
        return "", "no website given"
    if not url.startswith("http"):
        url = "https://" + url

    try:
        import httpx
        async with httpx.AsyncClient(timeout=12.0, follow_redirects=True) as client:
            r = await client.get(url, headers={"User-Agent": SOURCE_BROWSER_UA})
    except Exception as exc:
        return "", f"could not reach the site ({type(exc).__name__})"

    if r.status_code != 200:
        return "", f"the site returned {r.status_code}"

    parser = _PageText()
    try:
        parser.feed(r.text)
    except Exception:
        return "", "the page markup could not be parsed"

    text = parser.text()
    if len(text) < 200:
        return "", ("the page has almost no readable text - it is most likely "
                    "rendered by JavaScript")
    return text[:6000], ""


# Which language buyers in this market actually type. Prompts generated in the
# wrong language measure a market the client does not sell in.
MARKET_LANGUAGES = {
    "US": "English", "GB": "English", "CA": "English", "AU": "English",
    "IE": "English", "NZ": "English", "IN": "English", "SG": "English",
    "RU": "Russian", "BY": "Russian", "KZ": "Russian",
    "DE": "German", "AT": "German", "CH": "German",
    "FR": "French", "BE": "French",
    "ES": "Spanish", "MX": "Spanish", "AR": "Spanish",
    "IT": "Italian", "PT": "Portuguese", "BR": "Portuguese",
    "NL": "Dutch", "PL": "Polish", "TR": "Turkish",
    "JP": "Japanese", "KR": "Korean", "CN": "Chinese",
    "UA": "Ukrainian", "SE": "Swedish", "NO": "Norwegian", "DK": "Danish",
}


def market_language(country: str) -> str:
    return MARKET_LANGUAGES.get((country or "US").upper(), "English")


PROMPT_COUNT = int(os.getenv("PROMPT_COUNT", "20"))
# Every prompt costs money, so hand-written lists get a ceiling too.
MAX_CUSTOM_PROMPTS = int(os.getenv("MAX_CUSTOM_PROMPTS", "20"))
# Added prompts cost the same per run as generated ones, so they get a ceiling
# of their own rather than borrowing the one above.
MAX_EXTRA_PROMPTS = int(os.getenv("MAX_EXTRA_PROMPTS", "20"))
# Cap answer length. NOTE: on reasoning models this budget also covers internal
# reasoning tokens, so setting it too low returns an EMPTY answer with no error.
MAX_ANSWER_TOKENS = int(os.getenv("MAX_ANSWER_TOKENS", "2500"))

MODEL_ERRORS: dict[str, str] = {}

# gpt-4o-search-preview was shut down on 2026-07-23. Web search now runs through
# the Responses API web_search tool on a standard model. Override with env vars
# if these names change again.
OPENAI_SEARCH_MODEL = os.getenv("OPENAI_SEARCH_MODEL", "gpt-5.6-luna")
OPENAI_SEARCH_FALLBACKS = ["gpt-5.6-terra", "gpt-5.6", "gpt-4.1"]
OPENAI_UTILITY_MODEL = os.getenv("OPENAI_UTILITY_MODEL", "gpt-5.6-luna")
OPENAI_UTILITY_FALLBACKS = ["gpt-5-nano", "gpt-4.1-mini"]
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
MAX_ANSWER_TOKENS = int(os.getenv("MAX_ANSWER_TOKENS", "2500"))
GEMINI_ENABLED = os.getenv("GEMINI_ENABLED", "false").lower() == "true"

# Perplexity. `sonar` is the cheap grounded model: a request fee plus tokens,
# roughly $0.007 an answer, so a 50-prompt audit costs about 35 cents. It always
# searches, which is why it belongs in the visibility score at all.
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "")
PERPLEXITY_MODEL = os.getenv("PERPLEXITY_MODEL", "sonar")
PERPLEXITY_ENABLED = (os.getenv("PERPLEXITY_ENABLED", "false").lower() == "true"
                      and bool(PERPLEXITY_API_KEY))

# DeepSeek. Its Responses endpoint runs the search on DeepSeek's own servers, so
# there is no second vendor to wire up. Deliberately no fallback to
# /chat/completions: that endpoint answers from memory, which measures what the
# model remembers rather than what it tells someone asking today — a different
# quantity wearing the same label.
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
DEEPSEEK_ENABLED = (os.getenv("DEEPSEEK_ENABLED", "false").lower() == "true"
                    and bool(DEEPSEEK_API_KEY))

# Google's AI Overviews. Google publishes no API for them, so this goes through
# DataForSEO, which runs the search and parses the block. Two consequences worth
# knowing before reading the numbers:
#   - it costs about $0.004 a prompt, unlike the models;
#   - Google shows an AI Overview for only some queries, so a prompt can come
#     back with nothing. That is "no overview existed", not "you were not named",
#     and it drops out of the denominator rather than counting against the brand.
DATAFORSEO_LOGIN = os.getenv("DATAFORSEO_LOGIN", "")
DATAFORSEO_PASSWORD = os.getenv("DATAFORSEO_PASSWORD", "")
AI_OVERVIEW_ENABLED = (os.getenv("AI_OVERVIEW_ENABLED", "false").lower() == "true"
                       and bool(DATAFORSEO_LOGIN) and bool(DATAFORSEO_PASSWORD))

# DataForSEO wants a place, not a country code. Overridable in one env var
# rather than needing a code change for a market we have not listed.
DATAFORSEO_LOCATIONS = {
    "US": "United States", "GB": "United Kingdom", "UK": "United Kingdom",
    "RU": "Russia", "TR": "Turkey", "DE": "Germany", "FR": "France",
    "ES": "Spain", "IT": "Italy", "NL": "Netherlands", "PL": "Poland",
    "AE": "United Arab Emirates", "KZ": "Kazakhstan", "CA": "Canada",
    "AU": "Australia", "IN": "India", "BR": "Brazil",
}
DATAFORSEO_DEFAULT_LOCATION = os.getenv("DATAFORSEO_LOCATION", "United States")

# The SERP endpoint requires a language; the keyword-volume endpoint must not be
# given one. Those are opposite rules on the same vendor, and treating them as
# one rule is how the language got dropped from the SERP call and broke it.
#
# Derived from the market rather than from the audit's interface language: a
# search in Germany is answered in German whichever language the dashboard is
# in, and the overview we are measuring is the one a German buyer sees.
DATAFORSEO_LANGUAGES = {
    "US": "en", "GB": "en", "UK": "en", "CA": "en", "AU": "en", "IN": "en",
    "RU": "ru", "KZ": "ru", "TR": "tr", "DE": "de", "FR": "fr", "ES": "es",
    "IT": "it", "NL": "nl", "PL": "pl", "BR": "pt", "AE": "en",
}
DATAFORSEO_DEFAULT_LANGUAGE = os.getenv("DATAFORSEO_LANGUAGE", "en")

# How often the phrase is searched on Google. Billed per request rather than per
# keyword, so the whole prompt set costs one call — about nine cents — and the
# figure is a monthly average that barely moves, so it does not need refreshing
# every audit.
#
# This is Google search volume, not prompt volume. Nobody sells prompt volume:
# the model vendors do not publish it. It is a proxy for which prompts matter,
# and the screen has to say which of the two it is showing.
KEYWORD_VOLUME_ENABLED = (os.getenv("KEYWORD_VOLUME_ENABLED", "false").lower() == "true"
                          and bool(DATAFORSEO_LOGIN) and bool(DATAFORSEO_PASSWORD))

# Hard limits from the API, not guesses: a keyword over either is rejected
# rather than returned empty. Most conversational prompts fail the word count,
# which is the real ceiling on how much of a set this can cover without an
# agent rewriting the prompts into search phrases first.
KEYWORD_MAX_WORDS = 10
KEYWORD_MAX_CHARS = 80

# Newest path first, legacy second: DataForSEO says to migrate off the old one,
# and an account may be on either.
# Once DataForSEO rejects or suspends the credentials, stop calling it.
#
# ask_ai_overview runs per prompt, so an audit of twenty prompts with a wrong
# password fired twenty parallel rejected requests from a datacentre IP against
# a days-old account. That is indistinguishable from credential stuffing, and
# their risk system suspended the account for it — a self-inflicted outage
# caused by retrying rather than by the wrong password itself.
#
# Any auth-shaped failure now latches, and the rest of the audit skips the call
# entirely. It clears on restart, which a redeploy after fixing the credentials
# provides anyway.
_DATAFORSEO_LOCKED = ""

# The latch alone does not help against a full fan-out: twenty prompts all start
# before the first one fails, so all twenty requests go out anyway — measured,
# not assumed. Limiting how many are in flight is what gives the latch something
# to catch. It also suits what this is: a paid, rate-limited, fraud-monitored
# SERP vendor rather than a model API built for parallel load.
DATAFORSEO_CONCURRENCY = int(os.getenv("DATAFORSEO_CONCURRENCY", "3"))
_dataforseo_gate = asyncio.Semaphore(DATAFORSEO_CONCURRENCY)

# Auth and account-state failures, as opposed to "this query had no overview".
# 40100 unauthorized, 40200 payment required, 40201 account suspended.
DATAFORSEO_FATAL = {40100, 40101, 40200, 40201, 40202}

KEYWORD_VOLUME_PATHS = [
    "/v3/keywords_data/google_ads/search_volume/live",
    "/v3/keywords_data/google/search_volume/live",
]

# The registry. Every model the audit knows about is declared here once, in
# display order, with the switch that says whether it can run at all.
#
# Adding a model means two edits: a line here and an ask_* function registered
# in ASK. Availability, per-audit selection, how the prompt is put, scoring,
# competitor counts and status all read from the registry, so a new model
# cannot be added and then quietly left out of the numbers.
MODEL_SWITCHES: dict[str, "callable[[], bool]"] = {
    "chatgpt": lambda: True,                 # no switch: without it there is no audit
    "gemini": lambda: GEMINI_ENABLED,
    "perplexity": lambda: PERPLEXITY_ENABLED,
    "deepseek": lambda: DEEPSEEK_ENABLED,
    "ai_overview": lambda: AI_OVERVIEW_ENABLED,
}
ALL_MODELS = list(MODEL_SWITCHES)

# The exact model string each one answers with, read at request time rather than
# at import: these are env vars, and a run has to record what actually answered.
MODEL_VERSIONS: dict[str, "callable[[], str]"] = {
    "chatgpt": lambda: OPENAI_SEARCH_MODEL,
    "gemini": lambda: GEMINI_MODEL,
    "perplexity": lambda: PERPLEXITY_MODEL,
    "deepseek": lambda: DEEPSEEK_MODEL,
    "ai_overview": lambda: "google ai overview via dataforseo",
}

# How to ask each model. Filled in below as the ask_* functions are defined;
# run_prompt reads this rather than naming models one by one, so a model added
# to the registry is actually put to the question.
ASK: dict[str, "callable"] = {}


def available_models() -> list[str]:
    """Models that could run right now, before the client narrows them."""
    return [m for m, on in MODEL_SWITCHES.items() if on()]


def selected_models(requested: list[str] | None) -> list[str]:
    """What this audit will actually run.

    A requested model that is switched off globally stays off: the request
    narrows, it never enables. An empty intersection falls back to everything
    available, because measuring nothing and reporting 0% would read as "you are
    invisible" when the truth is "you measured nothing".
    """
    avail = available_models()
    if not requested:
        return avail
    want = {m.strip().lower() for m in requested if m and m.strip()}
    return [m for m in avail if m in want] or avail

def _collect_gemini_urls(response) -> list[str]:
    """Grounding citations, read defensively: the shape of this metadata has moved
    between SDK versions, and a missing attribute must cost us the sources, not the
    whole answer."""
    urls: list[str] = []
    try:
        for cand in (getattr(response, "candidates", None) or []):
            meta = getattr(cand, "grounding_metadata", None)
            for chunk in (getattr(meta, "grounding_chunks", None) or []):
                uri = getattr(getattr(chunk, "web", None), "uri", None)
                if uri:
                    urls.append(uri)
    except Exception as e:
        print(f"Gemini grounding metadata unreadable: {e}")
    return list(dict.fromkeys(urls))


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
        # Same shape as the OpenAI path, so link extraction and the Sources list on
        # screen do not have to know which model answered.
        urls = _collect_gemini_urls(response)
        if urls:
            text += "\n\nSources: " + " ".join(urls)
        print(f"Gemini OK: {len(text)} chars, {len(urls)} cited urls")
        return text
    except Exception as e:
        print(f"Gemini error: {e}")
        MODEL_ERRORS["gemini"] = str(e)[:300]
        return ""

"""Perplexity moved Sonar off /chat/completions onto the Agent API, and the two
descriptions of where it went disagree: the live API's own 403 says
`/v1/responses`, while the migration guide's curl example posts to `/v1/agent`
with a `preset` instead of a `model`. Both may be true — one OpenAI-compatible,
one native — and neither can be confirmed from here without a key.

So the shapes are a list, tried in order until one answers, and the one that
worked is remembered for the rest of the process. The server's own error message
is first because it is the more direct evidence. If Perplexity changes this
again, the fix is a line in this list rather than a rewrite.
"""

PERPLEXITY_PRESET = os.getenv("PERPLEXITY_PRESET", "fast")

PERPLEXITY_SHAPES: list[tuple[str, str]] = [
    # (path, how the body names the thing to run)
    ("/v1/responses", "model"),   # what the live 403 pointed at
    ("/v1/agent", "preset"),      # what the migration guide shows
    ("/chat/completions", "chat"),  # the old one, for a key still on it
]
_perplexity_shape: int | None = None   # index of the one that answered


def _perplexity_body(kind: str, prompt: str) -> dict:
    if kind == "chat":
        return {"model": PERPLEXITY_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": MAX_ANSWER_TOKENS}
    if kind == "preset":
        return {"preset": PERPLEXITY_PRESET, "input": prompt}
    return {"model": PERPLEXITY_MODEL, "input": prompt}


def _perplexity_text(data: dict) -> str:
    """The answer, wherever this version of the API puts it."""
    if isinstance(data.get("output_text"), str) and data["output_text"]:
        return data["output_text"]
    # The typed trace: find the message item and join its text parts.
    parts: list[str] = []
    for item in (data.get("output") or []):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for c in (item.get("content") or []):
            if isinstance(c, dict) and isinstance(c.get("text"), str):
                parts.append(c["text"])
    if parts:
        return "\n".join(parts)
    # The old chat shape.
    return ((data.get("choices") or [{}])[0].get("message", {}) or {}).get("content", "") or ""


def _perplexity_urls(data: dict) -> list[str]:
    """Every source, from whichever place this version lists them.

    The migration guide describes the results list but not the name of the field
    holding the URL, so a result contributes its `url` if it has one and
    otherwise the first value in it that looks like a link. Guessing the key
    would mean silently losing the sources the moment it is not `url`.
    """
    urls: list[str] = []

    def take(results) -> None:
        for item in (results or []):
            if isinstance(item, str) and item.startswith("http"):
                urls.append(item)
            elif isinstance(item, dict):
                u = item.get("url")
                if not (isinstance(u, str) and u.startswith("http")):
                    u = next((v for v in item.values()
                              if isinstance(v, str) and v.startswith("http")), None)
                if u:
                    urls.append(u)

    for item in (data.get("output") or []):
        if isinstance(item, dict) and item.get("type") == "search_results":
            take(item.get("results"))
    take(data.get("search_results"))
    take(data.get("citations"))
    return list(dict.fromkeys(urls))


async def ask_perplexity(prompt: str, country: str = "") -> str:
    """Perplexity always searches, so there is no tool to switch on.

    The sources come back as structured fields rather than inside the prose, so
    they are folded into the same "Sources:" block the other models produce.
    Without that the answer would reach the screen with no sources at all —
    Perplexity cites with bare [1] markers and rarely inlines a URL.

    `country` is accepted and not sent: the old web_search_options field has no
    documented equivalent here, and inventing a key risks a 400 on every call.
    Results will be Perplexity's default market until the live shape is known.
    """
    global _perplexity_shape
    if not PERPLEXITY_ENABLED:
        MODEL_ERRORS["perplexity"] = "disabled"
        return ""

    order = list(range(len(PERPLEXITY_SHAPES)))
    if _perplexity_shape is not None:
        order.remove(_perplexity_shape)
        order.insert(0, _perplexity_shape)

    last = ""
    for i in order:
        path, kind = PERPLEXITY_SHAPES[i]
        try:
            async with httpx.AsyncClient(timeout=90) as client:
                resp = await client.post(
                    f"https://api.perplexity.ai{path}",
                    headers={"Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                             "Content-Type": "application/json"},
                    json=_perplexity_body(kind, prompt),
                )
            if resp.status_code >= 400:
                last = f"{resp.status_code} on {path}: {resp.text[:200]}"
                print(f"Perplexity {last}")
                # 401 is the key, not the shape, and no other path will fix it —
                # stop rather than turn one clear failure into three. 403 is not
                # treated that way: the migration notice arrives as a 403.
                if resp.status_code == 401:
                    break
                continue

            data = resp.json()
            text = _perplexity_text(data)
            if not text:
                last = f"{path} answered with no text: {str(data)[:200]}"
                print(f"Perplexity {last}")
                continue

            urls = _perplexity_urls(data)
            if urls:
                text += "\n\nSources: " + " ".join(urls)
            if _perplexity_shape != i:
                print(f"Perplexity: using {path} ({kind})")
            _perplexity_shape = i
            print(f"Perplexity OK: {len(text)} chars, {len(urls)} cited urls")
            MODEL_ERRORS.pop("perplexity", None)
            return text
        except Exception as e:
            last = f"{path}: {str(e)[:200]}"
            print(f"Perplexity error on {last}")
            continue

    MODEL_ERRORS["perplexity"] = last or "no response"
    return ""


async def ask_deepseek(prompt: str, country: str = "") -> str:
    """DeepSeek, with the search it runs on its own servers.

    The Responses endpoint plus a web_search tool; no second vendor. There is
    deliberately no fallback to /chat/completions — that endpoint answers from
    memory, and an answer from memory is a different measurement wearing the
    same label. Better a model that reports a failure than one that quietly
    starts measuring what it remembers.
    """
    if not DEEPSEEK_ENABLED:
        MODEL_ERRORS["deepseek"] = "disabled"
        return ""
    try:
        async with httpx.AsyncClient(timeout=90) as client:
            resp = await client.post(
                "https://api.deepseek.com/responses",
                headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                         "Content-Type": "application/json"},
                json={"model": DEEPSEEK_MODEL,
                      "tools": [{"type": "web_search"}],
                      "input": prompt},
            )
        if resp.status_code >= 400:
            detail = f"{resp.status_code}: {resp.text[:200]}"
            print(f"DeepSeek error {detail}")
            MODEL_ERRORS["deepseek"] = detail
            return ""
        data = resp.json()
        # Same family of shapes as the Perplexity Agent API, so the same readers
        # work and there is one place to fix if either moves again.
        text = _perplexity_text(data)
        if not text:
            MODEL_ERRORS["deepseek"] = f"answered with no text: {str(data)[:200]}"
            return ""
        urls = _perplexity_urls(data)
        if urls:
            text += "\n\nSources: " + " ".join(urls)
        print(f"DeepSeek OK: {len(text)} chars, {len(urls)} cited urls")
        MODEL_ERRORS.pop("deepseek", None)
        return text
    except Exception as e:
        print(f"DeepSeek error: {e}")
        MODEL_ERRORS["deepseek"] = str(e)[:300]
        return ""


def keyword_eligible(prompt: str) -> bool:
    """Whether this prompt can be asked about at all.

    The API rejects anything over 80 characters or 10 words, so a long
    conversational prompt is not "no data" — it never gets looked up. Checking
    here rather than letting the call fail keeps one bad prompt from costing the
    whole request.
    """
    p = (prompt or "").strip()
    return bool(p) and len(p) <= KEYWORD_MAX_CHARS and len(p.split()) <= KEYWORD_MAX_WORDS


async def keyword_volumes(prompts: list[str], country: str = "") -> dict:
    """Google search volume for the prompts that are short enough to ask about.

    One request for the whole set, because billing is per request. Returns a map
    keyed by the lowercased prompt — the API lowercases keywords, so matching the
    response back to the prompt has to as well.

    A prompt that is missing from the result, or carries a null volume, is left
    out of the map entirely rather than stored as zero. Zero means "nobody
    searches this", and that is a different statement from "Google has no data",
    which is what a null actually means.
    """
    global _DATAFORSEO_LOCKED
    if not KEYWORD_VOLUME_ENABLED or _DATAFORSEO_LOCKED:
        return {}
    eligible = [p for p in prompts if keyword_eligible(p)]
    if not eligible:
        print("keyword volume: no prompt is short enough to look up")
        return {}

    location = DATAFORSEO_LOCATIONS.get((country or "").upper(), DATAFORSEO_DEFAULT_LOCATION)
    # language_code is deliberately not sent: DataForSEO warns that setting it
    # makes Google return null volumes.
    body = [{"keywords": eligible[:700], "location_name": location}]

    for path in KEYWORD_VOLUME_PATHS:
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(f"https://api.dataforseo.com{path}",
                                         auth=(DATAFORSEO_LOGIN, DATAFORSEO_PASSWORD),
                                         json=body)
            if resp.status_code >= 400:
                print(f"keyword volume {resp.status_code} on {path}: {resp.text[:200]}")
                if resp.status_code in (401, 402, 403, 429):
                    # Not a wrong path — a refused account. Trying the other
                    # path doubles the rejected attempts for no chance of
                    # success.
                    _DATAFORSEO_LOCKED = f"{resp.status_code}: {resp.text[:200]}"
                    return {}
                continue
            data = resp.json()
            task = (data.get("tasks") or [{}])[0]
            if task.get("status_code", 20000) >= 40000:
                print(f"keyword volume task failed: {task.get('status_code')} "
                      f"{task.get('status_message', '')[:200]}")
                if task.get("status_code") in DATAFORSEO_FATAL:
                    _DATAFORSEO_LOCKED = (f"{task.get('status_code')}: "
                                          f"{task.get('status_message', '')[:200]}")
                    return {}
                continue

            out: dict = {}
            for item in (task.get("result") or []):
                if not isinstance(item, dict):
                    continue
                kw = (item.get("keyword") or "").strip().lower()
                vol = item.get("search_volume")
                if kw and isinstance(vol, int):
                    out[kw] = {"search_volume": vol,
                               "competition": item.get("competition"),
                               "cpc": item.get("cpc")}
            print(f"keyword volume: {len(out)} of {len(eligible)} eligible prompts "
                  f"have data ({len(prompts) - len(eligible)} too long to ask)")
            return out
        except Exception as e:
            print(f"keyword volume error on {path}: {e}")
            continue
    return {}


async def ask_ai_overview(prompt: str, country: str = "") -> str:
    """The AI Overview Google shows for this query, via DataForSEO.

    Not a model and not an API Google offers: a real search is run and the block
    is parsed out of the page. Which means two things the other entries in the
    registry do not have.

    An empty answer here is ordinary, not a failure. Google shows an AI Overview
    for only some queries — informational and longer ones most of all — so a
    prompt that triggers none returns "". That is "no overview existed", not
    "the overview did not name you", and the scoring already drops an empty
    answer out of the denominator rather than counting it against the brand.

    And the prompt is sent as a search query, because that is what it becomes.
    A conversational prompt that reads well to a chat model may produce no
    overview at all here; that is a fact about the query, and the audit should
    show it rather than hide it.
    """
    global _DATAFORSEO_LOCKED
    if not AI_OVERVIEW_ENABLED:
        MODEL_ERRORS["ai_overview"] = "disabled"
        return ""
    location = DATAFORSEO_LOCATIONS.get((country or "").upper(), DATAFORSEO_DEFAULT_LOCATION)
    language = DATAFORSEO_LANGUAGES.get((country or "").upper(), DATAFORSEO_DEFAULT_LANGUAGE)
    try:
        async with _dataforseo_gate:
            # Checked inside the gate, not before it: a prompt that waited its
            # turn must see a refusal that arrived while it waited.
            if _DATAFORSEO_LOCKED:
                MODEL_ERRORS["ai_overview"] = _DATAFORSEO_LOCKED
                return ""
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    "https://api.dataforseo.com/v3/serp/google/organic/live/advanced",
                    auth=(DATAFORSEO_LOGIN, DATAFORSEO_PASSWORD),
                    json=[{"keyword": prompt,
                           "location_name": location,
                           "language_code": language,
                           "device": "desktop",
                           # Without this the overview is missing from the
                           # response even when Google shows one on the page.
                           "load_async_ai_overview": True}],
                )
        if resp.status_code >= 400:
            detail = f"{resp.status_code}: {resp.text[:200]}"
            print(f"AI Overview error {detail}")
            MODEL_ERRORS["ai_overview"] = detail
            if resp.status_code in (401, 402, 403, 429):
                _DATAFORSEO_LOCKED = detail
                print("AI Overview: credentials refused, not calling again this run")
            return ""
        data = resp.json()

        # DataForSEO reports per-task failures inside a 200, so the status code
        # alone would read a quota problem as "no overview for this query".
        tasks = data.get("tasks") or []
        if not tasks:
            MODEL_ERRORS["ai_overview"] = f"no tasks in response: {str(data)[:200]}"
            return ""
        task = tasks[0]
        if task.get("status_code", 20000) >= 40000:
            detail = f"{task.get('status_code')}: {task.get('status_message', '')[:200]}"
            print(f"AI Overview task failed {detail}")
            MODEL_ERRORS["ai_overview"] = detail
            # A per-task failure about the account, not about this query, must
            # stop the remaining prompts too.
            if task.get("status_code") in DATAFORSEO_FATAL:
                _DATAFORSEO_LOCKED = detail
                print("AI Overview: account refused, not calling again this run")
            return ""

        text, urls = _parse_ai_overview(task)
        MODEL_ERRORS.pop("ai_overview", None)
        if not text:
            print(f"AI Overview: Google showed none for {prompt[:60]!r}")
            return ""
        if urls:
            text += "\n\nSources: " + " ".join(urls)
        print(f"AI Overview OK: {len(text)} chars, {len(urls)} references")
        return text
    except Exception as e:
        print(f"AI Overview error: {e}")
        MODEL_ERRORS["ai_overview"] = str(e)[:300]
        return ""


def _parse_ai_overview(task: dict) -> tuple[str, list[str]]:
    """The overview's prose and the pages it cites, out of a SERP response.

    Read defensively: this is a parsed page, not a documented model output, and
    the shape changes when Google changes the block.
    """
    parts: list[str] = []
    urls: list[str] = []
    for result in (task.get("result") or []):
        for item in ((result or {}).get("items") or []):
            if not isinstance(item, dict) or item.get("type") != "ai_overview":
                continue
            for el in (item.get("items") or []):
                if not isinstance(el, dict):
                    continue
                chunk = el.get("markdown") or el.get("text") or ""
                if chunk:
                    parts.append(chunk)
                for link in (el.get("links") or []):
                    if isinstance(link, dict) and str(link.get("url", "")).startswith("http"):
                        urls.append(link["url"])
            for ref in (item.get("references") or []):
                if isinstance(ref, dict) and str(ref.get("url", "")).startswith("http"):
                    urls.append(ref["url"])
    return "\n\n".join(p for p in parts if p).strip(), list(dict.fromkeys(urls))


def search_tool(country: str = "") -> dict:
    """Web search results are location-influenced, so the same query returns
    different sources in London and Moscow. Without this the audit measures
    whatever market OpenAI defaults to, not the client's."""
    tool: dict = {"type": "web_search"}
    if country:
        tool["user_location"] = {"type": "approximate", "country": country.upper()}
    return tool


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

async def ask_openai(prompt: str, use_search: bool = False, country: str = "") -> str:
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
                tools=[search_tool(country)],
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


# Every model takes (prompt, country) so run_prompt does not have to know which
# one it is calling. Registered here, after the functions exist, and keyed by the
# registry — a model declared in MODEL_SWITCHES with no entry here would be
# selected, counted in the denominator and never asked anything, which is the
# one failure mode the registry exists to prevent.
ASK.update({
    "chatgpt": lambda prompt, country: ask_openai(prompt, use_search=True, country=country),
    "gemini": lambda prompt, country: ask_gemini(prompt),
    "perplexity": lambda prompt, country: ask_perplexity(prompt, country=country),
    "deepseek": lambda prompt, country: ask_deepseek(prompt, country=country),
    "ai_overview": lambda prompt, country: ask_ai_overview(prompt, country=country),
})
assert set(ASK) == set(ALL_MODELS), (
    f"registry and ASK disagree: {set(ALL_MODELS) ^ set(ASK)}")
assert set(MODEL_VERSIONS) == set(ALL_MODELS), (
    f"registry and MODEL_VERSIONS disagree: {set(ALL_MODELS) ^ set(MODEL_VERSIONS)}")


async def enrich_brand(brand: str, description: str = "", website: str = "") -> dict:
    """Work out which market this brand competes in.

    Order matters: the site is the most reliable source, a typed description
    next, the model's own recall last. Asking the model to recall a small brand
    it has never seen is how categories came back as "Unknown".
    """
    clean_brand = brand
    if brand.startswith("http"):
        clean_brand = extract_brand_from_url(brand)

    site_url = website or (brand if brand.startswith("http") else "")
    site_text, site_reason = await fetch_site_text(site_url)

    if site_text:
        source = "site"
        context = (f"Brand: {clean_brand}\n"
                   f"Text from their website:\n{site_text}")
        if description:
            context += f"\n\nWhat the owner says they do: {description}"
    elif description:
        source = "description"
        context = f"Brand: {clean_brand}\nDescription: {description}"
        if site_reason:
            print(f"  site not readable: {site_reason}")
    else:
        source = "model_knowledge"
        if site_reason:
            print(f"  site not readable: {site_reason}")
        context = (f"Brand: {clean_brand}\n"
                   "Identify this company from your own knowledge. If you do not "
                   'recognise it, set known to false and category to "Unknown".')

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
        data["original_brand"] = brand
        data["clean_brand"] = data.get("clean_name", clean_brand)
        data["category_source"] = source
        # A category we could not actually determine should say so rather than
        # quietly becoming a guess the prompts are then built on.
        if not data.get("category") or data["category"].lower() in ("unknown", "n/a"):
            data["category"] = "Unknown"
            data["needs_description"] = source != "description"
            data["site_issue"] = site_reason
        return data
    except Exception:
        return {"category": description or "Unknown",
                "known": False,
                "original_brand": brand,
                "clean_brand": clean_brand,
                "category_source": source,
                "needs_description": not description,
                "site_issue": site_reason}


PROMPT_CACHE: dict[str, list[dict]] = {}

# What the audit asks, and in what proportion. Only 'brand' style queries were
# being generated before, which is why the audit kept surfacing vendor docs
# instead of independent write-ups.


# The six kinds of query the audit asks, and in what proportion. Weights must
# add up to 1.0, and the LAST entry absorbs the rounding in split_counts(), so
# the smallest share goes last. At PROMPT_COUNT=20 this gives 5/4/4/3/2/2.
PROMPT_MIX = [
    ("comparison", 0.25),
    ("commercial", 0.20),
    ("problem",    0.20),
    ("brand",      0.15),
    ("vertical",   0.12),
    ("technical",  0.08),
]

# How those weights change with how well known the brand is.
#
# The weights are not cosmetic. In a broad comparison query the model answers
# with the most frequently written-about names, so a brand nobody writes about
# cannot appear there however good its site is. In a narrow query frequency
# stops helping, because almost nobody has written about the narrow thing — so
# whoever has the matching text wins. Measuring a mid-size brand mostly on
# broad comparisons therefore spends the audit on ground it cannot take.
#
# 'brand' rises as the brand gets smaller: for an unknown company the useful
# question is whether the model knows it at all, not where it places. 'problem'
# is flat at 20% everywhere — symptom-shaped queries do not care about size.
#
# Same key order in all three, so the same type always absorbs the rounding.
PROMPT_MIXES: dict[str, list[tuple[str, float]]] = {
    # Broad queries are winnable, so they keep real weight. This is PROMPT_MIX.
    "large": PROMPT_MIX,
    # Mid-size with little or no SEO: the model knows the brand but never
    # recalls it unprompted. Weight moves to the long tail.
    "mid": [
        ("comparison", 0.10),
        ("commercial", 0.15),
        ("problem",    0.20),
        ("brand",      0.15),
        ("vertical",   0.25),
        ("technical",  0.15),
    ],
    # Small or new: broad comparisons are close to unwinnable, so they get the
    # minimum, and the priority is establishing whether the model knows anything.
    "small": [
        ("comparison", 0.05),
        ("commercial", 0.10),
        ("problem",    0.20),
        ("brand",      0.20),
        ("vertical",   0.30),
        ("technical",  0.15),
    ],
}

# An audit that does not say the size gets this. Most brands buying visibility
# measurement are mid-size without SEO, and of the three profiles 'mid' is the
# least wrong in either direction.
DEFAULT_BRAND_SIZE = "mid"


def resolve_mix(size: str | None) -> list[tuple[str, float]]:
    """Pick a mix by size, tolerating whatever the client sends."""
    key = (size or "").strip().lower()
    key = {
        "big": "large", "enterprise": "large", "known": "large", "well-known": "large",
        "medium": "mid", "mid-size": "mid", "midsize": "mid",
        "startup": "small", "new": "small",
        "unknown": DEFAULT_BRAND_SIZE, "": DEFAULT_BRAND_SIZE,
    }.get(key, key)
    return PROMPT_MIXES.get(key, PROMPT_MIXES[DEFAULT_BRAND_SIZE])


# One entry per PROMPT_MIX key — _build_generation_prompt() indexes this
# directly, so a missing key is a KeyError at request time, not at import.
PROMPT_TYPE_GUIDE = {
    "comparison": (
        "Someone who knows the category and is weighing options against each other.\n"
        "Examples: best ai visibility tracking tools / "
        "ai visibility tools compared / "
        "cheapest way to monitor brand mentions in ai"
    ),
    "commercial": (
        "Someone ready to buy, asking about price, plans or how to start.\n"
        "Examples: how much does ai visibility tracking cost / "
        "ai brand monitoring free trial / "
        "is there a cheap ai visibility tool"
    ),
    "problem": (
        "Someone with the symptom who does not know this category of tool exists. "
        "They describe what is wrong, not what to buy.\n"
        "Examples: why does my brand not show up in chatgpt / "
        "how do i know what ai says about my company / "
        "my competitors appear in ai answers and i don't"
    ),
    "brand": (
        "Someone who already names a specific product in this category.\n"
        "Examples: profound alternatives / is otterly worth it / "
        "peec ai vs profound"
    ),
    "vertical": (
        "Someone asking about this category for a specific industry, company size "
        "or role.\n"
        "Examples: ai visibility tracking for ecommerce / "
        "brand monitoring for small agencies / "
        "ai seo tools for saas startups"
    ),
    "technical": (
        "Someone asking how it works or how it fits their stack — integrations, "
        "APIs, data, setup.\n"
        "Examples: ai visibility tool with api / "
        "how to track llm mentions programmatically / "
        "export ai search data to looker"
    ),
}


def split_counts(total: int, size: str | None = None) -> dict:
    """Turn the mix into whole numbers that add up to `total` exactly."""
    mix = resolve_mix(size)
    counts = {}
    assigned = 0
    for i, (name, share) in enumerate(mix):
        if i == len(mix) - 1:
            counts[name] = total - assigned      # last one absorbs the rounding
        else:
            n = max(1, round(total * share)) if total >= len(mix) else 0
            counts[name] = n
            assigned += n
    if counts[mix[-1][0]] < 0:
        counts[mix[-1][0]] = 0
    return counts


def _is_junk_prompt(p: str) -> bool:
    low = p.lower()
    if re.search(r'\b(x vs\.? y|\[.*?\]|<.*?>|tool a|brand a)\b', low):
        return True
    if len(p.split()) > 12 or len(p.split()) < 2:
        return True
    return False


def _build_generation_prompt(brand: str, category: str, counts: dict,
                            competitors: list[str] | None = None,
                            language: str = "English") -> str:
    blocks = []
    for name, n in counts.items():
        if n <= 0:
            continue
        blocks.append(f"{n} of type {name.upper()}:\n{PROMPT_TYPE_GUIDE[name]}")
    rivals = ""
    if competitors:
        # Without the real names the model invents plausible-sounding rivals,
        # and the brand-type prompts end up measuring companies that do not exist.
        rivals = ("Real competitors in this market: " + ", ".join(competitors[:6]) +
                  ". Use these names in BRAND queries, not invented ones.\n\n")
    return (
        f"Category: {category}\n\n"
        + rivals +
        f"Write search queries real people type into ChatGPT about this category.\n\n"
        + "\n\n".join(blocks) +
        "\n\nRules:\n"
        f"- Write every query in {language}. These are the words buyers in this "
        f"market actually type.\n"
        "- 3 to 10 words each. Plain lowercase, how people actually type.\n"
        "- NEVER use placeholders like 'X vs Y' or brackets.\n"
        f"- Do not mention {brand} in any query EXCEPT the BRAND ones - those are "
        f"about {brand} by definition.\n"
        "- Stay inside the stated category, do not drift to adjacent markets.\n\n"
        "Output format, one per line, nothing else. Use the label that matches "
        "the type you were asked for:\n"
        "COMMERCIAL: the query\n"
        "COMPARISON: the query\n"
        "PROBLEM: the query\n"
        "BRAND: the query\n"
        "VERTICAL: the query\n"
        "TECHNICAL: the query"
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
    """Used when generation returns nothing usable. Spread across types rather
    than collapsing into one, so a failed generation still produces a mix."""
    base = [
        {"text": f"Best {category} tools", "type": "commercial"},
        {"text": f"How much does {category} cost", "type": "commercial"},
        {"text": f"{category} compared", "type": "comparison"},
        {"text": f"Alternatives to leading {category} tools", "type": "comparison"},
        {"text": f"How to choose a {category} tool", "type": "problem"},
        {"text": f"What is {category} and why does it matter", "type": "problem"},
        {"text": f"{category} for small business", "type": "vertical"},
        {"text": f"{category} with open API", "type": "technical"},
    ]
    out = []
    while len(out) < total and base:
        out.extend(base[: total - len(out)])
    return out[:total]


async def generate_prompts(brand: str, category: str,
                           competitors: list[str] | None = None,
                           country: str = "US",
                           language: str = "",
                           brand_size: str = "") -> list[dict]:
    # Competitors change the brand-type prompts, so they belong in the key.
    # Market changes both the language and the rivals, so it belongs in the key.
    language = language or market_language(country)
    # Language is in the key as well as country: same market, different audience
    # language means different prompts.
    # Size belongs in the key too. Two brands in the same category at different
    # sizes get different mixes, and without this the first one audited would
    # serve its prompts to the second from cache.
    size_key = (brand_size or DEFAULT_BRAND_SIZE).strip().lower()
    cache_key = "|".join([category.strip().lower(), (country or "US").upper(), language,
                          size_key,
                          ",".join(sorted(c.lower() for c in (competitors or [])))])
    if cache_key in PROMPT_CACHE:
        print(f"Prompt cache hit: {cache_key}")
        return PROMPT_CACHE[cache_key]

    counts = split_counts(PROMPT_COUNT, brand_size)
    response = await ask_openai(_build_generation_prompt(brand, category, counts, competitors, language))
    prompts = _parse_generated(response, counts)

    if len(prompts) < 2:
        print("  prompt generation produced too little, using fallback")
        prompts = _fallback_prompts(category, PROMPT_COUNT)

    by_type = {}
    for p in prompts:
        by_type[p["type"]] = by_type.get(p["type"], 0) + 1
    print(f"  prompts by type ({size_key}): {by_type}")

    PROMPT_CACHE[cache_key] = prompts
    return prompts
@app.post("/check-brand", dependencies=[Depends(caller)])
async def check_brand(request: dict):
    brand = request.get("brand", "")
    info = await enrich_brand(brand)
    return {"known": info.get("known", False), "category": info.get("category", "")}

AUDIT_CACHE: dict = {}
AUDIT_CACHE_TTL = int(os.getenv("AUDIT_CACHE_TTL", "3600"))

async def discover_brands(results: list[dict], known: list[str], category: str) -> list[str]:
    """Product names the models mentioned that nobody asked us to look for.

    Run once over all answers rather than once per prompt: the extra call costs
    about half a cent, twenty would not be worth it. These are often the more
    useful finding - a rival the client did not know they had.
    """
    blob = "\n\n".join(
        (r["chatgpt"].get("answer") or "")[:1200] for r in results
    )[:14000]
    if not blob.strip():
        return []

    prompt = (
        f"Below are AI answers about: {category}.\n\n"
        "List the product, tool or company names that are recommended or compared "
        "in them. One per line, nothing else.\n"
        "Rules:\n"
        "- Real product names only. Not generic terms like 'CRM' or 'loyalty program'.\n"
        "- Write each name exactly as it appears in the text.\n"
        "- If there are none, reply NONE.\n\n"
        f"Answers:\n{blob}"
    )
    try:
        raw = await ask_openai(prompt)
    except Exception as exc:
        print(f"  brand discovery failed: {type(exc).__name__}")
        return []

    known_flat = {_strip_separators(k)[0] for k in known if k}
    found, seen = [], set()
    for line in (raw or "").split("\n"):
        name = line.strip().lstrip("-*0123456789. ").strip()
        if not name or name.upper() == "NONE" or len(name) > 40:
            continue
        flat = _strip_separators(name)[0]
        # Already tracked, or already found under a different spelling.
        if not flat or flat in known_flat or flat in seen:
            continue
        seen.add(flat)
        found.append(name)
    return found[:12]


def aggregate_audit(results: list[dict], running: list[str], brand: str,
                    clean_brand: str, competitors: list[str],
                    competitor_domains: dict, brand_domain: str) -> dict:
    """Every number the dashboard shows, derived from the results alone.

    Extracted so an audit and an extended audit cannot drift apart. It reads
    only fields that also reach the client, which is what makes recomputing a
    merged set possible at all: the raw answers are discarded before the
    response is sent, so a count that re-scans them could never be reproduced.

    Counting a competitor from `competitors_found` instead of re-scanning the
    raw answer gives identical numbers — process_model already did that scan,
    and this just stops repeating it.
    """
    # Every count below runs over the models that actually answered. It used to
    # run over the literal pair ("gemini", "chatgpt"), which meant a third model
    # could be added, could answer, could name the brand — and appear in none of
    # the numbers. The registry promises that adding a model takes four edits;
    # this is what has to be true for that promise to hold.
    counted = [m for m in running if m in ALL_MODELS]
    per_model = {m: sum(1 for r in results if r.get(m, {}).get("mentioned"))
                 for m in ALL_MODELS}
    named_count = sum(per_model[m] for m in counted)

    # Divide by answers, not by attempts.
    #
    # This used to be len(results) * len(running) — every prompt times every
    # model asked. A model that was selected and then failed returns an empty
    # answer for each prompt, and those empties were counted as measurements:
    # "the model did not name you" where the truth is "the model never spoke".
    # One model failing out of three took a third off the score, and off every
    # competitor's rate, without anything on screen saying the figure had moved
    # for that reason. The same mistake as dividing by the models that exist
    # rather than the models that ran, one level further in.
    answers_by_model = {
        m: sum(1 for r in results if (r.get(m, {}).get("answer") or "").strip())
        for m in counted
    }
    answered_models = [m for m in counted if answers_by_model[m]]
    total_checks = sum(answers_by_model.values())
    visibility_score = (int(named_count / total_checks * 100)
                        if total_checks > 0 else 0)

    mentions_with_link_count = sum(1 for r in results for m in counted
                                   if r.get(m, {}).get("mentioned_with_link"))
    mentions_without_link_count = sum(1 for r in results for m in counted
                                      if r.get(m, {}).get("mentioned_without_link"))

    def get_search_name(b):
        return extract_brand_from_url(b) if b.startswith("http") else b

    competitor_stats = []
    for b in [clean_brand] + competitors:
        search_b = get_search_name(b)
        is_you = b == brand or b == clean_brand
        if is_you:
            by_model = dict(per_model)
            with_link = mentions_with_link_count
            without_link = mentions_without_link_count
        else:
            by_model = {m: sum(1 for r in results
                               if search_b in r.get(m, {}).get("competitors_found", []))
                        for m in ALL_MODELS}
            with_link = sum(1 for r in results for m in counted
                            if search_b in r.get(m, {}).get("competitors_with_link", []))
            without_link = sum(1 for r in results for m in counted
                               if search_b in r.get(m, {}).get("competitors_without_link", []))
        total = sum(by_model[m] for m in counted)
        competitor_stats.append({
            "name": search_b,
            "is_your_brand": is_you,
            # Carried through so the page check can compare this brand's page
            # with the pages of the rivals that actually get named.
            "domain": (brand_domain if is_you else competitor_domains.get(search_b, "")),
            # The named fields stay for stored runs and the existing dashboard;
            # mentions_by_model is the one that grows with the registry.
            "gemini_mentions": by_model.get("gemini", 0),
            "chatgpt_mentions": by_model.get("chatgpt", 0),
            "mentions_by_model": by_model,
            "total_mentions": total,
            "mention_rate": round(total / total_checks * 100, 1) if total_checks > 0 else 0,
            "mentions_with_link": with_link,
            "mentions_without_link": without_link,
        })
    competitor_stats.sort(key=lambda x: x["total_mentions"], reverse=True)
    for i, stat in enumerate(competitor_stats):
        stat["rank"] = i + 1

    return {
        "visibility_score": visibility_score,
        # Two models have a named field each because stored runs and the
        # Supabase columns carry them. Everything beyond the pair lives in
        # model_scores, which is what a new model gets for free.
        "gemini_score": per_model.get("gemini", 0),
        "chatgpt_score": per_model.get("chatgpt", 0),
        "model_scores": {m: per_model[m] for m in counted},
        # The denominator itself, so the dashboard shows the figure this score
        # was computed from rather than recomputing it from prompts × models and
        # disagreeing whenever a model was asked and did not answer.
        "answers": total_checks,
        "models_answered": answered_models,
        "total_prompts": len(results),
        "mentions_score": mentions_with_link_count,
        "citations_score": mentions_without_link_count,
        "competitor_ranking": competitor_stats,
    }


@app.post("/audit")
async def run_audit(request: AuditRequest, who: Caller | None = Depends(within_quota)):
    brand = request.brand
    # Both shapes collapse to: a list of names, plus domains where we know them.
    if request.competitor_list:
        competitors = []
        competitor_domains = {}
        for c in request.competitor_list:
            # People paste a comma-separated list into one field out of habit.
            # Left as-is it becomes a single "brand" that appears in no answer,
            # and every competitor silently scores zero.
            names = [n.strip() for n in c.name.split(",") if n.strip()]
            sites = [w.strip() for w in c.website.split(",") if w.strip()]
            for i, n in enumerate(names):
                competitors.append(n)
                site = sites[i] if i < len(sites) else (sites[0] if len(sites) == 1 and len(names) == 1 else "")
                if site:
                    competitor_domains[n] = extract_domain(
                        site if site.startswith("http") else "https://" + site)
    else:
        competitors = [n.strip() for c in request.competitors
                       for n in c.split(",") if n.strip()]
        competitor_domains = {}
    description = request.description
    website = request.website.strip()

    cache_key = "|".join([brand.strip().lower(),
                          ",".join(sorted(c.lower() for c in competitors)),
                          website.lower(), (request.country or "US").upper(),
                          (request.language or ""),
                          # Size changes which prompts run, so the same brand at
                          # two sizes is two different audits.
                          (request.brand_size or "").strip().lower(),
                          # Two audits of the same brand on different models are
                          # different audits, not a cache hit.
                          ",".join(sorted(selected_models(request.models))),
                          "|".join(sorted(p.strip() for p in request.custom_prompts)),
                          "|".join(sorted(p.strip() for p in request.extra_prompts))])
    cached = AUDIT_CACHE.get(cache_key)
    if cached and (asyncio.get_event_loop().time() - cached[0]) < AUDIT_CACHE_TTL:
        print(f"Audit cache hit: {cache_key}")
        return {**cached[1], "from_cache": True}

    MODEL_ERRORS.clear()

    brand_info = await enrich_brand(brand, description, website)
    # An undetermined category should not silently become "<brand> category":
    # prompts built on a made-up market measure nothing.
    category = brand_info.get("category") or "Unknown"
    clean_brand = brand_info.get("clean_brand", brand)
    language = (request.language or "").strip() or market_language(request.country)

    custom = [p.strip() for p in request.custom_prompts if p.strip()][:MAX_CUSTOM_PROMPTS]
    extra = [p.strip() for p in request.extra_prompts if p.strip()][:MAX_EXTRA_PROMPTS]
    if custom:
        # Hand-written prompts replace generation entirely. The point of choosing
        # "manual" is to measure exactly these, so quietly topping them up with
        # generated ones would defeat it.
        print(f"  using {len(custom)} custom prompt(s), generation skipped")
        prompt_specs = [{"text": p, "type": "custom"} for p in custom]
    else:
        prompt_specs = await generate_prompts(clean_brand, category, competitors,
                                              request.country, language,
                                              request.brand_size)
    # Added prompts sit on top of whatever the branch above produced, in both
    # modes. A tracked query is watched ALONGSIDE the rest; replacing the whole
    # set with it was the old behaviour and it silently changed what the
    # visibility score was a percentage of.
    if extra:
        seen = {p["text"].strip().lower() for p in prompt_specs}
        added = 0
        for text in extra:
            if text.lower() in seen:
                continue          # already being measured; running it twice costs twice
            seen.add(text.lower())
            prompt_specs.append({"text": text, "type": "tracked"})
            added += 1
        print(f"  plus {added} tracked prompt(s), {len(prompt_specs)} in total")

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

    running = selected_models(request.models)
    print(f"  models for this run: {', '.join(running)}")

    async def _skip() -> str:
        """A model that was not selected returns nothing and costs nothing."""
        return ""

    async def run_prompt(prompt):
        """Every registered model, asked in parallel, keyed by name.

        Was a two-element gather naming gemini and chatgpt. A model added to the
        registry was then selected and counted but never actually asked, which
        would have scored it as a silent zero for every prompt.
        """
        answers = await asyncio.gather(*[
            ASK[m](prompt, request.country) if m in running else _skip()
            for m in ALL_MODELS
        ])
        return prompt, dict(zip(ALL_MODELS, answers))

    # The volume lookup is one request for the whole set and does not depend on
    # the answers, so it runs alongside them rather than adding to the wait.
    prompt_results, volumes = await asyncio.gather(
        asyncio.gather(*[run_prompt(p) for p in prompts]),
        keyword_volumes(prompts, request.country),
    )

    results = []
    all_urls = {}
    sample_quote = ""

    def process_model(prompt, answer, competitors):
        nonlocal all_urls
        answer_lower = answer.lower()
        name_mentioned = brand_in_text(answer, clean_brand)
        urls = extract_urls(answer)
        has_own_link = any(url_matches_name(u, brand_domain, clean_brand) for u in urls)

        competitors_with_link = []
        competitors_without_link = []
        for c in competitors:
            if not brand_in_text(answer, c):
                continue
            # A known domain is an exact test; without one we fall back to
            # matching the name against the domain, which is a guess.
            if any(url_matches_name(u, competitor_domains.get(c, ""), c) for u in urls):
                competitors_with_link.append(c)
            else:
                competitors_without_link.append(c)

        for u in urls:
            domain = extract_domain(u)
            if domain not in all_urls:
                all_urls[domain] = {"url": u, "domain": domain, "total": 0,
                                    "prompt": prompt,
                                    # A counter per registered model. The two
                                    # named ones are read by the stored-citation
                                    # merge in /audit/extend, so they keep their
                                    # keys whether or not those models ran.
                                    **{f"{m}_count": 0 for m in ALL_MODELS}}

        return {
            "name_mentioned": name_mentioned,
            "has_own_link": has_own_link,
            "urls": urls,
            "competitors_with_link": competitors_with_link,
            "competitors_without_link": competitors_without_link,
        }

    for prompt, answers in prompt_results:
        # One block per registered model, built the same way for all of them.
        # This used to be two hand-written blocks, which is why a third model
        # could be asked a question and still appear nowhere in the output.
        row: dict = {
            "prompt": prompt,
            "prompt_type": prompt_type_by_text.get(prompt, "comparison"),
        }
        # Absent when the phrase is too long to ask about or Google has no data
        # for it. Absent, not zero: zero would claim nobody searches it.
        vol = volumes.get(prompt.strip().lower())
        if vol:
            row["search_volume"] = vol["search_volume"]
            row["keyword_source"] = "exact"   # the prompt itself, not a rewrite
        for m in ALL_MODELS:
            answer = answers.get(m, "")
            facts = process_model(prompt, answer, competitors)

            for u in facts["urls"]:
                domain = extract_domain(u)
                all_urls[domain][f"{m}_count"] += 1
                all_urls[domain]["total"] += 1

            mentioned = facts["name_mentioned"]
            with_link = mentioned and facts["has_own_link"]
            if not sample_quote and mentioned:
                sample_quote = find_mention_sentence(answer, clean_brand) or sample_quote

            row[f"_{m}_raw"] = answer
            row[m] = {
                "mentioned": mentioned,
                "mentioned_with_link": with_link,
                "mentioned_without_link": mentioned and not with_link,
                "competitors_found": list(set(facts["competitors_with_link"]
                                              + facts["competitors_without_link"])),
                "competitors_with_link": facts["competitors_with_link"],
                "competitors_without_link": facts["competitors_without_link"],
                "answer": trim_answer(answer),
                "sources": split_sources(answer)[1][:12],
                "cited_domains": unique_domains(facts["urls"]),
            }
        results.append(row)

    totals = aggregate_audit(results, running, brand, clean_brand, competitors,
                             competitor_domains, brand_domain)
    gemini_mentions = totals["gemini_score"]
    openai_mentions = totals["chatgpt_score"]
    visibility_score = totals["visibility_score"]
    answers_count = totals["answers"]
    mentions_with_link_count = totals["mentions_score"]
    mentions_without_link_count = totals["citations_score"]
    competitor_stats = totals["competitor_ranking"]

    discovered = await discover_brands(results, competitors + [clean_brand], category)
    if discovered:
        print(f"  discovered {len(discovered)} brand(s) nobody asked about: {', '.join(discovered)}")
        for r in results:
            for model_key in ALL_MODELS:
                answer = r[model_key].get("answer") or ""
                r[model_key]["discovered_brands"] = [
                    d for d in discovered if brand_in_text(answer, d)
                ]
    else:
        for r in results:
            for model_key in ALL_MODELS:
                r[model_key]["discovered_brands"] = []

    citations = sorted(all_urls.values(), key=lambda x: x["total"], reverse=True)[:10]
    summary = "Brand: " + clean_brand + ", Category: " + category + ", Score: " + str(visibility_score) + "%"

    rec_response = "" if request.skip_recommendations else await ask_openai(
        "You are an AI visibility consultant. Give exactly 3 specific recommendations to improve " + clean_brand + " visibility in AI models. Category: " + category + """ Format each as:
PRIORITY: [High/Medium/Low]
ACTION: [specific action]
WHY: [one sentence]
EFFORT: [Easy/Medium/Hard]
Data: """ + summary)

    # Raw answers are working data: they are large, and the payload is stored.
    # Driven by the registry so a new model's raw text cannot be left behind.
    for r in results:
        for m in ALL_MODELS:
            r.pop(f"_{m}_raw", None)

    payload = {
        "brand": clean_brand,
        "category": category,
        "brand_domain": brand_domain,
        "category_source": brand_info.get("category_source", ""),
        "needs_description": brand_info.get("needs_description", False),
        "site_issue": brand_info.get("site_issue", ""),
        "country": (request.country or "US").upper(),
        "language": language,
        "run_at": datetime.now(timezone.utc).isoformat(),
        # Which exact model answered, so two runs made with different versions
        # are not read as comparable. Registry-driven for the same reason as
        # everything else here.
        "models_used": {m: MODEL_VERSIONS[m]() for m in ALL_MODELS},
        "keyword_volume": {
            "enabled": KEYWORD_VOLUME_ENABLED,
            "with_data": sum(1 for r in results if r.get("search_volume") is not None),
            "eligible": sum(1 for p in prompts if keyword_eligible(p)),
            "total": len(prompts),
        },
        "models_run": running,
        # What was asked, and what actually came back. A model can be in the
        # first list and absent from the second: selected, called, and silent.
        "models_answered": totals["models_answered"],
        "models_available": available_models(),
        # Built from the registry, so a model added later reports its status
        # without this block being touched. "did not run" and "ran and found
        # nothing" are different facts and must not look alike.
        "model_status": {
            m: {"ok": m in running and m not in MODEL_ERRORS,
                "enabled": MODEL_SWITCHES[m](),
                "selected": m in running,
                "error": (MODEL_ERRORS.get(m) if m in running
                          else "not selected for this audit")}
            for m in ALL_MODELS
        },
        "visibility_score": visibility_score,
        "answers": answers_count,
        "model_scores": totals["model_scores"],
        "gemini_score": gemini_mentions,
        "chatgpt_score": openai_mentions,
        "total_prompts": len(prompts),
        "mentions_score": mentions_with_link_count,
        "citations_score": mentions_without_link_count,
        "results": results,
        "competitor_ranking": competitor_stats,
        "citations": citations,
        "discovered_brands": discovered,
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

    got_answers = any(r[m].get("answer") for r in results for m in running)
    if "chatgpt" not in MODEL_ERRORS and got_answers:
        AUDIT_CACHE[cache_key] = (asyncio.get_event_loop().time(), payload)

    return payload

# ─────────────────────────────────────────────────────────────
# Extending an audit: measure a few more prompts and fold them in
# ─────────────────────────────────────────────────────────────

class ExtendRequest(AuditRequest):
    """An audit already seen, plus the prompts to add to it.

    previous_results are the result rows the caller already holds, exactly as
    /audit returned them. They are not re-measured and not re-charged.
    """
    previous_results: list[dict] = []
    previous_citations: list[dict] = []


@app.post("/audit/extend", dependencies=[Depends(caller)])
async def extend_audit(request: ExtendRequest):
    new_prompts = [p.strip() for p in request.extra_prompts if p.strip()][:MAX_EXTRA_PROMPTS]
    previous = request.previous_results or []

    if not new_prompts:
        return {"error": "no prompts to add"}
    if not previous:
        return {"error": "nothing to extend",
                "detail": "Send the results you already have, or run a normal audit."}

    # Skip anything already measured. Adding a prompt twice would charge twice
    # and weight it double in the score.
    seen = {r.get("prompt", "").strip().lower() for r in previous}
    todo = [p for p in new_prompts if p.lower() not in seen]
    if not todo:
        return {"error": "those prompts are already in this audit"}

    # Measure ONLY the added prompts, through the ordinary audit path so that a
    # prompt measured here is measured exactly as one measured there.
    sub = AuditRequest(**{**request.model_dump(
        exclude={"previous_results", "previous_citations", "extra_prompts", "custom_prompts"}),
        "custom_prompts": todo, "extra_prompts": [], "skip_recommendations": True})
    measured = await run_audit(sub)
    if not isinstance(measured, dict) or "results" not in measured:
        return {"error": "could not measure the added prompts", "detail": str(measured)[:300]}

    merged = previous + measured["results"]

    # The same function the ordinary audit uses, so an extended audit and a
    # fresh one of the same prompts cannot report different numbers.
    running = measured.get("models_run") or selected_models(request.models)
    competitors, competitor_domains = [], {}
    for c in request.competitor_list:
        for i, n in enumerate([x.strip() for x in c.name.split(",") if x.strip()]):
            competitors.append(n)
            sites = [w.strip() for w in c.website.split(",") if w.strip()]
            site = sites[i] if i < len(sites) else (sites[0] if len(sites) == 1 else "")
            if site:
                competitor_domains[n] = extract_domain(
                    site if site.startswith("http") else "https://" + site)
    if not competitors:
        competitors = [n.strip() for c in request.competitors for n in c.split(",") if n.strip()]

    clean_brand = measured.get("brand") or request.brand
    brand_domain = measured.get("brand_domain", "")
    totals = aggregate_audit(merged, running, request.brand, clean_brand,
                             competitors, competitor_domains, brand_domain)

    # Citation counts add up by domain; the list stays capped the same way.
    by_domain = {c["domain"]: dict(c) for c in (request.previous_citations or [])}
    for c in measured.get("citations", []):
        cur = by_domain.get(c["domain"])
        if cur:
            for k in ("gemini_count", "chatgpt_count", "total"):
                cur[k] = cur.get(k, 0) + c.get(k, 0)
        else:
            by_domain[c["domain"]] = dict(c)
    citations = sorted(by_domain.values(), key=lambda x: x.get("total", 0), reverse=True)[:10]

    print(f"  extended audit: {len(previous)} + {len(measured['results'])} = {len(merged)} prompts")

    return {**measured, **totals,
            "results": merged,
            "citations": citations,
            "added_prompts": todo,
            # The caller keeps its own recommendations: these were not regenerated.
            "recommendations": "",
            "extended_at": datetime.now(timezone.utc).isoformat()}


@app.post("/clear-cache", dependencies=[Depends(caller)])
async def clear_cache():
    n = len(AUDIT_CACHE) + len(PROMPT_CACHE)
    AUDIT_CACHE.clear()
    PROMPT_CACHE.clear()
    return {"cleared": n}

@app.get("/debug-prompt", dependencies=[Depends(caller)])
async def debug_prompt(q: str = "best AI visibility tracking tools", country: str = ""):
    try:
        response = await openai_client.responses.create(
            model=OPENAI_SEARCH_MODEL,
            tools=[search_tool(country)],
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


# ─────────────────────────────────────────────────────────────
# Page check: why the models do not name you, and what to change
#
# The visibility audit answers "are we named". This answers "why not", by
# reading the page the way a retrieval system reads it and comparing it with
# the pages of competitors the audit found ARE named.
#
# Everything here except the fetching is a pure function, so it can be tested
# on fixtures instead of on live sites.
# ─────────────────────────────────────────────────────────────

PAGE_CHECK_TIMEOUT = 12.0

# The crawlers that actually matter for being named in an assistant's answer.
# Grouped by who operates them, because a robots.txt fix is written per group.
AI_CRAWLERS = {
    "OpenAI": ["GPTBot", "OAI-SearchBot", "ChatGPT-User"],
    "Anthropic": ["ClaudeBot", "Claude-Web", "Claude-User", "Claude-SearchBot", "anthropic-ai"],
    "Perplexity": ["PerplexityBot", "Perplexity-User"],
    "Google": ["Google-Extended"],
    "Common Crawl": ["CCBot"],
}
AI_CRAWLER_FLAT = [a for group in AI_CRAWLERS.values() for a in group]


class _PageFacts(HTMLParser):
    """Everything about a page that affects whether it gets retrieved.

    Deliberately not just the text: where a phrase sits matters far more than
    how often it occurs, and that is exactly what the text alone cannot show.
    """

    SKIP = {"script", "style", "noscript", "svg"}
    HEADINGS = {"h1", "h2", "h3"}

    def __init__(self):
        super().__init__()
        self.title = ""
        self.meta_description = ""
        self.meta_robots = ""
        self.headings: list[tuple[str, str]] = []     # (tag, text)
        self.jsonld: list[str] = []
        self.body_parts: list[str] = []
        self._skip = 0
        self._capture = None                          # "title" | "h1".. | "jsonld"
        self._buf: list[str] = []

    def handle_starttag(self, tag, attrs):
        a = {k.lower(): (v or "") for k, v in attrs}
        if tag == "meta":
            name = a.get("name", "").lower()
            if name == "description" and not self.meta_description:
                self.meta_description = a.get("content", "").strip()
            elif name == "robots":
                self.meta_robots = a.get("content", "").strip().lower()
        elif tag == "script":
            if a.get("type", "").lower() == "application/ld+json":
                self._capture, self._buf = "jsonld", []
                return                                # capture, do not skip
            self._skip += 1
        elif tag in self.SKIP:
            self._skip += 1
        elif tag == "title":
            self._capture, self._buf = "title", []
        elif tag in self.HEADINGS:
            self._capture, self._buf = tag, []

    def handle_endtag(self, tag):
        if tag == "script" and self._capture == "jsonld":
            self.jsonld.append("".join(self._buf).strip())
            self._capture, self._buf = None, []
            return
        if tag in self.SKIP and self._skip:
            self._skip -= 1
            return
        if self._capture == tag or (tag == "title" and self._capture == "title"):
            text = " ".join(" ".join(self._buf).split())
            if tag == "title":
                self.title = text
            elif text:
                self.headings.append((tag, text))
            self._capture, self._buf = None, []

    def handle_data(self, data):
        if self._capture:
            self._buf.append(data)
        if not self._skip:
            t = data.strip()
            if t:
                self.body_parts.append(t)

    # ── derived ──
    def body_text(self) -> str:
        return " ".join(self.body_parts)

    def h1(self) -> str:
        return next((t for tag, t in self.headings if tag == "h1"), "")

    def questions(self) -> list[str]:
        return [t for _, t in self.headings if t.rstrip().endswith("?")]

    def has_faq_schema(self) -> bool:
        return any("faqpage" in b.lower() for b in self.jsonld)

    def schema_types(self) -> list[str]:
        found = []
        for block in self.jsonld:
            for m in re.finditer(r'"@type"\s*:\s*"([^"]+)"', block):
                if m.group(1) not in found:
                    found.append(m.group(1))
        return found


def _stem(word: str) -> str:
    """Crude stem so Russian cases match: программа / программы / программу.

    A real morphological analyser would be better, but it is a dependency and a
    deployment risk for a gain that does not change any verdict here — the
    endings we need to survive are two or three characters long.
    """
    w = word.lower().strip("«»\"'.,:;!?()[]")
    if len(w) <= 4:
        return w
    return w[:-2] if len(w) <= 7 else w[:-3]


def term_pattern(term: str) -> re.Pattern:
    """Match a multi-word term across inflections and a little word order slack."""
    stems = [re.escape(_stem(w)) for w in term.split() if w.strip()]
    if not stems:
        return re.compile(r"(?!)")                    # matches nothing
    return re.compile(r"\w*".join(f"{s}\\w*" for s in stems).replace(r"\w*\w*", r"\w*[\s\-]*"),
                      re.IGNORECASE | re.UNICODE)


def term_placement(facts: _PageFacts, terms: list[str]) -> dict:
    """Where the category term sits, which is what decides retrieval.

    Frequency is recorded but deliberately not scored: a page can name the
    category forty times in its case studies and still declare itself to be
    something else in the only three places that are read as a declaration.
    """
    pats = [(t, term_pattern(t)) for t in terms if t.strip()]
    body = facts.body_text()
    hit = lambda s: any(p.search(s or "") for _, p in pats)
    return {
        "in_title": hit(facts.title),
        "in_h1": hit(facts.h1()),
        "in_meta_description": hit(facts.meta_description),
        "in_any_heading": any(hit(t) for _, t in facts.headings),
        "body_occurrences": sum(len(p.findall(body)) for _, p in pats),
        "title": facts.title,
        "h1": facts.h1(),
        "meta_description": facts.meta_description,
    }


# ── robots.txt ──

def parse_robots(text: str) -> dict:
    """{'agents': {ua_lower: [(allow|disallow, path)]}, 'sitemaps': [...]}"""
    agents: dict[str, list[tuple[str, str]]] = {}
    sitemaps: list[str] = []
    current: list[str] = []
    for raw in (text or "").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        field, _, value = line.partition(":")
        field, value = field.strip().lower(), value.strip()
        if field == "user-agent":
            # Consecutive User-agent lines share one group of rules.
            if current and agents.get(current[-1]):
                current = []
            current.append(value.lower())
            agents.setdefault(value.lower(), [])
        elif field in ("allow", "disallow") and current:
            for ua in current:
                agents[ua].append((field, value))
        elif field == "sitemap":
            sitemaps.append(value)
    return {"agents": agents, "sitemaps": sitemaps}


def _robots_path_match(rule: str, path: str) -> int:
    """Length of the matched prefix, or -1. Supports * and a trailing $."""
    if rule == "":
        return -1                                      # an empty Disallow allows all
    anchored = rule.endswith("$")
    body = rule[:-1] if anchored else rule
    regex = "".join(".*" if ch == "*" else re.escape(ch) for ch in body)
    m = re.match(regex + ("$" if anchored else ""), path)
    return len(body) if m else -1


def robots_allows(parsed: dict, agent: str, path: str) -> tuple[bool, str]:
    """Longest-match wins, Allow wins ties — the convention crawlers follow.

    Returns (allowed, which_group). The group matters for the report: 'no rules
    for you, falling back to *' is a different situation from 'you are named'.
    """
    agents = parsed["agents"]
    key = agent.lower() if agent.lower() in agents else ("*" if "*" in agents else "")
    if not key:
        return True, "no robots.txt rules at all"
    best_allow = best_disallow = -1
    for kind, rule in agents[key]:
        n = _robots_path_match(rule, path)
        if n < 0:
            continue
        if kind == "allow":
            best_allow = max(best_allow, n)
        else:
            best_disallow = max(best_disallow, n)
    allowed = best_allow >= best_disallow
    return allowed, ("named explicitly" if key == agent.lower() else "falls back to *")


async def fetch_page_facts(client, url: str) -> tuple[_PageFacts | None, str]:
    try:
        r = await client.get(url, follow_redirects=True, timeout=PAGE_CHECK_TIMEOUT,
                             headers={"User-Agent": SOURCE_BROWSER_UA})
    except httpx.TimeoutException:
        return None, f"took longer than {PAGE_CHECK_TIMEOUT:.0f}s to respond"
    except httpx.ConnectError:
        return None, "could not connect"
    except Exception as exc:
        return None, f"fetch failed ({type(exc).__name__})"
    if r.status_code != 200:
        return None, _reason_for_status(r.status_code)
    facts = _PageFacts()
    try:
        facts.feed(r.text)
    except Exception:
        return None, "page markup could not be parsed"
    return facts, ""


async def fetch_robots(client, origin: str) -> tuple[dict, str]:
    try:
        r = await client.get(origin.rstrip("/") + "/robots.txt",
                             follow_redirects=True, timeout=PAGE_CHECK_TIMEOUT,
                             headers={"User-Agent": SOURCE_BROWSER_UA})
    except Exception as exc:
        return {"agents": {}, "sitemaps": []}, f"could not read robots.txt ({type(exc).__name__})"
    if r.status_code == 404:
        # Not an error: no robots.txt means everything is permitted.
        return {"agents": {}, "sitemaps": [], "missing": True}, ""
    if r.status_code != 200:
        return {"agents": {}, "sitemaps": []}, f"robots.txt returned {r.status_code}"
    return parse_robots(r.text), ""


# ── the checks ──

def check_self_declaration(place: dict, rivals: list[dict], brand: str,
                           terms: list[str], missed: list[dict]) -> dict | None:
    """The page says what it is in three places. Does it say the category?"""
    if place["in_title"] and place["in_h1"] and place["in_meta_description"]:
        return None
    term = terms[0] if terms else ""
    missing = [f for f, k in (("title", "in_title"), ("H1", "in_h1"),
                              ("meta description", "in_meta_description")) if not place[k]]
    rivals_with = [r["name"] for r in rivals if r["placement"].get("in_title")]

    proposed_title = f"{brand} — {term}".strip(" —")
    if place["title"] and term:
        tail = place["title"].split("—")[-1].strip()
        # Titles very often already end in the brand ("… — BATTLE START"). Carrying
        # that tail over would print the brand twice in the suggested title, and the
        # suggestion is meant to be pasted in as-is.
        if tail and name_to_slug(tail) != name_to_slug(brand):
            proposed_title = f"{brand} — {term}: {tail}"

    # The term can be absent from the body as well as the three declarations, and
    # then there is no contrast to draw: "appears 0 times in the body, but not in
    # the title" asserts a difference between two absences.
    n = place["body_occurrences"]
    if n:
        found = (f"'{term}' appears {n} time(s) in the body, "
                 f"but not in the {join_list(missing, 'or')}.")
    elif len(missing) == 3:
        found = (f"'{term}' does not appear anywhere on the page — not in the "
                 f"title, the H1, the meta description, or the body.")
    else:
        found = (f"'{term}' is missing from the {join_list(missing, 'and')}, "
                 f"and does not appear in the body either.")

    return {
        "id": "self_declaration",
        "severity": "high",
        # The facts the wording was derived from. A client renders its own
        # sentence from these; the English below is a fallback, not the source
        # of truth, so a Russian interface does not end up showing English.
        "facts": {"term": term, "body_occurrences": place["body_occurrences"],
                  "missing": missing},
        "title": "The page does not declare the category it competes in",
        "found": found,
        "why": ("A model answering a category question matches on what the page declares "
                "itself to be, not on how often a phrase occurs further down. Frequency in "
                "body copy and case studies does not substitute for the title, the H1 and "
                "the description — and a page can lead on those three with far fewer "
                "mentions overall and still be the one named."),
        "affects_prompts": [m["text"] for m in missed
                            if m.get("type") in ("comparison", "commercial", "vertical")][:6],
        "competitors_doing_it": rivals_with,
        "changes": [
            {"field": "title", "where": "<head><title>",
             "current": place["title"], "proposed": proposed_title},
            {"field": "h1", "where": "first <h1> on the page",
             "current": place["h1"], "proposed": term.capitalize() if term else ""},
            {"field": "meta_description", "where": '<meta name="description">',
             "current": place["meta_description"],
             "proposed": (f"{brand} — {term}. " + (place["meta_description"] or ""))[:300].strip()},
        ],
    }


def check_question_content(facts: _PageFacts, rivals: list[dict],
                           missed: list[dict]) -> dict | None:
    """Question-shaped prompts need question-shaped content to land on."""
    asked = [m for m in missed if m.get("type") in ("problem", "commercial", "technical")]
    if not asked:
        return None
    own_questions = facts.questions()
    if own_questions and facts.has_faq_schema():
        return None
    rivals_with = [r["name"] for r in rivals
                   if r["questions"] or r["has_faq_schema"]]
    return {
        "id": "question_content",
        "severity": "high" if not own_questions else "medium",
        "facts": {"own_questions": len(own_questions),
                  "has_faq_schema": facts.has_faq_schema(),
                  "lost_questions": len(asked)},
        "title": "No question-shaped content for the questions you are losing",
        "found": (f"{len(own_questions)} heading(s) on the page are phrased as a question"
                  + ("" if facts.has_faq_schema() else ", and there is no FAQPage markup")
                  + f". {len(asked)} of the prompts you were not named in are questions."),
        "why": ("A model answering a question prefers a passage that is itself that question "
                "followed by an answer. Prose that contains the answer without ever asking "
                "the question is harder to retrieve for it, which is why an FAQ outperforms "
                "a better-written page without one."),
        "affects_prompts": [m["text"] for m in asked][:8],
        "competitors_doing_it": rivals_with,
        "changes": [
            {"field": "faq", "where": "a new FAQ section, with FAQPage JSON-LD",
             "current": "; ".join(own_questions[:5]),
             # The questions are the prompts the brand actually lost, not invented ones.
             "proposed": "\n".join(m["text"] for m in asked[:8])},
        ],
    }


def check_ai_crawlers(parsed: dict, path: str, rivals: list[dict]) -> dict | None:
    blocked, unnamed = [], []
    for operator, agents in AI_CRAWLERS.items():
        for agent in agents:
            allowed, how = robots_allows(parsed, agent, path)
            if not allowed:
                blocked.append(f"{agent} ({operator})")
            elif how == "falls back to *":
                unnamed.append(agent)
    if not blocked and not unnamed:
        return None
    rivals_naming = [r["name"] for r in rivals if r.get("names_ai_crawlers")]
    lines = ["# Assistants that answer questions about your category",
             *[f"User-agent: {a}" for a in AI_CRAWLER_FLAT],
             f"Allow: {path}", ""]
    if not parsed.get("sitemaps"):
        lines.append("Sitemap: https://YOUR-DOMAIN/sitemap.xml")
    return {
        "id": "ai_crawlers",
        "severity": "high" if blocked else "low",
        "facts": {"blocked": blocked, "unnamed": len(unnamed),
                  "total_agents": len(AI_CRAWLER_FLAT),
                  "has_sitemap": bool(parsed.get("sitemaps"))},
        "title": ("AI crawlers are blocked from this page" if blocked
                  else ("No robots.txt rules name the AI crawlers" if len(unnamed) == len(AI_CRAWLER_FLAT)
                        else "Some AI crawlers have no rules of their own")),
        "found": (f"Blocked: {', '.join(blocked)}. " if blocked else "")
                 + (f"{len(unnamed)} AI crawler(s) have no rules of their own and fall back "
                    f"to the wildcard group. " if unnamed else "")
                 + ("No sitemap is declared." if not parsed.get("sitemaps") else ""),
        "why": ("A blocked crawler cannot be fixed by any amount of content work — it is the "
                "one failure that makes everything else pointless. Crawlers with no rules of "
                "their own are not blocked, so this is not urgent on its own; it matters "
                "because a wildcard group written for search engines can disallow paths for "
                "reasons that no longer apply, and because naming them is how competitors "
                "signal they are paying attention."),
        "affects_prompts": [],
        "competitors_doing_it": rivals_naming,
        "changes": [{"field": "robots.txt", "where": "/robots.txt",
                     "current": "", "proposed": "\n".join(lines)}],
    }


def check_js_rendered(facts: _PageFacts) -> dict | None:
    body = facts.body_text()
    if len(body) >= 400:
        return None
    return {
        "id": "js_rendered",
        "severity": "high",
        "facts": {"body_chars": len(body)},
        "title": "The page has almost no text until JavaScript runs",
        "found": f"Only {len(body)} characters of text are present in the HTML itself.",
        "why": ("Most crawlers that feed assistants do not run JavaScript, so they see an "
                "empty page. Nothing else on this list can help while this is true."),
        "affects_prompts": [],
        "competitors_doing_it": [],
        "changes": [{"field": "rendering", "where": "the build",
                     "current": f"{len(body)} characters server-rendered",
                     "proposed": "Server-render or prerender this page so the text is in the HTML."}],
    }


# ── endpoint ──

class KeywordResearchRequest(BaseModel):
    # One per line in the UI; split before it gets here.
    keywords: list[str] = []
    country: str = ""


async def keyword_difficulty(keywords: list[str], country: str = "") -> dict:
    """How hard each phrase is to rank for organically, 0-100.

    A different number from the `competition` that comes with search volume:
    that one counts advertisers bidding, this one estimates the chance of
    reaching the organic top ten. They are routinely confused, and a tool that
    shows one under the other's name is worse than a tool that shows neither.

    This endpoint requires a language, unlike the volume one, which must not be
    given one.
    """
    global _DATAFORSEO_LOCKED
    if not KEYWORD_VOLUME_ENABLED or _DATAFORSEO_LOCKED or not keywords:
        return {}
    location = DATAFORSEO_LOCATIONS.get((country or "").upper(), DATAFORSEO_DEFAULT_LOCATION)
    language = DATAFORSEO_LANGUAGES.get((country or "").upper(), DATAFORSEO_DEFAULT_LANGUAGE)
    try:
        async with _dataforseo_gate:
            if _DATAFORSEO_LOCKED:
                return {}
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    "https://api.dataforseo.com/v3/dataforseo_labs/google/bulk_keyword_difficulty/live",
                    auth=(DATAFORSEO_LOGIN, DATAFORSEO_PASSWORD),
                    json=[{"keywords": keywords[:1000],
                           "location_name": location,
                           "language_code": language}],
                )
        if resp.status_code >= 400:
            print(f"keyword difficulty {resp.status_code}: {resp.text[:200]}")
            if resp.status_code in (401, 402, 403, 429):
                _DATAFORSEO_LOCKED = f"{resp.status_code}: {resp.text[:200]}"
            return {}
        task = (resp.json().get("tasks") or [{}])[0]
        if task.get("status_code", 20000) >= 40000:
            print(f"keyword difficulty task failed: {task.get('status_code')} "
                  f"{task.get('status_message', '')[:200]}")
            if task.get("status_code") in DATAFORSEO_FATAL:
                _DATAFORSEO_LOCKED = (f"{task.get('status_code')}: "
                                      f"{task.get('status_message', '')[:200]}")
            return {}
        out: dict = {}
        for result in (task.get("result") or []):
            for item in ((result or {}).get("items") or []):
                if not isinstance(item, dict):
                    continue
                kw = (item.get("keyword") or "").strip().lower()
                kd = item.get("keyword_difficulty")
                # Absent rather than zero: 0 is a real difficulty score meaning
                # "trivial to rank for", not "we do not know".
                if kw and isinstance(kd, int):
                    out[kw] = kd
        return out
    except Exception as e:
        print(f"keyword difficulty error: {e}")
        return {}


@app.get("/markets", dependencies=[Depends(admin_only)])
def markets():
    """The markets keyword data can be asked for.

    Served rather than hardcoded in the dashboard so there is one list. Every
    other model list in this codebase was duplicated into the frontend at some
    point, and every one of them drifted.
    """
    return {"markets": sorted(
        ({"code": code,
          "name": name,
          "language": DATAFORSEO_LANGUAGES.get(code, DATAFORSEO_DEFAULT_LANGUAGE)}
         for code, name in DATAFORSEO_LOCATIONS.items()
         # UK and GB are the same place under two codes; keep the ISO one.
         if code != "UK"),
        key=lambda m: m["name"])}


@app.post("/keyword-research", dependencies=[Depends(admin_only)])
async def keyword_research(request: KeywordResearchRequest):
    """Volume and difficulty for a list of phrases, for planning work.

    Separate from an audit: this answers "is this phrase worth going after",
    which is the question before "does a model name us for it".
    """
    seen, keywords = set(), []
    for k in request.keywords:
        k = (k or "").strip()
        low = k.lower()
        if k and low not in seen:
            seen.add(low)
            keywords.append(k)
    if not keywords:
        return {"error": "no keywords"}
    if not KEYWORD_VOLUME_ENABLED:
        return {"error": "keyword data is switched off",
                "detail": "Set KEYWORD_VOLUME_ENABLED and the DataForSEO credentials."}

    too_long = [k for k in keywords if not keyword_eligible(k)]
    askable = [k for k in keywords if keyword_eligible(k)]

    # Two endpoints, two prices, run together.
    volumes, difficulty = await asyncio.gather(
        keyword_volumes(askable, request.country),
        keyword_difficulty(askable, request.country),
    )

    rows = []
    for k in keywords:
        low = k.strip().lower()
        v = volumes.get(low) or {}
        rows.append({
            "keyword": k,
            "search_volume": v.get("search_volume"),
            "keyword_difficulty": difficulty.get(low),
            "cpc": v.get("cpc"),
            # Advertiser competition, kept under its own name so it is never
            # read as ranking difficulty.
            "ad_competition": v.get("competition"),
            "too_long": k in too_long,
        })
    rows.sort(key=lambda r: (r["search_volume"] is None, -(r["search_volume"] or 0)))
    return {"results": rows,
            "country": (request.country or "US").upper(),
            "error": _DATAFORSEO_LOCKED or ""}


class PageCheckRequest(BaseModel):
    # The page to check. Not the domain: a loyalty product competes as a page.
    website: str
    brand: str
    # Words a buyer uses for the category. Falls back to splitting `category`.
    category: str = ""
    category_terms: list[str] = []
    competitors: list[Competitor] = []
    # The prompts the brand was NOT named in, with their type. This is what
    # turns generic advice into "add these four questions": the FAQ we propose
    # is built from questions the brand actually lost, not invented ones.
    missed_prompts: list[dict] = []


def _origin_and_path(url: str) -> tuple[str, str]:
    if not url.startswith("http"):
        url = "https://" + url
    m = re.match(r"(https?://[^/]+)(/.*)?$", url)
    if not m:
        return url.rstrip("/"), "/"
    return m.group(1), (m.group(2) or "/")


def _terms_for(req: "PageCheckRequest") -> list[str]:
    if req.category_terms:
        return [t.strip() for t in req.category_terms if t.strip()]
    cat = (req.category or "").strip()
    return [cat] if cat else []


async def _rival_profile(client, name: str, url: str, terms: list[str]) -> dict:
    facts, reason = await fetch_page_facts(client, url if url.startswith("http") else "https://" + url)
    if not facts:
        return {"name": name, "url": url, "unreadable": reason,
                "placement": {}, "questions": [], "has_faq_schema": False,
                "names_ai_crawlers": False}
    origin, path = _origin_and_path(url)
    robots, _ = await fetch_robots(client, origin)
    named = any(a.lower() in robots.get("agents", {}) for a in AI_CRAWLER_FLAT)
    return {"name": name, "url": url, "unreadable": "",
            "placement": term_placement(facts, terms),
            "questions": facts.questions(),
            "has_faq_schema": facts.has_faq_schema(),
            "names_ai_crawlers": named}


@app.post("/page-check", dependencies=[Depends(caller)])
async def page_check(request: PageCheckRequest):
    """Why the models do not name this page, and the exact change to make.

    Deliberately separate from /audit: it is far cheaper, and it is meant to be
    re-run after a fix. Re-running it is how a before-and-after gets measured
    without paying for a whole audit again.
    """
    terms = _terms_for(request)
    if not terms:
        return {"error": "no category to check against",
                "detail": "Pass category or category_terms — without them there is "
                          "nothing to look for on the page."}
    if not request.website.strip():
        return {"error": "no page to check"}

    url = request.website.strip()
    if not url.startswith("http"):
        url = "https://" + url
    origin, path = _origin_and_path(url)

    async with httpx.AsyncClient() as client:
        facts, reason = await fetch_page_facts(client, url)
        if not facts:
            return {"error": "could not read the page", "detail": reason, "url": url}

        robots, robots_reason = await fetch_robots(client, origin)

        # A page with no server-rendered text makes checks 1 and 2 measure
        # nothing, so it is reported alone rather than alongside them.
        gate = check_js_rendered(facts)
        if gate:
            return {"url": url, "brand": request.brand, "terms": terms,
                    "blocked_by": gate["id"], "findings": [gate],
                    "competitors": [], "robots_note": robots_reason}

        rivals = []
        for c in request.competitors[:5]:
            if c.website and c.website.strip():
                rivals.append(await _rival_profile(client, c.name, c.website.strip(), terms))

    readable = [r for r in rivals if not r["unreadable"]]
    place = term_placement(facts, terms)
    missed = [m for m in request.missed_prompts if m.get("text")]

    findings = [f for f in (
        check_self_declaration(place, readable, request.brand, terms, missed),
        check_question_content(facts, readable, missed),
        check_ai_crawlers(robots, path, readable),
    ) if f]
    order = {"high": 0, "medium": 1, "low": 2}
    findings.sort(key=lambda f: order.get(f["severity"], 3))

    return {
        "url": url,
        "brand": request.brand,
        "terms": terms,
        "page": {"title": facts.title, "h1": facts.h1(),
                 "meta_description": facts.meta_description,
                 "questions": facts.questions(),
                 "schema_types": facts.schema_types(),
                 "body_chars": len(facts.body_text())},
        "placement": place,
        "robots": {"sitemaps": robots.get("sitemaps", []),
                   "named_ai_agents": [a for a in AI_CRAWLER_FLAT
                                       if a.lower() in robots.get("agents", {})],
                   "note": robots_reason},
        "findings": findings,
        "competitors": [{"name": r["name"], "url": r["url"], "unreadable": r["unreadable"],
                         "in_title": r["placement"].get("in_title", False),
                         "in_h1": r["placement"].get("in_h1", False),
                         "in_meta_description": r["placement"].get("in_meta_description", False),
                         "questions": len(r["questions"]),
                         "has_faq_schema": r["has_faq_schema"],
                         "names_ai_crawlers": r["names_ai_crawlers"]} for r in rivals],
    }


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
        # Reddit answers 200 with an HTML "Blocked" page rather than an error,
        # so a JSON parse failure here almost always means we were blocked.
        return "", "", ("reddit blocked the request - add REDDIT_CLIENT_ID and "
                        "REDDIT_CLIENT_SECRET in Railway to read it")
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
    """Same rule as the audit uses, so a page and an answer are judged alike."""
    return brand_in_text(text, name)


class SourceGapRequest(BaseModel):
    urls: list[str]
    brand: str
    competitors: list[str] = []
    brand_domain: str = ""


@app.post("/source-gap", dependencies=[Depends(caller)])
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
@app.get("/health-models", dependencies=[Depends(caller)])
async def health_models():
    """Quick check of which AI models actually respond. Use this to tell a real
    'low visibility' result apart from 'the model never answered'."""
    MODEL_ERRORS.clear()
    probe = "What is the best project management software? Name two tools."
    # Every registered model, including the ones switched off — a disabled model
    # reporting "disabled" here is the answer the endpoint exists to give.
    texts = await asyncio.gather(*[ASK[m](probe, "") for m in ALL_MODELS])
    return {
        m: {
            "ok": bool(text),
            "enabled": MODEL_SWITCHES[m](),
            "chars": len(text),
            "model": MODEL_VERSIONS[m](),
            "urls_found": len(split_sources(text)[1]),
            "error": MODEL_ERRORS.get(m),
        }
        for m, text in zip(ALL_MODELS, texts)
    }

@app.get("/")
def root():
    # Open on purpose: a health check that needs a token is not a health check.
    # Reporting whether the door is locked saves guessing after a deploy.
    return {"status": "GetCited API is running", "auth_required": REQUIRE_AUTH}
