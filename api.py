from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
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

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class AuditRequest(BaseModel):
    brand: str
    competitors: list[str] = []

@app.post("/audit")
async def run_audit(request: AuditRequest):
    brand = request.brand
    competitors = request.competitors

    prompts = [
        f"what is the best tool for {brand}'s category?",
        f"top alternatives to {brand}",
        f"best software similar to {brand}",
        f"recommend a tool like {brand}",
        f"{brand} competitors and alternatives",
    ]

    results = []
    for prompt in prompts:
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt
        )
        answer = response.text.lower()
        brand_mentioned = brand.lower() in answer
        competitors_found = [c for c in competitors if c.lower() in answer]
        results.append({
            "prompt": prompt,
            "brand_mentioned": brand_mentioned,
            "competitors_found": competitors_found,
        })

    mentions = sum(1 for r in results if r["brand_mentioned"])
    visibility_score = int(mentions / len(prompts) * 100)

    summary = f"""
    Brand: {brand}
    Visibility score: {visibility_score}%
    Competitors found: {competitors}
    Missed prompts: {[r['prompt'] for r in results if not r['brand_mentioned']]}
    """

    rec_response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=f"""You are an AI visibility consultant.
        Give exactly 3 specific recommendations to improve {brand}'s visibility in AI models.
        Format each as:
        PRIORITY: [High/Medium/Low]
        ACTION: [specific action]
        WHY: [one sentence]
        EFFORT: [Easy/Medium/Hard]
        
        Data: {summary}"""
    )

    return {
        "brand": brand,
        "visibility_score": visibility_score,
        "total_prompts": len(prompts),
        "mentions": mentions,
        "results": results,
        "recommendations": rec_response.text,
    }

@app.get("/")
def root():
    return {"status": "GetCited API is running"}