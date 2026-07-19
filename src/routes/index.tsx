import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
import { ArrowRight, Check, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

export const Route = createFileRoute("/")({
  component: Landing,
});

const BRAND = "GetCited";

const AI_LOGOS = [
  "ChatGPT", "Claude", "Gemini", "Perplexity",
  "YandexGPT", "AI Overview", "Copilot", "Mistral", "Grok",
];

function Reveal({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const [shown, setShown] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(
      ([e]) => { if (e.isIntersecting) { setShown(true); io.disconnect(); } },
      { threshold: 0.15 },
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);
  return (
    <div ref={ref} className={`transition-all duration-700 ease-out ${shown ? "translate-y-0 opacity-100" : "translate-y-6 opacity-0"} ${className}`}>
      {children}
    </div>
  );
}

function Landing() {
  const [brand, setBrand] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const startAudit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!brand.trim()) return;
    setLoading(true);
    setError("");
    try {
      const res = await fetch("https://web-production-b2168.up.railway.app/audit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ brand: brand.trim(), competitors: [] }),
      });
      const data = await res.json();
      navigate({
        to: "/dashboard",
        search: {
          brand: data.brand,
          score: data.visibility_score,
          results: JSON.stringify(data.results),
          recommendations: data.recommendations,
        } as never,
      });
    } catch {
      setError("Something went wrong. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-white text-neutral-900">
      <header className="sticky top-0 z-40 border-b border-neutral-100 bg-white/80 backdrop-blur">
        <div className="mx-auto grid max-w-6xl grid-cols-[minmax(0,1fr)_auto] items-center gap-4 px-6 py-4 md:grid-cols-3">
          <Link to="/" className="flex items-center gap-2">
            <div className="flex h-7 w-7 items-center justify-center rounded-md bg-neutral-900 text-white">
              <span className="text-sm font-bold">G</span>
            </div>
            <span className="text-lg font-semibold tracking-tight">{BRAND}</span>
          </Link>
          <nav className="hidden items-center justify-center gap-8 text-sm text-neutral-600 md:flex">
            <a href="#how" className="hover:text-neutral-900">How it works</a>
            <a href="#pricing" className="hover:text-neutral-900">Pricing</a>
            <a href="#blog" className="hover:text-neutral-900">Blog</a>
          </nav>
          <div className="flex items-center justify-end gap-2">
            <div className="mr-2 hidden items-center gap-1 text-xs text-neutral-500 sm:flex">
              <Link to="/" className="font-semibold text-neutral-900">🇬🇧 EN</Link>
              <span className="text-neutral-300">|</span>
              <Link to="/ru" className="hover:text-neutral-900">🇷🇺 RU</Link>
            </div>
            <Button asChild variant="ghost" size="sm" className="hidden sm:inline-flex">
              <Link to="/dashboard">Sign In</Link>
            </Button>
            <Button asChild size="sm" className="bg-[var(--brand)] text-white hover:bg-[var(--brand)]/90">
              <Link to="/dashboard">Start Free</Link>
            </Button>
          </div>
        </div>
      </header>

      <style>{`:root{--brand:#5B4BFF;}`}</style>

      <section className="mx-auto max-w-5xl px-6 pt-20 pb-16 text-center md:pt-28">
        <Reveal>
          <span className="inline-flex items-center gap-2 rounded-full border border-neutral-200 bg-neutral-50 px-3 py-1 text-xs text-neutral-600">
            <span className="h-1.5 w-1.5 rounded-full bg-[var(--brand)]" />
            Now tracking ChatGPT, Claude, Gemini & YandexGPT
          </span>
        </Reveal>
        <Reveal>
          <h1 className="mx-auto mt-6 max-w-4xl text-balance text-5xl font-bold tracking-tight text-neutral-900 md:text-7xl">
            See how often AI recommends your brand
          </h1>
        </Reveal>
        <Reveal>
          <p className="mx-auto mt-6 max-w-2xl text-lg text-neutral-600 md:text-xl">
            Get actionable recommendations to outflank your competitors.
          </p>
        </Reveal>
        <Reveal>
          <form onSubmit={startAudit} className="mx-auto mt-10 flex w-full max-w-xl flex-col gap-3 sm:flex-row">
            <Input
              value={brand}
              onChange={(e) => setBrand(e.target.value)}
              placeholder="Enter your brand name"
              className="h-12 flex-1 border-neutral-200 bg-white text-base"
            />
            <Button
              type="submit"
              size="lg"
              disabled={loading}
              className="h-12 bg-[var(--brand)] px-6 text-white hover:bg-[var(--brand)]/90"
            >
              {loading ? "Running audit..." : "Start Free Audit"}
              {!loading && <ArrowRight className="ml-1 h-4 w-4" />}
            </Button>
          </form>
          {error && <p className="mt-2 text-sm text-red-500">{error}</p>}
          <p className="mt-4 text-sm text-neutral-500">Free to start · no credit card</p>
        </Reveal>
      </section>

      <section className="border-y border-neutral-100 bg-white py-8">
        <div className="mb-4 text-center text-xs uppercase tracking-widest text-neutral-400">
          Tracking visibility across
        </div>
        <div className="relative overflow-hidden">
          <div className="flex w-max animate-[marquee_30s_linear_infinite] gap-14 pr-14">
            {[...AI_LOGOS, ...AI_LOGOS].map((logo, i) => (
              <span key={i} className="whitespace-nowrap text-xl font-semibold tracking-tight text-neutral-400">
                {logo}
              </span>
            ))}
          </div>
        </div>
        <style>{`@keyframes marquee{from{transform:translateX(0)}to{transform:translateX(-50%)}}`}</style>
      </section>

      <section id="how" className="bg-neutral-50/50 py-24 md:py-32">
        <div className="mx-auto max-w-6xl px-6">
          <Reveal>
            <h2 className="max-w-3xl text-balance text-4xl font-bold tracking-tight md:text-5xl">
              See {BRAND} in action
            </h2>
            <p className="mt-4 max-w-2xl text-lg text-neutral-600">
              Six ways to understand — and grow — your presence inside AI answers.
            </p>
          </Reveal>
          <div className="mt-20 space-y-24 md:space-y-32">
            <FeatureRow index={1} title="How often AI mentions your brand vs competitors" description="Track your mention rate as a percentage across all major AI models. See exactly where competitors outrank you." visual={<MentionRateVisual />} />
            <FeatureRow index={2} title="Which prompts your buyers are using — and where you're missing" description="Discover the exact questions people ask AI in your category. See which topics competitors cover that you don't." visual={<PromptListVisual />} flip />
            <FeatureRow index={3} title="Which AI models mention you most — and which ignore you" description="ChatGPT vs Claude vs Gemini vs YandexGPT — your visibility score per model, side by side." visual={<ModelCardsVisual />} />
            <FeatureRow index={4} title="Which pages on your site AI actually reads — and why" description="Not all your content gets cited by AI. See which pages appear in AI answers and get a score for each." visual={<PagesVisual />} flip />
            <FeatureRow index={5} title="How many visitors from AI became customers" description="Track conversions from AI referral traffic. Know which AI channels actually drive revenue." visual={<FunnelVisual />} badge="Coming Soon" />
            <FeatureRow index={6} title="Actionable recommendations — automated + expert audit" description="Every page gets auto-recommendations. Upgrade for a personal audit and expert call." visual={<RecommendationVisual />} flip />
          </div>
        </div>
      </section>

      <section id="pricing" className="mx-auto max-w-6xl px-6 py-24 md:py-32">
        <Reveal>
          <div className="mx-auto max-w-2xl text-center">
            <h2 className="text-balance text-4xl font-bold tracking-tight md:text-5xl">Simple pricing</h2>
            <p className="mt-4 text-lg text-neutral-600">Start free. Upgrade when you're ready.</p>
          </div>
        </Reveal>
        <div className="mt-16 grid gap-6 md:grid-cols-3">
          <PricingCard name="Free Trial" price="$0" period="3 days" features={["1 brand", "10 prompts", "2 AI models", "1 run"]} cta="Start Free" />
          <PricingCard name="Starter" price="$9" period="/month" features={["1 brand", "20 prompts", "3 AI models", "4 runs / month"]} cta="Get Started" />
          <PricingCard name="Pro" price="$29" period="/month" features={["5 brands", "50 prompts", "5 AI models", "12 runs / month", "CSV export", "Team seats"]} cta="Go Pro" highlighted />
        </div>
      </section>

      <footer className="border-t border-neutral-100 bg-white">
        <div className="mx-auto max-w-6xl px-6 py-12">
          <div className="flex flex-col items-start justify-between gap-6 md:flex-row md:items-center">
            <div className="flex items-center gap-2">
              <div className="flex h-7 w-7 items-center justify-center rounded-md bg-neutral-900 text-white">
                <span className="text-sm font-bold">G</span>
              </div>
              <span className="text-lg font-semibold tracking-tight">{BRAND}</span>
            </div>
            <nav className="flex flex-wrap gap-6 text-sm text-neutral-600">
              <a href="#how" className="hover:text-neutral-900">How it works</a>
              <a href="#pricing" className="hover:text-neutral-900">Pricing</a>
              <a href="#blog" className="hover:text-neutral-900">Blog</a>
              <Link to="/dashboard" className="hover:text-neutral-900">Sign In</Link>
            </nav>
          </div>
          <div className="mt-8 flex flex-col items-start justify-between gap-2 border-t border-neutral-100 pt-6 text-sm text-neutral-500 md:flex-row md:items-center">
            <span>Free to start · no credit card · set up in minutes</span>
            <span>© 2026 {BRAND}</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

function FeatureRow({ index, title, description, visual, flip, badge }: { index: number; title: string; description: string; visual: React.ReactNode; flip?: boolean; badge?: string; }) {
  return (
    <Reveal>
      <div className="grid items-center gap-10 md:grid-cols-2 md:gap-16">
        <div className={flip ? "md:order-2" : ""}>
          <div className="text-sm font-mono text-[var(--brand)]">{String(index).padStart(2, "0")}</div>
          <h3 className="mt-3 text-3xl font-bold tracking-tight md:text-4xl">{title}</h3>
          <p className="mt-4 text-lg text-neutral-600">{description}</p>
          {badge && <span className="mt-4 inline-block rounded-full border border-neutral-200 bg-white px-3 py-1 text-xs font-medium text-neutral-600">{badge}</span>}
        </div>
        <div className={flip ? "md:order-1" : ""}>
          <div className="rounded-2xl border border-neutral-200 bg-white p-6 shadow-sm">{visual}</div>
        </div>
      </div>
    </Reveal>
  );
}

function MentionRateVisual() {
  const rows = [{ name: "Your brand", pct: 34, you: true }, { name: "Competitor A", pct: 67 }, { name: "Competitor B", pct: 45 }, { name: "Competitor C", pct: 22 }];
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between text-xs text-neutral-500"><span>Mention rate</span><span>Last 30 days</span></div>
      {rows.map((r) => (
        <div key={r.name}>
          <div className="mb-1 flex justify-between text-sm">
            <span className={r.you ? "font-semibold text-neutral-900" : "text-neutral-600"}>{r.name}</span>
            <span className="font-mono text-neutral-500">{r.pct}%</span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-neutral-100">
            <div className={`h-full rounded-full ${r.you ? "bg-[var(--brand)]" : "bg-neutral-300"}`} style={{ width: `${r.pct}%` }} />
          </div>
        </div>
      ))}
    </div>
  );
}

function PromptListVisual() {
  const prompts = [{ q: "Best CRM for small teams", covered: false }, { q: "Alternatives to Salesforce", covered: true }, { q: "Cheapest sales tools 2026", covered: false }, { q: "AI sales assistants", covered: true }, { q: "How to track pipeline", covered: false }];
  return (
    <div className="space-y-2">
      <div className="mb-2 flex items-center justify-between text-xs text-neutral-500"><span>Buyer prompts</span><span>You vs Competitor A</span></div>
      {prompts.map((p) => (
        <div key={p.q} className="flex items-center justify-between rounded-lg border border-neutral-100 bg-neutral-50/60 px-3 py-2.5 text-sm">
          <span className="truncate text-neutral-800">{p.q}</span>
          {p.covered ? <span className="flex items-center gap-2 text-emerald-600"><Check className="h-4 w-4" /><span className="text-xs">Covered</span></span> : <span className="flex items-center gap-2 text-red-500"><X className="h-4 w-4" /><span className="text-xs">Missing</span></span>}
        </div>
      ))}
    </div>
  );
}

function ModelCardsVisual() {
  const models = [{ name: "ChatGPT", score: 62 }, { name: "Claude", score: 41 }, { name: "Gemini", score: 28 }, { name: "YandexGPT", score: 12 }];
  return (
    <div className="grid grid-cols-2 gap-3">
      {models.map((m) => (
        <div key={m.name} className="rounded-xl border border-neutral-100 bg-neutral-50/60 p-4">
          <div className="text-xs text-neutral-500">{m.name}</div>
          <div className="mt-2 text-3xl font-bold tracking-tight">{m.score}</div>
          <div className="mt-1 text-xs text-neutral-500">visibility score</div>
        </div>
      ))}
    </div>
  );
}

function PagesVisual() {
  const pages = [{ url: "/pricing", score: 82 }, { url: "/blog/best-crm", score: 74 }, { url: "/features", score: 41 }, { url: "/about", score: 12 }];
  return (
    <div className="space-y-2">
      <div className="mb-2 flex items-center justify-between text-xs text-neutral-500"><span>Page</span><span>AI citation score</span></div>
      {pages.map((p) => (
        <div key={p.url} className="flex items-center justify-between rounded-lg border border-neutral-100 bg-neutral-50/60 px-3 py-2.5">
          <span className="font-mono text-sm text-neutral-800">{p.url}</span>
          <span className="text-sm font-semibold text-neutral-900">{p.score}</span>
        </div>
      ))}
    </div>
  );
}

function FunnelVisual() {
  const steps = [{ label: "AI referral", val: 12400 }, { label: "Visit", val: 8100 }, { label: "Conversion", val: 340 }];
  const max = steps[0].val;
  return (
    <div className="space-y-3">
      <div className="mb-2 text-xs text-neutral-500">AI-driven conversions</div>
      {steps.map((s) => (
        <div key={s.label}>
          <div className="mb-1 flex justify-between text-sm"><span className="text-neutral-700">{s.label}</span><span className="font-mono text-neutral-500">{s.val.toLocaleString()}</span></div>
          <div className="h-3 overflow-hidden rounded-md bg-neutral-100"><div className="h-full rounded-md bg-[var(--brand)]" style={{ width: `${(s.val / max) * 100}%` }} /></div>
        </div>
      ))}
    </div>
  );
}

function RecommendationVisual() {
  return (
    <div className="space-y-3">
      <div className="rounded-xl border border-neutral-100 bg-neutral-50/60 p-4">
        <div className="flex items-center justify-between">
          <span className="rounded-full bg-red-50 px-2 py-0.5 text-xs font-semibold text-red-600">Priority: High</span>
          <span className="text-xs text-neutral-500">Effort · Low</span>
        </div>
        <p className="mt-3 text-sm font-medium text-neutral-900">Add a comparison table to /pricing — 3 competitors cite theirs when ChatGPT is asked "best X alternatives".</p>
      </div>
      <div className="rounded-xl border border-neutral-100 bg-neutral-50/60 p-4">
        <div className="flex items-center justify-between">
          <span className="rounded-full bg-amber-50 px-2 py-0.5 text-xs font-semibold text-amber-600">Priority: Medium</span>
          <span className="text-xs text-neutral-500">Effort · Medium</span>
        </div>
        <p className="mt-3 text-sm font-medium text-neutral-900">Publish a "vs" page targeting the top 3 buyer prompts you're missing.</p>
      </div>
    </div>
  );
}

function PricingCard({ name, price, period, features, cta, highlighted }: { name: string; price: string; period: string; features: string[]; cta: string; highlighted?: boolean; }) {
  return (
    <div className={`relative flex flex-col rounded-2xl border p-8 ${highlighted ? "border-neutral-900 bg-neutral-900 text-white" : "border-neutral-200 bg-white"}`}>
      {highlighted && <span className="absolute -top-3 left-6 rounded-full bg-[var(--brand)] px-2.5 py-0.5 text-xs font-medium text-white">Most popular</span>}
      <h3 className={`text-sm font-medium ${highlighted ? "text-neutral-300" : "text-neutral-500"}`}>{name}</h3>
      <div className="mt-3 flex items-baseline gap-1">
        <span className="text-5xl font-bold tracking-tight">{price}</span>
        <span className={highlighted ? "text-neutral-400" : "text-neutral-500"}>{period}</span>
      </div>
      <ul className="mt-8 flex-1 space-y-3 text-sm">
        {features.map((f) => (
          <li key={f} className="flex items-start gap-2">
            <Check className={`mt-0.5 h-4 w-4 shrink-0 ${highlighted ? "text-[var(--brand)]" : "text-neutral-900"}`} />
            <span>{f}</span>
          </li>
        ))}
      </ul>
      <Button asChild className={`mt-8 h-11 ${highlighted ? "bg-white text-neutral-900 hover:bg-neutral-100" : "bg-neutral-900 text-white hover:bg-neutral-800"}`}>
        <Link to="/dashboard">{cta}</Link>
      </Button>
    </div>
  );
}