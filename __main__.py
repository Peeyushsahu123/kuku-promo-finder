import argparse
import sys

from .llm import LLM
from .pipeline import PackError, run


def main(argv=None):
    ap = argparse.ArgumentParser(prog="promofinder", description="Find promo clips in timestamped microdrama screenplays.")
    ap.add_argument("input", help="folder of episode .txt screenplays (or a single file)")
    ap.add_argument("-o", "--out", default="out", help="output folder (default: out)")
    ap.add_argument("--per-episode", type=int, default=2, help="max promos per episode (default 2)")
    ap.add_argument("--top", type=int, default=20, help="size of the 'post first' shortlist used for tiers (default 20)")
    ap.add_argument("--no-stitched", action="store_true", help="only single continuous clips")
    ap.add_argument("--title", default="", help="show title (helps the LLM judge)")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="process the valid episodes even if some files have errors (they stay listed in diagnostics.json)")
    ap.add_argument("--llm", default=None, help="gemini | openai | anthropic | none (default: env PROMO_LLM_PROVIDER or none)")
    ap.add_argument("--model", default=None, help="model name for the LLM judge")
    ap.add_argument("--base-url", default=None, help="OpenAI-compatible base URL (Groq, OpenRouter, Ollama...)")
    ap.add_argument("--llm-pool", type=int, default=5, help="candidates per episode sent to the LLM (default 5)")
    ap.add_argument("--llm-timeout", type=float, default=None, help="seconds per LLM request (default 30)")
    a = ap.parse_args(argv)
    llm = LLM(a.llm, a.model, None, a.base_url, a.llm_timeout)
    if llm.provider not in ("none", "gemini", "openai", "anthropic"):
        print(f"error: unknown --llm provider {llm.provider!r}", file=sys.stderr)
        return 2
    if llm.provider != "none" and not llm.enabled:
        print(f"[warn] provider {llm.provider} selected but no API key found; running rules-only (recorded in run_summary.json)")
    try:
        run(a.input, a.out, per_episode=a.per_episode, stitched=not a.no_stitched, top=a.top,
            llm=llm if llm.provider != "none" else None, llm_pool=a.llm_pool, title=a.title,
            continue_on_error=a.continue_on_error)
    except PackError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
