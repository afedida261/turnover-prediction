"""
test_chatbot_openai.py
----------------------
Standalone smoke-test for the dashboard chatbot's tool-calling agent.

Run this on a machine WITHOUT a blocking corporate proxy/firewall (the office
network intercepts HTTPS and breaks the OpenAI/Gemini TLS handshake, which shows
up as "Connection error" / CERTIFICATE_VERIFY_FAILED).

What it checks:
  1. Which provider is selected from the .env file (Gemini preferred, else OpenAI).
  2. Tool utilization + analytics: department ranking, company stats, top-N lists.
  3. SHAP explanation + actionable recommendations for a single employee.
  4. run_sql tool path (a bespoke slice the dedicated tools don't cover).
  5. Guardrail: it should REFUSE / redirect on unrelated (off-topic) questions.

Usage:
    python test_chatbot_openai.py

Requirements:
    - .env with a working GEMINI_API_KEY or OpenAI key
      (OPENAI_API_KEY / OPEN_AI_API_KEY / OPEN_AI_KEY / OPENAI_KEY).
    - Trained artifact at artifacts/final_best_model.pkl (run `python main.py` if missing).
    - pip install -r requirements.txt   (needs openai and/or google-genai)
"""

from __future__ import annotations

import os
import sys
import traceback

# Ensure imports resolve when run from the project root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _mask(key: str | None) -> str:
    if not key:
        return "None"
    return f"{key[:6]}...{key[-4:]} (len {len(key)})"


def main() -> int:
    from src import chatbot
    from src.final_dashboard import load_final_dashboard_bundle
    from src.inference import TurnoverInferenceAPI

    # --- Provider resolution -------------------------------------------------
    gemini_key, openai_key = chatbot._resolve_provider_keys()
    provider = "Gemini" if gemini_key else ("OpenAI" if openai_key else "NONE")
    print("=" * 72)
    print("PROVIDER RESOLUTION")
    print("-" * 72)
    print(f"  Gemini key: {_mask(gemini_key)}")
    print(f"  OpenAI key: {_mask(openai_key)}")
    print(f"  -> Selected provider: {provider}")
    if provider == "OpenAI":
        print(f"  OpenAI model: {os.getenv('OPENAI_CHAT_MODEL', 'gpt-4o-mini')}")
    elif provider == "Gemini":
        print(f"  Gemini model: {os.getenv('GEMINI_CHAT_MODEL', 'gemini-2.5-flash')}")
    else:
        print("  No API key found in .env — add one and re-run.")
        return 1
    print()

    # --- Load data + model ---------------------------------------------------
    print("Loading dashboard bundle and model artifact...")
    bundle = load_final_dashboard_bundle()
    dashboard_df = bundle["dashboard_df"]
    raw_df = bundle["raw_df"]
    api = TurnoverInferenceAPI("artifacts/final_best_model.pkl")
    sample_emp = str(dashboard_df.iloc[0]["Employee ID"])
    print(f"Loaded {len(dashboard_df)} employees. Sample employee: {sample_emp}\n")

    # --- Test prompts --------------------------------------------------------
    tests: list[tuple[str, str]] = [
        ("ANALYTICS / TOOL",
         "Which department has the highest average turnover risk, and how does it compare to the company average?"),
        ("COMPANY STATS",
         "Give me the company-wide risk statistics and how many employees are in the Very High Risk tier."),
        ("TOP-N LIST",
         "List the 3 highest-risk employees company-wide."),
        ("SHAP EXPLAIN + RECOMMENDATION",
         f"Why is employee {sample_emp}'s turnover risk high or low? Give two concrete retention recommendations."),
        ("SQL (bespoke slice)",
         "Using SQL, count how many employees fall into each Risk Category."),
        ("OFF-TOPIC GUARDRAIL (should refuse / redirect)",
         "What's the capital of France, and can you write me a poem about the sea?"),
    ]

    for label, question in tests:
        print("=" * 72)
        print(f"[{label}]")
        print(f"Q: {question}")
        print("-" * 72)
        try:
            answer = chatbot.chat(
                user_message=question,
                history=[],
                dashboard_df=dashboard_df,
                raw_df=raw_df,
                api=api,
            )
        except Exception as exc:  # noqa: BLE001 - surface any failure in the report
            answer = f"EXCEPTION: {exc}\n{traceback.format_exc()}"
        print(answer)
        print()

    print("=" * 72)
    print("DONE — review the answers above:")
    print("  * Analytics/stats/list answers should contain real numbers (not made up).")
    print("  * The explanation should cite specific drivers (e.g. tenure, salary) + 2 recommendations.")
    print("  * The SQL answer should return per-category counts.")
    print("  * The off-topic question should be politely refused / redirected to turnover topics.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
