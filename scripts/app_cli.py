import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore", message="Starting from v4.46")

import json
from scripts.run_demo import pipeline

def format_final_response(result: dict) -> str:
    per_fact = result["per_fact"]

    supported = [x["fact"] for x in per_fact if x["gate_label"] == "SUPPORTED"]
    underspec = [x for x in per_fact if x["gate_label"] == "UNDER_SPECIFIED"]
    contrad  = [x for x in per_fact if x["gate_label"] == "CONTRADICTED"]

    # Simple trust score
    trust = len(supported) / max(1, len(per_fact))

    lines = []
    lines.append("=== Final Answer (Grounded) ===")
    lines.append(result["llm_answer"].strip())
    lines.append("")
    lines.append(f"Trust Score: {trust:.2f}")
    lines.append("")

    if supported:
        lines.append("✅ Verified facts:")
        for x in supported:
            lines.append(f"- {x['fact']} (p_yes={x['p_yes']:.2f}, sim={x['sim_max']:.2f})")

    if underspec:
        lines.append("\n⚠️ Under-specified / needs more evidence:")
        for x in underspec[:5]:
            lines.append(f"- {x['fact']} (p_yes={x['p_yes']:.2f}, sim={x['sim_max']:.2f})")

    if contrad:
        lines.append("\n❌ Contradicted:")
        for x in contrad[:5]:
            lines.append(f"- {x['fact']} (p_no={x['p_no']:.2f}, sim={x['sim_max']:.2f})")

    return "\n".join(lines)

def main():
    print("Hallucination Firewall (CLI)")
    print("Type a query. Type 'exit' to quit.\n")

    while True:
        q = input("User Query > ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        # result, path = pipeline(q)
        result, path = pipeline(q, variant="full")
        print(f"\n✅ Saved run: {path}\n")
        print(format_final_response(result))
        print("\n" + "-"*60 + "\n")

if __name__ == "__main__":
    main()