import os, json, time
from pathlib import Path

def save_run(payload: dict, out_dir="outputs/runs"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(out_dir, f"run_{ts}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path