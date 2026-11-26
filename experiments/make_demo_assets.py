"""Copy the data the web demo needs into docs/assets/ (showcase JSON, figure PNGs, headline numbers)."""
import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, FIG, OUT = ROOT / "results", ROOT / "figures", ROOT / "docs" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

if (RES / "e8_showcase.json").exists():
    shutil.copy(RES / "e8_showcase.json", OUT / "showcase.json")
for png in FIG.glob("fig_*.png"):
    shutil.copy(png, OUT / png.name)

summary = {}
if (RES / "e7_timing.json").exists():
    t = json.loads((RES / "e7_timing.json").read_text())
    summary["n_problems"] = t["n_problems"]
if (RES / "e6_sweep.json").exists():
    sw = pd.DataFrame(json.loads((RES / "e6_sweep.json").read_text()))
    q = sw[(sw["mode"] == "semi") & (sw.tau == sw.tau.max()) & (sw.eps == 0.01)].groupby("solver").iters.median()
    if "Sinkhorn" in q and "AA-TI-Sinkhorn" in q:
        summary["speedup"] = f"{q['Sinkhorn'] / q['AA-TI-Sinkhorn']:,.0f}×"
(OUT / "summary.json").write_text(json.dumps(summary))
print("assets:", sorted(p.name for p in OUT.iterdir()), summary)
