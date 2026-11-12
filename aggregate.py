import json, glob, os, numpy as np
from io import BytesIO
from pathlib import Path
from generation_pairing import load_generations, select_baseline, winrate_vs_base
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
runs = {}
for f in sorted(glob.glob("runs/*/result.json")):
    r = json.load(open(f)); runs[os.path.basename(os.path.dirname(f))] = r
base = select_baseline(runs)
generations = {k: load_generations(f"runs/{k}/generations.jsonl") for k in runs}
basegen = generations[base] if base else None
# Fail before writing summaries if any comparison has incomplete/ambiguous prompts.
winrates = {k: winrate_vs_base(generations[k], basegen)
            for k, r in runs.items() if basegen and r["config"]["beta"] > 0}
# pooled linear fit of reward-model score on response length (all runs), for a length-adjusted score
allg = []
for k in runs:
    allg += generations[k]
L = np.array([g["len"] for g in allg], float); R = np.array([g["rm"] for g in allg])
if np.all(L == L[0]):
    raise ValueError("length adjustment requires at least two distinct lengths")
constant_reward = bool(np.all(R == R[0]))
if constant_reward:
    # With varying lengths, a constant reward has slope zero but undefined r.
    slope, icpt, corr = 0.0, float(R[0]), None
else:
    slope, icpt = np.polyfit(L, R, 1)
    corr = float(np.corrcoef(L, R)[0, 1])
    if not np.all(np.isfinite([slope, icpt, corr])):
        raise ValueError("length adjustment produced nonfinite regression statistics")
rows = []
for k, r in runs.items():
    b = r["config"]["beta"]; g = r["gen"]; p = r["pref"]
    win = None
    if basegen and b > 0:
        win = winrates[k]
    mine_all = generations[k]
    adj = (0.0 if constant_reward else
           float(np.mean([m["rm"] - slope * m["len"] for m in mine_all]) - icpt))
    rows.append({"rm_len_adjusted": adj, "run": k, "beta": b, "seed": r["config"]["seed"],
                 "heldout_implicit_acc": p.get("implicit_reward_acc"), "acc_ci95": p.get("implicit_reward_acc_ci95"),
                 "implicit_ece": p.get("implicit_reward_ece"), "lennorm_logp_acc": p["lennorm_logp_acc"],
                 "kl": g["kl_mean"], "rm": g["rm_mean"], "len": g["len_mean"], "hit_max": g["hit_max_len_frac"],
                 "rm_winrate_vs_base": win, "minutes": r["minutes"], "gpu": r["gpu"]})
rows.sort(key=lambda x: (x["beta"] == 0 and -1 or 1 / x["beta"], x["seed"]))
# Validate strict JSON before opening either existing output for replacement.
summary_text = json.dumps(
    {"rows": rows, "rm_vs_len_slope_per_100tok": float(slope * 100), "rm_len_corr": corr},
    indent=1, allow_nan=False)
print("RM~len slope per 100 tokens", slope * 100, "corr", corr)
for x in rows: print(x)
d = [x for x in rows if x["beta"] > 0]
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for x in rows:
    ax[0].scatter(x["kl"], x["rm"]); ax[0].annotate(f"b={x['beta']}", (x["kl"], x["rm"]))
    ax[1].scatter(x["kl"], x["len"]); ax[1].annotate(f"b={x['beta']}", (x["kl"], x["len"]))
ax[0].set_xlabel("KL to reference (nats/response)"); ax[0].set_ylabel("reward-model score")
ax[1].set_xlabel("KL to reference"); ax[1].set_ylabel("mean response length (tokens)")
for x in d:
    ax[2].bar(f"b={x['beta']}\ns{x['seed']}", x["heldout_implicit_acc"])
ax[2].set_ylabel("held-out implicit reward accuracy"); ax[2].set_ylim(0.4, 0.8)
plt.tight_layout()
figure_bytes = BytesIO()
plt.savefig(figure_bytes, format="png", dpi=150)
plt.close(fig)
# Rendering and numeric validation have succeeded; filesystem failures remain
# ordinary I/O errors, and these two writes are not an atomic pair transaction.
Path("summary.json").write_text(summary_text, encoding="utf-8")
Path("dpo_audit.png").write_bytes(figure_bytes.getvalue())
print("saved dpo_audit.png")
