import json, glob, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
runs = {}
for f in sorted(glob.glob("runs/*/result.json")):
    r = json.load(open(f)); runs[os.path.basename(os.path.dirname(f))] = r
base = next((k for k in runs if k.startswith("beta0_")), None)
basegen = [json.loads(l) for l in open(f"runs/{base}/generations.jsonl")] if base else None
# pooled linear fit of reward-model score on response length (all runs), for a length-adjusted score
allg = []
for k in runs:
    allg += [json.loads(l) for l in open(f"runs/{k}/generations.jsonl")]
L = np.array([g["len"] for g in allg], float); R = np.array([g["rm"] for g in allg])
slope, icpt = np.polyfit(L, R, 1)
corr = float(np.corrcoef(L, R)[0, 1])
rows = []
for k, r in runs.items():
    b = r["config"]["beta"]; g = r["gen"]; p = r["pref"]
    win = None
    if basegen and b > 0:
        mine = [json.loads(l) for l in open(f"runs/{k}/generations.jsonl")]
        win = float(np.mean([m["rm"] > q["rm"] for m, q in zip(mine, basegen)]))
    mine_all = [json.loads(l) for l in open(f"runs/{k}/generations.jsonl")]
    adj = float(np.mean([m["rm"] - slope * m["len"] for m in mine_all]) - icpt)
    rows.append({"rm_len_adjusted": adj, "run": k, "beta": b, "seed": r["config"]["seed"],
                 "heldout_implicit_acc": p.get("implicit_reward_acc"), "acc_ci95": p.get("implicit_reward_acc_ci95"),
                 "implicit_ece": p.get("implicit_reward_ece"), "lennorm_logp_acc": p["lennorm_logp_acc"],
                 "kl": g["kl_mean"], "rm": g["rm_mean"], "len": g["len_mean"], "hit_max": g["hit_max_len_frac"],
                 "rm_winrate_vs_base": win, "minutes": r["minutes"], "gpu": r["gpu"]})
rows.sort(key=lambda x: (x["beta"] == 0 and -1 or 1 / x["beta"], x["seed"]))
json.dump({"rows": rows, "rm_vs_len_slope_per_100tok": float(slope * 100), "rm_len_corr": corr},
          open("summary.json", "w"), indent=1)
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
plt.tight_layout(); plt.savefig("dpo_audit.png", dpi=150)
print("saved dpo_audit.png")
