"""DPO with LoRA on Qwen2.5-0.5B-Instruct, plus a reliability audit.

Self-contained DPO (no TRL): the reference model is the same base with the
LoRA adapter disabled, so only one copy of the weights sits on the GPU.

Usage:
  python dpo_lora.py --beta 0.1 --seed 0 --out runs/b0.1_s0
  python dpo_lora.py --beta 0 --out runs/base        # base model, eval only
"""
import os, json, math, time, random, argparse
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--rm", default="OpenAssistant/reward-model-deberta-v3-large-v2")
ap.add_argument("--data", default="HuggingFaceH4/ultrafeedback_binarized")
ap.add_argument("--beta", type=float, default=0.1)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n_train", type=int, default=5000)
ap.add_argument("--n_test", type=int, default=500)
ap.add_argument("--n_gen", type=int, default=200)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--micro", type=int, default=2)
ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--max_prompt", type=int, default=256)
ap.add_argument("--max_resp", type=int, default=256)
ap.add_argument("--gen_tokens", type=int, default=512)
ap.add_argument("--out", required=True)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(args.seed)
dev = "cuda"
t0 = time.time()

tok = AutoTokenizer.from_pretrained(args.model)
END = tok.convert_tokens_to_ids("<|im_end|>")
PAD = tok.pad_token_id if tok.pad_token_id is not None else END

def prompt_ids(text):
    return tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True)

def encode_pairs(split, n, skip=0):
    ds = load_dataset(args.data, split=split)
    idx = list(range(len(ds)))
    random.Random(1234).shuffle(idx)          # fixed data order, independent of training seed
    out, seen = [], 0
    for i in idx:
        ex = ds[i]
        p = prompt_ids(ex["prompt"])
        if len(p) > args.max_prompt:
            continue
        c = tok(ex["chosen"][-1]["content"], add_special_tokens=False)["input_ids"][: args.max_resp] + [END]
        r = tok(ex["rejected"][-1]["content"], add_special_tokens=False)["input_ids"][: args.max_resp] + [END]
        if c == r:
            continue
        seen += 1
        if seen <= skip:
            continue
        out.append({"prompt": ex["prompt"], "p": p, "c": c, "r": r})
        if len(out) >= n:
            break
    return out

def collate(seqs):
    """seqs: list of (prompt_ids, resp_ids). Right padded, labels mask prompt."""
    L = max(len(p) + len(r) for p, r in seqs)
    ids = torch.full((len(seqs), L), PAD, dtype=torch.long)
    lab = torch.full((len(seqs), L), -100, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for k, (p, r) in enumerate(seqs):
        s = p + r
        ids[k, : len(s)] = torch.tensor(s)
        lab[k, len(p): len(s)] = torch.tensor(r)
        att[k, : len(s)] = 1
    return ids.to(dev), att.to(dev), lab.to(dev)

def seq_logp(model, ids, att, lab):
    with torch.autocast("cuda", dtype=torch.float16):
        logits = model(input_ids=ids, attention_mask=att).logits[:, :-1]
    tgt = lab[:, 1:]
    mask = tgt != -100
    lp = torch.log_softmax(logits.float(), -1).gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (lp * mask).sum(-1), mask.sum(-1)

def pair_logps(model, batch, use_ref):
    seqs = [(b["p"], b["c"]) for b in batch] + [(b["p"], b["r"]) for b in batch]
    ids, att, lab = collate(seqs)
    n = len(batch)
    if use_ref:
        with torch.no_grad(), model.disable_adapter():
            lp, cnt = seq_logp(model, ids, att, lab)
    else:
        lp, cnt = seq_logp(model, ids, att, lab)
    return lp[:n], lp[n:], cnt[:n], cnt[n:]

base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(dev)
base.config.use_cache = False
base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
base.enable_input_require_grads()
model = get_peft_model(base, LoraConfig(
    r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

train_log = []
if args.beta > 0:
    train = encode_pairs("train_prefs", args.n_train)
    random.Random(args.seed).shuffle(train)
    steps = len(train) // (args.micro * args.accum)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    warm = max(1, int(0.1 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))
    scaler = torch.cuda.amp.GradScaler()
    model.train()
    acc_loss, acc_acc, acc_margin, acc_n = 0.0, 0.0, 0.0, 0
    for step in range(steps):
        for a in range(args.accum):
            b = train[(step * args.accum + a) * args.micro: (step * args.accum + a + 1) * args.micro]
            rc, rr, _, _ = pair_logps(model, b, use_ref=True)
            pc, pr, _, _ = pair_logps(model, b, use_ref=False)
            m = args.beta * ((pc - rc) - (pr - rr))
            loss = -F.logsigmoid(m).mean()
            scaler.scale(loss / args.accum).backward()
            acc_loss += loss.item(); acc_acc += (m > 0).float().mean().item(); acc_margin += m.mean().item(); acc_n += 1
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
        if (step + 1) % 20 == 0 or step == steps - 1:
            rec = {"step": step + 1, "loss": acc_loss / acc_n, "reward_acc": acc_acc / acc_n,
                   "margin": acc_margin / acc_n, "lr": sched.get_last_lr()[0], "min": (time.time() - t0) / 60}
            train_log.append(rec); print(json.dumps(rec), flush=True)
            acc_loss, acc_acc, acc_margin, acc_n = 0.0, 0.0, 0.0, 0
    model.save_pretrained(os.path.join(args.out, "adapter"))

# ---------------- held-out preference evaluation ----------------
model.eval()
test = encode_pairs("test_prefs", args.n_test)
pc_l, pr_l, rc_l, rr_l, nc_l, nr_l = [], [], [], [], [], []
with torch.no_grad():
    for i in range(0, len(test), 4):
        b = test[i: i + 4]
        pc, pr, nc, nr = pair_logps(model, b, use_ref=False)
        rc, rr, _, _ = pair_logps(model, b, use_ref=True)
        for L, v in ((pc_l, pc), (pr_l, pr), (rc_l, rc), (rr_l, rr), (nc_l, nc), (nr_l, nr)):
            L += v.float().cpu().tolist()
pc, pr, rc, rr, nc, nr = map(np.array, (pc_l, pr_l, rc_l, rr_l, nc_l, nr_l))

def ece(p, y, bins=15):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if k.any():
            e += k.mean() * abs(p[k].mean() - y[k].mean())
    return float(e)

pref = {"n": int(len(pc)),
        "lennorm_logp_acc": float(np.mean(pc / nc > pr / nr)),
        "sum_logp_acc": float(np.mean(pc > pr))}
if args.beta > 0:
    margin = args.beta * ((pc - rc) - (pr - rr))
    p = 1 / (1 + np.exp(-margin))
    # symmetrize: each pair seen in both orders, so labels are balanced
    ps, ys = np.concatenate([p, 1 - p]), np.concatenate([np.ones_like(p), np.zeros_like(p)])
    boot = [np.mean(margin[np.random.default_rng(k).integers(0, len(margin), len(margin))] > 0) for k in range(1000)]
    pref.update({"implicit_reward_acc": float(np.mean(margin > 0)),
                 "implicit_reward_acc_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                 "implicit_margin_mean": float(margin.mean()),
                 "implicit_reward_ece": ece(ps, ys),
                 "implicit_reward_mean_conf": float(np.mean(np.maximum(p, 1 - p)))})
print("PREF", json.dumps(pref), flush=True)

# ---------------- generation: KL to reference, reward-model score, length ----------------
gen_src = encode_pairs("test_prefs", args.n_gen, skip=args.n_test)
tok.padding_side = "left"
gens = []
for i in range(0, len(gen_src), 16):
    b = gen_src[i: i + 16]
    L = max(len(x["p"]) for x in b)
    ids = torch.full((len(b), L), PAD, dtype=torch.long); att = torch.zeros((len(b), L), dtype=torch.long)
    for k, x in enumerate(b):
        ids[k, L - len(x["p"]):] = torch.tensor(x["p"]); att[k, L - len(x["p"]):] = 1
    torch.manual_seed(10_000 + i)          # same sampling noise across models
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        model.config.use_cache = True
        out = model.generate(input_ids=ids.to(dev), attention_mask=att.to(dev), do_sample=True, temperature=1.0,
                             top_p=1.0, max_new_tokens=args.gen_tokens, eos_token_id=END, pad_token_id=PAD)
    for k, x in enumerate(b):
        resp = out[k, L:].tolist()
        if END in resp:
            resp = resp[: resp.index(END) + 1]
        resp = [t for t in resp if t != PAD] or [END]
        gens.append({"prompt": x["prompt"], "p": x["p"], "resp_ids": resp,
                     "text": tok.decode(resp, skip_special_tokens=True), "len": len(resp)})
with torch.no_grad():
    for i in range(0, len(gens), 8):
        b = gens[i: i + 8]
        ids, att, lab = collate([(x["p"], x["resp_ids"]) for x in b])
        lp, _ = seq_logp(model, ids, att, lab)
        with model.disable_adapter():
            lr_, _ = seq_logp(model, ids, att, lab)
        for x, a, c in zip(b, lp.float().cpu().tolist(), lr_.float().cpu().tolist()):
            x["logp"], x["ref_logp"] = a, c
del model, base
torch.cuda.empty_cache()

rm_tok = AutoTokenizer.from_pretrained(args.rm)
rm = AutoModelForSequenceClassification.from_pretrained(args.rm).to(dev).eval()
with torch.no_grad():
    for i in range(0, len(gens), 8):
        b = gens[i: i + 8]
        enc = rm_tok([x["prompt"] for x in b], [x["text"] for x in b], return_tensors="pt",
                     padding=True, truncation=True, max_length=768).to(dev)
        s = rm(**enc).logits[:, 0].float().cpu().tolist()
        for x, v in zip(b, s):
            x["rm"] = v

kl = np.array([x["logp"] - x["ref_logp"] for x in gens])
gen = {"n": len(gens), "rm_mean": float(np.mean([x["rm"] for x in gens])),
       "len_mean": float(np.mean([x["len"] for x in gens])),
       "kl_mean": float(kl.mean()), "hit_max_len_frac": float(np.mean([x["len"] >= args.gen_tokens for x in gens]))}
print("GEN", json.dumps(gen), flush=True)

with open(os.path.join(args.out, "generations.jsonl"), "w") as f:
    for x in gens:
        f.write(json.dumps({k: x[k] for k in ("prompt", "text", "len", "logp", "ref_logp", "rm")}) + "\n")
res = {"config": vars(args), "trainable_params": trainable, "train_log": train_log, "pref": pref, "gen": gen,
       "minutes": (time.time() - t0) / 60, "torch": torch.__version__,
       "gpu": torch.cuda.get_device_name(0)}
json.dump(res, open(os.path.join(args.out, "result.json"), "w"), indent=1)
print("DONE", args.out, f"{res['minutes']:.1f} min", flush=True)
