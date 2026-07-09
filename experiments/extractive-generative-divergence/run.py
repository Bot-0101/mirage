# %% [markdown]
# # extractive-generative-divergence — Kaggle notebook (run.py)
#
# Is the **semantic divergence between Qwen2.5-3B's open-book extractive answer and its
# closed-book generative answer** a label-free confabulation detector, competitive with
# semantic entropy at matched forward-pass compute?
#
# Source of truth: `impl-plan.md`. Each `# %%` block is one notebook cell = a number or a plot.
# Library: HF transformers · Model: Qwen2.5-3B-Instruct · Data: SQuAD 2.0 · Compute: Kaggle GPU · Seed: 42.
#
# **Run order (per plan):** Cells 0–4 on `SMOKE_N=20` first (plumbing + compliance gate),
# then set `SMOKE=False` and run Cells 2–9 on the 500-item slice, then Cell 10 (ablations)
# only if the headline clears the bar. Pure scoring logic is in `metrics_lib.py` (unit-tested off-GPU).

# %%
# ======================================================================
# Cell 0 — config & seed
# ======================================================================
import os, json, random, math
import numpy as np

SEED = 42
random.seed(SEED); np.random.seed(SEED)

import torch
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          AutoModelForSequenceClassification, set_seed)
set_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Auto-detect the writable output dir: Kaggle -> /kaggle/working, Colab -> /content,
# else a local ./results. Each platform's dir persists as its downloadable output.
if os.path.isdir("/kaggle/working"):
    RESULTS_DIR = "/kaggle/working/results"
elif os.path.isdir("/content"):
    RESULTS_DIR = "/content/results"
else:
    RESULTS_DIR = os.path.join(os.getcwd(), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
print("RESULTS_DIR:", RESULTS_DIR)

# Toggle: run the smoke slice first, then flip to False for the full 500-item slice.
SMOKE = True
SMOKE_N = 20
N_ANSWERABLE = 250
N_UNANSWERABLE = 250

MODEL = "Qwen/Qwen2.5-3B-Instruct"
NLI_CKPT = "microsoft/deberta-v2-xlarge-mnli"   # ONE checkpoint for divergence AND SE clustering
# Robustness-swap checkpoint (Cell 10). Microsoft never published a v3-large MNLI head;
# this is the de-facto v3-large NLI checkpoint. Its label order is entailment-first
# (differs from deberta-v2's), but Cell 10 reads id2label dynamically, so that's handled.
NLI_CKPT_ALT = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
EMB_MODEL = "sentence-transformers/all-MiniLM-L12-v2"

# Sampling params for semantic entropy (pin & document — sources disagree greedy vs T=0.1).
SE_N = 10
SE_TEMPERATURE = 1.0
SE_TOP_P = 0.9
SE_TOP_K = 50
F1_TAU = 0.5          # correctness threshold; robustness pass at 0.3
F1_TAU_ROBUST = 0.3
MAX_NEW_TOKENS = 32

# metrics_lib is unit-tested (test_metrics_lib.py); import the pure scoring helpers.
import metrics_lib as M

print(f"device={device}  SMOKE={SMOKE}  seed={SEED}")


# %%
# ======================================================================
# Cell 1 — load model + SANITY (tokenizer-first). GATE: do not proceed until
# the rendered chat prompt is correct. A silent chat-format bug corrupts everything.
# ======================================================================
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, device_map="auto")
model.eval()

def encode_chat(msgs):
    # return_dict=True -> BatchEncoding with input_ids + attention_mask. Robust across
    # transformers versions; newer ones return a tokenizers Encoding from bare
    # return_tensors="pt", which breaks tok.decode(ids[0]) and model.generate(ids).
    return tok.apply_chat_template(msgs, add_generation_prompt=True,
                                   tokenize=True, return_dict=True, return_tensors="pt")

def render(msgs):
    ids = encode_chat(msgs)["input_ids"]
    return tok.decode(ids[0]), ids.shape[-1]

_ctx = "The Eiffel Tower was completed in 1889 for the World's Fair in Paris."
_extractive_msgs = [{"role": "user", "content":
    f"Context: {_ctx}\nQuestion: When was the Eiffel Tower completed?\n"
    "Answer using ONLY a verbatim span from the context, or say exactly 'unanswerable'."}]
_generative_msgs = [{"role": "user", "content":
    "Question: When was the Eiffel Tower completed? Answer concisely."}]

for name, msgs in [("EXTRACTIVE", _extractive_msgs), ("GENERATIVE", _generative_msgs)]:
    text, n = render(msgs)
    print(f"\n===== {name} rendered prompt (n_tokens={n}) =====")
    print(repr(text))
# ^ Verify: Qwen's silent default system prompt is present and is what we expect.

# Trivial known-QA reproduction (greedy) as a smoke behavior check.
def _greedy_once(msgs):
    enc = encode_chat(msgs).to(device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

print("\nSanity greedy answer:", _greedy_once(_generative_msgs))
# GATE: eyeball the rendered prompts above before running any downstream cell.


# %%
# ======================================================================
# Cell 2 — build_dataset. Stratified answerable / unanswerable pools, seed 42.
# Persist exact ids + indices to results/slice_manifest.json for reproducibility.
# ======================================================================
from datasets import load_dataset

ds = load_dataset("rajpurkar/squad_v2")["validation"]

answerable_idx, unanswerable_idx = [], []
for i, ex in enumerate(ds):
    (answerable_idx if len(ex["answers"]["text"]) > 0 else unanswerable_idx).append(i)

rng = random.Random(SEED)
if SMOKE:
    n_ans = n_una = SMOKE_N // 2
else:
    n_ans, n_una = N_ANSWERABLE, N_UNANSWERABLE

pick_ans = rng.sample(answerable_idx, n_ans)
pick_una = rng.sample(unanswerable_idx, n_una)
slice_idx = pick_ans + pick_una

items = []
for i in slice_idx:
    ex = ds[i]
    items.append({
        "id": ex["id"],
        "ds_index": i,
        "question": ex["question"],
        "context": ex["context"],
        "golds": ex["answers"]["text"],          # [] for unanswerable
        "answerable": len(ex["answers"]["text"]) > 0,
    })

manifest = {"seed": SEED, "smoke": SMOKE, "model": MODEL,
            "n_answerable": n_ans, "n_unanswerable": n_una,
            "ids": [it["id"] for it in items], "ds_indices": slice_idx}
with open(f"{RESULTS_DIR}/slice_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
print(f"slice: {len(items)} items ({n_ans} answerable / {n_una} unanswerable). manifest saved.")


# %%
# ======================================================================
# Cell 3 — prompt builders + generate(). Three templates; greedy wrapper returns
# text + mean token log-prob. Print 3 RANDOM rendered prompts.
# ======================================================================
def build_extractive(question, context):
    return [{"role": "user", "content":
        f"Context: {context}\nQuestion: {question}\n"
        "Answer using ONLY a verbatim span copied from the context. "
        "If the context does not contain the answer, respond with exactly 'unanswerable'. "
        "Give only the answer span, no explanation."}]

def build_gen_closed(question):
    return [{"role": "user", "content":
        f"Question: {question}\nAnswer concisely with just the answer, no explanation."}]

def build_gen_openbook(question, context):
    return [{"role": "user", "content":
        f"Context: {context}\nQuestion: {question}\n"
        "Answer concisely with just the answer, no explanation."}]

@torch.no_grad()
def generate_greedy(msgs, max_new_tokens=MAX_NEW_TOKENS):
    """Greedy decode; return (text, mean_token_logprob) over generated tokens."""
    enc = encode_chat(msgs).to(device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id,
                         return_dict_in_generate=True, output_scores=True)
    gen_ids = out.sequences[0, enc["input_ids"].shape[-1]:]
    text = tok.decode(gen_ids, skip_special_tokens=True).strip()
    # mean log-prob of the greedily chosen tokens
    logps = []
    for step, score in enumerate(out.scores):
        if step >= len(gen_ids):
            break
        logp = torch.log_softmax(score[0].float(), dim=-1)[gen_ids[step]]
        logps.append(logp.item())
    mean_logprob = float(np.mean(logps)) if logps else float("nan")
    return text, mean_logprob

@torch.no_grad()
def sample_answers(msgs, n=SE_N, max_new_tokens=MAX_NEW_TOKENS):
    """N stochastic samples for semantic entropy."""
    enc = encode_chat(msgs).to(device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True,
                         temperature=SE_TEMPERATURE, top_p=SE_TOP_P, top_k=SE_TOP_K,
                         num_return_sequences=n, pad_token_id=tok.eos_token_id)
    plen = enc["input_ids"].shape[-1]
    return [tok.decode(out[k, plen:], skip_special_tokens=True).strip()
            for k in range(out.shape[0])]

# Print 3 random rendered prompts (mixed templates) as a format check.
_sample = random.Random(SEED).sample(items, min(3, len(items)))
for it in _sample:
    txt, n = render(build_extractive(it["question"], it["context"]))
    print(f"\n--- RANDOM extractive prompt (id={it['id']}, n_tokens={n}) ---\n{repr(txt[:400])} ...")


# %%
# ======================================================================
# Cell 4 — elicit_answers. Run extractive / gen_closed / gen_openbook for every item.
# Print extractive substring-compliance + abstention rates. GATE: compliance < ~70% => fix/pivot.
# ======================================================================
answers = []
for it in items:
    ext_text, ext_lp = generate_greedy(build_extractive(it["question"], it["context"]))
    clo_text, clo_lp = generate_greedy(build_gen_closed(it["question"]))
    opn_text, opn_lp = generate_greedy(build_gen_openbook(it["question"], it["context"]))
    answers.append({
        "id": it["id"], "answerable": it["answerable"],
        "extractive": ext_text, "extractive_logprob": ext_lp,
        "gen_closed": clo_text, "gen_closed_logprob": clo_lp,
        "gen_openbook": opn_text, "gen_openbook_logprob": opn_lp,
    })

with open(f"{RESULTS_DIR}/answers.jsonl", "w") as f:
    for a in answers:
        f.write(json.dumps(a) + "\n")

ctx_by_id = {it["id"]: it["context"] for it in items}
def _is_abstain(s):
    return M.normalize_answer(s) in {"unanswerable", "unknown", "no answer", "cannot answer"}

compliant = [M.is_substring_compliant(a["extractive"], ctx_by_id[a["id"]])
             for a in answers if not _is_abstain(a["extractive"])]
compliance_rate = float(np.mean(compliant)) if compliant else 0.0
abstain_rate_ext = float(np.mean([_is_abstain(a["extractive"]) for a in answers]))
abstain_rate_clo = float(np.mean([_is_abstain(a["gen_closed"]) for a in answers]))
print(f"extractive substring-compliance (non-abstain): {compliance_rate:.1%}")
print(f"extractive abstention rate: {abstain_rate_ext:.1%}  |  closed-book abstention: {abstain_rate_clo:.1%}")
# GATE: if compliance < ~70%, fix the extractive prompt or pivot to open-book BEFORE trusting AUROC.


# %%
# ======================================================================
# Cell 5 — score_divergence. Load the ONE NLI checkpoint. Bidirectional NLI with
# Q+A templating -> d_nli. Also d_cos (MiniLM) and d_lex (token-F1). Save divergence.jsonl.
# ======================================================================
_nli_tok = AutoTokenizer.from_pretrained(NLI_CKPT)
_nli_model = AutoModelForSequenceClassification.from_pretrained(
    NLI_CKPT, torch_dtype=torch.float16).to(device).eval()
# DeBERTa-MNLI label order is [contradiction, neutral, entailment]; verify from config.
_id2label = {int(k): v.lower() for k, v in _nli_model.config.id2label.items()}
_ENTAIL_IDX = next(i for i, l in _id2label.items() if "entail" in l)
print("NLI id2label:", _id2label, "entail_idx:", _ENTAIL_IDX)

@torch.no_grad()
def nli_probs(premise, hypothesis):
    enc = _nli_tok(premise, hypothesis, return_tensors="pt", truncation=True,
                   max_length=256).to(device)
    logits = _nli_model(**enc).logits[0].float()
    return torch.softmax(logits, dim=-1).cpu().numpy()

def nli_label(premise, hypothesis):
    return _id2label[int(nli_probs(premise, hypothesis).argmax())]

def nli_entail_prob(premise, hypothesis):
    return float(nli_probs(premise, hypothesis)[_ENTAIL_IDX])

from sentence_transformers import SentenceTransformer
_embedder = SentenceTransformer(EMB_MODEL, device=device)

def cos_div(a, b):
    e = _embedder.encode([a, b], normalize_embeddings=True)
    return float(1.0 - np.dot(e[0], e[1]))

q_by_id = {it["id"]: it["question"] for it in items}
divergence = []
for a in answers:
    q = q_by_id[a["id"]]
    # Q+A templating (load-bearing) — ablated off in Cell 10.
    ext_t = M.qa_template(q, a["extractive"])
    clo_t = M.qa_template(q, a["gen_closed"])
    opn_t = M.qa_template(q, a["gen_openbook"])
    row = {"id": a["id"], "answerable": a["answerable"]}
    # headline: extractive vs closed-book
    row["d_nli"] = M.bidirectional_divergence(ext_t, clo_t, nli_entail_prob)
    # PS-2: extractive vs open-book (grounding-vs-format)
    row["d_nli_openbook"] = M.bidirectional_divergence(ext_t, opn_t, nli_entail_prob)
    # untemplated variant for the Cell 10 ablation
    row["d_nli_untemplated"] = M.bidirectional_divergence(
        a["extractive"], a["gen_closed"], nli_entail_prob)
    row["d_cos"] = cos_div(a["extractive"], a["gen_closed"])
    row["d_lex"] = 1.0 - M.token_f1(a["extractive"], a["gen_closed"])
    divergence.append(row)

with open(f"{RESULTS_DIR}/divergence.jsonl", "w") as f:
    for r in divergence:
        f.write(json.dumps(r) + "\n")
print(f"divergence scored for {len(divergence)} items.")


# %%
# ======================================================================
# Cell 6 — label_correctness. Positive class = INCORRECT. Uses GOLD only (no leakage
# from the extractive arm). Answerable: y = token_f1(gen_closed, gold) < tau.
# Unanswerable: y = closed-book did NOT abstain (any confident answer is a confabulation).
# ======================================================================
golds_by_id = {it["id"]: it["golds"] for it in items}
labels = []
for a in answers:
    golds = golds_by_id[a["id"]]
    if a["answerable"]:
        f1 = M.squad_f1_max(a["gen_closed"], golds)
        f1r = f1
        y = int(f1 < F1_TAU)
        y_robust = int(f1r < F1_TAU_ROBUST)
    else:
        f1 = float("nan"); f1r = float("nan")
        y = int(not _is_abstain(a["gen_closed"]))     # answered an unanswerable Q => confabulation
        y_robust = y
    labels.append({"id": a["id"], "answerable": a["answerable"],
                   "gen_closed_f1": f1, "y": y, "y_robust_f1_0.3": y_robust})

with open(f"{RESULTS_DIR}/labels.jsonl", "w") as f:
    for l in labels:
        f.write(json.dumps(l) + "\n")
pos = float(np.mean([l["y"] for l in labels]))
print(f"positive (INCORRECT) rate @tau={F1_TAU}: {pos:.1%}  over {len(labels)} items.")


# %%
# ======================================================================
# Cell 7 — baseline_signals. (a) closed-book mean token-logprob (already in answers);
# (b) p_true prompt-only self-eval; (c) semantic entropy (N samples, SAME NLI rule).
# Track forward-pass counts per method for the matched-compute claim.
# ======================================================================
@torch.no_grad()
def p_true(question, proposed):
    """P(True) that `proposed` answers `question`, prompt-only self-eval."""
    msgs = [{"role": "user", "content":
        f"Question: {question}\nProposed answer: {proposed}\n"
        "Is the proposed answer correct? Reply with a single word: True or False."}]
    enc = encode_chat(msgs).to(device)
    logits = model(**enc).logits[0, -1].float()
    logp = torch.log_softmax(logits, dim=-1)
    def _first_id(word):
        return tok(word, add_special_tokens=False)["input_ids"][0]
    lp_true = logp[_first_id(" True")].item()
    lp_false = logp[_first_id(" False")].item()
    # normalize over the two options
    m_ = max(lp_true, lp_false)
    pt = math.exp(lp_true - m_) / (math.exp(lp_true - m_) + math.exp(lp_false - m_))
    return float(pt)

baselines = []
pass_counts = {"divergence": 2, "token_logprob": 1, "p_true": 2, "semantic_entropy": 1 + SE_N}
for a in answers:
    q = q_by_id[a["id"]]
    samples = sample_answers(build_gen_closed(q), n=SE_N)
    se = M.semantic_entropy(
        [M.qa_template(q, s) for s in samples], nli_label, rule="non_defeating")
    baselines.append({
        "id": a["id"], "answerable": a["answerable"],
        "neg_token_logprob": -a["gen_closed_logprob"],   # higher => less confident => more likely wrong
        "p_false": 1.0 - p_true(q, a["gen_closed"]),      # higher => more likely wrong
        "semantic_entropy": se,
        "se_samples": samples,
    })

with open(f"{RESULTS_DIR}/baselines.jsonl", "w") as f:
    for b in baselines:
        f.write(json.dumps(b) + "\n")
print("baselines done. forward-pass counts per method:", pass_counts)


# %%
# ======================================================================
# Cell 8 — metrics & plots. AUROC + bootstrap CI + AUPRC for every signal.
# Headline plot: SE AUROC-vs-N curve with the divergence 2-pass point overlaid.
# Calibration/reliability diagram for d_nli. Save PNGs + metrics.json.
# ======================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_curve

by_id = lambda rows: {r["id"]: r for r in rows}
D, L, B = by_id(divergence), by_id(labels), by_id(baselines)
ids = [l["id"] for l in labels]
y = np.array([L[i]["y"] for i in ids])

signals = {
    "d_nli":            np.array([D[i]["d_nli"] for i in ids]),
    "d_cos":            np.array([D[i]["d_cos"] for i in ids]),
    "d_lex":            np.array([D[i]["d_lex"] for i in ids]),
    "neg_token_logprob": np.array([B[i]["neg_token_logprob"] for i in ids]),
    "p_false":          np.array([B[i]["p_false"] for i in ids]),
    "semantic_entropy": np.array([B[i]["semantic_entropy"] for i in ids]),
}

metrics = {"model": MODEL, "nli_ckpt": NLI_CKPT, "seed": SEED, "smoke": SMOKE,
           "n": len(ids), "positive_rate": float(y.mean()),
           "se_sampling": {"N": SE_N, "T": SE_TEMPERATURE, "top_p": SE_TOP_P, "top_k": SE_TOP_K},
           "forward_passes": pass_counts, "signals": {}}
for name, s in signals.items():
    boot = M.bootstrap_auroc(y, s, n_boot=1000, seed=SEED)
    auprc = float(average_precision_score(y, s)) if len(np.unique(y)) > 1 else float("nan")
    metrics["signals"][name] = {**boot, "auprc": auprc}
    print(f"{name:18s} AUROC={boot['auroc']:.3f} "
          f"[{boot['ci_low']:.3f},{boot['ci_high']:.3f}]  AUPRC={auprc:.3f}")

# Headline: SE AUROC-vs-N curve; divergence as a single 2-pass point.
se_samples_by_id = {i: B[i]["se_samples"] for i in ids}
q_lookup = q_by_id
N_grid = [1, 2, 5, 10]
se_auroc_by_N = []
for N in N_grid:
    se_N = []
    for i in ids:
        s = se_samples_by_id[i][:N]
        se_N.append(M.semantic_entropy([M.qa_template(q_lookup[i], x) for x in s],
                                       nli_label, rule="non_defeating"))
    se_N = np.array(se_N)
    se_auroc_by_N.append(M.bootstrap_auroc(y, se_N, seed=SEED)["auroc"])
metrics["se_auroc_by_N"] = dict(zip(map(str, N_grid), se_auroc_by_N))

plt.figure(figsize=(6, 4))
plt.plot([1 + n for n in N_grid], se_auroc_by_N, "o-", label="semantic entropy (1+N passes)")
plt.scatter([2], [metrics["signals"]["d_nli"]["auroc"]], color="red", zorder=5,
            label="divergence (2 passes)")
plt.xlabel("forward passes"); plt.ylabel("AUROC"); plt.title("Divergence vs SE at matched compute")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/headline_auroc_vs_compute.png", dpi=120); plt.close()

# Calibration/reliability for d_nli (min-max scaled to [0,1] as a pseudo-probability).
d = signals["d_nli"]; d01 = (d - d.min()) / (d.max() - d.min() + 1e-9)
bins = np.linspace(0, 1, 11); idx = np.clip(np.digitize(d01, bins) - 1, 0, 9)
xs, ys = [], []
for b_ in range(10):
    mask = idx == b_
    if mask.sum() > 0:
        xs.append(d01[mask].mean()); ys.append(y[mask].mean())
plt.figure(figsize=(5, 5))
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.plot(xs, ys, "o-"); plt.xlabel("mean scaled d_nli"); plt.ylabel("empirical P(incorrect)")
plt.title("d_nli reliability"); plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/d_nli_calibration.png", dpi=120); plt.close()

with open(f"{RESULTS_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("saved metrics.json + headline/calibration PNGs.")


# %%
# ======================================================================
# Cell 9 — read_data (mandatory). Dump 15 RANDOM items + FP / FN quadrants.
# During /analyze, delegate this to the data-reader subagent instead.
# ======================================================================
rd = random.Random(SEED)
sample_ids = rd.sample(ids, min(15, len(ids)))
print("=== 15 RANDOM items (q | gold | extractive | closed | d_nli | y) ===")
for i in sample_ids:
    it = next(x for x in items if x["id"] == i)
    a = next(x for x in answers if x["id"] == i)
    print(f"\nid={i}  y={L[i]['y']}  d_nli={D[i]['d_nli']:.3f}")
    print(f"  Q: {it['question']}")
    print(f"  gold: {it['golds']}")
    print(f"  extractive: {a['extractive']!r}")
    print(f"  closed:     {a['gen_closed']!r}")

thr = np.median(signals["d_nli"])
print(f"\n=== error quadrants (d_nli threshold = median {thr:.3f}) ===")
fp = [i for i in ids if D[i]["d_nli"] > thr and L[i]["y"] == 0]
fn = [i for i in ids if D[i]["d_nli"] <= thr and L[i]["y"] == 1]
print(f"False positives (high divergence, closed-book CORRECT): {len(fp)}")
print(f"False negatives (low divergence, closed-book WRONG):   {len(fn)}")
for label, bucket in [("FP", fp), ("FN", fn)]:
    for i in rd.sample(bucket, min(3, len(bucket))):
        a = next(x for x in answers if x["id"] == i)
        it = next(x for x in items if x["id"] == i)
        print(f"  [{label}] id={i} d_nli={D[i]['d_nli']:.3f}  Q={it['question']!r}")
        print(f"        ext={a['extractive']!r}  closed={a['gen_closed']!r}  gold={it['golds']}")


# %%
# ======================================================================
# Cell 10 — ablations & control (only if headline clears the bar).
#   (a) Q+A templating on/off AUROC.
#   (b) PS-2 grounding-vs-format: extractive-span vs open-book free-form as grounded arm.
#   (c) PS-3 unanswerable control: d_nli vs pure extractive-abstention detector.
#   (d) correctness-threshold robustness F1@0.5 vs @0.3.
#   (e) NLI checkpoint swap (deberta-v2-xlarge vs deberta-v3-large).
# ======================================================================
ablations = {}

# (a) templating on/off
d_tmpl = np.array([D[i]["d_nli"] for i in ids])
d_untmpl = np.array([D[i]["d_nli_untemplated"] for i in ids])
ablations["templating"] = {
    "on":  M.bootstrap_auroc(y, d_tmpl, seed=SEED),
    "off": M.bootstrap_auroc(y, d_untmpl, seed=SEED)}

# (b) PS-2 grounding vs format
d_openbook = np.array([D[i]["d_nli_openbook"] for i in ids])
ablations["grounding_vs_format"] = {
    "extractive_span": M.bootstrap_auroc(y, d_tmpl, seed=SEED),
    "openbook_freeform": M.bootstrap_auroc(y, d_openbook, seed=SEED)}

# (c) PS-3 control on UNANSWERABLE items: divergence vs extractive-abstention prob.
una_ids = [i for i in ids if not L[i]["answerable"]]
if len(una_ids) > 0 and len(set(L[i]["y"] for i in una_ids)) > 1:
    y_una = np.array([L[i]["y"] for i in una_ids])
    d_una = np.array([D[i]["d_nli"] for i in una_ids])
    A = by_id(answers)
    abstain_prob = np.array([1.0 if _is_abstain(A[i]["extractive"]) else 0.0 for i in una_ids])
    ablations["ps3_unanswerable"] = {
        "d_nli": M.bootstrap_auroc(y_una, d_una, seed=SEED),
        "extractive_abstention": M.bootstrap_auroc(y_una, abstain_prob, seed=SEED)}
else:
    ablations["ps3_unanswerable"] = {"note": "degenerate (single-class or empty) — see abstention rate"}

# (d) correctness-threshold robustness
y_robust = np.array([L[i]["y_robust_f1_0.3"] for i in ids])
ablations["threshold_robustness"] = {
    "f1_0.5": M.bootstrap_auroc(y, d_tmpl, seed=SEED),
    "f1_0.3": M.bootstrap_auroc(y_robust, d_tmpl, seed=SEED)}

# (e) NLI checkpoint swap — recompute d_nli with the alt checkpoint.
# Non-fatal: a download/repo problem here must not wipe out ablations (a)-(d) above.
try:
    _alt_tok = AutoTokenizer.from_pretrained(NLI_CKPT_ALT)
    _alt_model = AutoModelForSequenceClassification.from_pretrained(
        NLI_CKPT_ALT, torch_dtype=torch.float16).to(device).eval()
    _alt_id2label = {int(k): v.lower() for k, v in _alt_model.config.id2label.items()}
    _alt_entail = next(i for i, l in _alt_id2label.items() if "entail" in l)
    print("alt NLI id2label:", _alt_id2label, "entail_idx:", _alt_entail)
    @torch.no_grad()
    def _alt_entail_prob(premise, hypothesis):
        enc = _alt_tok(premise, hypothesis, return_tensors="pt", truncation=True,
                       max_length=256).to(device)
        p = torch.softmax(_alt_model(**enc).logits[0].float(), dim=-1).cpu().numpy()
        return float(p[_alt_entail])
    A = by_id(answers)
    d_alt = np.array([
        M.bidirectional_divergence(M.qa_template(q_by_id[i], A[i]["extractive"]),
                                   M.qa_template(q_by_id[i], A[i]["gen_closed"]), _alt_entail_prob)
        for i in ids])
    ablations["nli_checkpoint"] = {
        NLI_CKPT: M.bootstrap_auroc(y, d_tmpl, seed=SEED),
        NLI_CKPT_ALT: M.bootstrap_auroc(y, d_alt, seed=SEED)}
except Exception as e:
    ablations["nli_checkpoint"] = {"error": f"{type(e).__name__}: {e}"}
    print("NLI checkpoint swap skipped:", ablations["nli_checkpoint"]["error"])

with open(f"{RESULTS_DIR}/ablations.json", "w") as f:
    json.dump(ablations, f, indent=2)
print("saved ablations.json")
for k, v in ablations.items():
    print(k, "->", {kk: (vv.get("auroc") if isinstance(vv, dict) else vv) for kk, vv in v.items()})


# %%
# ======================================================================
# Cell 11 — JUDGE-LABEL RE-ANALYSIS (the decisive follow-up).
# Replaces the lexical token-F1<0.5 correctness label with a NON-LEXICAL
# NLI-vs-gold label, then re-scores every signal. Breaks the circularity
# between the lexical label and the lexical d_lex signal, and makes the
# semantic-entropy comparison fair (Farquhar/Nature grades with F1+judge, not F1).
#
# Self-contained: reads results/*.jsonl + reloads squad + reloads the NLI checkpoint.
# Runs against saved outputs with NO Qwen regeneration. Judge = DeBERTa (a DIFFERENT
# model from Qwen -> no self-enhancement bias, unlike a Qwen-judges-Qwen setup).
# ======================================================================
import json as _json, numpy as _np
from datasets import load_dataset as _load_dataset
from transformers import (AutoTokenizer as _AT,
                          AutoModelForSequenceClassification as _AM)
import torch as _torch
import metrics_lib as _M

JUDGE_THRESH = 0.5   # correct iff max_gold min-bidirectional-entailment >= this

def _read_jsonl(name):
    return {r["id"]: r for r in
            (_json.loads(l) for l in open(f"{RESULTS_DIR}/{name}").read().splitlines())}
_ans = _read_jsonl("answers.jsonl")
_div = _read_jsonl("divergence.jsonl")
_bas = _read_jsonl("baselines.jsonl")
_lab = _read_jsonl("labels.jsonl")
_manifest = _json.load(open(f"{RESULTS_DIR}/slice_manifest.json"))
_ids = [i for i in _manifest["ids"] if i in _ans]

# question + gold per id, from squad (keyed by id; reproducible via the manifest)
_sq = _load_dataset("rajpurkar/squad_v2")["validation"]
_meta = {ex["id"]: {"q": ex["question"], "golds": ex["answers"]["text"]}
         for ex in _sq if ex["id"] in set(_ids)}

# reload the SAME NLI checkpoint used for divergence (no confound)
_jt = _AT.from_pretrained(NLI_CKPT)
_jm = _AM.from_pretrained(NLI_CKPT, torch_dtype=_torch.float16).to(device).eval()
_j_id2label = {int(k): v.lower() for k, v in _jm.config.id2label.items()}
_J_ENTAIL = next(i for i, l in _j_id2label.items() if "entail" in l)

@_torch.no_grad()
def _judge_entail_prob(premise, hypothesis):
    enc = _jt(premise, hypothesis, return_tensors="pt", truncation=True, max_length=256).to(device)
    p = _torch.softmax(_jm(**enc).logits[0].float(), dim=-1).cpu().numpy()
    return float(p[_J_ENTAIL])

def _is_abstain(s):
    return _M.normalize_answer(s) in {"unanswerable", "unknown", "no answer", "cannot answer"}

# --- build the judge label (positive = INCORRECT) ---
y_judge, y_f1, flipped = [], [], 0
for i in _ids:
    a = _ans[i]; m = _meta[i]
    if _lab[i]["answerable"]:
        correct = _M.answer_correct_vs_golds(a["gen_closed"], m["golds"], m["q"],
                                             _judge_entail_prob, thresh=JUDGE_THRESH)
        yj = int(not correct)
    else:
        yj = int(not _is_abstain(a["gen_closed"]))   # unchanged: unanswerable
    y_judge.append(yj); y_f1.append(_lab[i]["y"])
    if _lab[i]["answerable"] and yj != _lab[i]["y"]:
        flipped += 1

y_judge = _np.array(y_judge); y_f1 = _np.array(y_f1)
_ans_mask = _np.array([_lab[i]["answerable"] for i in _ids])
print(f"judge label (thresh={JUDGE_THRESH}): positive={y_judge.mean():.3f} "
      f"(was {y_f1.mean():.3f} under token-F1)")
print(f"answerable labels FLIPPED F1->judge: {flipped}/{int(_ans_mask.sum())} "
      f"({flipped/max(1,int(_ans_mask.sum())):.1%})  <- magnitude of lexical-label noise")

# --- re-score every signal under BOTH labels, ALL and ANSWERABLE-only ---
_sig = {
    "d_nli": _np.array([_div[i]["d_nli"] for i in _ids]),
    "d_cos": _np.array([_div[i]["d_cos"] for i in _ids]),
    "d_lex": _np.array([_div[i]["d_lex"] for i in _ids]),
    "neg_token_logprob": _np.array([_bas[i]["neg_token_logprob"] for i in _ids]),
    "p_false": _np.array([_bas[i]["p_false"] for i in _ids]),
    "semantic_entropy": _np.array([_bas[i]["semantic_entropy"] for i in _ids]),
}
reanalysis = {"judge_thresh": JUDGE_THRESH, "nli_ckpt": NLI_CKPT,
              "positive_rate": {"token_f1": float(y_f1.mean()), "judge": float(y_judge.mean())},
              "answerable_flipped": flipped, "signals": {}}
print(f"\n{'signal':18s} {'F1 all':>7s} {'JUDGE all':>9s} {'F1 ans':>7s} {'JUDGE ans':>9s}")
for name, s in _sig.items():
    def _au(y, mask=None):
        yy, ss = (y, s) if mask is None else (y[mask], s[mask])
        return _M.bootstrap_auroc(yy, ss, seed=SEED)
    r = {"f1_all": _au(y_f1), "judge_all": _au(y_judge),
         "f1_ans": _au(y_f1, _ans_mask), "judge_ans": _au(y_judge, _ans_mask)}
    reanalysis["signals"][name] = r
    g = lambda d: f"{d['auroc']:.3f}" if d["auroc"] == d["auroc"] else "  n/a"
    print(f"{name:18s} {g(r['f1_all']):>7s} {g(r['judge_all']):>9s} "
          f"{g(r['f1_ans']):>7s} {g(r['judge_ans']):>9s}")

with open(f"{RESULTS_DIR}/judge_reanalysis.json", "w") as f:
    _json.dump(reanalysis, f, indent=2)

# --- the verdict this experiment exists to deliver ---
_dnli = reanalysis["signals"]["d_nli"]["judge_ans"]["auroc"]
_dlex = reanalysis["signals"]["d_lex"]["judge_ans"]["auroc"]
_se = reanalysis["signals"]["semantic_entropy"]["judge_ans"]["auroc"]
print("\n=== VERDICT (answerable-only, judge label) ===")
print(f"d_nli={_dnli:.3f}  d_lex={_dlex:.3f}  semantic_entropy={_se:.3f}")
print(f"NLI still loses to lexical? {'YES -> detector deflated' if _dnli < _dlex else 'NO -> circularity was the story'}")
print(f"d_nli still beats SE?       {'YES -> wedge survives' if _dnli > _se else 'NO -> headline dead'}")
print("saved judge_reanalysis.json")


# %%
# ======================================================================
# Cell 12 — INDEPENDENT-JUDGE CHECK (rules out the symmetric confound).
# Cell 11's judge used the SAME DeBERTa (deberta-v2) as d_nli, so d_nli's win
# could be circular (detector & judge share weights + the extractive arm ≈ gold).
# Here the correctness label is built from a DIFFERENT model (NLI_CKPT_ALT =
# MoritzLaurer v3: different weights/training/arch) while d_nli KEEPS its
# deberta-v2 scores from divergence.jsonl. Detector on model A, judge on model B.
#
# Decision: if d_nli STILL beats d_lex, SE, AND token-logprob under this independent
# judge -> the shared-weights circularity is broken and the result is real. If d_nli
# collapses toward the token-logprob floor -> it was judge-circular.
# Self-contained; reads saved results/*.jsonl; no Qwen. ~2 min (loads MoritzLaurer).
# ======================================================================
import json as _json2, numpy as _np2
from datasets import load_dataset as _load_ds2
from transformers import (AutoTokenizer as _AT2,
                          AutoModelForSequenceClassification as _AM2)
import torch as _torch2, metrics_lib as _M2

INDEP_JUDGE_CKPT = NLI_CKPT_ALT           # different model from d_nli's NLI_CKPT
assert INDEP_JUDGE_CKPT != NLI_CKPT, "independent judge must differ from the detector's NLI"
INDEP_THRESH = 0.5

def _rj(name):
    return {r["id"]: r for r in
            (_json2.loads(l) for l in open(f"{RESULTS_DIR}/{name}").read().splitlines())}
_ans2, _div2, _bas2, _lab2 = _rj("answers.jsonl"), _rj("divergence.jsonl"), _rj("baselines.jsonl"), _rj("labels.jsonl")
_man2 = _json2.load(open(f"{RESULTS_DIR}/slice_manifest.json"))
_ids2 = [i for i in _man2["ids"] if i in _ans2]
_sq2 = _load_ds2("rajpurkar/squad_v2")["validation"]
_meta2 = {ex["id"]: {"q": ex["question"], "golds": ex["answers"]["text"]}
          for ex in _sq2 if ex["id"] in set(_ids2)}

# load the INDEPENDENT judge; read its entail index dynamically (v3 label order differs)
_it = _AT2.from_pretrained(INDEP_JUDGE_CKPT)
_im = _AM2.from_pretrained(INDEP_JUDGE_CKPT, torch_dtype=_torch2.float16).to(device).eval()
_i_id2label = {int(k): v.lower() for k, v in _im.config.id2label.items()}
_I_ENTAIL = next(i for i, l in _i_id2label.items() if "entail" in l)
print(f"independent judge = {INDEP_JUDGE_CKPT}\n  id2label={_i_id2label} entail_idx={_I_ENTAIL}")

@_torch2.no_grad()
def _indep_entail_prob(premise, hypothesis):
    enc = _it(premise, hypothesis, return_tensors="pt", truncation=True, max_length=256).to(device)
    p = _torch2.softmax(_im(**enc).logits[0].float(), dim=-1).cpu().numpy()
    return float(p[_I_ENTAIL])

def _abst2(s):
    return _M2.normalize_answer(s) in {"unanswerable", "unknown", "no answer", "cannot answer"}

# independent judge label
y_ind, y_f1b, flip_vs_f1 = [], [], 0
for i in _ids2:
    a = _ans2[i]; m = _meta2[i]
    if _lab2[i]["answerable"]:
        correct = _M2.answer_correct_vs_golds(a["gen_closed"], m["golds"], m["q"],
                                              _indep_entail_prob, thresh=INDEP_THRESH)
        yi = int(not correct)
    else:
        yi = int(not _abst2(a["gen_closed"]))
    y_ind.append(yi); y_f1b.append(_lab2[i]["y"])
    if _lab2[i]["answerable"] and yi != _lab2[i]["y"]:
        flip_vs_f1 += 1
y_ind = _np2.array(y_ind); y_f1b = _np2.array(y_f1b)
_amask2 = _np2.array([_lab2[i]["answerable"] for i in _ids2])
print(f"independent-judge positive={y_ind.mean():.3f}  answerable flipped vs F1={flip_vs_f1}")

_sig2 = {k: _np2.array([( _div2 if k in ('d_nli','d_cos','d_lex') else _bas2)[i][k] for i in _ids2])
         for k in ("d_nli","d_cos","d_lex","neg_token_logprob","p_false","semantic_entropy")}

# try to load Cell 11's deberta-v2 judge numbers for a side-by-side
try:
    _v2 = _json2.load(open(f"{RESULTS_DIR}/judge_reanalysis.json"))["signals"]
except Exception:
    _v2 = None

indep = {"judge_ckpt": INDEP_JUDGE_CKPT, "detector_nli": NLI_CKPT, "thresh": INDEP_THRESH,
         "positive_rate": float(y_ind.mean()), "answerable_flipped_vs_f1": flip_vs_f1, "signals": {}}
print(f"\n{'signal':18s} {'v2-judge ans':>12s} {'INDEP-judge ans':>16s}")
for name, s in _sig2.items():
    au = _M2.bootstrap_auroc(y_ind[_amask2], s[_amask2], seed=SEED)
    indep["signals"][name] = au
    prev = f"{_v2[name]['judge_ans']['auroc']:.3f}" if _v2 else "   —"
    print(f"{name:18s} {prev:>12s} {au['auroc']:>10.3f} [{au['ci_low']:.3f},{au['ci_high']:.3f}]")

with open(f"{RESULTS_DIR}/judge_independent.json", "w") as f:
    _json2.dump(indep, f, indent=2)

# --- the decisive verdict ---
_dn = indep["signals"]["d_nli"]["auroc"]
_dl = indep["signals"]["d_lex"]["auroc"]
_s = indep["signals"]["semantic_entropy"]["auroc"]
_lp = indep["signals"]["neg_token_logprob"]["auroc"]
print("\n=== INDEPENDENT-JUDGE VERDICT (answerable-only) ===")
print(f"d_nli={_dn:.3f}  d_lex={_dl:.3f}  SE={_s:.3f}  token-logprob={_lp:.3f}")
_beats_all = _dn > _dl and _dn > _s and _dn > _lp
if _beats_all:
    print("d_nli beats d_lex, SE, AND token-logprob under an INDEPENDENT judge")
    print("-> shared-weights circularity BROKEN; result holds. Real finding.")
else:
    loser_to = [n for n, v in (("d_lex",_dl),("SE",_s),("token-logprob",_lp)) if _dn <= v]
    print(f"d_nli no longer beats: {loser_to}")
    print("-> d_nli's win was (partly) judge-circular; deflate accordingly.")
print("saved judge_independent.json")


# %%
# ======================================================================
# Cell 13 — PAIRED SIGNIFICANCE TESTS (turns "point estimate wins" into "wins, p<.05").
# Comparing two detectors' marginal CIs is the WRONG test — they score the SAME items,
# so the AUROCs are correlated. The paired bootstrap of the DIFFERENCE is correct: its
# CI can exclude 0 even when the marginal CIs overlap. Runs d_nli vs {SE, token-logprob,
# d_lex} on answerable-only, under the INDEPENDENT-judge label (the cleanest one).
#
# Caches per-item independent-judge labels to results/judge_independent_labels.json on
# first run -> subsequent runs are pure-numpy (no model, no GPU needed).
# ======================================================================
import json as _json3, numpy as _np3, os as _os3, metrics_lib as _M3

def _rj3(name):
    return {r["id"]: r for r in
            (_json3.loads(l) for l in open(f"{RESULTS_DIR}/{name}").read().splitlines())}
_ans3, _div3, _bas3, _lab3 = _rj3("answers.jsonl"), _rj3("divergence.jsonl"), _rj3("baselines.jsonl"), _rj3("labels.jsonl")
_man3 = _json3.load(open(f"{RESULTS_DIR}/slice_manifest.json"))
_ids3 = [i for i in _man3["ids"] if i in _ans3]
_LABELS_PATH = f"{RESULTS_DIR}/judge_independent_labels.json"

if _os3.path.exists(_LABELS_PATH):
    _ylab = _json3.load(open(_LABELS_PATH))           # {id: y_independent_judge}
    print(f"loaded cached independent-judge labels ({len(_ylab)} items) — no model needed")
else:
    print("no cached labels -> computing independent-judge labels once (loads MoritzLaurer)...")
    from datasets import load_dataset as _ld3
    from transformers import (AutoTokenizer as _AT3, AutoModelForSequenceClassification as _AM3)
    import torch as _torch3
    _sq3 = _ld3("rajpurkar/squad_v2")["validation"]
    _meta3 = {ex["id"]: {"q": ex["question"], "golds": ex["answers"]["text"]}
              for ex in _sq3 if ex["id"] in set(_ids3)}
    _jt3 = _AT3.from_pretrained(NLI_CKPT_ALT)
    _jm3 = _AM3.from_pretrained(NLI_CKPT_ALT, torch_dtype=_torch3.float16).to(device).eval()
    _j3 = {int(k): v.lower() for k, v in _jm3.config.id2label.items()}
    _E3 = next(i for i, l in _j3.items() if "entail" in l)
    @_torch3.no_grad()
    def _ep3(premise, hypothesis):
        enc = _jt3(premise, hypothesis, return_tensors="pt", truncation=True, max_length=256).to(device)
        return float(_torch3.softmax(_jm3(**enc).logits[0].float(), dim=-1).cpu().numpy()[_E3])
    def _ab3(s): return _M3.normalize_answer(s) in {"unanswerable","unknown","no answer","cannot answer"}
    _ylab = {}
    for i in _ids3:
        if _lab3[i]["answerable"]:
            ok = _M3.answer_correct_vs_golds(_ans3[i]["gen_closed"], _meta3[i]["golds"],
                                             _meta3[i]["q"], _ep3, thresh=0.5)
            _ylab[i] = int(not ok)
        else:
            _ylab[i] = int(not _ab3(_ans3[i]["gen_closed"]))
    _json3.dump(_ylab, open(_LABELS_PATH, "w"))
    print(f"saved {_LABELS_PATH}")

# answerable-only arrays
_aids = [i for i in _ids3 if _lab3[i]["answerable"]]
_y = _np3.array([_ylab[i] for i in _aids])
def _S(key, src):
    return _np3.array([({"div": _div3, "bas": _bas3}[src])[i][key] for i in _aids])
_scores = {"d_nli": _S("d_nli","div"), "semantic_entropy": _S("semantic_entropy","bas"),
           "neg_token_logprob": _S("neg_token_logprob","bas"), "d_lex": _S("d_lex","div")}

# marginal AUROCs (for context) + the paired difference tests
_marg = {k: _M3.bootstrap_auroc(_y, s, seed=SEED)["auroc"] for k, s in _scores.items()}
print(f"\nanswerable n={len(_aids)}  positives={int(_y.sum())}  negatives={int((_y==0).sum())}")
print("marginal AUROC:", {k: round(v, 3) for k, v in _marg.items()})

paired = {}
print(f"\n{'comparison':32s} {'dAUROC':>7s} {'95% CI':>18s} {'p':>7s}  sig")
for other in ["semantic_entropy", "neg_token_logprob", "d_lex"]:
    r = _M3.paired_bootstrap_auroc_diff(_y, _scores["d_nli"], _scores[other], n_boot=2000, seed=SEED)
    paired[f"d_nli_vs_{other}"] = r
    ci = f"[{r['ci_low']:+.3f},{r['ci_high']:+.3f}]"
    print(f"{'d_nli - ' + other:32s} {r['diff']:>+7.3f} {ci:>18s} {r['p_value']:>7.3f}  "
          f"{'YES' if r['significant'] else 'no'}")

_json3.dump({"label": "independent_judge", "marginal_auroc": _marg, "paired": paired},
            open(f"{RESULTS_DIR}/paired_tests.json", "w"), indent=2)
print("\nsaved paired_tests.json")
print("Read: 'sig YES' on d_nli - semantic_entropy => Claim 2 holds with significance.")
print("      'sig no'  on d_nli - neg_token_logprob => efficiency-over-logprob NOT established (need larger n).")
