# impl-plan.md — extractive-generative-divergence

**Phase:** planned (2026-07-08) · Source of truth for `/implement`. Read with `experiment.config.md` + `workflow-plan.md`.
**Library:** HF `transformers` · **Compute:** Kaggle GPU · **Model:** Qwen2.5-3B-Instruct · **Seed:** 42
**Divergence NLI + SE-clustering NLI:** ONE checkpoint (default `microsoft/deberta-v2-xlarge-mnli`) — using different checkpoints for method vs baseline is a confound.

## Grounded implementation choices (from paper-scout; cite in writeup)
- **Semantic entropy = DISCRETE variant** (`cluster_assignment_entropy`, Shannon entropy over cluster
  frequencies) — the only SE variant valid in a pure black-box HF-generate setup. Sampling N=10 @ T=1.0,
  top-p=0.9, top-k=50. Main answer = greedy (pin & document; sources disagree greedy vs T=0.1).
- **Bidirectional-NLI clustering rule:** implement Farquhar's default **non-defeating** (no contradiction
  in either direction, not mutually neutral) AND strict (both-directions-entail); pick non-defeating as
  primary, document explicitly (blogs conflate the two).
- **Short-answer NLI templating (load-bearing):** never feed bare spans to DeBERTa. Wrap both answers as
  `"Q: {question} A: {answer}"` before NLI (Farquhar/Kuhn practice). Ablate on/off.
- **Correctness label:** SQuAD token-F1 ≥ 0.5 vs gold (robustness pass at 0.3). Positive class = INCORRECT.
- **Extractive compliance:** verify `answer.strip()` is a normalized substring of context; log
  non-compliance + fallback rate — do NOT assume compliance.
- **DBA judge rubric (Table 14)** reused only as the tertiary LLM-judge equivalence cross-check.

## Data flow
`build_slice → elicit_answers → {score_divergence, label_correctness, baseline_signals} → metrics/plots → read_data`
Everything keyed by SQuAD `id`. Persist each stage to `results/` as JSONL so cells are restartable.

## Notebook cells (each = a vertical slice producing a number or plot)

**Cell 0 — config & seed.** Imports; `SEED=42`; seed `random/numpy/torch` + HF `set_seed`; `device`;
`RESULTS_DIR="/kaggle/working/results"`. (Snippet already in `experiment.config.md`.)

**Cell 1 — load model + SANITY (tokenizer-first).** Load Qwen2.5-3B-Instruct (fp16). Print the
`apply_chat_template(...)` RENDERED string for one extractive + one generative prompt (verify Qwen's
silent default system prompt is what we expect) and its token count. Reproduce one trivial known QA
greedily. **Gate: do not proceed until the rendered prompt is correct.**

**Cell 2 — build_dataset.** `load_dataset("rajpurkar/squad_v2")["validation"]`; split into answerable
(`len(answers.text)>0`) / unanswerable pools; stratified `random.sample(seed=42)` → 250 + 250. Save
`results/slice_manifest.json` (exact ids + indices) for reproducibility.

**Cell 3 — prompt builders + generate().** Three prompt templates:
`extractive` ("answer with a verbatim span from context, or exactly 'unanswerable'"),
`gen_closed` (question only, free-form), `gen_openbook` (question + context, free-form; for PS-2 ablation).
Greedy `generate()` wrapper returning text + mean token log-prob (via `output_scores`). Print 3 RANDOM
rendered prompts.

**Cell 4 — elicit_answers.** For every item: run `extractive`, `gen_closed`, `gen_openbook`. Save
`results/answers.jsonl`. Compute & print **extractive substring-compliance rate** and abstention rate.
(Gate per workflow-plan: compliance <~70% ⇒ fix prompt / pivot before trusting downstream numbers.)

**Cell 5 — score_divergence.** Load the ONE NLI checkpoint. `bi_nli(a,b)` with Q+A templating →
`d_nli` for (extractive, gen_closed) [headline] and (extractive, gen_openbook) [PS-2]. Also `d_cos`
(all-MiniLM-L12-v2 cosine) and `d_lex` (token-F1 overlap between the two answers, the dumb baseline).
Save `results/divergence.jsonl`.

**Cell 6 — label_correctness.** `token_f1(gen_closed, gold)`; `y = (f1 < 0.5)` for answerable; for
unanswerable, `y = not_abstained(gen_closed)`. Also store f1@0.3. Save `results/labels.jsonl`.

**Cell 7 — baseline_signals.** (a) closed-book mean token-logprob (from Cell 3); (b) `p_true` prompt-only
self-eval ("Is the proposed answer correct? True/False", take P(True)); (c) **semantic entropy**: N=10
samples of `gen_closed`, cluster via the SAME NLI rule, discrete cluster entropy. Track forward-pass
counts per method. Save `results/baselines.jsonl`.

**Cell 8 — metrics & plots.** `roc_auc_score` + bootstrap CI + AUPRC for {d_nli, d_cos, d_lex,
token-logprob, p_true, semantic_entropy}. **Headline plot:** SE AUROC-vs-N curve (N=1,2,5,10) with the
divergence 2-pass point overlaid (convention: arXiv:2504.03579). Calibration/reliability diagram for
d_nli. Save PNGs + `results/metrics.json`.

**Cell 9 — read_data (mandatory).** Dump 15 RANDOM items (q, gold, extractive, closed, d_nli, y). Print
the false-positive and false-negative quadrants separately. (Delegate to `data-reader` during `/analyze`.)

**Cell 10 — ablations & control.** Templating on/off AUROC; PS-2 grounding-vs-format AUROC; PS-3
unanswerable: d_nli vs abstention-prob AUROC; F1@0.5 vs @0.3; NLI-checkpoint swap. Save `results/ablations.json`.

## Vertical build order
1. Cells 0–4 on **20 items** (smoke) → plumbing + compliance number.
2. Cells 5–9 on the **500-item** slice → headline AUROC + SE plot + random-data read.
3. Cell 10 → ablations/control (only if headline clears the bar).

## Risks → how each is caught
| Risk | Caught by |
|---|---|
| Chat-template silent corruption (Qwen default sys prompt) | Cell 1 prints rendered prompt; gate before proceeding |
| Extractive arm doesn't comply (paraphrases span) | Cell 4 substring-compliance rate + fallback policy; pivot rule |
| NLI misjudges bare short spans | Q+A templating (Cell 5) + on/off ablation (Cell 10) |
| Weak model never abstains on unanswerable | Cell 4 abstention rate; if ~0, PS-3 control degenerates — note it |
| Divergence just re-encodes answerability / logprob | token-logprob baseline (Cell 7) + PS-3 control (Cell 10) |
| SE reproduction wrong | sanity-check SE AUROC vs published ballpark (~0.75–0.79 on QA); pin sampling params |
| Matched-compute unfairness | count generation passes explicitly (Cell 7); report NLI overhead separately; be fair (SE main+N vs 2 elicitations) |
| Correctness-label leakage | label uses GOLD only, never the extractive arm; keep separate (verify in Cell 6) |
| Class imbalance inflates AUROC | report AUPRC alongside (Cell 8) |

## Reproducibility & outputs
Seed 42 pinned in Cell 0; `slice_manifest.json` records exact ids; all stage outputs → `/kaggle/working/results/`
(persist as Kaggle output); every plot PNG + `metrics.json`/`ablations.json` saved. Record model revision + NLI
checkpoint + SE sampling params in `metrics.json`.
