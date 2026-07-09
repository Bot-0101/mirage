# workflow-plan.md — extractive-generative-divergence

**Phase:** planned (2026-07-08) · Model: Qwen2.5-3B-Instruct · Data: SQuAD 2.0 · Compute: Kaggle · Seed: 42
Read alongside `problem-statement.md`, `novelty-result.md`, `impl-plan.md`.

## Hypothesis (from problem-statement)
The semantic divergence between Qwen2.5-3B's **open-book extractive** answer and its **closed-book
generative** answer predicts whether the closed-book answer is a confabulation, and does so
competitively with **semantic entropy at matched forward-pass compute**.

## Falsifiers (any one kills / deflates it)
1. AUROC(d_NLI) ≈ 0.5 on the ~500-item slice → signal is fake. **Stop.**
2. AUROC(d_NLI) ≤ AUROC(semantic entropy) at matched compute → not worth it (SE already exists).
3. AUROC(d_NLI) ≤ AUROC(closed-book token-logprob) → re-encoding a near-free signal, not new.
4. On the unanswerable control, divergence ≤ a pure **extractive-abstention** detector → it's just abstention.
5. AUROC(d_NLI) ≈ AUROC(d_lexical-F1) → the NLI machinery is not load-bearing (method over-engineered).

## explore → understand → distill

### EXPLORE (fast slices; goal = is there any signal at all?)
- **Slice A — smoke (~20 items):** run the whole pipeline end-to-end. Success = it runs, chat-template
  print looks right, extractive substring-compliance is measured, a plausible AUROC comes out. No belief yet.
- **Slice B — headline (~500 items, 250 answerable / 250 unanswerable, stratified, seed 42):** compute
  AUROC(d_NLI → closed-book-wrong) with bootstrap CI. Eyeball 15 RANDOM items.
- **Stopping / pivot rules (set before looking):**
  - Extractive substring-compliance < ~70% → the grounded arm is unreliable; fix the prompt or pivot to
    open-book free-form BEFORE trusting any AUROC. (Don't interpret a signal built on a broken arm.)
  - AUROC ≤ 0.55 on Slice B, OR ≤ token-logprob baseline → signal is dead/derivative. Pivot or stop;
    do not spend effort on ablations.
  - Any AUROC that looks *great* (>0.85) → 5-min "how is this an artifact?" timer (per repo rule); prime
    suspects: divergence re-encoding answerability, length/format confound, or a leaked correctness label.

### UNDERSTAND (only if EXPLORE clears the bar: AUROC clearly >0.6 and > token-logprob)
- **Headline experiment:** semantic entropy (discrete variant) AUROC-vs-N curve; plot divergence (2
  generation passes) as a single point against it (convention from arXiv:2504.03579). The claim lives here.
- **Ablations (always ablate — report which parts are load-bearing):**
  - *Q+A templating on/off* for the NLI divergence (the method hinges on this; scout flags it as
    common-practice-not-isolated). If AUROC collapses without templating, say so.
  - *Grounding vs format (PS-2):* extractive-span vs free-form-with-context as the grounded arm. This
    **isolates the delta over CDD** (which used free-form "extract-implied" answers). If the span
    constraint doesn't move AUROC, the contribution narrows to detector-framing + SE benchmark.
  - *NLI checkpoint robustness:* deberta-v2-xlarge-mnli vs deberta-v3-large-mnli (same checkpoint used for
    BOTH divergence and SE clustering — differing checkpoints = confound).
  - *Correctness threshold:* F1≥0.5 vs F1≥0.3 (and/or LLM-judge) — AUROC should be threshold-stable.
- **Control (PS-3, kill the triviality confound):** on unanswerable items, does d_NLI beat a pure
  extractive-abstention-probability detector? If abstention alone matches it, the headline is deflated.
- **Baselines (comparable effort on each — repo rule):** semantic entropy [headline], closed-book
  token-logprob [floor], P(True) prompt-only self-eval [obvious black-box alt], lexical-F1 divergence
  [is NLI needed?]. Report all on one AUROC table + PR curves.

### Reading the data (what "read the data" means here)
Dump **15 random (not cherry-picked)** items: question, gold, extractive answer, closed-book answer,
d_NLI, correctness label. Then specifically inspect the two error quadrants:
- **False positives** (high divergence, but closed-book actually correct) — what is divergence firing on?
  (paraphrase? extra detail? format?)
- **False negatives** (low divergence, but closed-book wrong) — confident agreement on a shared error.
These reveal what the signal *actually* measures vs. what we claim.

### DISTILL (candidate claims, each tied to the evidence that isolates it)
1. Divergence detects closed-book confabulation on SQuAD 2.0 (AUROC = X, CI), > token-logprob floor.
2. **Headline:** at matched forward-pass compute (~2 passes), divergence rivals/beats semantic entropy
   (SE needs N passes for equal AUROC) — the compute-efficiency contribution.
3. The verbatim-span constraint contributes Δ over free-form context answers (the CDD delta), per PS-2.
Drop any claim whose evidence is too thin; state exactly what was and wasn't shown.

## Confirm vs. kill (decision at end of UNDERSTAND)
- **Confirm** = AUROC(d_NLI) ≥ SE-at-matched-compute (overlapping/upper CI) AND > token-logprob AND
  survives the PS-3 control AND isn't explained by an inspected confound.
- **Kill/deflate** = any falsifier above fires. Log it honestly in `/logbook` — a rigorous negative
  (e.g. "divergence works but never beats SE at matched compute") is a real result, not a failure.
