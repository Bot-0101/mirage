# problem-statement.md — extractive-generative-divergence

**Phase:** scoped (2026-07-07) · next: `/novelty`
**Model:** Qwen2.5-3B-Instruct · **Data:** SQuAD 2.0 · **Compute:** Kaggle · **Seed:** 42

## Framing
Building on my published extractive-vs-generative QA comparison. The signal under test is
**grounding divergence**: for one fixed model run in two modes, how far its *grounded* answer
drifts from its *ungrounded* answer, used as a **label-free confabulation detector**.
"Label-free" is a property of the *deployed* signal — gold labels are used ONLY to validate the
detector here, never to compute it.

Two knobs are bundled in "extractive vs generative"; we make **grounding** the manipulated
variable and demote **format** to an ablation:
- **Extractive / grounded arm:** context provided, answer constrained to a verbatim span (or abstain).
- **Generative / parametric arm:** no context, free-form answer from parametric knowledge.

## Hypothesis (one sentence)
For Qwen2.5-3B-Instruct, the **semantic divergence** between its open-book extractive answer and
its closed-book generative answer **predicts whether the closed-book answer is a confabulation**,
and does so competitively with semantic entropy at a fraction of the compute.

## The one experiment (minimal, ~500 items)
1. Sample ~500 **answerable** SQuAD 2.0 items (balanced by article; seed 42).
2. **Arm A (grounded/extractive):** prompt with context, "answer using only a verbatim span from
   the context, or say unanswerable." Verify the answer is a context substring (else flag as
   non-compliant, don't silently treat as extractive).
3. **Arm B (parametric/generative):** prompt with the question only, no context, free-form answer.
4. **Divergence score** `d = 1 − BiNLI(A,B)` where BiNLI = min of both-direction entailment probs
   from a small NLI model (DeBERTa-MNLI). Cross-check: `1 − MiniLM-cosine(A,B)`.
5. **Validation label** `y = 1` if arm B is wrong vs gold (token-F1 < τ AND NLI-contradicts gold),
   else 0.
6. **Primary metric:** AUROC(`d` → `y`). Report AUPRC + calibration too.

## Baseline it MUST beat (repo rule)
**Semantic entropy** (Farquhar/Kuhn) over N closed-book samples, at **matched forward-pass budget**
(divergence ≈ 2 passes; semantic entropy ≈ N passes). Also compare against the closed-book answer's
own mean token log-prob (cheap confidence baseline). Divergence is only interesting if it rivals
semantic entropy at materially lower compute.

## Built-in ablation + control (repo rules: always ablate, kill the confound)
- **Ablation (format knob):** give arm B the context too (span-constrained vs free-form, both grounded).
  Tells us whether grounding or format is load-bearing.
- **Control (triviality):** SQuAD 2.0 **unanswerable** items — does divergence beat a pure
  *extractive-abstention* detector? If abstention alone matches its AUROC, the signal is just an
  abstention detector, not a divergence signal.

## Falsifiers (what kills the hypothesis)
- AUROC ≈ 0.5 on PS-1 → signal is fake, stop.
- Divergence AUROC ≤ semantic-entropy AUROC at matched compute → not worth it.
- Divergence adds nothing over (a) the closed-book answer's token log-prob, or (b) extractive
  abstention probability → it's re-encoding a cheaper signal, not a new one.

## Assumptions (to check early)
1. On answerable SQuAD, the **open-book extractive answer is a reliable correctness proxy** (the
   grounded arm is usually right) — verify by spot-checking arm A vs gold; if arm A is often wrong,
   the whole "grounded = trustworthy anchor" premise weakens.
2. Qwen2.5-3B **follows both instructions reliably** (span-constrain/abstain vs free-form) — verify
   compliance rate in the first slice; low compliance confounds divergence.
3. BiNLI on short QA answers gives sensible equivalence judgments — eyeball ~20 random pairs before
   trusting the AUROC.

## Deliberately out of scope (for now)
Mechanistic "where in the residual stream do the arms diverge" (earned only if PS-1 survives);
larger models; the lecture-video domain (reintroduce as a later robustness axis).
