# REFERENCES.md — extractive-generative-divergence

Legend: ✓ verified-by-me · ◐ scout-reported (abstract/snippet only, unverified) · ⚠ flagged (verify ID/claim)

## Nearest collisions (verify these first)
**arXiv:2605.14473** — *Does RAG Know When Retrieval Is Wrong? Diagnosing Context Compliance under Knowledge Conflict (CDD).* ✓ VERIFIED (read full text 2026-07-08) — elicits same-model contextual vs parametric answers + NLI-contradiction scoring, BUT diagnostic only: no AUROC, no SE benchmark, free-form (not span) grounded arm. Nearest collision on mechanism; not a detector.
**arXiv:2602.04853** — *Decomposed Prompting Does Not Fix Knowledge Gaps, But Helps Models Say "I Don't Know" (DBA).* ✓ VERIFIED (read full text 2026-07-08) — training-free disagreement→abstention detector WITH AUROC, but both arms closed-book (Direct vs Decomposed), LLM-judge scoring, baselines = Self-Consistency (NOT semantic entropy). Right mechanic, wrong axis.
**arXiv:2603.25450** — *Cross-Model Disagreement as a Label-Free Correctness Signal.* ◐ SCOUT — label-free divergence detector with explicit compute-cheap (single verifier pass) framing; cross-model not cross-mode.

## Baseline / compute rival
**arXiv:2406.15927** — *Semantic Entropy Probes (SEP): Robust and Cheap Hallucination Detection in LLMs.* ◐ SCOUT — single-pass cheap-SE, SQuAD-evaluated; the direct compute-cost rival to beat/match.
**PMC11186750 (Nature 2024)** — *Detecting hallucinations in LLMs using semantic entropy* (Farquhar/Kuhn). ◐ SCOUT — sampling-based SE; primary benchmark target; source of the bidirectional-NLI machinery.
**arXiv:2303.08896** — *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection.* ◐ SCOUT — sampling-consistency + NLI contradiction machinery.

## White-box parametric-vs-context cousins
**arXiv:2505.07528** — *SEReDeEP: Hallucination Detection in RAG via Semantic Entropy and Context-Parameter Fusion.* ◐ SCOUT — parametric-vs-context disequilibrium, white-box.
**arXiv:2410.11414** — *ReDeEP: Detecting Hallucination in RAG via Mechanistic Interpretability* (ICLR 2025). ◐ SCOUT — white-box FFN/copy-head parametric-vs-context detector.
**arXiv:2509.21875** — *LUMINA: Detecting Hallucinations in RAG with Context–Knowledge Signals.* ◐ SCOUT — white-box context/knowledge utilization signals.

## Analysis-only (conflict behavior, not detectors)
**arXiv:2409.08435** — *When Context Leads but Parametric Memory Follows.* ◐ SCOUT — no-context answer as parametric baseline vs context answer (analysis, not a detector).
**arXiv:2211.05655** — *DisentQA.* ◐ SCOUT — disentangling parametric vs contextual answers.
**arXiv:2109.05052** — *Entity-Based Knowledge Conflicts in QA* (Longpre 2021). ⚠ FLAG (ID inferred) — foundational parametric-vs-contextual answer-pair lineage.

## Correctness-label / eval-confound prior art (added 2026-07-08, novelty-scout)
Context: our token-F1<0.5 label is (a) noisy on verbose-correct answers and (b) circular with the lexical
`d_lex` signal. Checked whether "de-lexicalizing the label reorders hallucination detectors" is known.
**Verdict: largely already done** — 2508.08285 is the direct pre-emption. Our contribution reduces to a
sharper mechanism-level instantiation (token-F1-divergence vs NLI). VERIFIED manually 2026-07-08 (2 WebFetch
reads, abstract+HTML, agree; table-level numbers still fetch-grade). Pre-emption is *stronger* than the scout
thought: they use the SAME dataset+regime (SQuAD 2.0 nocontext), so "SQuAD 2.0 is a different testbed" is NOT a
real differentiator. Remaining daylight is narrow but real (see row).
**arXiv:2508.08285** — *The Illusion of Progress: Re-evaluating Hallucination Detection in LLMs* (EMNLP 2025). ✓ VERIFIED (WebFetch abstract+HTML, 2026-07-08; qualitative claims solid, exact table figures fetch-grade). Switching ROUGE→LLM-judge correctness label drops detector AUROC up to **45.9%** and reorders rankings; on **SQuAD 2.0 rc.nocontext** (4,150 val), Llama-3.1-8B + Mistral-7B. Detectors = Perplexity, LN-Entropy, Semantic Entropy, EigenScore, eRank, LogDet + length heuristics — **all uncertainty signals, NONE a lexical answer-divergence detector.** Confound attributed to **response LENGTH** correlating with ROUGE; **no explicit label↔detector circularity claim** ("metric–method alignment bias" was the scout's paraphrase, NOT the paper's term). ⇒ Our daylight: (a) `d_lex` is a lexical *divergence detector* (not in their lineup); (b) our circularity mechanism ≠ their length mechanism. Confirmatory/sharper instance of a published pitfall — not a standalone contribution.
**arXiv:2504.13677** — *Revisiting UQ Evaluation… Spurious Interactions with Response Length Bias.* ◐ SCOUT — label-side confound (length × correctness metric) in UQ-detector eval; possible statistical framing for the writeup.
**s41586-024-07421-0 (Nature 630:625, 2024)** — *Detecting hallucinations… using semantic entropy* (Farquhar). ◐ SCOUT — **grades correctness with SQuAD-F1 PLUS GPT-4 gold comparison.** ⇒ our F1-only label likely handicapped the SE baseline. Confirm in Methods/SI.
**arXiv:2302.09664** — *Semantic Uncertainty* (Kuhn 2023, ICLR). ◐ SCOUT — reports robustness to correctness measure but only *within lexical* (EM, Rouge-1) — so it could not have caught the lexical circularity. Supports our motivation.
**arXiv:2202.07654** — *Tomayto, Tomahto / BEM* (Bulian 2022, EMNLP). ◐ SCOUT — canonical "replace F1 because it undercounts correct answers"; learned answer-equivalence + 23k human AE judgments. Q1 done.
**arXiv:2402.11161** — *CFMatch.* ◐ SCOUT — cheaper interpretable BEM successor (answer-equivalence metric).
**arXiv:2305.06984** — *Evaluating ODQA in the Era of LLMs.* ◐ SCOUT — EM/F1 under-counts LLM QA correctness; argues for judge-based eval.
**arXiv:2306.05685** — *Judging LLM-as-a-Judge / MT-Bench* (Zheng 2023). ◐ SCOUT — Q4: GPT-4 judge ~80% human agreement; position/verbosity/self-enhancement biases (caveats if we adopt a judge label).
**arXiv:2606.19544** / **arXiv:2512.16041** — recent LLM-as-judge reliability/consistency/bias audits. ◐ SCOUT — Q4 successors; modern judge caveats.
