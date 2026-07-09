"""Pure-logic helpers for extractive-generative-divergence.

No torch / no model dependencies — numpy + sklearn only, so the load-bearing
scoring logic (SQuAD F1, discrete semantic-entropy clustering, bootstrap AUROC)
is unit-testable off-GPU. The DeBERTa / MiniLM / Qwen wrappers live in run.py and
inject their NLI/entailment functions into the clustering + divergence helpers here.

Seed 42 pinned by callers; bootstrap takes an explicit seed.
"""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Callable, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------- SQuAD F1

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(s: str) -> str:
    """SQuAD normalization: lowercase, strip punctuation, articles, extra whitespace."""
    s = s.lower()
    s = s.translate(_PUNCT_TABLE)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def token_f1(pred: str, gold: str) -> float:
    """SQuAD token-level F1 between two strings."""
    pred_toks = normalize_answer(pred).split()
    gold_toks = normalize_answer(gold).split()
    # Match official SQuAD edge case: if either side is empty, F1 is 1.0 only when both empty.
    if len(pred_toks) == 0 or len(gold_toks) == 0:
        return float(pred_toks == gold_toks)
    common = Counter(pred_toks) & Counter(gold_toks)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(pred_toks)
    recall = n_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def squad_f1_max(pred: str, golds: Sequence[str]) -> float:
    """Max token-F1 over the list of gold answers (SQuAD reports the best-matching gold)."""
    if len(golds) == 0:
        return 0.0
    return max(token_f1(pred, g) for g in golds)


# ---------------------------------------------------------------- compliance

def is_substring_compliant(answer: str, context: str) -> bool:
    """True if the (normalized) answer is a contiguous normalized substring of context.

    Normalization matters: raw substring checks fail on trivial casing/punctuation
    differences even when the model did extract a verbatim span.
    """
    a = normalize_answer(answer)
    if a == "":
        return False
    return a in normalize_answer(context)


# ------------------------------------------------ semantic-entropy clustering

def cluster_answers(
    answers: Sequence[str],
    equivalent_fn: Callable[[str, str], bool],
) -> list[int]:
    """Greedy first-match clustering (Kuhn/Farquhar): assign each answer to the first
    existing cluster whose representative it is equivalent to, else start a new cluster.

    Returns a list of cluster ids (ints) aligned with `answers`.
    """
    reps: list[str] = []          # representative (first member) of each cluster
    assignments: list[int] = []
    for ans in answers:
        placed = False
        for cid, rep in enumerate(reps):
            if equivalent_fn(ans, rep):
                assignments.append(cid)
                placed = True
                break
        if not placed:
            reps.append(ans)
            assignments.append(len(reps) - 1)
    return assignments


def make_equivalence_fn(
    nli_label_fn: Callable[[str, str], str],
    rule: str = "non_defeating",
) -> Callable[[str, str], bool]:
    """Build a symmetric equivalence predicate from a directional NLI label function.

    `nli_label_fn(premise, hypothesis)` returns one of {"entailment","neutral","contradiction"}.
    Two answers a, b are judged equivalent under:
      - "non_defeating" (Farquhar default): neither direction is a contradiction.
      - "strict":                            both directions entail.
    Callers are expected to have already wrapped answers with Q+A templating.
    """
    if rule not in ("non_defeating", "strict"):
        raise ValueError(f"unknown clustering rule: {rule!r}")

    def equivalent(a: str, b: str) -> bool:
        lab_ab = nli_label_fn(a, b)
        lab_ba = nli_label_fn(b, a)
        if rule == "strict":
            return lab_ab == "entailment" and lab_ba == "entailment"
        # non_defeating: equivalent unless a contradiction appears in either direction
        return lab_ab != "contradiction" and lab_ba != "contradiction"

    return equivalent


def cluster_assignment_entropy(cluster_ids: Sequence[int]) -> float:
    """Discrete semantic entropy = Shannon entropy (nats) of the cluster-frequency
    distribution over samples. This is the black-box-valid SE variant (Farquhar 2024).
    """
    n = len(cluster_ids)
    if n == 0:
        return 0.0
    counts = np.array(list(Counter(cluster_ids).values()), dtype=float)
    p = counts / n
    return float(-np.sum(p * np.log(p)))


def semantic_entropy(
    samples: Sequence[str],
    nli_label_fn: Callable[[str, str], str],
    rule: str = "non_defeating",
) -> float:
    """Discrete semantic entropy over sampled generations (cluster then entropy)."""
    equivalent = make_equivalence_fn(nli_label_fn, rule=rule)
    ids = cluster_answers(samples, equivalent)
    return cluster_assignment_entropy(ids)


# ---------------------------------------------------------------- divergence

def bidirectional_divergence(
    a: str,
    b: str,
    nli_entail_prob_fn: Callable[[str, str], float],
) -> float:
    """d = 1 - min(P_entail(a->b), P_entail(b->a)).

    High when the two answers fail to entail each other in at least one direction.
    Callers wrap a, b with Q+A templating before calling.
    """
    p_ab = nli_entail_prob_fn(a, b)
    p_ba = nli_entail_prob_fn(b, a)
    return 1.0 - min(p_ab, p_ba)


def qa_template(question: str, answer: str) -> str:
    """Wrap a short answer as 'Q: {question} A: {answer}' before NLI (load-bearing;
    feeding bare spans to DeBERTa is unreliable). Ablated on/off in Cell 10."""
    return f"Q: {question} A: {answer}"


def answer_correct_vs_golds(
    answer: str,
    golds: Sequence[str],
    question: str,
    nli_entail_prob_fn: Callable[[str, str], float],
    thresh: float = 0.5,
) -> bool | None:
    """Non-lexical correctness: is `answer` semantically equivalent to ANY gold?

    Used for the judge-label re-analysis (Cell 11) to replace the token-F1 label,
    breaking the circularity between a lexical label and the lexical d_lex signal.
    Equivalence = high bidirectional entailment (Q+A-templated) with the best-matching
    gold: correct iff max_g min(entail(ans->gold_g), entail(gold_g->ans)) >= thresh.

    Returns None when there are no golds (unanswerable items are labeled separately).
    """
    if len(golds) == 0:
        return None
    a = qa_template(question, answer)
    best = max(
        min(nli_entail_prob_fn(a, qa_template(question, g)),
            nli_entail_prob_fn(qa_template(question, g), a))
        for g in golds
    )
    return best >= thresh


# ---------------------------------------------------------------- metrics

def bootstrap_auroc(
    y: Sequence[int],
    scores: Sequence[float],
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """AUROC with a percentile bootstrap 95% CI.

    Positive class = 1 (INCORRECT / confabulation). Returns point estimate + CI.
    Bootstrap resamples that end up single-class are skipped (AUROC undefined).
    """
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(y)) < 2:
        return {"auroc": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n": int(len(y)), "note": "single-class"}
    point = float(roc_auc_score(y, scores))
    rng = np.random.default_rng(seed)
    n = len(y)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yi = y[idx]
        if len(np.unique(yi)) < 2:
            continue
        boots.append(roc_auc_score(yi, scores[idx]))
    if not boots:
        lo = hi = float("nan")
    else:
        lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"auroc": point, "ci_low": float(lo), "ci_high": float(hi), "n": int(n)}


def paired_bootstrap_auroc_diff(
    y: Sequence[int],
    score_a: Sequence[float],
    score_b: Sequence[float],
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    """Paired bootstrap of the AUROC difference AUROC(a) - AUROC(b) on the SAME items.

    The correct test when comparing two detectors on one labelled set: because both
    scores see the same items, their AUROCs are correlated, so the *difference* can be
    significant even when the marginal CIs overlap. Each bootstrap resample draws ONE
    index set and scores both a and b on it (paired). Returns the point difference, a
    95% CI, and a two-sided bootstrap p-value (H0: diff = 0). significant = CI excludes 0.
    """
    y = np.asarray(y, dtype=int)
    a = np.asarray(score_a, dtype=float)
    b = np.asarray(score_b, dtype=float)
    if len(np.unique(y)) < 2:
        return {"diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "p_value": float("nan"), "significant": False, "note": "single-class"}
    point = float(roc_auc_score(y, a) - roc_auc_score(y, b))
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yi = y[idx]
        if len(np.unique(yi)) < 2:
            continue
        diffs.append(roc_auc_score(yi, a[idx]) - roc_auc_score(yi, b[idx]))
    diffs = np.asarray(diffs)
    if diffs.size == 0:
        return {"diff": point, "ci_low": float("nan"), "ci_high": float("nan"),
                "p_value": float("nan"), "significant": False, "note": "no valid resamples"}
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # two-sided bootstrap p-value: twice the smaller tail mass at 0
    p = 2.0 * min(float(np.mean(diffs <= 0)), float(np.mean(diffs >= 0)))
    p = min(1.0, p)
    return {"diff": point, "ci_low": float(lo), "ci_high": float(hi),
            "p_value": p, "significant": bool(lo > 0 or hi < 0), "n": int(n)}
