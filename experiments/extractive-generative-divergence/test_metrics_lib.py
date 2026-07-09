"""Local sanity tests for metrics_lib (no torch/GPU). Run: python test_metrics_lib.py"""
import numpy as np

import metrics_lib as m


def test_token_f1():
    assert m.token_f1("the cat", "cat") == 1.0                 # articles stripped
    assert m.token_f1("Paris.", "paris") == 1.0                # punct + case
    assert m.token_f1("blue", "red") == 0.0
    assert abs(m.token_f1("x y z", "x y") - 0.8) < 1e-9        # P=2/3,R=1 -> 0.8
    assert m.token_f1("", "") == 1.0
    assert m.token_f1("x", "") == 0.0


def test_squad_f1_max():
    assert m.squad_f1_max("Einstein", ["Newton", "Albert Einstein", "Einstein"]) == 1.0
    assert m.squad_f1_max("Einstein", []) == 0.0


def test_substring_compliance():
    ctx = "The Eiffel Tower is located in Paris, France."
    assert m.is_substring_compliant("Paris", ctx)
    assert m.is_substring_compliant("in paris, FRANCE", ctx)   # normalized match
    assert not m.is_substring_compliant("London", ctx)
    assert not m.is_substring_compliant("", ctx)


def _mock_nli(groups):
    """Return an nli_label_fn that says 'entailment' iff both strings share a group,
    else 'contradiction'. groups: dict str->group_id."""
    def fn(premise, hypothesis):
        if groups.get(premise) == groups.get(hypothesis):
            return "entailment"
        return "contradiction"
    return fn


def test_clustering_and_entropy():
    samples = ["A1", "A2", "B1", "A3", "C1"]
    groups = {"A1": 0, "A2": 0, "A3": 0, "B1": 1, "C1": 2}   # clusters {3,1,1}
    nli = _mock_nli(groups)
    ids = m.cluster_answers(samples, m.make_equivalence_fn(nli, "non_defeating"))
    # three distinct clusters, first-match assignment
    assert len(set(ids)) == 3
    ent = m.cluster_assignment_entropy(ids)
    # entropy of distribution [3/5, 1/5, 1/5] in nats
    p = np.array([3, 1, 1]) / 5
    assert abs(ent - (-(p * np.log(p)).sum())) < 1e-9
    # all-identical -> zero entropy
    assert m.cluster_assignment_entropy([7, 7, 7, 7]) == 0.0


def test_strict_vs_nondefeating():
    # neutral in one direction: non_defeating clusters them, strict does not
    def nli(premise, hypothesis):
        return "neutral"
    eq_nd = m.make_equivalence_fn(nli, "non_defeating")
    eq_st = m.make_equivalence_fn(nli, "strict")
    assert eq_nd("x", "y") is True
    assert eq_st("x", "y") is False


def test_divergence():
    # symmetric high entailment -> low divergence
    d_same = m.bidirectional_divergence("a", "b", lambda p, h: 0.95)
    assert abs(d_same - 0.05) < 1e-9
    # asymmetric: min direction dominates
    probs = {("a", "b"): 0.9, ("b", "a"): 0.2}
    d = m.bidirectional_divergence("a", "b", lambda p, h: probs[(p, h)])
    assert abs(d - 0.8) < 1e-9


def test_answer_correct_vs_golds():
    # entail prob keyed by whether the two templated strings share a "fact" token.
    def entail_fn(premise, hypothesis):
        # crude: high entailment if the answer word appears in both
        return 0.9 if premise.split("A: ")[-1] == hypothesis.split("A: ")[-1] else 0.1
    q = "who?"
    # exact match to one of several golds -> correct
    assert m.answer_correct_vs_golds("Einstein", ["Newton", "Einstein"], q, entail_fn) is True
    # matches none -> incorrect
    assert m.answer_correct_vs_golds("Tesla", ["Newton", "Einstein"], q, entail_fn) is False
    # no golds (unanswerable) -> None, handled separately by caller
    assert m.answer_correct_vs_golds("anything", [], q, entail_fn) is None
    # threshold is respected: constant 0.4 entailment fails default 0.5
    assert m.answer_correct_vs_golds("x", ["y"], q, lambda p, h: 0.4) is False
    assert m.answer_correct_vs_golds("x", ["y"], q, lambda p, h: 0.4, thresh=0.3) is True


def test_bootstrap_auroc():
    rng = np.random.default_rng(0)
    y = np.array([0] * 50 + [1] * 50)
    scores = np.concatenate([rng.normal(0, 1, 50), rng.normal(2, 1, 50)])
    out = m.bootstrap_auroc(y, scores, n_boot=500, seed=42)
    assert 0.85 < out["auroc"] <= 1.0
    assert out["ci_low"] <= out["auroc"] <= out["ci_high"]
    # single-class guard
    bad = m.bootstrap_auroc([1, 1, 1], [0.1, 0.2, 0.3])
    assert bad["note"] == "single-class"
    assert np.isnan(bad["auroc"])


def test_paired_bootstrap_auroc_diff():
    rng = np.random.default_rng(0)
    y = np.array([0] * 60 + [1] * 60)
    strong = np.concatenate([rng.normal(0, 1, 60), rng.normal(2.2, 1, 60)])   # good detector
    weak = np.concatenate([rng.normal(0, 1, 60), rng.normal(0.5, 1, 60)])      # weak detector
    out = m.paired_bootstrap_auroc_diff(y, strong, weak, n_boot=800, seed=42)
    assert out["diff"] > 0                       # strong beats weak
    assert out["significant"] is True            # CI excludes 0
    assert out["ci_low"] > 0
    assert out["p_value"] < 0.05
    # identical signals -> no difference, not significant
    same = m.paired_bootstrap_auroc_diff(y, strong, strong, n_boot=800, seed=42)
    assert abs(same["diff"]) < 1e-9
    assert same["significant"] is False
    # single-class guard
    bad = m.paired_bootstrap_auroc_diff([1, 1, 1], [0.1, 0.2, 0.3], [0.3, 0.2, 0.1])
    assert bad["note"] == "single-class"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
