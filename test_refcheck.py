"""Tests for refcheck.check_reference — the regression guard used by the M1
diagnostic scripts to assert that re-runs reproduce the numbers reported in the
paper (e.g. zero-shot AUROC@3 = 0.383, within-doc AUROC@3 = 0.687).
"""
from refcheck import check_reference


def test_within_tolerance_returns_true():
    assert check_reference("zeroshot@3", 0.383, 0.383, 0.02) is True


def test_small_deviation_within_tolerance_passes():
    assert check_reference("zeroshot@3", 0.390, 0.383, 0.02) is True


def test_outside_tolerance_raises():
    raised = False
    try:
        check_reference("zeroshot@3", 0.900, 0.383, 0.02)
    except AssertionError:
        raised = True
    assert raised, "expected AssertionError when got is far from expected"


def test_boundary_exactly_at_tolerance_passes():
    assert check_reference("z", 0.403, 0.383, 0.02) is True


def test_predicate_form_below_threshold_passes():
    # N@3 must stay below 0.5 ("not a normalization artifact")
    assert check_reference("renorm@3<0.5", 0.408, None, None, upper=0.5) is True


def test_predicate_form_above_threshold_raises():
    raised = False
    try:
        check_reference("renorm@3<0.5", 0.55, None, None, upper=0.5)
    except AssertionError:
        raised = True
    assert raised, "expected AssertionError when value violates upper bound"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nall {len(fns)} tests passed")
