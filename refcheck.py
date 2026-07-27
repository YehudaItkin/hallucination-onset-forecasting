"""Regression guard for the M1 diagnostic scripts.

check_reference encodes a number reported in the paper as an executable contract:
a re-run that no longer reproduces it (or violates a claimed bound) raises, so a
silent pipeline regression cannot slip into the manuscript's claims.
"""


def check_reference(name, got, expected, tol, upper=None, lower=None):
    """Assert `got` matches a paper reference value.

    Tolerance form:  abs(got - expected) <= tol.
    Bound form:      got < upper   (or)   got > lower.
    Returns True on success; raises AssertionError otherwise.
    """
    if upper is not None:
        ok = got < upper
        detail = f"< {upper}"
    elif lower is not None:
        ok = got > lower
        detail = f"> {lower}"
    else:
        ok = abs(got - expected) <= tol + 1e-9  # epsilon: inclusive boundary, float-safe
        detail = f"= {expected} +- {tol}"
    if not ok:
        raise AssertionError(f"[refcheck] {name}: got {got:.4f}, expected {detail}")
    print(f"[refcheck] OK  {name}: {got:.4f} ({detail})")
    return True
