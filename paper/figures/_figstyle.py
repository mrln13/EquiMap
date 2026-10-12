"""Shared figure styling for the manuscript figures.

Keeps element colours consistent across every figure: a given element maps to
the same colour regardless of which elements are present or how they are ordered
in a particular panel. Import from the sibling figure scripts.
"""
import matplotlib.pyplot as plt

# Canonical element order. The colour of an element is its index here (mod 10)
# in the tab10 palette, so e.g. Si is always blue and Fe always red, in every
# figure, even when some elements are absent.
ELEMENT_ORDER = ["Si", "Ti", "Al", "Fe", "Mn", "Mg", "Ca", "Na", "K",
                 "Cr", "Ni", "P"]
_CMAP = plt.get_cmap("tab10")


def element_colour(el):
    """A stable tab10 colour for element ``el``, identical across all figures."""
    if el in ELEMENT_ORDER:
        return _CMAP(ELEMENT_ORDER.index(el) % 10)
    # Unknown element: deterministic (hash-free) fallback so reruns are stable.
    return _CMAP(sum(ord(c) for c in str(el)) % 10)


def element_colours(elements):
    """``{el: colour}`` for a list of elements."""
    return {el: element_colour(el) for el in elements}


def order_elements(elements):
    """Canonical-order the known elements first, then the rest alphabetically."""
    known = [e for e in ELEMENT_ORDER if e in elements]
    rest = sorted(e for e in elements if e not in ELEMENT_ORDER)
    return known + rest
