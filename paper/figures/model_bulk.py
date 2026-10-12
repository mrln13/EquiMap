"""The bulk composition passed to MAGEMin for the worked example (one place).

Every modelling script of the worked example (the Fig 5 grid, both Monte
Carlo runs, both scale sweeps and the per-window inversions) builds its
MAGEMin input here, so the recipe of the paper's Table is one function:

  elements passed   Si, Al, Ca, Mg, Fe, K, Na, Ti, Mn (as oxides). These are
                    the measured elements that are components of the
                    metapelite database ("mpe": SiO2 Al2O3 CaO MgO FeO K2O
                    Na2O TiO2 O MnO H2O; CO2 and S are not used). Anything
                    else is dropped: P (not a component), and on the SPX
                    quantification the automatic-peak-ID artefacts Sn, Tl, Zr
                    and traces (3.3 wt% of the oxide total of the section bulk).
  ferric iron       Fe3+/sum(Fe) = 0.1 (enters as the "O" component).
  water             4 wt% H2O OF THE BULK, i.e. of the eleven components
                    passed. (Until 2026-10-05 the toolkit added "4" to a
                    cation-normalised oxide total, which is 2.1 wt% of the
                    bulk; the best fit and accepted region of Fig 5 are the
                    same at 2.1, 3.9 and 7.4 wt%, the hot, melt-bearing end
                    of the stability field is not.)
"""
from thermo import Composition

MODEL_ELEMENTS = ("Si", "Al", "Ca", "Mg", "Fe", "K", "Na", "Ti", "Mn")
FE3_FRACTION = 0.1
H2O_WT_PCT = 4.0
DATABASE = "mpe"


def split_elements(comp_el):
    """(kept, dropped) element-wt% dicts; oxygen is neither (it is recomputed)."""
    kept = {e: float(v) for e, v in comp_el.items() if e in MODEL_ELEMENTS and float(v) > 0}
    dropped = {e: float(v) for e, v in comp_el.items()
               if e not in MODEL_ELEMENTS and e != "O" and float(v) > 0}
    return kept, dropped


def model_oxides(comp_el, fe3=FE3_FRACTION, h2o=H2O_WT_PCT):
    """Oxide wt% (summing to 100, H2O = ``h2o``) of the modelled bulk."""
    kept, _ = split_elements(comp_el)
    extra = {"H2O": float(h2o)} if h2o and float(h2o) > 0 else None
    return Composition.from_element_wt(kept).to_oxide_wt(
        fe3_fraction=fe3, extra_oxides=extra, normalize=True, extra_basis="bulk")


def model_composition(comp_el, fe3=FE3_FRACTION, h2o=H2O_WT_PCT):
    """``Composition`` of the modelled bulk (what the backend receives)."""
    return Composition.from_oxide_wt(model_oxides(comp_el, fe3=fe3, h2o=h2o))
