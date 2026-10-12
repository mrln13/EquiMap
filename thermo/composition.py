# thermo/composition.py
"""
Composition — the central data type for bulk chemical compositions.

Internally stores element moles normalised so Σ(mol_i × AW_i) = 100,
which is numerically equivalent to element weight percent.  All other
representations (oxide wt%, mole fractions, cation fractions) are derived
on demand.

Iron oxidation state is *not* stored internally.  Fe is kept as total Fe.
Use the `fe3_fraction` parameter on `to_oxide_wt()` / `to_cation_mol_frac()`
to split FeO / Fe₂O₃ at output time.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Atomic weights  (IUPAC 2021, abridged to elements relevant for geo + alloys)
# ---------------------------------------------------------------------------
ATOMIC_WEIGHTS: Dict[str, float] = {
    'H': 1.008,   'He': 4.0026, 'Li': 6.94,   'Be': 9.0122,
    'B': 10.81,   'C': 12.011,  'N': 14.007,  'O': 15.999,
    'F': 18.998,  'Ne': 20.180, 'Na': 22.990, 'Mg': 24.305,
    'Al': 26.982, 'Si': 28.086, 'P': 30.974,  'S': 32.06,
    'Cl': 35.45,  'Ar': 39.948, 'K': 39.098,  'Ca': 40.078,
    'Sc': 44.956, 'Ti': 47.867, 'V': 50.942,  'Cr': 51.996,
    'Mn': 54.938, 'Fe': 55.845, 'Co': 58.933, 'Ni': 58.693,
    'Cu': 63.546, 'Zn': 65.38,  'Ga': 69.723, 'Ge': 72.630,
    'As': 74.922, 'Se': 78.971, 'Br': 79.904, 'Kr': 83.798,
    'Rb': 85.468, 'Sr': 87.62,  'Y': 88.906,  'Zr': 91.224,
    'Nb': 92.906, 'Mo': 95.95,  'Ru': 101.07, 'Rh': 102.91,
    'Pd': 106.42, 'Ag': 107.87, 'Cd': 112.41, 'In': 114.82,
    'Sn': 118.71, 'Sb': 121.76, 'Te': 127.60, 'I': 126.90,
    'Xe': 131.29, 'Cs': 132.91, 'Ba': 137.33, 'La': 138.91,
    'Ce': 140.12, 'Pr': 140.91, 'Nd': 144.24, 'Sm': 150.36,
    'Eu': 151.96, 'Gd': 157.25, 'Tb': 158.93, 'Dy': 162.50,
    'Ho': 164.93, 'Er': 167.26, 'Tm': 168.93, 'Yb': 173.05,
    'Lu': 174.97, 'Hf': 178.49, 'Ta': 180.95, 'W': 183.84,
    'Re': 186.21, 'Os': 190.23, 'Ir': 192.22, 'Pt': 195.08,
    'Au': 196.97, 'Hg': 200.59, 'Tl': 204.38, 'Pb': 207.2,
    'Bi': 208.98, 'Th': 232.04, 'U': 238.03,
}

# ---------------------------------------------------------------------------
# Oxide formulas  (formula → {element: stoich_count, 'O': stoich_count})
# ---------------------------------------------------------------------------
OXIDE_FORMULAS: Dict[str, Dict[str, int]] = {
    'SiO2':  {'Si': 1, 'O': 2},
    'Al2O3': {'Al': 2, 'O': 3},
    'FeO':   {'Fe': 1, 'O': 1},
    'Fe2O3': {'Fe': 2, 'O': 3},
    'MgO':   {'Mg': 1, 'O': 1},
    'CaO':   {'Ca': 1, 'O': 1},
    'Na2O':  {'Na': 2, 'O': 1},
    'K2O':   {'K': 2,  'O': 1},
    'TiO2':  {'Ti': 1, 'O': 2},
    'MnO':   {'Mn': 1, 'O': 1},
    'MnO2':  {'Mn': 1, 'O': 2},
    'P2O5':  {'P': 2,  'O': 5},
    'Cr2O3': {'Cr': 2, 'O': 3},
    'NiO':   {'Ni': 1, 'O': 1},
    'CoO':   {'Co': 1, 'O': 1},
    'ZnO':   {'Zn': 1, 'O': 1},
    'BaO':   {'Ba': 1, 'O': 1},
    'SrO':   {'Sr': 1, 'O': 1},
    'CuO':   {'Cu': 1, 'O': 1},
    'Cu2O':  {'Cu': 2, 'O': 1},
    'V2O5':  {'V': 2,  'O': 5},
    'V2O3':  {'V': 2,  'O': 3},
    'ZrO2':  {'Zr': 1, 'O': 2},
    'H2O':   {'H': 2,  'O': 1},
    'CO2':   {'C': 1,  'O': 2},
    'SO3':   {'S': 1,  'O': 3},
    'SO2':   {'S': 1,  'O': 2},
    'Li2O':  {'Li': 2, 'O': 1},
    'B2O3':  {'B': 2,  'O': 3},
    'MoO3':  {'Mo': 1, 'O': 3},
    'WO3':   {'W': 1,  'O': 3},
    'ThO2':  {'Th': 1, 'O': 2},
    'UO2':   {'U': 1,  'O': 2},
    'La2O3': {'La': 2, 'O': 3},
    'Ce2O3': {'Ce': 2, 'O': 3},
    'CeO2':  {'Ce': 1, 'O': 2},
    'Y2O3':  {'Y': 2,  'O': 3},
    'Nb2O5': {'Nb': 2, 'O': 5},
    'HfO2':  {'Hf': 1, 'O': 2},
    'Ta2O5': {'Ta': 2, 'O': 5},
    'Sc2O3': {'Sc': 2, 'O': 3},
    'Ga2O3': {'Ga': 2, 'O': 3},
    'GeO2':  {'Ge': 1, 'O': 2},
    'In2O3': {'In': 2, 'O': 3},
    'SnO2':  {'Sn': 1, 'O': 2},
    'PbO':   {'Pb': 1, 'O': 1},
    'PbO2':  {'Pb': 1, 'O': 2},
    'Bi2O3': {'Bi': 2, 'O': 3},
}

# Default oxide per element (element wt% → oxide wt% conversion)
ELEMENT_DEFAULT_OXIDE: Dict[str, str] = {
    'Si': 'SiO2',  'Al': 'Al2O3', 'Fe': 'FeO',   'Mg': 'MgO',
    'Ca': 'CaO',   'Na': 'Na2O',  'K': 'K2O',    'Ti': 'TiO2',
    'Mn': 'MnO',   'P': 'P2O5',   'Cr': 'Cr2O3', 'Ni': 'NiO',
    'Co': 'CoO',   'Zn': 'ZnO',   'Ba': 'BaO',   'Sr': 'SrO',
    'Cu': 'CuO',   'V': 'V2O5',   'Zr': 'ZrO2',  'H': 'H2O',
    'C': 'CO2',    'S': 'SO3',    'Li': 'Li2O',  'B': 'B2O3',
    'La': 'La2O3', 'Ce': 'Ce2O3', 'Y': 'Y2O3',   'Nb': 'Nb2O5',
    'Hf': 'HfO2',  'Ta': 'Ta2O5', 'Sc': 'Sc2O3', 'Mo': 'MoO3',
    'W': 'WO3',    'Th': 'ThO2',  'U': 'UO2',    'Ga': 'Ga2O3',
    'Ge': 'GeO2',  'In': 'In2O3', 'Sn': 'SnO2',  'Pb': 'PbO',
    'Bi': 'Bi2O3',
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def oxide_mw(oxide: str) -> float:
    """Molecular weight of an oxide formula."""
    formula = OXIDE_FORMULAS.get(oxide)
    if formula is None:
        raise ValueError(f"Unknown oxide '{oxide}'. Add it to OXIDE_FORMULAS.")
    return sum(ATOMIC_WEIGHTS[el] * n for el, n in formula.items())


def element_in_oxide_mass_fraction(oxide: str, element: str) -> float:
    """Mass fraction of *element* in *oxide* (0 < result ≤ 1)."""
    formula = OXIDE_FORMULAS[oxide]
    n_el = formula.get(element, 0)
    if n_el == 0:
        raise ValueError(f"Element '{element}' not present in oxide '{oxide}'.")
    return (n_el * ATOMIC_WEIGHTS[element]) / oxide_mw(oxide)


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

class Composition:
    """
    Bulk chemical composition.

    Internal representation: ``{element: moles}`` normalised so that
    ``Σ(mol_i × AW_i) = 100``.  This is numerically equal to element
    weight-percent, so ``to_element_wt()`` is just a dict copy with ×AW.

    Examples
    --------
    From EDS output (element wt %)::

        comp = Composition.from_element_wt(
            {'Si': 21.2, 'Al': 6.4, 'Ca': 6.7, 'Mg': 4.6, 'Fe': 6.2,
             'Na': 2.0, 'Ti': 0.57, 'K': 0.17, 'H': 1.1}
        )

    From geochemical analysis (oxide wt %)::

        comp = Composition.from_oxide_wt(
            {'SiO2': 45.4, 'Al2O3': 12.1, 'FeO': 8.0, 'MgO': 7.6,
             'CaO': 9.4, 'Na2O': 2.7, 'TiO2': 0.95, 'H2O': 10.0}
        )
    """

    # ---- Construction -------------------------------------------------------

    def __init__(self, moles: Dict[str, float]):
        """
        Low-level constructor.  Pass element → moles (any scale; O is ignored).
        Use the class-methods below for typical use.
        """
        clean = {k: float(v) for k, v in moles.items()
                 if k != 'O' and float(v) > 0.0}
        if not clean:
            raise ValueError("Composition must contain at least one positive element abundance.")
        unknown = [el for el in clean if el not in ATOMIC_WEIGHTS]
        if unknown:
            raise ValueError(
                f"Unknown element(s): {unknown}. Add them to ATOMIC_WEIGHTS."
            )
        total_wt = sum(m * ATOMIC_WEIGHTS[el] for el, m in clean.items())
        if total_wt <= 0:
            raise ValueError("Total weight is zero; check input values.")
        scale = 100.0 / total_wt
        self._moles: Dict[str, float] = {el: m * scale for el, m in clean.items()}

    @classmethod
    def from_element_wt(cls, data: Dict[str, float]) -> 'Composition':
        """Create from element weight percent (e.g. EDS/WDS output)."""
        moles = {}
        for el, wt in data.items():
            if el == 'O' or float(wt) <= 0:
                continue
            if el not in ATOMIC_WEIGHTS:
                raise ValueError(f"Unknown element '{el}'.")
            moles[el] = float(wt) / ATOMIC_WEIGHTS[el]
        return cls(moles)

    @classmethod
    def from_oxide_wt(cls, data: Dict[str, float]) -> 'Composition':
        """
        Create from oxide weight percent.
        FeO and Fe₂O₃ are both accepted and merged as total Fe internally.
        """
        elem_wt: Dict[str, float] = {}
        for oxide, wt in data.items():
            if float(wt) <= 0:
                continue
            if oxide not in OXIDE_FORMULAS:
                raise ValueError(f"Unknown oxide '{oxide}'. Add it to OXIDE_FORMULAS.")
            mw = oxide_mw(oxide)
            for el, n in OXIDE_FORMULAS[oxide].items():
                if el == 'O':
                    continue
                elem_wt[el] = elem_wt.get(el, 0.0) + float(wt) * (n * ATOMIC_WEIGHTS[el]) / mw
        return cls.from_element_wt(elem_wt)

    @classmethod
    def from_element_mol_frac(cls, data: Dict[str, float]) -> 'Composition':
        """Create from element mole fractions (values need not sum to 1)."""
        return cls(data)

    # ---- Output representations ---------------------------------------------

    def to_element_wt(self) -> Dict[str, float]:
        """Element weight percent (sums to 100)."""
        return {el: m * ATOMIC_WEIGHTS[el] for el, m in self._moles.items()}

    def to_element_mol(self) -> Dict[str, float]:
        """Element moles (normalised to Σ wt = 100)."""
        return dict(self._moles)

    def to_element_mol_frac(self) -> Dict[str, float]:
        """Element mole fractions (sum to 1)."""
        total = sum(self._moles.values())
        return {el: m / total for el, m in self._moles.items()}

    def to_oxide_wt(
        self,
        fe3_fraction: float = 0.0,
        extra_oxides: Optional[Dict[str, float]] = None,
        filter_to: Optional[Sequence[str]] = None,
        normalize: bool = True,
        extra_basis: str = "bulk",
    ) -> Dict[str, float]:
        """
        Convert to oxide weight percent.

        Parameters
        ----------
        fe3_fraction :
            Fraction of total Fe reported as Fe₂O₃ (remainder as FeO).
            0 = all FeO, 1 = all Fe₂O₃.
        extra_oxides :
            Oxides the composition does not carry (e.g. ``{'H2O': 4.0}``).
            With ``extra_basis="bulk"`` (default) each value is that oxide's
            weight percent **in the final bulk**: the measured oxides are
            scaled to ``100 − Σ extras`` and the result sums to 100 whatever
            ``normalize`` says.
        extra_basis :
            ``"bulk"`` (above) or ``"legacy"``. Legacy reproduces the
            arithmetic used until 2026-10-05, which added the raw values to
            the oxide sum of a composition normalised over the *cations*
            (that sum is ~180 for a silicate), so a nominal 4 wt% H₂O became
            ~2.1 wt% of the bulk. Kept only so stored runs replay unchanged.
        filter_to :
            If given, return only these oxides in this order (missing ones
            are returned as 0.0).  Useful for matching a database oxide list.
        normalize :
            Normalise to 100 wt% (default True).
        """
        elem_wt = self.to_element_wt()
        out: Dict[str, float] = {}

        for el, wt in elem_wt.items():
            if el == 'Fe':
                f3 = float(fe3_fraction)
                fe2_wt = wt * (1.0 - f3)
                fe3_wt = wt * f3
                if fe2_wt > 0:
                    f = oxide_mw('FeO') / (1 * ATOMIC_WEIGHTS['Fe'])
                    out['FeO'] = out.get('FeO', 0.0) + fe2_wt * f
                if fe3_wt > 0:
                    f = oxide_mw('Fe2O3') / (2 * ATOMIC_WEIGHTS['Fe'])
                    out['Fe2O3'] = out.get('Fe2O3', 0.0) + fe3_wt * f
            else:
                default = ELEMENT_DEFAULT_OXIDE.get(el)
                if default is None:
                    continue  # element has no default oxide (e.g. noble gases)
                n_el = OXIDE_FORMULAS[default].get(el, 0)
                f = oxide_mw(default) / (n_el * ATOMIC_WEIGHTS[el])
                out[default] = out.get(default, 0.0) + wt * f

        extras = {ox: float(v) for ox, v in (extra_oxides or {}).items() if v and float(v) > 0}
        if extras and extra_basis == "legacy":
            for ox, v in extras.items():
                out[ox] = out.get(ox, 0.0) + v
        elif extras:
            if extra_basis != "bulk":
                raise ValueError(f"extra_basis must be 'bulk' or 'legacy', got {extra_basis!r}")
            share = sum(extras.values())
            if share >= 100.0:
                raise ValueError(f"extra oxides sum to {share:g} wt% of the bulk (must be < 100)")
            dry = sum(out.values())
            if dry > 0:
                out = {k: v * (100.0 - share) / dry for k, v in out.items()}
            for ox, v in extras.items():
                out[ox] = out.get(ox, 0.0) + v
            normalize = False            # already sums to 100 with the extras as given

        if normalize:
            total = sum(out.values())
            if total > 0:
                out = {k: 100.0 * v / total for k, v in out.items()}

        if filter_to is not None:
            return {ox: out.get(ox, 0.0) for ox in filter_to}

        return out

    def to_oxide_mol(
        self,
        fe3_fraction: float = 0.0,
        extra_oxides: Optional[Dict[str, float]] = None,
        filter_to: Optional[Sequence[str]] = None,
        normalize: bool = True,
        extra_basis: str = "bulk",
    ) -> Dict[str, float]:
        """
        Convert to oxide mole percent.
        Shares the same parameters as ``to_oxide_wt``.
        """
        ox_wt = self.to_oxide_wt(
            fe3_fraction=fe3_fraction,
            extra_oxides=extra_oxides,
            filter_to=filter_to,
            normalize=False,
            extra_basis=extra_basis,
        )
        ox_mol = {ox: wt / oxide_mw(ox) for ox, wt in ox_wt.items() if wt > 0}

        if normalize:
            total = sum(ox_mol.values())
            if total > 0:
                ox_mol = {k: 100.0 * v / total for k, v in ox_mol.items()}

        if filter_to is not None:
            return {ox: ox_mol.get(ox, 0.0) for ox in filter_to}

        return ox_mol

    def to_cation_mol_frac(self, fe3_fraction: float = 0.0) -> Dict[str, float]:
        """
        Cation mole fractions (Fe²⁺ / Fe³⁺ distinguished when fe3_fraction > 0).
        """
        ox_wt = self.to_oxide_wt(fe3_fraction=fe3_fraction, normalize=False)
        cations: Dict[str, float] = {}
        for oxide, wt in ox_wt.items():
            if wt <= 0:
                continue
            mw = oxide_mw(oxide)
            for el, n in OXIDE_FORMULAS[oxide].items():
                if el == 'O':
                    continue
                moles = (wt / mw) * n
                if oxide == 'FeO':
                    key = 'Fe2+'
                elif oxide == 'Fe2O3':
                    key = 'Fe3+'
                else:
                    key = el
                cations[key] = cations.get(key, 0.0) + moles
        total = sum(cations.values())
        if total > 0:
            cations = {k: v / total for k, v in cations.items()}
        return cations

    # ---- Utility ------------------------------------------------------------

    def elements(self) -> List[str]:
        """Sorted list of elements present."""
        return sorted(self._moles.keys())

    def mix(self, other: 'Composition', fraction: float = 0.5) -> 'Composition':
        """
        Mix this composition with *other* by mass.
        ``fraction=0`` → self, ``fraction=1`` → other.
        """
        a = self.to_element_wt()
        b = other.to_element_wt()
        keys = set(a) | set(b)
        mixed = {k: (1 - fraction) * a.get(k, 0.0) + fraction * b.get(k, 0.0)
                 for k in keys}
        return Composition.from_element_wt(mixed)

    def __repr__(self) -> str:
        wt = self.to_element_wt()
        parts = ", ".join(
            f"{el}={v:.2f}" for el, v in sorted(wt.items(), key=lambda x: -x[1])
        )
        return f"Composition({parts})"
