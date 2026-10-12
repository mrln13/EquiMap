# thermo/result.py
"""Output data types for thermodynamic equilibrium calculations."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

# ``to_dict`` appends ``_<k>`` to the k-th repeated phase name so that a
# backend reporting the same solution phase twice (a solvus pair, e.g. two
# ``fsp`` = plagioclase + K-feldspar) keeps both instances in a flat row.
_INSTANCE_SUFFIX = re.compile(r"^(.*)_(\d+)$")


def phase_base_name(column_phase: str) -> str:
    """Strip the ``_<k>`` instance suffix from a per-phase column name.

    ``phase_base_name("fsp_2") == "fsp"``; names without a suffix are returned
    unchanged. Use this when reading the ``mol_frac_*``/``wt_frac_*``/
    ``vol_frac_*`` columns written by :meth:`ThermoResult.to_dict` back into
    :class:`PhaseResult` objects, so alias matching still sees ``fsp``.
    """
    m = _INSTANCE_SUFFIX.match(column_phase)
    return m.group(1) if m else column_phase


@dataclass
class PhaseResult:
    """A single stable phase in an equilibrium assemblage."""
    name: str
    mol_fraction: float                            # modal fraction (moles)
    wt_fraction: Optional[float] = None            # modal fraction (mass), if available
    vol_fraction: Optional[float] = None           # modal fraction (volume), if available
    composition_oxide_wt: Optional[Dict[str, float]] = None  # oxide wt% of this phase
    composition_oxide_mol: Optional[Dict[str, float]] = None
    is_melt: bool = False
    endmembers: Optional[Dict[str, float]] = None  # endmember mole fractions


@dataclass
class ThermoResult:
    """
    Result of a single-point thermodynamic equilibrium calculation.

    Attributes
    ----------
    p_gpa :     Pressure [GPa]
    t_c :       Temperature [°C]
    stable_phases : List of stable phases with modal fractions and compositions.
    bulk_properties : Scalar properties of the bulk assemblage, e.g.
                      ``{'rho_kgm3': 2800., 'G_J': -1.2e6}``.
    backend :   Name of the backend that produced this result.
    converged : False if the minimisation did not converge.
    raw :       Backend-specific raw output object (for advanced post-processing).
    """
    p_gpa: float
    t_c: float
    stable_phases: List[PhaseResult]
    bulk_properties: Dict[str, float] = field(default_factory=dict)
    backend: str = ""
    converged: bool = True
    raw: Any = None

    # ---- Convenience properties ------------------------------------------

    @property
    def phase_names(self) -> List[str]:
        return [p.name for p in self.stable_phases]

    @property
    def n_phases(self) -> int:
        return len(self.stable_phases)

    def phase(self, name: str) -> Optional[PhaseResult]:
        """Return the PhaseResult for *name*, or None if absent."""
        for p in self.stable_phases:
            if p.name == name:
                return p
        return None

    def has_melt(self) -> bool:
        return any(p.is_melt for p in self.stable_phases)

    # ---- Serialisation ---------------------------------------------------

    def to_dict(self) -> Dict:
        """Flat dict suitable for building a DataFrame row.

        A phase name that occurs more than once in the assemblage (MAGEMin
        reports a solvus pair as two instances of the same solution model,
        e.g. ``fsp`` + ``fsp`` for plagioclase + K-feldspar) is written as
        ``<name>``, ``<name>_2``, ``<name>_3`` ... so that no instance's
        fraction is lost; :func:`phase_base_name` recovers the base name.
        """
        d: Dict = {
            'P_GPa': self.p_gpa,
            'T_C': self.t_c,
            'n_phases': self.n_phases,
            'phases': ' '.join(self.phase_names),
            'converged': self.converged,
            'backend': self.backend,
        }
        seen: Dict[str, int] = {}
        for p in self.stable_phases:
            k = seen.get(p.name, 0)
            seen[p.name] = k + 1
            col = p.name if k == 0 else f"{p.name}_{k + 1}"
            d[f'mol_frac_{col}'] = p.mol_fraction
            if p.wt_fraction is not None:
                d[f'wt_frac_{col}'] = p.wt_fraction
            if p.vol_fraction is not None:
                d[f'vol_frac_{col}'] = p.vol_fraction
        d.update(self.bulk_properties)
        return d

    def __repr__(self) -> str:
        phases = ', '.join(
            f"{p.name}({p.mol_fraction:.3f})" for p in self.stable_phases
        )
        return (f"ThermoResult(P={self.p_gpa:.3f} GPa, T={self.t_c:.0f} °C, "
                f"phases=[{phases}])")


def results_to_dataframe(results: List[ThermoResult]) -> pd.DataFrame:
    """Convert a list of ThermoResult objects to a tidy DataFrame."""
    rows = [r.to_dict() for r in results]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Portable (JSON / pickle-safe) round-trip
# ---------------------------------------------------------------------------
# These produce/consume plain dicts with no backend-specific ``raw`` object,
# so a ThermoResult can cross a process boundary (e.g. a ProcessPoolExecutor
# worker that ran MAGEMin in its own Julia session — see thermo/parallel.py).
# The schema is intentionally identical to the viewer's session serialiser
# (_serialize_thermod_result in viewer_thermo_helpers.py) so the two are
# interchangeable.

def _phase_to_portable_dict(p: PhaseResult) -> Dict:
    return {
        "name": p.name,
        "mol_fraction": (float(p.mol_fraction) if p.mol_fraction is not None else None),
        "wt_fraction": (float(p.wt_fraction) if p.wt_fraction is not None else None),
        "vol_fraction": (float(p.vol_fraction) if p.vol_fraction is not None else None),
        "composition_oxide_wt": (dict(p.composition_oxide_wt)
                                 if p.composition_oxide_wt else None),
        "composition_oxide_mol": (dict(p.composition_oxide_mol)
                                  if p.composition_oxide_mol else None),
        "is_melt": bool(p.is_melt),
        "endmembers": dict(p.endmembers) if p.endmembers else None,
    }


def _phase_from_portable_dict(d: Dict) -> PhaseResult:
    return PhaseResult(
        name=d.get("name", ""),
        mol_fraction=float(d.get("mol_fraction") or 0.0),
        wt_fraction=(float(d["wt_fraction"]) if d.get("wt_fraction") is not None else None),
        vol_fraction=(float(d["vol_fraction"]) if d.get("vol_fraction") is not None else None),
        composition_oxide_wt=d.get("composition_oxide_wt") or None,
        composition_oxide_mol=d.get("composition_oxide_mol") or None,
        is_melt=bool(d.get("is_melt", False)),
        endmembers=d.get("endmembers") or None,
    )


def result_to_portable_dict(r: ThermoResult) -> Dict:
    """Serialise a ThermoResult to a plain dict (drops the ``raw`` object)."""
    return {
        "p_gpa": float(r.p_gpa),
        "t_c": float(r.t_c),
        "stable_phases": [_phase_to_portable_dict(p) for p in r.stable_phases],
        "bulk_properties": {k: (None if v is None else float(v))
                            for k, v in (r.bulk_properties or {}).items()},
        "backend": str(r.backend or ""),
        "converged": bool(r.converged),
    }


def result_from_portable_dict(d: Dict) -> ThermoResult:
    """Reconstruct a ThermoResult from :func:`result_to_portable_dict` output."""
    return ThermoResult(
        p_gpa=float(d.get("p_gpa", 0.0)),
        t_c=float(d.get("t_c", 0.0)),
        stable_phases=[_phase_from_portable_dict(p) for p in d.get("stable_phases", [])],
        bulk_properties=dict(d.get("bulk_properties") or {}),
        backend=str(d.get("backend", "")),
        converged=bool(d.get("converged", True)),
        raw=None,
    )
