# thermo/backends/pycalphad.py
"""
pycalphad backend — for alloys, ceramics, and any system with a TDB database.

Install:  pip install pycalphad
Docs:     https://pycalphad.org

Pressure units : Pa    (101 325 Pa = 1 atm)
Temperature    : K     (°C + 273.15)
Composition    : element mole fractions

Usage example::

    from thermo import ThermoCalc, Composition

    comp = Composition.from_element_wt({'Fe': 70.0, 'Cr': 18.0, 'Ni': 12.0})
    calc = ThermoCalc(
        backend='pycalphad',
        tdb_path='steel.tdb',
        phases=['BCC_A2', 'FCC_A1', 'SIGMA'],   # optional subset
    )
    result = calc.run(comp, p_gpa=1e-4, t_c=900)

Notes
-----
* pycalphad uses UPPERCASE element symbols and requires 'VA' (vacancy) as a
  component for metallic systems.  The backend adds 'VA' automatically.
* A TDB database file must be supplied via ``tdb_path``.  Many open TDB files
  are available from the NIST CALPHAD database repository and alloy suppliers.
* For geological silicate systems, pycalphad can in principle be used if a
  silicate TDB is available, but most geological thermodynamic datasets are
  not distributed in TDB format.  Use the ``magemin`` or ``perplex`` backends
  for geological work.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from thermo.backends.base import AbstractBackend
from thermo.composition import Composition
from thermo.result import PhaseResult, ThermoResult


class PycalphadBackend(AbstractBackend):
    """
    Thermodynamic equilibrium via pycalphad.

    Parameters
    ----------
    tdb_path :
        Path to a TDB (Thermodynamic Data Base) file.
    phases :
        Subset of phases to consider.  If None, all phases in the database
        are used (can be slow for large databases).
    pressure_pa :
        Default pressure in Pa.  Overridden by ``p_gpa`` in ``run()``.
    add_va :
        Automatically add 'VA' (vacancy) to the component list (needed for
        most metallic databases).
    """

    name = "pycalphad"

    def __init__(
        self,
        tdb_path: str | Path,
        phases: Optional[Sequence[str]] = None,
        pressure_pa: float = 101_325.0,
        add_va: bool = True,
    ):
        self.tdb_path = Path(tdb_path)
        if not self.tdb_path.exists():
            raise FileNotFoundError(f"TDB file not found: {self.tdb_path}")
        self._phases_override = list(phases) if phases else None
        self.pressure_pa = float(pressure_pa)
        self.add_va = add_va
        self._db = None   # lazily loaded

    # ---- Public API ---------------------------------------------------------

    def run(
        self,
        composition: Composition,
        p_gpa: float,
        t_c: float,
    ) -> ThermoResult:
        """Single-point equilibrium via pycalphad."""
        from pycalphad import equilibrium, variables as v

        db = self._get_db()
        components, phases, cond = self._build_conditions(composition, p_gpa, t_c)

        try:
            eq = equilibrium(db, components, phases, cond)
            result = self._parse_result(eq, p_gpa, t_c)
        except Exception as exc:
            warnings.warn(f"[pycalphad] Minimisation failed at "
                          f"P={p_gpa:.3f} GPa, T={t_c:.0f}°C: {exc}")
            result = ThermoResult(
                p_gpa=p_gpa, t_c=t_c,
                stable_phases=[],
                backend=self.name,
                converged=False,
                raw=None,
            )
        return result

    def run_grid(
        self,
        composition: Composition,
        p_range: Tuple[float, float],
        t_range: Tuple[float, float],
        n_p: int = 20,
        n_t: int = 20,
    ) -> List[ThermoResult]:
        """P-T pseudosection via pycalphad's vectorised equilibrium.

        Passes the full T and P arrays to a single ``equilibrium()`` call,
        which uses pycalphad's numba-JIT loop internally — much faster than
        calling ``run()`` once per point.  Results are extracted by slicing
        the returned xarray Dataset.
        """
        from pycalphad import equilibrium, variables as v

        db = self._get_db()
        p_vals = np.linspace(p_range[0], p_range[1], n_p)   # GPa
        t_vals = np.linspace(t_range[0], t_range[1], n_t)   # °C

        # Build shared composition conditions
        db_elements = {c.upper() for c in db.elements}
        mol_frac = composition.to_element_mol_frac()
        cond_comps = {
            el.upper(): frac for el, frac in mol_frac.items()
            if el.upper() in db_elements
        }
        if not cond_comps:
            raise ValueError(
                "None of the composition elements are present in the TDB database. "
                f"Database elements: {sorted(db_elements)}"
            )

        components = list(cond_comps.keys()) + (['VA'] if self.add_va else [])
        components = [c for c in components if c in db_elements]
        phases = self._phases_override or list(db.phases.keys())
        elements_sorted = sorted(cond_comps.keys())

        conditions = {
            v.T: t_vals + 273.15,   # K
            v.P: p_vals * 1e9,      # Pa
        }
        for el in elements_sorted[:-1]:
            conditions[v.X(el)] = cond_comps[el]

        print(
            f"[pycalphad] Computing {n_p * n_t} P-T points ({n_p}×{n_t})…",
            flush=True,
        )
        try:
            eq = equilibrium(db, components, phases, conditions, verbose=False)
        except Exception as exc:
            warnings.warn(f"[pycalphad] Grid calculation failed: {exc}")
            return []

        # Discover the T and P dimension names in the result Dataset.
        # pycalphad 0.11 uses 'T' and 'P'; guard against future renames.
        dims = list(eq.dims)
        p_dim = next((d for d in dims if d.upper() == "P"), None)
        t_dim = next((d for d in dims if d.upper() == "T"), None)

        print(f"[pycalphad] Parsing {n_p * n_t} results…", flush=True)
        results: List[ThermoResult] = []
        for i, p_gpa in enumerate(p_vals):
            for j, t_c in enumerate(t_vals):
                sel: dict = {}
                if p_dim:
                    sel[p_dim] = i
                if t_dim:
                    sel[t_dim] = j
                eq_pt = eq.isel(**sel) if sel else eq
                results.append(self._parse_result(eq_pt, p_gpa, t_c))

        return results

    def available(self) -> bool:
        try:
            import pycalphad  # noqa: F401
            return True
        except ImportError:
            return False

    # ---- Internal -----------------------------------------------------------

    def _get_db(self):
        if self._db is None:
            from pycalphad import Database
            self._db = Database(str(self.tdb_path))
        return self._db

    def _build_conditions(
        self, composition: Composition, p_gpa: float, t_c: float
    ) -> Tuple:
        from pycalphad import variables as v

        db = self._get_db()
        db_elements = {c.upper() for c in db.elements}

        # Map composition elements → pycalphad uppercase
        mol_frac = composition.to_element_mol_frac()
        cond_comps = {
            el.upper(): frac for el, frac in mol_frac.items()
            if el.upper() in db_elements
        }
        if not cond_comps:
            raise ValueError(
                "None of the composition elements are present in the TDB database. "
                f"Database elements: {sorted(db_elements)}"
            )

        # Build component list
        components = [el for el in cond_comps] + (['VA'] if self.add_va else [])
        components = [c for c in components if c in db_elements]

        # Phases
        phases = self._phases_override or list(db.phases.keys())

        # Conditions dict (all but one element specified as X)
        elements_sorted = sorted(cond_comps.keys())
        conditions: Dict = {
            v.T: t_c + 273.15,           # K
            v.P: p_gpa * 1e9,            # Pa
        }
        # Specify all but the last element (pycalphad derives the remainder)
        for el in elements_sorted[:-1]:
            conditions[v.X(el)] = cond_comps[el]

        return components, phases, conditions

    def _parse_result(self, eq, p_gpa: float, t_c: float) -> ThermoResult:
        """Convert pycalphad equilibrium Dataset to ThermoResult."""
        import xarray as xr

        try:
            # Phase array: shape (n_P, n_T, n_vertices)
            phase_arr = eq['Phase'].squeeze()
            np_phases = list(phase_arr.values.ravel())
            stable = [p for p in np_phases if p and p != '']
            stable = list(dict.fromkeys(stable))  # unique, order preserved

            # Mole fractions of each phase (NP_PHASE axis if present)
            try:
                np_frac = eq['NP'].squeeze()
                phase_fracs: Dict[str, float] = {}
                for p, f in zip(np_phases, np_frac.values.ravel()):
                    if p and p != '':
                        phase_fracs[p] = float(f)
            except (KeyError, AttributeError):
                n = len(stable)
                phase_fracs = {p: 1.0 / n for p in stable} if n else {}

            phases_out = [
                PhaseResult(
                    name=p,
                    mol_fraction=phase_fracs.get(p, float('nan')),
                )
                for p in stable
            ]

        except Exception as exc:
            warnings.warn(f"[pycalphad] Result parsing error: {exc}")
            phases_out = []

        return ThermoResult(
            p_gpa=p_gpa,
            t_c=t_c,
            stable_phases=phases_out,
            backend=self.name,
            converged=True,
            raw=eq,
        )

    def __repr__(self) -> str:
        return f"PycalphadBackend(tdb='{self.tdb_path.name}')"
