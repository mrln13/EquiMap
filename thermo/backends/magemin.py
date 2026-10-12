# thermo/backends/magemin.py
"""
MAGEMin_C.jl direct backend — full database support.

Bypasses PetThermoTools / MAGEMinCalc and calls MAGEMin_C.jl v2.x
directly via juliacall, giving access to all supported databases:

    "ig"   — Igneous (Green et al. 2025 / Holland et al. 2018)
    "igad" — Igneous, Weller et al. 2024
    "mp"   — Metapelite, White et al. 2014
    "mb"   — Metabasalt
    "um"   — Ultramafic
    "mpe"  — Metapelite extended
    "mbe"  — Metabasalt extended
    "mtl"  — Metatropical limestone

Requires:
    pip install juliacall juliapkg   (installed automatically by petthermotools)
    Julia package MAGEMin_C v2.x installed in any Julia depot

Pressure : kbar internally  (1 GPa = 10 kbar);  public API uses GPa
Temperature : °C
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from thermo.backends.base import AbstractBackend
from thermo.composition import Composition
from thermo.result import PhaseResult, ThermoResult

# ---------------------------------------------------------------------------
# Database metadata
# ---------------------------------------------------------------------------

SUPPORTED_DB: Dict[str, str] = {
    "ig":   "Igneous — Green et al. 2025",
    "igad": "Igneous — Weller et al. 2024",
    "mp":   "Metapelite — White et al. 2014",
    "mb":   "Metabasalt",
    "um":   "Ultramafic",
    "mpe":  "Metapelite extended",
    "mbe":  "Metabasalt extended",
    "mtl":  "Metatropical limestone",
}

# Oxides that each database understands (used to build the Xoxides list).
# "O" is the ferric-iron / oxygen-fugacity component (not Fe2O3 directly).
DB_OXIDES: Dict[str, List[str]] = {
    "ig":   ["SiO2", "Al2O3", "CaO", "MgO", "FeO", "K2O", "Na2O", "TiO2", "O", "Cr2O3", "H2O"],
    "igad": ["SiO2", "Al2O3", "CaO", "MgO", "FeO", "K2O", "Na2O", "TiO2", "O", "Cr2O3"],
    "mp":   ["SiO2", "Al2O3", "CaO", "MgO", "FeO", "K2O", "Na2O", "TiO2", "O", "MnO", "H2O"],
    "mb":   ["SiO2", "Al2O3", "CaO", "MgO", "FeO", "K2O", "Na2O", "TiO2", "O", "H2O"],
    "um":   ["SiO2", "Al2O3", "MgO", "FeO", "O", "H2O", "S"],
    "mpe":  ["SiO2", "Al2O3", "CaO", "MgO", "FeO", "K2O", "Na2O", "TiO2", "O", "MnO", "H2O", "CO2", "S"],
    "mbe":  ["SiO2", "Al2O3", "CaO", "MgO", "FeO", "K2O", "Na2O", "TiO2", "O", "H2O"],
    "mtl":  ["SiO2", "Al2O3", "CaO", "MgO", "FeO", "K2O", "Na2O", "TiO2", "O", "MnO", "H2O", "CO2"],
}

# Molecular weights
_MW = {
    "Fe": 55.845,
    "O":  15.999,
    "Fe2O3": 2 * 55.845 + 3 * 15.999,  # 159.69
    "FeO":   55.845 + 15.999,           # 71.844
}


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

class MAGEMinDirectBackend(AbstractBackend):
    """
    Thermodynamic equilibrium via MAGEMin_C.jl (direct Julia call).

    Parameters
    ----------
    db :
        Database code.  See ``SUPPORTED_DB`` for options.
    fe3_fraction :
        Fe³⁺ / Fe_total molar ratio used to split total iron into
        FeO (Fe²⁺) and the "O" oxygen-fugacity component.
    """

    name = "MAGEMin"

    def __init__(self, db: str = "ig", fe3_fraction: float = 0.1):
        if db not in SUPPORTED_DB:
            raise ValueError(
                f"Unknown MAGEMin database '{db}'. "
                f"Choose one of: {list(SUPPORTED_DB)}"
            )
        self.db = db
        self.fe3_fraction = float(fe3_fraction)
        self._jl = None   # set on first _activate()

    # ---- Public API ---------------------------------------------------------

    def available(self) -> bool:
        try:
            import petthermotools  # noqa: F401
            from juliacall import Main  # noqa: F401
            return True
        except ImportError:
            return False

    # keep the old name that ThermoCalc.check_backend() calls
    check_backend = available

    def run(
        self,
        composition: Composition,
        p_gpa: float,
        t_c: float,
        **kw,
    ) -> ThermoResult:
        """Single-point equilibrium (implemented as a tiny 2-point grid)."""
        results = self.run_grid(
            composition,
            p_range=(p_gpa, p_gpa + 0.001),
            t_range=(t_c,   t_c   + 1.0),
            n_p=2, n_t=2,
        )
        if not results:
            return ThermoResult(
                p_gpa=p_gpa, t_c=t_c, stable_phases=[],
                backend=f"magemin/{self.db}", converged=False,
            )
        return results[0]

    def run_grid(
        self,
        composition: Composition,
        p_range: Tuple[float, float],
        t_range: Tuple[float, float],
        n_p: int = 20,
        n_t: int = 20,
    ) -> List[ThermoResult]:
        """P-T grid via MAGEMin_C.jl multi_point_minimization."""
        self._activate()
        jl = self._jl

        # --- Build composition -----------------------------------------------
        xoxides, x = self._build_composition(composition)

        # --- Build flat P-T arrays (P-major: for each p, all t values) ---------
        # juliacall requires Vector{Float64} — pass numpy float64 arrays.
        p_vals = np.linspace(p_range[0], p_range[1], n_p)
        t_vals = np.linspace(t_range[0], t_range[1], n_t)
        P_kbar_list: List[float] = []
        T_C_list:    List[float] = []
        for pv in p_vals:
            for tv in t_vals:
                P_kbar_list.append(float(pv) * 10.0)   # GPa → kbar
                T_C_list.append(float(tv))
        P_kbar_np = np.array(P_kbar_list, dtype=np.float64)
        T_C_np    = np.array(T_C_list,    dtype=np.float64)
        # Build X and Xoxides as proper Julia types via seval.
        # juliacall converts numpy arrays to PyArray{Float64}, not Vector{Float64}.
        # MAGEMin_C dispatches on isa(X, Vector{Float64}) to choose the
        # "same composition for all points" branch — PyArray fails that check.
        x_lit       = ", ".join(repr(float(v)) for v in x)
        jl_x        = jl.seval(f"Float64[{x_lit}]")
        xoxides_lit = ", ".join(f'"{s}"' for s in xoxides)
        jl_xoxides  = jl.seval(f"[{xoxides_lit}]")

        # --- Call MAGEMin_C.jl -----------------------------------------------
        n_pts = n_p * n_t
        print(
            f"[MAGEMin] Computing {n_pts} P-T points ({n_p}×{n_t}) "
            f"on database '{self.db}'…",
            flush=True,
        )
        data = jl.Initialize_MAGEMin(self.db, verbose=-1)
        try:
            out_list = jl.multi_point_minimization(
                P_kbar_np,
                T_C_np,
                data,
                X=jl_x,
                Xoxides=jl_xoxides,
                sys_in="wt",
                progressbar=True,
            )
        finally:
            jl.Finalize_MAGEMin(data)
        print(f"[MAGEMin] Done ({n_pts} points).", flush=True)

        # --- Convert results --------------------------------------------------
        return [self._out_to_result(out) for out in out_list]

    # ---- Internal -----------------------------------------------------------

    def _activate(self):
        """Load MAGEMin_C into the Julia session (lazy, once per process)."""
        if self._jl is not None:
            return
        import petthermotools as ptt
        ptt.activate_petthermotools_env()
        from juliacall import Main as jl
        jl.seval("using MAGEMin_C")
        self._jl = jl

    def _build_composition(
        self, composition: Composition
    ) -> Tuple[List[str], List[float]]:
        """
        Convert Composition to (Xoxides, X) lists expected by MAGEMin_C.jl.

        MAGEMin uses "O" as the ferric-iron/oxygen component, not "Fe2O3".
        We convert:
          FeO_total  = Fe²⁺ as FeO  +  Fe from Fe2O3 re-expressed as FeO
          O          = excess oxygen from Fe³⁺  (= O in Fe2O3 minus O in FeO)
        """
        # to_oxide_wt splits Fe into FeO (Fe²⁺) and Fe2O3 (Fe³⁺)
        ox = composition.to_oxide_wt(fe3_fraction=self.fe3_fraction)

        fe2o3_wt = ox.get("Fe2O3", 0.0)
        feo_wt   = ox.get("FeO",   0.0)

        mw_fe2o3 = _MW["Fe2O3"]
        mw_feo   = _MW["FeO"]
        mw_o     = _MW["O"]

        # Fe from Fe2O3, re-expressed as FeO (for the FeO component)
        feo_from_fe3 = fe2o3_wt * (2.0 * mw_feo) / mw_fe2o3
        # Excess oxygen (the "O" component in MAGEMin)
        o_from_fe3   = fe2o3_wt * mw_o / mw_fe2o3

        total_feo = feo_wt + feo_from_fe3

        # Build lists, including only non-zero values
        all_oxides = DB_OXIDES.get(self.db, DB_OXIDES["ig"])
        xoxides: List[str] = []
        x: List[float] = []
        for oxide in all_oxides:
            if oxide == "FeO":
                if total_feo > 0.0:
                    xoxides.append("FeO")
                    x.append(total_feo)
            elif oxide == "O":
                if o_from_fe3 > 0.0:
                    xoxides.append("O")
                    x.append(o_from_fe3)
            else:
                val = ox.get(oxide, 0.0)
                if val > 0.0:
                    xoxides.append(oxide)
                    x.append(val)

        return xoxides, x

    @staticmethod
    def _instance_compositions(ph_type, ph_id, ss_comps, pp_comps):
        """Per-instance compositions for the phases listed in ``out.ph``.

        MAGEMin_C stores compositions per *instance* in ``SS_vec`` (solution
        phases) and ``PP_vec`` (pure phases), and for the i-th entry of
        ``out.ph`` gives ``ph_type[i]`` (1 = solution, 0 = pure) and
        ``ph_id[i]`` (index into the matching vector). ``SS_syms``/``PP_syms``
        are unordered lists of *unique* names and must never be zipped with
        the vectors: doing so scrambled every reported phase composition
        (biotite shown as fluid, quartz as rutile) and collapsed a solvus
        pair (two ``fsp`` = plagioclase + K-feldspar) onto one composition.

        Returns a list aligned with ``ph_type``/``ph_id``; ``None`` where the
        lookup is impossible (missing ids or out-of-range index).
        """
        out: List[Optional[Dict[str, float]]] = []
        n = max(len(ph_type), len(ph_id))
        for i in range(n):
            try:
                vec = ss_comps if int(ph_type[i]) == 1 else pp_comps
                out.append(vec[int(ph_id[i])])
            except (IndexError, TypeError, ValueError):
                out.append(None)
        return out

    def _out_to_result(self, out) -> ThermoResult:
        """Convert a MAGEMin_C.jl gmin_struct to ThermoResult."""

        # ---- Per-phase oxide compositions -----------------------------------
        # Per-instance {oxide: wt%} dicts in SS_vec / PP_vec order, mapped onto
        # ``out.ph`` through ph_type/ph_id (see _instance_compositions).
        # Best-effort: an instance without a resolvable composition gets None.
        inst_comps: List[Optional[Dict[str, float]]] = []
        try:
            oxides = [str(o) for o in list(out.oxides)]

            def _comp_wt(entry):
                return {ox: float(v) * 100.0
                        for ox, v in zip(oxides, list(entry.Comp_wt))
                        if float(v) > 0.0}

            ss_comps = [_comp_wt(ss) for ss in list(out.SS_vec)]
            pp_comps = [_comp_wt(pp) for pp in list(out.PP_vec)]
            inst_comps = self._instance_compositions(
                list(out.ph_type), list(out.ph_id), ss_comps, pp_comps)
        except Exception:
            inst_comps = []

        # ---- Phase assemblage -----------------------------------------------
        ph_names    = list(out.ph)
        ph_frac     = list(out.ph_frac)
        ph_frac_wt  = list(out.ph_frac_wt)
        _vol        = getattr(out, "ph_frac_vol", None)
        ph_frac_vol = list(_vol) if _vol is not None else [None] * len(ph_names)

        phases: List[PhaseResult] = []
        for i, (name, mf, wf, vf) in enumerate(
                zip(ph_names, ph_frac, ph_frac_wt, ph_frac_vol)):
            sname = str(name)
            phases.append(PhaseResult(
                name=sname,
                mol_fraction=float(mf),
                wt_fraction=float(wf),
                vol_fraction=float(vf) if vf is not None else None,
                composition_oxide_wt=inst_comps[i] if i < len(inst_comps) else None,
                is_melt=any(k in sname.lower() for k in ("liq", "melt")),
            ))

        # ---- Bulk properties ------------------------------------------------
        # Helper: safely extract a scalar, handling 1-element Julia vectors.
        def _sc(field: str):
            v = getattr(out, field, None)
            if v is None:
                return None
            try:
                return float(v)
            except Exception:
                try:
                    return float(list(v)[0])
                except Exception:
                    return None

        props: Dict[str, float] = {}
        for key, field in [
            # Densities (kg/m³)
            ("rho_kgm3",        "rho"),
            ("rho_melt_kgm3",   "rho_M"),
            ("rho_solid_kgm3",  "rho_S"),
            ("rho_fluid_kgm3",  "rho_F"),
            # Seismic velocities (km/s)
            ("Vp_kms",          "Vp"),
            ("Vs_kms",          "Vs"),
            ("Vp_S_kms",        "Vp_S"),
            ("Vs_S_kms",        "Vs_S"),
            # Elastic moduli (GPa)
            ("bulkMod_GPa",     "bulkMod"),
            ("shearMod_GPa",    "shearMod"),
            # Thermodynamics
            ("G_system_kJ",     "G_system"),
            ("enthalpy_kJ",     "enthalpy"),
            ("entropy_kJK",     "entropy"),
            ("cp_Jkg_K",        "s_cp"),
            ("alpha_per_K",     "alpha"),
            # Oxygen fugacity
            ("fO2",             "fO2"),
            ("dQFM",            "dQFM"),
            # Melt / solid / fluid modal fractions (molar)
            ("frac_melt",       "frac_M"),
            ("frac_solid",      "frac_S"),
            ("frac_fluid",      "frac_F"),
            # Weight fractions
            ("frac_melt_wt",    "frac_M_wt"),
            ("frac_solid_wt",   "frac_S_wt"),
            ("frac_fluid_wt",   "frac_F_wt"),
        ]:
            v = _sc(field)
            if v is not None:
                props[key] = v

        return ThermoResult(
            p_gpa=float(out.P_kbar) / 10.0,
            t_c=float(out.T_C),
            stable_phases=phases,
            bulk_properties=props,
            backend=f"magemin/{self.db}",
            converged=(int(out.status) == 0),
            raw=out,
        )

    def __repr__(self) -> str:
        return f"MAGEMinDirectBackend(db='{self.db}', fe3_fraction={self.fe3_fraction})"
