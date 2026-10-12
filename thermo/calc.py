# thermo/calc.py
"""
ThermoCalc — the user-facing facade.

Examples
--------
Geological single point (MAGEMin direct, igneous database)::

    from thermo import ThermoCalc, Composition

    comp = Composition.from_oxide_wt({
        'SiO2': 45.4, 'Al2O3': 9.0, 'FeO': 8.0, 'MgO': 7.6,
        'CaO': 9.4, 'Na2O': 2.7, 'TiO2': 0.95, 'H2O': 10.0,
    })
    calc = ThermoCalc(backend='magemin', db='ig')
    result = calc.run(comp, p_gpa=1.0, t_c=700)
    print(result)

P-T pseudosection as a DataFrame::

    df = calc.to_dataframe(
        calc.run_grid(comp, p_range=(0.5, 2.0), t_range=(500, 900))
    )

From EDS element wt% (as stored in viewer.py)::

    comp = Composition.from_element_wt(panel.last_spx_composition)
    result = calc.run(comp, p_gpa=0.8, t_c=650)

Alloys with pycalphad::

    comp = Composition.from_element_wt({'Fe': 70.0, 'Cr': 18.0, 'Ni': 12.0})
    calc = ThermoCalc(backend='pycalphad', tdb_path='steel.tdb')
    result = calc.run(comp, p_gpa=1e-4, t_c=900)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from thermo.backends import get_backend_class, list_backends
from thermo.backends.base import AbstractBackend
from thermo.composition import Composition
from thermo.result import ThermoResult, results_to_dataframe


class ThermoCalc:
    """
    Facade for thermodynamic equilibrium calculations.

    Parameters
    ----------
    backend :
        Either a backend name string (``"magemin"``, ``"pycalphad"``,
        ``"perplex"``) or an already-constructed :class:`AbstractBackend`
        instance.
    **backend_kwargs :
        Passed to the backend constructor when *backend* is a string.
        E.g. ``db="mp"`` for magemin, or ``tdb_path="steel.tdb"``
        for pycalphad.
    """

    def __init__(
        self,
        backend: Union[str, AbstractBackend] = "magemin",
        **backend_kwargs: Any,
    ):
        if isinstance(backend, str):
            cls = get_backend_class(backend)
            self._backend: AbstractBackend = cls(**backend_kwargs)
            # Remembered so run_grid can build a stable cache key (see cache.py).
            self._backend_name: str = backend
            self._backend_kwargs: Dict[str, Any] = dict(backend_kwargs)
        elif isinstance(backend, AbstractBackend):
            self._backend = backend
            self._backend_name = getattr(backend, "name", type(backend).__name__)
            self._backend_kwargs = {}
        else:
            raise TypeError(
                f"backend must be a string or AbstractBackend instance, "
                f"got {type(backend).__name__}."
            )

    # ---- Core methods -------------------------------------------------------

    def run(
        self,
        composition: Composition,
        p_gpa: float,
        t_c: float,
        **kwargs: Any,
    ) -> ThermoResult:
        """
        Compute equilibrium at a single P-T point.

        Parameters
        ----------
        composition :   bulk composition
        p_gpa :         pressure [GPa]
        t_c :           temperature [°C]
        **kwargs :      forwarded to the backend's ``run()`` method.
        """
        return self._backend.run(composition, float(p_gpa), float(t_c), **kwargs)

    def run_grid(
        self,
        composition: Composition,
        p_range: Tuple[float, float] = (0.5, 1.5),
        t_range: Tuple[float, float] = (400.0, 800.0),
        n_p: int = 20,
        n_t: int = 20,
        *,
        use_cache: bool = True,
        cache: "Optional[Any]" = None,
        **kwargs: Any,
    ) -> List[ThermoResult]:
        """
        Compute equilibrium on a regular P-T grid.

        Returns a flat list of ThermoResult objects (P varies fastest).
        Use :meth:`to_dataframe` to convert to a tidy DataFrame.

        Results are memoised on disk (see :mod:`thermo.cache`); an identical
        grid (same backend config + composition + P-T range + resolution) is
        served instantly from cache.  Pass ``use_cache=False`` to force a fresh
        computation (still stored), or a :class:`~thermo.cache.GridCache`
        instance via ``cache`` to use a non-default location.
        """
        # Select the cache to use.  An explicit ``cache`` always provides the
        # store location; ``use_cache`` toggles whether we *read* from it
        # (False forces a fresh computation that is still stored).
        gc = None
        key = None
        if cache is not None:
            gc = cache
        elif use_cache:
            try:
                from thermo.cache import GridCache
                gc = GridCache()
            except Exception:
                gc = None

        if gc is not None and getattr(gc, "enabled", False):
            try:
                key = gc.key_for(
                    self._backend_name, self._backend_kwargs, composition,
                    p_range, t_range, n_p, n_t,
                    extra=(kwargs or None),
                )
            except Exception:
                key = None
            if use_cache and key is not None:
                try:
                    hit = gc.get_results(key)
                except Exception:
                    hit = None
                if hit is not None:
                    print(f"[thermo] grid served from cache ({self._backend_name}, "
                          f"{len(hit)} pts, key {key[:12]}…)", flush=True)
                    return hit

        results = self._backend.run_grid(
            composition,
            p_range=p_range,
            t_range=t_range,
            n_p=n_p,
            n_t=n_t,
            **kwargs,
        )

        if gc is not None and getattr(gc, "enabled", False) and key is not None:
            gc.put_results(key, results, meta={
                "backend": self._backend_name,
                "n_p": int(n_p), "n_t": int(n_t),
                "p_range": [float(p_range[0]), float(p_range[1])],
                "t_range": [float(t_range[0]), float(t_range[1])],
            })

        return results

    # ---- Convenience --------------------------------------------------------

    @staticmethod
    def to_dataframe(results: List[ThermoResult]) -> pd.DataFrame:
        """Convert a list of ThermoResult objects to a tidy DataFrame."""
        return results_to_dataframe(results)

    def pseudosection(
        self,
        composition: Composition,
        p_range: Tuple[float, float] = (0.5, 1.5),
        t_range: Tuple[float, float] = (400.0, 800.0),
        n_p: int = 30,
        n_t: int = 30,
        property: str = "n_phases",
        cmap: str = "tab20",
        ax=None,
        **run_kwargs,
    ):
        """
        Compute a P-T grid and produce a simple pseudosection plot.

        Parameters
        ----------
        property :
            Column name from ``to_dict()`` to plot.  E.g. ``"n_phases"``,
            ``"rho_kgm3"``, or ``"mol_frac_liq"``.
        ax :
            Existing matplotlib Axes to draw on.  Created if None.

        Returns
        -------
        fig, ax
        """
        import matplotlib.pyplot as plt
        import numpy as np

        results = self.run_grid(composition, p_range, t_range, n_p, n_t, **run_kwargs)
        df = self.to_dataframe(results)

        if property not in df.columns:
            available = [c for c in df.columns if c not in ('backend',)]
            raise ValueError(
                f"Property '{property}' not in results. Available: {available}"
            )

        grid = df[property].values.reshape(n_p, n_t)
        t_vals = np.linspace(t_range[0], t_range[1], n_t)
        p_vals = np.linspace(p_range[0], p_range[1], n_p)

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 5))
        else:
            fig = ax.figure

        im = ax.pcolormesh(t_vals, p_vals, grid, cmap=cmap, shading='auto')
        fig.colorbar(im, ax=ax, label=property)
        ax.set_xlabel("Temperature [°C]")
        ax.set_ylabel("Pressure [GPa]")
        ax.set_title(f"Pseudosection — {property} ({self._backend.name})")
        return fig, ax

    # ---- Info ---------------------------------------------------------------

    @property
    def backend(self) -> AbstractBackend:
        return self._backend

    @staticmethod
    def available_backends() -> List[str]:
        """Return names of all registered backends."""
        return list_backends()

    def check_backend(self) -> bool:
        """Return True if this backend's dependencies are installed."""
        ok = self._backend.available()
        if not ok:
            install_hints = {
                "MAGEMin":   "pip install petthermotools  (also requires Julia + MAGEMin_C.jl)",
                "pycalphad": "pip install pycalphad",
                "perplex":   "install Perple_X from https://www.perplex.ethz.ch/",
            }
            hint = install_hints.get(self._backend.name, "install the required package")
            print(
                f"[ThermoCalc] Backend '{self._backend.name}' is not available. "
                f"To fix: {hint}"
            )
        return ok

    def __repr__(self) -> str:
        return f"ThermoCalc(backend={self._backend!r})"
