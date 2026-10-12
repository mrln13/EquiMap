# thermo/backends/base.py
"""Abstract backend interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from thermo.composition import Composition
from thermo.result import ThermoResult


class AbstractBackend(ABC):
    """
    Common interface that every thermodynamic backend must implement.

    Backends convert a :class:`~thermo.composition.Composition` + P-T
    conditions into a :class:`~thermo.result.ThermoResult`.  Only
    :meth:`run` is mandatory; :meth:`run_grid` has a default implementation
    that loops over grid nodes.
    """

    #: Short identifier shown in ThermoResult.backend
    name: str = "base"

    @abstractmethod
    def run(self, composition: Composition, p_gpa: float, t_c: float) -> ThermoResult:
        """
        Compute equilibrium at a single P-T point.

        Parameters
        ----------
        composition :   bulk composition
        p_gpa :         pressure [GPa]
        t_c :           temperature [°C]
        """
        ...

    def run_grid(
        self,
        composition: Composition,
        p_range: Tuple[float, float],
        t_range: Tuple[float, float],
        n_p: int = 20,
        n_t: int = 20,
    ) -> List[ThermoResult]:
        """
        Compute equilibrium on a regular P-T grid.

        Default implementation loops over points sequentially.
        Backends that support batch calculations should override this.
        """
        try:
            from tqdm import tqdm
            _wrap = tqdm
        except ImportError:
            def _wrap(it, **kw):
                return it

        p_vals = np.linspace(p_range[0], p_range[1], n_p)
        t_vals = np.linspace(t_range[0], t_range[1], n_t)
        pts = [(float(p), float(t)) for p in p_vals for t in t_vals]

        results: List[ThermoResult] = []
        for p, t in _wrap(pts, desc=f"[{self.name}] P-T grid ({n_p}×{n_t})"):
            results.append(self.run(composition, p, t))
        return results

    def available(self) -> bool:
        """Return True if this backend's dependencies are importable."""
        return True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
