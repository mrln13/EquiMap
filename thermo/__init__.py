# thermo/__init__.py
"""
thermo — pluggable thermodynamic modelling for SEM/EDS compositions.

Quick start
-----------
Geological single point (MAGEMin direct, igneous database)::

    from thermo import ThermoCalc, Composition

    # From EDS element wt% (as returned by viewer.py)
    comp = Composition.from_element_wt(
        {'Si': 21.2, 'Al': 6.4, 'Ca': 6.7, 'Mg': 4.6, 'Fe': 6.2,
         'Na': 2.0, 'Ti': 0.57, 'K': 0.17}
    )

    # Or from geochemical oxide wt%
    comp = Composition.from_oxide_wt(
        {'SiO2': 45.4, 'Al2O3': 9.0, 'FeO': 8.0, 'MgO': 7.6,
         'CaO': 9.4, 'Na2O': 2.7, 'TiO2': 0.95, 'H2O': 2.0}
    )

    calc = ThermoCalc(backend='magemin', db='ig')

    # Single point
    result = calc.run(comp, p_gpa=1.0, t_c=700)
    print(result.phase_names, result.bulk_properties)

    # P-T grid → DataFrame
    df = calc.to_dataframe(
        calc.run_grid(comp, p_range=(0.5, 2.0), t_range=(500, 900))
    )

    # Pseudosection plot
    fig, ax = calc.pseudosection(comp, p_range=(0.5, 2.0), t_range=(500, 900))

Available backends
------------------
* ``"magemin"``   — geological silicate systems (igneous, metapelite,
  metabasalt, ultramafic, and more).  Calls MAGEMin_C.jl directly via
  juliacall.  Requires ``pip install petthermotools`` (installs Julia +
  MAGEMin_C.jl).
* ``"pycalphad"`` — alloys, ceramics, any TDB database.
  Requires ``pip install pycalphad`` and a TDB file.
* ``"perplex"``   — geological, drives Perple_X executables headlessly.
  Requires a local Perple_X 7.x installation.
"""

from thermo.composition import (
    Composition,
    ATOMIC_WEIGHTS,
    OXIDE_FORMULAS,
    ELEMENT_DEFAULT_OXIDE,
    oxide_mw,
    element_in_oxide_mass_fraction,
)
from thermo.result import PhaseResult, ThermoResult, results_to_dataframe
from thermo.calc import ThermoCalc
from thermo.backends import list_backends, get_backend_class

__all__ = [
    # Main classes
    "Composition",
    "ThermoCalc",
    "ThermoResult",
    "PhaseResult",
    # Utilities
    "results_to_dataframe",
    "list_backends",
    "get_backend_class",
    # Composition data tables (for users who need them)
    "ATOMIC_WEIGHTS",
    "OXIDE_FORMULAS",
    "ELEMENT_DEFAULT_OXIDE",
    "oxide_mw",
    "element_in_oxide_mass_fraction",
]
