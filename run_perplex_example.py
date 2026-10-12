"""
Example: running Perple_X via the THERMOD thermo package.

Requirements:
  - Perple_X 7.x installed (executables: build.exe, vertex.exe, werami.exe)
  - hp633ver.dat and solution_model.dat in C:\perplex\datafiles\

Usage:
  python run_perplex_example.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


from thermo.backends.perplex import PerplexBackend
from thermo.composition import Composition

# ---------------------------------------------------------------------------
# 1. Define bulk composition (oxide wt%)
# ---------------------------------------------------------------------------
comp = Composition.from_oxide_wt({
    "SiO2":  45.4,
    "Al2O3":  9.0,
    "FeO":    8.0,
    "MgO":    7.6,
    "CaO":    9.4,
    "Na2O":   2.7,
    "TiO2":   0.95,
})

# ---------------------------------------------------------------------------
# 2. Configure the backend
# ---------------------------------------------------------------------------
backend = PerplexBackend(
    perplex_dir=r"C:\perplex",          # directory with build/vertex/werami
    dataset="hp633ver.dat",             # thermodynamic dataset
    solution_model="solution_model.dat",
    outdir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "perplex_runs"),
    # Optional: override default solution phases
    # solution_phases=["Cpx(JH)", "Opx(JH)", "Pl(JH)", "Gt(W)", "melt(G)"],
)

print(f"Backend  : {backend}")
print(f"Available: {backend.available()}")
print(f"Phases   : {backend.solution_phases}")
print()

# ---------------------------------------------------------------------------
# 3a. Single-point calculation
# ---------------------------------------------------------------------------
print("=== Single-point run (1.0 GPa, 700 °C) ===")
result = backend.run(comp, p_gpa=1.0, t_c=700.0)
print(f"  P = {result.p_gpa:.3f} GPa")
print(f"  T = {result.t_c:.1f} °C")
print(f"  rho = {result.bulk_properties.get('rho_kgm3', 'N/A'):.1f} kg/m3")
print(f"  converged = {result.converged}")
print()

# ---------------------------------------------------------------------------
# 3b. P-T grid calculation (pseudosection)
# ---------------------------------------------------------------------------
print("=== Grid run (0.5–1.5 GPa, 600–800 °C, 5×5) ===")
results = backend.run_grid(
    comp,
    p_range=(0.5, 1.5),
    t_range=(600.0, 800.0),
    n_p=5,
    n_t=5,
)
print(f"  {len(results)} grid points returned")
print()
print(f"  {'P (GPa)':>8}  {'T (°C)':>7}  {'rho (kg/m3)':>12}")
print(f"  {'-'*35}")
for r in results:
    rho = r.bulk_properties.get("rho_kgm3", float("nan"))
    print(f"  {r.p_gpa:8.3f}  {r.t_c:7.1f}  {rho:12.1f}")
