"""
CLI entry point for the thermo package.

Usage
-----
Single point::

    python -m thermo run \\
        --composition SiO2:45.4,Al2O3:9.0,FeO:8.0,MgO:7.6,CaO:9.4,Na2O:2.7,TiO2:0.95,H2O:2.0 \\
        --p 1.0 --t 700 \\
        --backend magemin --database ig

Or from a JSON file (element wt% or oxide wt%)::

    python -m thermo run \\
        --comp-file roi_composition.json \\
        --p 1.0 --t 700

P-T grid -> CSV::

    python -m thermo grid \\
        --comp-file roi_composition.json \\
        --p-range 0.5 2.0 --t-range 400 900 \\
        --n-p 30 --n-t 30 \\
        --backend magemin --database ig \\
        --out pseudosection.csv

Pseudosection plot (saves PNG)::

    python -m thermo plot \\
        --comp-file roi_composition.json \\
        --p-range 0.5 2.0 --t-range 400 900 \\
        --backend magemin --database ig \\
        --property n_phases \\
        --out pseudosection.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict


def _parse_inline_composition(s: str) -> Dict[str, float]:
    """Parse 'SiO2:45.4,Al2O3:9.0,...' into a dict."""
    out: Dict[str, float] = {}
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            print(f"[warn] Ignoring malformed token '{token}' (expected 'key:value')")
            continue
        k, v = token.split(":", 1)
        out[k.strip()] = float(v.strip())
    return out


def _load_composition(args):
    """Return a Composition from CLI args."""
    from thermo import Composition

    if args.comp_file:
        p = Path(args.comp_file)
        data = json.loads(p.read_text(encoding="utf-8"))
        # Accept either {oxides_wt_percent: {...}} or flat dict
        if "oxides_wt_percent" in data:
            return Composition.from_oxide_wt(data["oxides_wt_percent"])
        elif "elements_wt_percent" in data:
            return Composition.from_element_wt(data["elements_wt_percent"])
        else:
            # Try to guess from keys: element symbols are 1-2 chars, oxides contain digits or O
            first_key = next(iter(data))
            if any(c.isdigit() for c in first_key) or first_key.endswith("O") or first_key == "H2O":
                return Composition.from_oxide_wt(data)
            else:
                return Composition.from_element_wt(data)
    elif args.composition:
        raw = _parse_inline_composition(args.composition)
        # Guess oxide vs element by checking OXIDE_FORMULAS
        from thermo.composition import OXIDE_FORMULAS
        if any(k in OXIDE_FORMULAS for k in raw):
            return Composition.from_oxide_wt(raw)
        else:
            return Composition.from_element_wt(raw)
    else:
        print("Error: provide --comp-file or --composition")
        sys.exit(1)


def _make_backend_kwargs(args) -> dict:
    kw: dict = {}
    backend = args.backend.lower()
    if backend == "magemin":
        kw["db"] = args.database
        kw["fe3_fraction"] = args.fe3
    elif backend == "perplex":
        if args.dataset:
            kw["dataset"] = args.dataset
        if args.perplex_dir:
            kw["perplex_dir"] = args.perplex_dir
        kw["fe3_fraction"] = args.fe3
    elif backend == "pycalphad":
        if not args.tdb:
            print("Error: --tdb is required for the pycalphad backend")
            sys.exit(1)
        kw["tdb_path"] = args.tdb
    return kw


def _add_common_args(p: argparse.ArgumentParser):
    p.add_argument("--comp-file", metavar="PATH",
                   help="JSON file with composition (oxide or element wt%%)")
    p.add_argument("--composition", metavar="K:V,...",
                   help="Inline composition, e.g. SiO2:45.4,Al2O3:9.0,...")
    p.add_argument("--backend", default="magemin",
                   choices=["magemin", "pycalphad", "perplex"],
                   help="Thermodynamic backend (default: magemin)")
    p.add_argument("--database", default="ig",
                   choices=["ig", "mp", "mb", "um", "ume", "mtl"],
                   help="MAGEMin database (default: ig, igneous)")
    p.add_argument("--fe3", type=float, default=0.0,
                   metavar="FRAC", help="Fe3+/Fe_total (default: 0.0)")
    p.add_argument("--tdb", metavar="PATH", help="TDB file for pycalphad backend")
    p.add_argument("--dataset", metavar="FILE",
                   help="Thermodynamic dataset for Perple_X (e.g. hp633ver.dat)")
    p.add_argument("--perplex-dir", metavar="DIR", help="Perple_X installation directory")


def cmd_run(args):
    from thermo import ThermoCalc
    comp = _load_composition(args)
    print(f"Composition: {comp}")

    calc = ThermoCalc(backend=args.backend, **_make_backend_kwargs(args))
    if not calc.check_backend():
        print(f"Error: backend '{args.backend}' is not available. Install it first.")
        sys.exit(1)

    result = calc.run(comp, p_gpa=args.p, t_c=args.t, fe3_fraction=args.fe3)
    print(result)
    print(f"\nStable phases ({result.n_phases}):")
    for ph in result.stable_phases:
        frac_str = f"{ph.mol_fraction:.4f}"
        wt_str = f"  (wt {ph.wt_fraction:.4f})" if ph.wt_fraction is not None else ""
        print(f"  {ph.name:<12s}  mol_frac={frac_str}{wt_str}")
    if result.bulk_properties:
        print("\nBulk properties:")
        for k, v in result.bulk_properties.items():
            print(f"  {k} = {v:.4g}")

    if args.out:
        import pandas as pd
        pd.DataFrame([result.to_dict()]).to_csv(args.out, index=False)
        print(f"\nSaved to {args.out}")


def cmd_grid(args):
    from thermo import ThermoCalc, results_to_dataframe
    comp = _load_composition(args)
    print(f"Composition: {comp}")
    print(f"Grid: P=[{args.p_range[0]}, {args.p_range[1]}] GPa × "
          f"T=[{args.t_range[0]}, {args.t_range[1]}] °C, "
          f"{args.n_p}×{args.n_t} points")

    calc = ThermoCalc(backend=args.backend, **_make_backend_kwargs(args))
    if not calc.check_backend():
        print(f"Error: backend '{args.backend}' is not available.")
        sys.exit(1)

    results = calc.run_grid(
        comp,
        p_range=(args.p_range[0], args.p_range[1]),
        t_range=(args.t_range[0], args.t_range[1]),
        n_p=args.n_p,
        n_t=args.n_t,
    )
    df = results_to_dataframe(results)
    out = args.out or "pseudosection.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} grid points to {out}")
    print(df.head())


def cmd_plot(args):
    from thermo import ThermoCalc
    comp = _load_composition(args)

    calc = ThermoCalc(backend=args.backend, **_make_backend_kwargs(args))
    if not calc.check_backend():
        print(f"Error: backend '{args.backend}' is not available.")
        sys.exit(1)

    fig, ax = calc.pseudosection(
        comp,
        p_range=(args.p_range[0], args.p_range[1]),
        t_range=(args.t_range[0], args.t_range[1]),
        n_p=args.n_p,
        n_t=args.n_t,
        property=args.property,
    )
    out = args.out or "pseudosection.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out}")

    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()


def cmd_cache(args):
    from thermo.cache import GridCache

    gc = GridCache()
    if args.cache_action == "path":
        print(gc.cache_dir)
        return
    if args.cache_action == "clear":
        n = gc.clear()
        print(f"Removed {n} cached grid(s) from {gc.cache_dir}")
        return
    # status (default)
    s = gc.stats()
    mb = s["total_bytes"] / (1024 * 1024)
    print(f"Grid result cache")
    print(f"  dir        : {s['dir']}")
    print(f"  exists     : {s['exists']}")
    print(f"  enabled    : {s['enabled']}")
    print(f"  version    : {s['version']}")
    print(f"  entries    : {s['n_entries']}")
    print(f"  total size : {mb:.2f} MiB")


def main():
    parser = argparse.ArgumentParser(
        prog="python -m thermo",
        description="Thermodynamic equilibrium from EDS/bulk compositions.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- run (single point) ---
    p_run = sub.add_parser("run", help="Single P-T point")
    _add_common_args(p_run)
    p_run.add_argument("--p", type=float, required=True, metavar="GPa",
                       help="Pressure [GPa]")
    p_run.add_argument("--t", type=float, required=True, metavar="C",
                       help="Temperature [°C]")
    p_run.add_argument("--out", metavar="CSV", help="Save result to CSV")
    p_run.set_defaults(func=cmd_run)

    # --- grid ---
    p_grid = sub.add_parser("grid", help="P-T grid -> CSV")
    _add_common_args(p_grid)
    p_grid.add_argument("--p-range", nargs=2, type=float, default=[0.5, 1.5],
                        metavar=("P_MIN", "P_MAX"), help="Pressure range [GPa]")
    p_grid.add_argument("--t-range", nargs=2, type=float, default=[400, 800],
                        metavar=("T_MIN", "T_MAX"), help="Temperature range [°C]")
    p_grid.add_argument("--n-p", type=int, default=20, help="Grid points in P")
    p_grid.add_argument("--n-t", type=int, default=20, help="Grid points in T")
    p_grid.add_argument("--out", metavar="CSV", default="pseudosection.csv")
    p_grid.set_defaults(func=cmd_grid)

    # --- plot ---
    p_plot = sub.add_parser("plot", help="P-T grid -> PNG pseudosection plot")
    _add_common_args(p_plot)
    p_plot.add_argument("--p-range", nargs=2, type=float, default=[0.5, 1.5],
                        metavar=("P_MIN", "P_MAX"))
    p_plot.add_argument("--t-range", nargs=2, type=float, default=[400, 800],
                        metavar=("T_MIN", "T_MAX"))
    p_plot.add_argument("--n-p", type=int, default=30)
    p_plot.add_argument("--n-t", type=int, default=30)
    p_plot.add_argument("--property", default="n_phases",
                        help="Property to plot (default: n_phases)")
    p_plot.add_argument("--out", metavar="PNG", default="pseudosection.png")
    p_plot.add_argument("--no-show", action="store_true",
                        help="Don't open interactive plot window")
    p_plot.set_defaults(func=cmd_plot)

    # --- cache (inspect / clear the grid result cache) ---
    p_cache = sub.add_parser("cache", help="Inspect or clear the grid result cache")
    p_cache.add_argument("cache_action", nargs="?", default="status",
                         choices=["status", "clear", "path"],
                         help="status (default) | clear | path")
    p_cache.set_defaults(func=cmd_cache)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
