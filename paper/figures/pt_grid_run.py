"""Compute the Fig 5 P-T grid of the worked example headlessly.

The section's SPX bulk (the composition of session run #3, the whole-section
SPX quantification) is turned into the modelled bulk by ``model_bulk`` and
run through MAGEMin's metapelite database on the grid of pt_grid_spec.py (Fig 5).

Writes
  pt_grid.csv         one row per cell (``ThermoResult.to_dict``): what
                      fig_pt.py, fig_reconstruction.py and the scorers read
  pt_grid_bulk.json   provenance: the SPX bulk, the elements passed and
                      dropped, the oxide composition given to MAGEMin

Usage:  python pt_grid_run.py [--session <viewer_session.json>]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(HERE))
import model_bulk                                                  # noqa: E402
from thermo import ThermoCalc                                      # noqa: E402

from pt_grid_spec import P_RANGE, T_RANGE, NP, NT                  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=r"D:/Actief/C23S1A_Quant/viewer_session_22_07_26.json")
    ap.add_argument("--run-index", type=int, default=3, help="session run holding the section's SPX bulk")
    args = ap.parse_args()
    t0 = time.time()
    s = json.load(open(args.session, encoding="utf-8"))
    run = {r["index"]: r for r in s["thermod_runs"]}[args.run_index]
    spx_el = {k: float(v) for k, v in run["comp_el"].items()}
    kept, dropped = model_bulk.split_elements(spx_el)
    oxides = model_bulk.model_oxides(spx_el)
    dry_all = sum(v for v in spx_el.values()) - spx_el.get("O", 0.0)
    print("elements passed:", {k: round(v, 3) for k, v in kept.items()})
    print("elements dropped:", {k: round(v, 3) for k, v in sorted(dropped.items(), key=lambda kv: -kv[1]) if v >= 0.001},
          f"({100 * sum(dropped.values()) / dry_all:.1f} % of the cation total)")
    print("oxides to MAGEMin (wt%):", {k: round(v, 3) for k, v in oxides.items()}, flush=True)

    bk = {"db": model_bulk.DATABASE, "fe3_fraction": model_bulk.FE3_FRACTION}
    calc = ThermoCalc("magemin", **bk)
    results = calc.run_grid(model_bulk.model_composition(spx_el), P_RANGE, T_RANGE, n_p=NP, n_t=NT)
    pd.DataFrame([r.to_dict() for r in results]).to_csv(HERE / "pt_grid.csv", index=False)
    json.dump({"source": f"session run #{args.run_index} (whole-section SPX quantification)",
               "spx_bulk_el": spx_el, "elements_passed": kept, "elements_dropped": dropped,
               "oxides_wt_pct": oxides, "fe3_fraction": model_bulk.FE3_FRACTION,
               "h2o_wt_pct_of_bulk": model_bulk.H2O_WT_PCT, "backend": bk,
               "p_range_GPa": P_RANGE, "t_range_C": T_RANGE, "n_p": NP, "n_t": NT},
              open(HERE / "pt_grid_bulk.json", "w"), indent=1)
    print(f"wrote pt_grid.csv ({len(results)} cells) and pt_grid_bulk.json in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
