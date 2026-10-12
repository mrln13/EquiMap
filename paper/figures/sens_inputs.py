# -*- coding: utf-8 -*-
"""Sensitivity of the Fig-5 estimate to the two model inputs the user sets:
H2O (2, 4, 8 wt% of the bulk) and Fe3+/sum(Fe) (0, 0.1, 0.2, 0.3), on the
section's SPX bulk, scored with the adopted modes. Output: sens_inputs.json (the input rows of Table A.2)."""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
FIG = REPO / "_Manuscript" / "figures"
OUT = FIG
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(FIG))


def main():
    import fig_pt
    import model_bulk
    from thermo.parallel import iter_grid_results
    from thermo.result import result_from_portable_dict
    from viewer_thermo_helpers import _modal_misfit_grid, _phase_matches, _is_fluid_phase

    P_RANGE, T_RANGE, NP, NT = (0.1, 1.5), (300.0, 800.0), 20, 20
    el = json.load(open(FIG / "pt_grid_bulk.json"))["spx_bulk_el"]
    cases = [("Fe3+ 0.0, H2O 4", 0.0, 4.0), ("Fe3+ 0.1, H2O 4 (adopted)", 0.1, 4.0), ("Fe3+ 0.2, H2O 4", 0.2, 4.0),
             ("Fe3+ 0.3, H2O 4", 0.3, 4.0), ("Fe3+ 0.1, H2O 2", 0.1, 2.0), ("Fe3+ 0.1, H2O 8", 0.1, 8.0)]
    modes, weights = fig_pt.observed_from_exports()
    p_ax = np.linspace(*P_RANGE, NP); t_ax = np.linspace(*T_RANGE, NT)
    rows = []
    # one backend per ferric fraction (it is a backend argument)
    for fe3 in sorted({c[1] for c in cases}):
        sub = [c for c in cases if c[1] == fe3]
        jobs = [{"comp": model_bulk.model_composition(el, fe3=fe3, h2o=h2o), "p_range": P_RANGE, "t_range": T_RANGE,
                 "n_p": NP, "n_t": NT} for _, _, h2o in sub]
        res = {}
        for idx, dicts, err in iter_grid_results(jobs, "magemin", {"db": model_bulk.DATABASE, "fe3_fraction": fe3}, max_workers=2):
            if err is not None:
                print("FAILED", sub[idx][0], err, flush=True); continue
            res[idx] = [result_from_portable_dict(d) for d in dicts]
        for i, (tag, _, h2o) in enumerate(sub):
            if i not in res:
                continue
            r = res[i]
            mis = np.asarray(_modal_misfit_grid(r, modes, NP, NT, weights=weights, presence_keys=fig_pt.PRESENCE)[0], float)
            j = int(np.nanargmin(mis)); ip, it = divmod(j, NT)
            acc = mis <= np.nanmin(mis) + fig_pt.REGION_DELTA; pp, tt = np.where(acc)
            fl = np.array([sum((ph.vol_fraction or 0) for ph in x.stable_phases if _is_fluid_phase(ph.name)) for x in r]).reshape(NP, NT)
            melt = np.array([sum((ph.vol_fraction or 0) for ph in x.stable_phases if ph.is_melt) for x in r]).reshape(NP, NT)
            stab = np.array([all(any(_phase_matches(k, ph.name) for ph in x.stable_phases) for k in list(modes) + list(fig_pt.PRESENCE))
                             for x in r]).reshape(NP, NT)
            sp, st = np.where(stab)
            row = dict(case=tag, fe3=fe3, h2o=h2o, best_T=float(t_ax[it]), best_P=float(p_ax[ip]), misfit=float(mis[ip, it]),
                       T_lo=float(t_ax[tt].min()), T_hi=float(t_ax[tt].max()), P_lo=float(p_ax[pp].min()), P_hi=float(p_ax[pp].max()),
                       n_acc=int(acc.sum()), n_stab=int(stab.sum()),
                       stab_T=(float(t_ax[st].min()), float(t_ax[st].max())) if stab.any() else None,
                       stab_P=(float(p_ax[sp].min()), float(p_ax[sp].max())) if stab.any() else None,
                       acc_with_fluid=int((acc & (fl > 0)).sum()), acc_with_melt=int((acc & (melt > 0)).sum()),
                       stab_with_fluid=int((stab & (fl > 0)).sum()), fluid_cells=int((fl > 0).sum()))
            rows.append(row)
            print(f"{tag:28s} best {row['best_T']:.0f}/{row['best_P']:.2f} ({row['misfit']:.3f}); region {row['T_lo']:.0f}-{row['T_hi']:.0f} / "
                  f"{row['P_lo']:.2f}-{row['P_hi']:.2f} ({row['n_acc']} cells, {row['acc_with_fluid']} with fluid, {row['acc_with_melt']} with melt); "
                  f"stability field {row['n_stab']} cells {row['stab_T']} {row['stab_P']}; fluid in {row['fluid_cells']}/400", flush=True)
    json.dump(rows, open(OUT / "sens_inputs.json", "w"), indent=1)


if __name__ == "__main__":
    main()
