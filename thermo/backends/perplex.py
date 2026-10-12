# thermo/backends/perplex.py
"""
Perple_X backend — drives the BUILD → VERTEX → WERAMI pipeline headlessly.

Unlike the meemum approach, this backend computes a full P-T pseudosection
grid and caches the VERTEX solution so that multiple WERAMI queries over the
same grid are cheap.  Single-point ``run()`` is implemented as a 1×1 grid.

For a fresh start on a composition, VERTEX must run once.  Subsequent calls
with the same composition (matched by SHA-1) reuse the cached solution.

Perple_X executables must be on PATH or the ``perplex_dir`` supplied.

Install:  https://www.perplex.ethz.ch/
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from thermo.backends.base import AbstractBackend
from thermo.composition import Composition
from thermo.result import PhaseResult, ThermoResult

# ---------------------------------------------------------------------------
# Perple_X dataset descriptions
# ---------------------------------------------------------------------------
DATASET_DESCRIPTIONS = {
    "hp02ver.dat":  "Holland & Powell (2002)",
    "hp633ver.dat": "Holland & Powell (2011, ds633)",
    "hp62ver.dat":  "Holland & Powell (2011, ds62)",
    "hp11ver.dat":  "Holland & Powell (2011, ds6x)",
    "stx21ver.dat": "Stixrude & Lithgow-Bertelloni (2021)",
    "solution_model.dat": "Standard solution model file",
}

# ---------------------------------------------------------------------------
# Valid oxide components per dataset
# ---------------------------------------------------------------------------
# Invalid oxide names in the BUILD dialog cause "Try again:" prompts that
# consume subsequent stdin tokens, shifting all later dialog entries and
# ultimately producing a truncated/invalid proj.dat.  Pre-filter the
# composition to only include components present in the chosen dataset.
DATASET_VALID_OXIDES: Dict[str, frozenset] = {
    "hp633ver.dat": frozenset({
        "SiO2", "Al2O3", "FeO", "MgO", "CaO", "Na2O", "K2O", "TiO2",
        "MnO", "NiO", "ZrO2", "Cr2O3", "CuO", "H2O", "CO2", "O2",
        "Cl2", "S2", "F2", "N2",
    }),
    "hp62ver.dat": frozenset({
        "SiO2", "Al2O3", "FeO", "MgO", "CaO", "Na2O", "K2O", "TiO2",
        "MnO", "NiO", "ZrO2", "Cr2O3", "CuO", "H2O", "CO2", "O2",
        "Cl2", "S2", "F2", "N2",
    }),
    "hp11ver.dat": frozenset({
        "SiO2", "Al2O3", "FeO", "MgO", "CaO", "Na2O", "K2O", "TiO2",
        "MnO", "NiO", "ZrO2", "Cr2O3", "CuO", "H2O", "CO2", "O2",
        "Cl2", "S2", "F2", "N2",
    }),
    "hp02ver.dat": frozenset({
        "SiO2", "Al2O3", "FeO", "MgO", "CaO", "Na2O", "K2O", "TiO2",
        "MnO", "NiO", "Cr2O3", "H2O", "CO2", "O2",
    }),
    "stx21ver.dat": frozenset({
        "SiO2", "MgO", "FeO", "Al2O3", "CaO", "Na2O", "K2O", "H2O", "CO2",
    }),
}
_DEFAULT_VALID_OXIDES: frozenset = frozenset()


class PerplexBackend(AbstractBackend):
    """
    Thermodynamic equilibrium via Perple_X (BUILD → VERTEX → WERAMI).

    Parameters
    ----------
    perplex_dir :
        Directory containing Perple_X executables (``build``, ``vertex``,
        ``werami``) and data files.  Falls back to PATH if not given.
    dataset :
        Thermodynamic dataset filename (e.g. ``"hp633ver.dat"``).
    solution_model :
        Solution-model filename (e.g. ``"solution_model.dat"``).
    outdir :
        Directory where Perple_X runs are cached.  Each unique composition
        gets a sub-directory named by SHA-1 hash.
    p_range, t_range :
        Default P [GPa] and T [°C] ranges for grid calculations.
    n_p, n_t :
        Default grid resolution.
    fe3_fraction :
        Fe³⁺ / Fe_total for BUILD composition.
    fluid :
        Fluid model: ``"none"``, ``"H2O-sat"``, ``"H2O-CO2"``.
    """

    name = "perplex"

    def __init__(
        self,
        perplex_dir: Optional[str | Path] = r"C:/perplex",
        dataset: str = "hp633ver.dat",
        solution_model: str = "solution_model.dat",
        outdir: str | Path = "./perplex_runs",
        p_range: Tuple[float, float] = (0.2, 1.5),
        t_range: Tuple[float, float] = (400.0, 800.0),
        n_p: int = 51,
        n_t: int = 51,
        fe3_fraction: float = 0.1,
        fluid: str = "none",
        solution_phases: Optional[Sequence[str]] = None,
    ):
        self.perplex_dir = Path(perplex_dir).resolve() if perplex_dir else None
        self.dataset = dataset
        self.solution_model = solution_model
        self.outdir = Path(outdir).resolve()
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.default_p_range = p_range
        self.default_t_range = t_range
        self.default_n_p = n_p
        self.default_n_t = n_t
        self.fe3_fraction = float(fe3_fraction)
        self.fluid = fluid
        # Default solution phases for hp633ver.dat.
        # Note: Chl(W) and cAmph(G) are NOT in the standard Perple_X 7.x solution_model.dat.
        # Users needing chlorite/amphibole should supply a solution_model.dat that includes them.
        self.solution_phases: List[str] = list(solution_phases) if solution_phases is not None else [
            "Cpx(JH)", "Opx(JH)", "Pl(JH)", "Gt(W)", "IlHm(A)", "melt(G)",
        ]

    # ---- Public API ---------------------------------------------------------

    def run(
        self,
        composition: Composition,
        p_gpa: float,
        t_c: float,
        **kwargs,
    ) -> ThermoResult:
        """
        Single-point calculation.

        Implemented as a 1×1 grid (BUILD+VERTEX once, then WERAMI).
        The solution is cached so repeated calls with the same composition
        at different P-T are cheap.
        """
        results = self.run_grid(
            composition,
            p_range=(p_gpa, p_gpa),
            t_range=(t_c, t_c),
            n_p=1, n_t=1,
        )
        return results[0] if results else ThermoResult(
            p_gpa=p_gpa, t_c=t_c, stable_phases=[],
            backend=self.name, converged=False,
        )

    def run_grid(
        self,
        composition: Composition,
        p_range: Optional[Tuple[float, float]] = None,
        t_range: Optional[Tuple[float, float]] = None,
        n_p: Optional[int] = None,
        n_t: Optional[int] = None,
        fe3_fraction: Optional[float] = None,
    ) -> List[ThermoResult]:
        """
        Run a full P-T pseudosection via BUILD → VERTEX → WERAMI.

        The VERTEX solution is cached by composition hash.  Only WERAMI
        reruns if the requested property set changes.
        """
        pr = p_range or self.default_p_range
        tr = t_range or self.default_t_range
        np_ = n_p or self.default_n_p
        nt_ = n_t or self.default_n_t
        f3 = self.fe3_fraction if fe3_fraction is None else float(fe3_fraction)

        run_dir = self._run_dir(composition, pr, tr, np_, nt_, f3)
        run_dir.mkdir(parents=True, exist_ok=True)

        n_pts = np_ * nt_
        print(
            f"[perplex] Computing {n_pts} P-T points ({np_}×{nt_}) "
            f"with dataset '{self.dataset}'…",
            flush=True,
        )

        self._run_build(composition, pr, tr, np_, nt_, f3, run_dir)
        self._run_vertex(run_dir)
        tab_path = self._run_werami(run_dir, np_, nt_)
        phases_tab = self._run_werami_phases(run_dir, np_, nt_)

        results = self._parse_werami(tab_path, pr, tr, np_, nt_)
        self._merge_phases(results, phases_tab)

        # Guard against a silent empty pseudosection. If VERTEX stabilised no
        # phase anywhere on the grid, every cell has an empty assemblage and the
        # viewer would render a single blank "field" with no explanation. That
        # almost always means the bulk composition is non-physical — most often
        # uncalibrated EDS net-count intensities passed as element wt% (raw
        # counts are not proportional to mass, so e.g. SiO2 comes out at a few
        # wt% → silica-undersaturated nonsense → no stable assemblage). Fail
        # loudly with the offending bulk so the cause is obvious.
        if results and not any(r.stable_phases for r in results):
            ox = composition.to_oxide_wt(fe3_fraction=f3)
            bulk = ", ".join(f"{k}={v:.2f}" for k, v in sorted(ox.items()) if v > 0)
            raise RuntimeError(
                "Perple_X VERTEX stabilised no phase anywhere on the P-T grid — "
                "the pseudosection is empty. This usually means the bulk "
                "composition is non-physical. A common cause is uncalibrated EDS "
                "net-count intensities used as wt% (silica is badly "
                "under-represented); use SPX-quantified or otherwise calibrated "
                f"wt% instead. Bulk passed (oxide wt%): {bulk}"
            )
        print(f"[perplex] Done ({n_pts} points).", flush=True)
        return results

    def available(self) -> bool:
        try:
            self._resolve_exe("build")
            return True
        except (SystemExit, FileNotFoundError):
            return False

    # ---- Internal -----------------------------------------------------------

    def _run_dir(self, composition, pr, tr, np_, nt_, f3) -> Path:
        key = {
            "composition": composition.to_oxide_wt(fe3_fraction=f3),
            "dataset": self.dataset, "solution": self.solution_model,
            "p_range": pr, "t_range": tr, "n_p": np_, "n_t": nt_,
        }
        h = hashlib.sha1(
            json.dumps(key, sort_keys=True).encode()
        ).hexdigest()[:12]
        return self.outdir / h

    def _resolve_exe(self, name: str) -> str:
        import shutil
        candidates = [name, name.upper(), name + ".exe", name.upper() + ".exe"]
        if self.perplex_dir:
            for sub in ("", "bin", "bin64"):
                root = self.perplex_dir / sub if sub else self.perplex_dir
                for c in candidates:
                    p = root / c
                    if p.exists():
                        return str(p)
        for c in candidates:
            w = shutil.which(c)
            if w:
                return w
        raise FileNotFoundError(
            f"Perple_X executable '{name}' not found. "
            "Set perplex_dir or add Perple_X to PATH."
        )

    def _resolve_data_file(self, filename: str) -> str:
        p = Path(filename)
        if p.exists():
            return str(p.resolve())
        if self.perplex_dir:
            for sub in ("", "data", "datafiles"):
                cand = (self.perplex_dir / sub / p.name) if sub \
                    else (self.perplex_dir / p.name)
                if cand.exists():
                    return str(cand.resolve())
        raise FileNotFoundError(
            f"Perple_X data file '{filename}' not found. "
            "Set perplex_dir or provide an absolute path."
        )

    def _resolve_option_file(self, filename: str) -> str:
        p = Path(filename)
        if p.exists():
            return str(p.resolve())
        if self.perplex_dir:
            for sub in ("", "optionfiles", "datafiles"):
                cand = (self.perplex_dir / sub / p.name) if sub \
                    else (self.perplex_dir / p.name)
                if cand.exists():
                    return str(cand.resolve())
        raise FileNotFoundError(
            f"Perple_X option file '{filename}' not found in {self.perplex_dir}."
        )

    def _run_proc(
        self,
        exe: str,
        stdin_text: Optional[str],
        cwd: Path,
        log: Path,
        stream: bool = False,
    ):
        if stream:
            # Stream stdout live (so the user sees VERTEX's own progress lines)
            # while also writing everything to the log file.
            proc = subprocess.Popen(
                [exe],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(cwd),
            )
            captured: list[str] = []
            if stdin_text:
                proc.stdin.write(stdin_text)
                proc.stdin.close()
            for line in proc.stdout:
                print(f"  [perplex] {line}", end="", flush=True)
                captured.append(line)
            proc.wait()
            log.write_text("".join(captured), encoding="utf-8")
            if proc.returncode != 0:
                raise RuntimeError(
                    f"Perple_X '{exe}' failed (code {proc.returncode}). "
                    f"See log: {log}"
                )
        else:
            result = subprocess.run(
                [exe],
                input=stdin_text,
                text=True,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            log.write_text(result.stdout or "", encoding="utf-8")
            if result.returncode != 0:
                raise RuntimeError(
                    f"Perple_X '{exe}' failed (code {result.returncode}). "
                    f"See log: {log}"
                )

    def _run_build(self, composition, pr, tr, np_, nt_, f3, run_dir: Path):
        dat_file = run_dir / "proj.dat"
        # A successful BUILD writes ~2-4 KB. Use 1000 bytes as the threshold.
        if dat_file.exists() and dat_file.stat().st_size > 1000:
            print("[perplex] BUILD: using cached project.", flush=True)
            return  # already built

        print("[perplex] BUILD: setting up project…", flush=True)

        # Delete any leftover proj.* files from a previous failed or killed attempt.
        # If proj.dat already exists when BUILD starts, Perple_X asks "overwrite it
        # (y/n)?" — an extra prompt not in our dialog sequence — shifting all
        # subsequent tokens and corrupting the entire dialog.
        for f in run_dir.glob("proj.*"):
            try:
                f.unlink()
            except OSError:
                pass

        import shutil

        # Use fe3_fraction=0 — Perple_X hp*ver.dat uses FeO (not Fe2O3) as component
        ox_wt = composition.to_oxide_wt(fe3_fraction=0.0)
        ox_wt = {k: v for k, v in ox_wt.items() if v > 0.0}

        # Filter to components valid for the selected dataset.
        # Invalid oxide names cause "Try again:" prompts in the BUILD dialog that
        # consume subsequent stdin tokens, shifting all later dialog entries and
        # producing a truncated proj.dat → VERTEX EOF error.
        valid = DATASET_VALID_OXIDES.get(self.dataset, _DEFAULT_VALID_OXIDES)
        if valid:
            removed = {k for k in ox_wt if k not in valid}
            if removed:
                print(
                    f"[perplex] Removed {len(removed)} oxide(s) not in {self.dataset}: "
                    f"{', '.join(sorted(removed))}. These elements will not be included.",
                    flush=True,
                )
            ox_wt = {k: v for k, v in ox_wt.items() if k in valid}

        if not ox_wt:
            raise ValueError(
                f"No valid components remain after filtering for dataset '{self.dataset}'."
            )

        oxides = sorted(ox_wt.keys())

        dataset_path = self._resolve_data_file(self.dataset)
        solution_path = self._resolve_data_file(self.solution_model)

        # Write the option file. x_nodes/y_nodes sets the VERTEX output grid:
        # WERAMI will produce exactly n_t × n_p data points, which must match
        # opts['n_t'] and opts['n_p'] so the viewer can reshape correctly.
        # BUILD internally enforces a minimum of 20×20 regardless of this value,
        # so setting small values here does not slow BUILD down.
        _xn = nt_
        _yn = np_
        (run_dir / "perplex_option.dat").write_text(
            "auto_refine off\n"
            f"x_nodes {_xn} {_xn}\n"
            f"y_nodes {_yn} {_yn}\n"
            "grid_levels 1 1\n"
        )

        # P range: GPa → bar; T range: °C → K
        pmin_bar = pr[0] * 10000.0
        pmax_bar = pr[1] * 10000.0
        tmin_K = tr[0] + 273.15
        tmax_K = tr[1] + 273.15

        # Ensure min != max for single-point runs
        if pmin_bar == pmax_bar:
            pmax_bar += 100.0
        if tmin_K == tmax_K:
            tmax_K += 1.0

        # Saturated fluid setting.
        # When sat_fluid="y", BUILD asks for the fluid component names (one per
        # line, blank to finish) before continuing.  Those components must also
        # be removed from the regular thermodynamic component list because
        # Perple_X does not list them again in the component-selection prompt.
        fluid_lower = self.fluid.lower()
        if fluid_lower == "h2o-sat":
            sat_fluid = "y"
            sat_fluid_components = ["H2O"]
        elif fluid_lower == "h2o-co2":
            sat_fluid = "y"
            sat_fluid_components = ["H2O", "CO2"]
        else:
            sat_fluid = "n"
            sat_fluid_components = []

        # Remove saturated-fluid components from the regular oxide list so they
        # are not sent again in the thermodynamic-component section.
        sat_set = set(sat_fluid_components)
        oxides = [o for o in oxides if o not in sat_set]
        comp_line = " ".join(f"{ox_wt[ox]:.4f}" for ox in oxides)

        # Extra tokens inserted after sat_fluid="y":
        #   component names (one per line) + blank line (end list).
        # NOTE: The EoS prompt ("Select the EoS...") appears AFTER the thermodynamic
        # component list, NOT immediately after the fluid component list.  "5" must
        # therefore be inserted after the end-of-component-list blank line below.
        sat_fluid_tokens = [*sat_fluid_components, ""] if sat_fluid_components else []

        # BUILD interactive dialog for Perple_X 7.x
        answers = "\n".join([
            "proj",                           # project name
            dataset_path,                     # thermodynamic dataset
            "",                               # option file: accept default
            "n",                              # transform components?
            "2",                              # mode: constrained minimization on 2d grid
            sat_fluid,                        # saturated fluid? (y/n)
            *sat_fluid_tokens,                # [fluid component names + ""] if y
            "n",                              # saturated components?
            "n",                              # chemical potentials?
            *oxides,                          # component names (one per line)
            "",                               # end component list
            # EoS for saturated fluid: appears HERE, after thermodynamic component
            # selection.  "5" = Holland-Powell 1998 H2O-CO2 (correct for H2O-sat and
            # H2O-CO2).  Omitted when no saturated fluid is used.
            *(["5"] if sat_fluid_components else []),
            "n",                              # make potential dependent (geothermal gradient)?
            "2",                              # x-axis: T(K)
            f"{tmin_K:.2f}",                  # T min in K
            f"{tmax_K:.2f}",                  # T max in K
            # When a saturated fluid is active, Y(CO2) becomes a potential axis
            # variable, so BUILD explicitly asks for the y-axis selection.
            # No-fluid runs auto-assign P(bar) as y-axis without this prompt.
            *(["2"] if sat_fluid_components else []),  # y-axis: P(bar) — fluid runs only
            f"{pmin_bar:.2f}",                # P min in bar
            f"{pmax_bar:.2f}",                # P max in bar
            # When a saturated fluid is active, BUILD asks for a fixed (sectioning)
            # value for Y(CO2) because it is not one of the two P-T axes.
            # For H2O-sat (pure water), Y(CO2) = 0.  Y=0 is within [0,1] so
            # no "Specify a different value?" warning prompt is issued.
            *(["0"] if sat_fluid_components else []),  # Y(CO2) sectioning = 0 (pure H2O)
            "y",                              # specify composition by mass (wt%)?
            comp_line,                        # bulk composition (all on one line)
            "n",                              # output print file?
            "n",                              # exclude pure/endmember phases?
            "y",                              # include solution models?
            solution_path,                    # solution model file
            *self.solution_phases,            # solution phase names (one per line)
            "",                               # end phase list
            "",                               # calculation title (blank)
            "",                               # trailing newline
        ])

        build_exe = self._resolve_exe("build")
        build_log = run_dir / "build.log"
        self._run_proc(build_exe, answers, run_dir, build_log)

        # Perple_X BUILD sometimes exits with returncode 0 even on failure (e.g.
        # "O.K., I quit too." when the dialog is misaligned). Verify that a
        # complete proj.dat was actually written.
        if not dat_file.exists() or dat_file.stat().st_size <= 1000:
            log_tail = ""
            if build_log.exists():
                lines = build_log.read_text(encoding="utf-8", errors="ignore").splitlines()
                log_tail = "\n".join(lines[-20:])
            raise RuntimeError(
                f"Perple_X BUILD appeared to succeed (returncode 0) but produced "
                f"an incomplete proj.dat ({dat_file.stat().st_size if dat_file.exists() else 0} bytes). "
                f"Last 20 lines of build.log:\n{log_tail}"
            )

    def _run_vertex(self, run_dir: Path):
        # Check for a non-empty plt (VERTEX output) — BUILD creates empty placeholder files,
        # so we must also verify the file is non-empty before skipping.
        plt = run_dir / "proj.plt"
        if plt.exists() and plt.stat().st_size > 0:
            print("[perplex] VERTEX: using cached solution.", flush=True)
            return  # already solved
        print("[perplex] VERTEX: computing pseudosection (this may take a few minutes)…", flush=True)
        vertex_exe = self._resolve_exe("vertex")
        self._run_proc(vertex_exe, "proj\n", run_dir, run_dir / "vertex.log", stream=True)
        print("[perplex] VERTEX: done.", flush=True)

    def _run_werami(self, run_dir: Path, n_p: int, n_t: int) -> Path:
        tab_path = run_dir / "werami.tab"
        if tab_path.exists():
            print("[perplex] WERAMI (properties): using cached output.", flush=True)
            return tab_path
        print("[perplex] WERAMI: extracting bulk properties…", flush=True)

        # Pre-delete any leftover proj_*.tab so we can identify the new output.
        for f in run_dir.glob("proj_*.tab"):
            f.unlink(missing_ok=True)

        # WERAMI 7.x interactive dialog: 2d grid, density
        query = "\n".join([
            "proj",    # project name
            "2",       # mode: properties on 2d grid
            "2",       # property: Density (kg/m3)
            "n",       # calculate individual phase properties?
            "n",       # include fluid in aggregate?
            "0",       # no more properties
            "n",       # change default variable range?
            "1",       # grid resolution: use default (first listed option)
            "0",       # exit WERAMI
            "",
        ])
        werami_exe = self._resolve_exe("werami")
        self._run_proc(werami_exe, query, run_dir, run_dir / "werami.log")

        # Rename whichever proj_*.tab appeared.
        for f in sorted(run_dir.glob("proj_*.tab")):
            f.rename(tab_path)
            break
        return tab_path

    def _run_werami_phases(self, run_dir: Path, n_p: int, n_t: int) -> Optional[Path]:
        tab_path = run_dir / "werami_phases.tab"
        if tab_path.exists():
            print("[perplex] WERAMI (phases): using cached output.", flush=True)
            return tab_path
        print("[perplex] WERAMI: extracting phase modes…", flush=True)

        # Pre-delete any leftover proj_*.tab so we can identify the new output.
        for f in run_dir.glob("proj_*.tab"):
            f.unlink(missing_ok=True)

        # WERAMI 7.x dialog: 2d grid, property 25 = Modes of all phases.
        # Unlike scalar properties (e.g. Density), property 25 produces a full
        # multi-column table in one shot and does NOT prompt
        # "Select an additional property or enter 0 to finish:". Its two extra
        # prompts come *before* that point:
        #   1. Output cumulative modes? → n
        #   2. Include fluid in computation of aggregate properties? → n
        # then it goes straight to "Change default variable range?". Sending a
        # spurious "0" here (copied from the scalar-property flow) shifted every
        # later token by one: "0" was eaten by the range prompt, "n" hit the
        # numeric grid-resolution prompt ("try again"), and for some
        # compositions the cascade corrupted the run so property 25 never
        # registered — yielding a phase tab with only T,P columns (no modes),
        # which renders as a single empty assemblage field. Do NOT add it back.
        query = "\n".join([
            "proj",    # project name
            "2",       # mode: properties on 2d grid
            "25",      # property: Modes of all phases
            "n",       # output cumulative modes?
            "n",       # include fluid in aggregate properties?
            "n",       # change default variable range?
            "1",       # grid resolution: use default
            "0",       # exit WERAMI
            "",
        ])
        werami_exe = self._resolve_exe("werami")
        self._run_proc(werami_exe, query, run_dir, run_dir / "werami_phases.log")

        for f in sorted(run_dir.glob("proj_*.tab")):
            f.rename(tab_path)
            break
        return tab_path if tab_path.exists() else None

    def _merge_phases(self, results: List[ThermoResult], tab_path: Optional[Path]) -> None:
        """Parse werami_phases.tab and attach PhaseResult objects to each ThermoResult."""
        if tab_path is None or not tab_path.exists():
            return

        lines = tab_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if not lines or not lines[0].startswith("|"):
            return

        # Parse Perple_X 7.x tab header.
        try:
            idx = 2  # skip version tag + filename
            n_vars = int(lines[idx].strip()); idx += 1
            for _ in range(n_vars):
                idx += 4  # skip var name, min, step, count
            n_cols = int(lines[idx].strip()); idx += 1
            headers = lines[idx].split(); idx += 1
        except (ValueError, IndexError):
            warnings.warn("[perplex] Could not parse werami_phases.tab header")
            return

        phase_names = headers[2:]  # first two cols are T(K) and P(bar)
        if not phase_names:
            # The tab has only the T(K)/P(bar) coordinate columns and no phase
            # mode columns — WERAMI's "Modes of all phases" query produced no
            # output. This means the dialog desynchronised (see _run_werami_phases)
            # or VERTEX produced no stable assemblage. Warn instead of silently
            # attaching zero phases (which renders as one empty phase field).
            warnings.warn(
                f"[perplex] werami_phases.tab has no phase columns "
                f"({tab_path}); the pseudosection will have no assemblage data. "
                "Check the WERAMI dialog / VERTEX solution."
            )
            return
        melt_keywords = {"melt", "liq", "fluid", "fl"}

        # Build a dict (T_K_rounded, P_bar_rounded) → phases for fast lookup.
        phase_map: Dict[Tuple[int, int], List] = {}
        for line in lines[idx:]:
            parts = line.split()
            if len(parts) != n_cols:
                continue
            try:
                t_k = float(parts[0])
                p_bar = float(parts[1])
                key = (round(t_k, 2), round(p_bar, 2))
                phases = []
                for name, val_str in zip(phase_names, parts[2:]):
                    if val_str.lower() == "nan":
                        continue
                    vol_frac = float(val_str) / 100.0
                    if vol_frac <= 0:
                        continue
                    is_melt = any(kw in name.lower() for kw in melt_keywords)
                    phases.append(PhaseResult(
                        name=name,
                        mol_fraction=vol_frac,   # best approximation available
                        vol_fraction=vol_frac,
                        is_melt=is_melt,
                    ))
                phase_map[key] = phases
            except (ValueError, IndexError):
                continue

        # Attach to ThermoResult objects matched by (T, P).
        for r in results:
            t_k = round(r.t_c + 273.15, 2)
            p_bar = round(r.p_gpa * 10000.0, 2)
            r.stable_phases = phase_map.get((t_k, p_bar), [])

    def _parse_werami(
        self, tab_path: Path, pr, tr, n_p: int, n_t: int
    ) -> List[ThermoResult]:
        """Parse WERAMI tabular output (Perple_X 7.x format) into ThermoResult list."""
        if not tab_path.exists():
            warnings.warn(f"[perplex] WERAMI output not found: {tab_path}")
            return []

        lines = tab_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if not lines:
            return []

        # Perple_X 7.x tab format starts with |6.x.x version tag
        # Header structure:
        #   |6.6.6
        #   <filename>
        #   <n_independent_vars>
        #   for each var: name, min, step, count  (4 lines each)
        #   <n_columns>
        #   <space-separated column headers>
        #   <data rows>
        if lines[0].startswith("|"):
            idx = 2  # skip version tag + filename
            try:
                n_vars = int(lines[idx].strip())
                idx += 1
                col_names = []
                for _ in range(n_vars):
                    idx += 4  # skip var name, min, step, count
                n_cols = int(lines[idx].strip())
                idx += 1
                col_names = lines[idx].split()
                idx += 1
            except (ValueError, IndexError) as exc:
                warnings.warn(f"[perplex] Failed to parse WERAMI 7.x header: {exc}")
                return []

            data_rows = []
            bad_value = float('nan')
            for line in lines[idx:]:
                parts = line.split()
                if len(parts) == n_cols:
                    try:
                        row = [float(x) if x.lower() != 'nan' else bad_value
                               for x in parts]
                        data_rows.append(row)
                    except ValueError:
                        continue

            results = []
            for row in data_rows:
                t_k = row[0]
                p_bar = row[1]
                t_c = t_k - 273.15
                p_gpa = p_bar / 10000.0
                props: Dict[str, float] = {}
                if len(row) > 2:
                    val = row[2]
                    if val == val:  # not NaN
                        props['rho_kgm3'] = val  # Density already in kg/m3
                results.append(ThermoResult(
                    p_gpa=p_gpa, t_c=t_c,
                    stable_phases=[],
                    bulk_properties=props,
                    backend=self.name,
                    converged=True,
                ))
            return results

        # Fallback: old 6.x format (no version tag, data after # comments)
        data_lines = [l for l in lines if not l.strip().startswith("#") and l.strip()]
        if not data_lines:
            return []

        from io import StringIO
        try:
            df = pd.read_csv(
                StringIO("\n".join(data_lines)),
                sep=r"\s+", header=None, engine="python"
            )
        except Exception as exc:
            warnings.warn(f"[perplex] Failed to parse WERAMI output: {exc}")
            return []

        results = []
        p_vals = np.linspace(pr[0], pr[1], n_p)
        t_vals = np.linspace(tr[0], tr[1], n_t)
        for i, (p, t) in enumerate(
            [(float(p), float(t)) for p in p_vals for t in t_vals]
        ):
            props: Dict[str, float] = {}
            if i < len(df):
                row = df.iloc[i]
                if len(row) > 2:
                    props['rho_kgm3'] = float(row.iloc[2]) * 1000.0  # g/cc → kg/m3
            results.append(ThermoResult(
                p_gpa=p, t_c=t,
                stable_phases=[],
                bulk_properties=props,
                backend=self.name,
                converged=True,
            ))
        return results

    def __repr__(self) -> str:
        return (f"PerplexBackend(dataset='{self.dataset}', "
                f"perplex_dir='{self.perplex_dir}')")
