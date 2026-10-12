#!/usr/bin/env python3
"""
perplex_run.py — headless Perple_X runner with --perplex-dir support.

New:
  • --perplex-dir (or PERPLEX_DIR env) to locate executables and datafiles.
    - Executables searched in: <dir>/bin, <dir>, then PATH.
    - Dataset/Solution files searched in: <dir>/data, <dir>, then as given.

Usage (example):
  python perplex_run.py \
    --perplex-dir /opt/perplex \
    --bulk bulk_oxides.json \
    --dataset hp02ver.dat \
    --solutions solution_model.dat \
    --pt 0.2 1.2 300 800 \
    --grid 101 101 \
    --project eds_test \
    --outdir ./perplex_runs \
    --props "P,T,rho,Vp,Vs" \
    --verbose
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# ------------------------------ Configuration ------------------------------

@dataclass(frozen=True)
class RunConfig:
    bulk_oxides: Dict[str, float]
    dataset_path: str
    solutions_path: str
    p_min: float
    p_max: float
    t_min: float
    t_max: float
    n_p: int
    n_t: int
    project_name: str
    fluid_spec: str = "none"
    build_exe: str = "build"
    vertex_exe: str = "vertex"
    werami_exe: str = "werami"
    props: Tuple[str, ...] = ("P", "T", "rho")
    extra_build_flags: Tuple[str, ...] = tuple()
    perplex_dir: str | None = None  # NEW

WERAMI_PROPERTY_MAP = {
    "P":   ("2", "1"),
    "T":   ("2", "2"),
    "rho": ("2", "501"),
    "Vp":  ("2", "941"),
    "Vs":  ("2", "942"),
}

# ------------------------------ Utilities ------------------------------

def sha1_of_dict(d: dict) -> str:
    import json as _json, hashlib as _hashlib
    msg = _json.dumps(d, sort_keys=True).encode("utf-8")
    return _hashlib.sha1(msg).hexdigest()[:12]

def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")

def resolve_executable(name_or_path: str, perplex_dir: Path | None) -> str:
    """
    Resolve Perple_X executable with lots of common variants:
      - exact path if it exists
      - <perplex_dir>/bin*/<candidate>, then <perplex_dir>/<candidate>
      - PATH lookup for each candidate
    Candidates include case/extension variants like BUILD, build.exe, etc.
    """
    import shutil as _shutil
    p = Path(name_or_path)

    # 1) If the user gave a path and it exists, use it
    if p.exists():
        return str(p.resolve())

    # 2) Build candidate names (case + extensions)
    base = p.name if p.name else name_or_path
    stem = base.replace(".exe", "").replace(".EXE", "")

    # canonical names for the three tools
    aliases = {
        "build":  ["build", "BUILD", "PerpleX-BUILD", "Perple_X-BUILD"],
        "vertex": ["vertex", "VERTEX", "PerpleX-VERTEX", "Perple_X-VERTEX"],
        "werami": ["werami", "WERAMI", "PerpleX-WERAMI", "Perple_X-WERAMI"],
    }
    key = "build" if "build" in stem.lower() else \
          "vertex" if "vertex" in stem.lower() else \
          "werami" if "werami" in stem.lower() else stem.lower()

    cands = []
    # If we recognize the tool, try known aliases; otherwise just the given stem
    names = aliases.get(key, [stem])
    # add Windows extensions variants
    for nm in names:
        cands.append(nm)
        cands.append(nm + ".exe")
        cands.append(nm + ".EXE")

    # 3) Look inside PERPLEX_DIR if provided
    if perplex_dir:
        for sub in ("bin", "bin64", "bin32", ""):
            root = perplex_dir / sub if sub else perplex_dir
            for nm in cands:
                cand = root / nm
                if cand.exists():
                    return str(cand.resolve())

    # 4) PATH lookup for each candidate
    for nm in cands:
        w = _shutil.which(nm)
        if w:
            return w

    # 5) Give a helpful error
    tried = []
    if perplex_dir:
        for sub in ("bin", "bin64", "bin32", ""):
            root = perplex_dir / sub if sub else perplex_dir
            for nm in cands:
                tried.append(str((root / nm).resolve()))
    tried.extend([f"[PATH]:{nm}" for nm in cands])
    raise SystemExit(
        "Executable not found for "
        f"'{name_or_path}'. Tried:\n  - " + "\n  - ".join(tried)
        + "\nHints: use --perplex-dir, or pass --build-exe/--vertex-exe/--werami-exe with the exact filename."
    )

def resolve_data_file(path_or_name: str, perplex_dir: Path | None) -> str:
    """
    Resolve dataset/solution files with candidates:
      1) As-given if exists (absolute or relative to CWD).
      2) <perplex_dir>/data/<name>
      3) <perplex_dir>/<name>
    """
    p = Path(path_or_name)
    if p.exists():
        return str(p.resolve())
    if perplex_dir:
        for cand in [perplex_dir / "data" / Path(path_or_name).name,
                     perplex_dir / Path(path_or_name).name]:
            if cand.exists():
                return str(cand.resolve())
    raise SystemExit(f"Data file not found: {path_or_name} (searched in PERPLEX_DIR/data and PERPLEX_DIR)")

def run_proc(exe: str, input_text: str | None, cwd: Path, log_path: Path) -> None:
    proc = subprocess.run(
        [exe],
        input=input_text,
        text=True,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False
    )
    write_text(log_path, proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"{exe} failed (code {proc.returncode}). See log: {log_path}")

def parse_tabular_output(tab_path: Path) -> pd.DataFrame:
    if not tab_path.exists():
        raise FileNotFoundError(f"WERAMI output not found: {tab_path}")
    lines = tab_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    # find first non-comment as data start
    data_start = 0
    for i, ln in enumerate(lines):
        if not ln.strip().startswith("#"):
            data_start = i
            break
    from io import StringIO
    df = pd.read_csv(StringIO("\n".join(lines[data_start:])), delim_whitespace=True, header=None)
    # try to grab last comment line as header
    for ln in reversed(lines[:data_start]):
        if ln.strip().startswith("#"):
            hdr = ln.lstrip("#").strip().split()
            if len(hdr) == df.shape[1]:
                df.columns = hdr
                break
    return df

# ------------------------------ BUILD answers ------------------------------

def build_answers(cfg: RunConfig) -> str:
    ox_list = sorted(cfg.bulk_oxides.items(), key=lambda kv: kv[0])
    oxide_names = [k for k, _ in ox_list]
    oxide_vals  = [v for _, v in ox_list]

    t = f"""\
{cfg.project_name}
{cfg.dataset_path}
{cfg.solutions_path}
1
1
{len(oxide_names)}
"""
    for name in oxide_names:
        t += f"{name}\n"
    t += "# Bulk (wt%)\n"
    for val in oxide_vals:
        t += f"{val:.6f}\n"

    if cfg.fluid_spec.lower() == "h2o-sat":
        t += "y\n1\n"
    elif cfg.fluid_spec.lower() in ("h2o-co2", "h2o-co2-grid"):
        t += "y\n2\n"
    else:
        t += "n\n"

    t += f"""{cfg.p_min:.6f}
{cfg.p_max:.6f}
{cfg.t_min:.2f}
{cfg.t_max:.2f}
{cfg.n_p}
{cfg.n_t}

"""
    if cfg.extra_build_flags:
        for line in cfg.extra_build_flags:
            t += f"{line}\n"
    return t

# ------------------------------ WERAMI query ------------------------------

def werami_query(props: Tuple[str, ...]) -> str:
    used = []
    for p in props:
        if p not in WERAMI_PROPERTY_MAP:
            raise SystemExit(f"Unknown property '{p}'. Update WERAMI_PROPERTY_MAP.")
        menu, code = WERAMI_PROPERTY_MAP[p]
        used.append((menu, code))
    lines: List[str] = []
    lines.append("2")  # grid tabulation
    lines.append("1")  # default computational results
    for _, code in used:
        lines.append(str(code))
    lines.append("0")
    lines.append("0")
    return "\n".join(lines) + "\n"

# ------------------------------ Runner ------------------------------

class PerplexRunner:
    def __init__(self, cfg: RunConfig, work_root: Path):
        self.cfg = cfg
        self.work_root = work_root

        perplex_dir = Path(cfg.perplex_dir).resolve() if cfg.perplex_dir else None

        # Resolve data files relative to perplex_dir if needed
        dataset_path = resolve_data_file(cfg.dataset_path, perplex_dir)
        solutions_path = resolve_data_file(cfg.solutions_path, perplex_dir)
        object.__setattr__(self.cfg, "dataset_path", dataset_path)
        object.__setattr__(self.cfg, "solutions_path", solutions_path)

        # Resolve executables
        self.build = resolve_executable(cfg.build_exe, perplex_dir)
        self.vertex = resolve_executable(cfg.vertex_exe, perplex_dir)
        self.werami = resolve_executable(cfg.werami_exe, perplex_dir)

        h = sha1_of_dict({"cfg": asdict(self.cfg), "bulk": self.cfg.bulk_oxides})
        self.run_dir = self.work_root / h
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def run(self, verbose: bool = False) -> Path:
        parsed = self.run_dir / "parsed.csv"
        if parsed.exists():
            if verbose:
                print(f"[cache] Using cached result at {parsed}")
            return parsed

        build_in = build_answers(self.cfg)
        write_text(self.run_dir / "build_answers.txt", build_in)
        if verbose: print("[run] BUILD…")
        run_proc(self.build, input_text=build_in, cwd=self.run_dir, log_path=self.run_dir / "build.log")

        if verbose: print("[run] VERTEX…")
        run_proc(self.vertex, input_text=None, cwd=self.run_dir, log_path=self.run_dir / "vertex.log")

        q = werami_query(self.cfg.props)
        write_text(self.run_dir / "werami_query.txt", q)
        if verbose: print("[run] WERAMI…")
        run_proc(self.werami, input_text=q, cwd=self.run_dir, log_path=self.run_dir / "werami.log")

        tab_path = self.run_dir / "tab.out"
        df = parse_tabular_output(tab_path)
        df.to_csv(parsed, index=False)
        if verbose: print(f"[ok] Parsed → {parsed}")
        return parsed

# ------------------------------ CLI ------------------------------

def load_bulk(bulk_json: str) -> Dict[str, float]:
    data = json.loads(Path(bulk_json).read_text(encoding="utf-8"))
    ox = data["oxides_wt_percent"] if "oxides_wt_percent" in data else data
    if not isinstance(ox, dict) or not ox:
        raise SystemExit("bulk JSON must contain an 'oxides_wt_percent' dict or be a dict itself.")
    s = sum(ox.values())
    if s <= 0:
        raise SystemExit("bulk composition sums to zero.")
    return {k: 100.0 * float(v) / s for k, v in ox.items()}

def main():
    ap = argparse.ArgumentParser(description="Headless Perple_X runner (BUILD→VERTEX→WERAMI) with --perplex-dir.")
    ap.add_argument("--bulk", required=True, help="bulk_oxides.json")
    ap.add_argument("--dataset", required=True, help="Thermodynamic dataset (file name or path)")
    ap.add_argument("--solutions", required=True, help="Solution model file (file name or path)")
    ap.add_argument("--pt", nargs=4, type=float, metavar=("P_MIN_GPa", "P_MAX_GPa", "T_MIN_C", "T_MAX_C"), required=True)
    ap.add_argument("--grid", nargs=2, type=int, metavar=("N_P", "N_T"), default=[101, 101])
    ap.add_argument("--project", default="proj")
    ap.add_argument("--outdir", default="./perplex_runs")
    ap.add_argument("--props", default="P,T,rho", help="Comma-separated list; keys in WERAMI_PROPERTY_MAP")
    ap.add_argument("--fluid", default="none", choices=["none","H2O-sat","H2O-CO2","H2O-CO2-grid"])
    ap.add_argument("--build-exe", default="build", help="Executable name or path")
    ap.add_argument("--vertex-exe", default="vertex", help="Executable name or path")
    ap.add_argument("--werami-exe", default="werami", help="Executable name or path")
    ap.add_argument("--perplex_dir", default=os.environ.get("PERPLEX_DIR", ""), help="Base dir for Perple_X (bin/, data/).")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    bulk = load_bulk(args.bulk)
    pmin, pmax, tmin, tmax = args.pt
    n_p, n_t = args.grid
    props = tuple(x.strip() for x in args.props.split(",") if x.strip())

    perplex_dir = Path(args.perplex_dir) if args.perplex_dir else None  # noqa: attribute name normalized below
    if isinstance(perplex_dir, Path):
        perplex_dir = perplex_dir.resolve()

    cfg = RunConfig(
        bulk_oxides=bulk,
        dataset_path=args.dataset,
        solutions_path=args.solutions,
        p_min=pmin, p_max=pmax, t_min=tmin, t_max=tmax,
        n_p=n_p, n_t=n_t,
        project_name=args.project,
        fluid_spec=args.fluid,
        build_exe=args.build_exe,
        vertex_exe=args.vertex_exe,
        werami_exe=args.werami_exe,
        props=props,
        perplex_dir=str(perplex_dir) if perplex_dir else None,
    )

    work_root = Path(args.outdir).resolve()
    runner = PerplexRunner(cfg, work_root)
    parsed_csv = runner.run(verbose=args.verbose)
    print(parsed_csv)

if __name__ == "__main__":
    main()
