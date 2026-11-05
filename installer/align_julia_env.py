"""Align juliacall's own Julia environment with the MAGEMin environment.

Two Julia environments meet in one session when EquiMap runs MAGEMin:

* juliacall boots from its own project (managed by ``juliapkg``, inside the
  conda env) and loads PythonCall and its dependencies, among them Parsers;
* petthermotools then activates ``~/.petthermotools_julia_env`` (MAGEMin_C,
  MAGEMinCalc, CSV, InlineStrings, ...) in the same session.

A package already loaded from the first environment is not reloaded from the
second. When the two resolve to different Parsers versions (juliacall's side
moved to Parsers 3.0, which removed ``xparse``, while MAGEMin's side needs 2.x),
InlineStrings fails to precompile and every MAGEMin run dies with
``UndefVarError: xparse not defined in Parsers``. This script pins juliacall's
Parsers to the exact version the MAGEMin environment uses, so both sides load
the same one. The pin is stored by juliapkg and survives later resolves; re-run
this script (install.bat does) after the MAGEMin environment changes.

Run it with the EquiMap environment's Python after install_MAGEMinCalc().
Exit code 0 also when there is nothing to align (MAGEMin not installed).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# Julia packages that both environments load (name -> registry UUID).
SHARED = {"Parsers": "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"}


def manifest_version(manifest: Path, name: str) -> str | None:
    """Version of ``name`` in a Julia Manifest.toml, or None."""
    text = manifest.read_text(encoding="utf-8")
    m = re.search(r"\[\[deps\.%s\]\](.*?)(?:\n\[\[|\Z)" % re.escape(name), text, re.S)
    if not m:
        return None
    v = re.search(r'^\s*version\s*=\s*"([^"]+)"', m.group(1), re.M)
    return v.group(1) if v else None


def main() -> int:
    manifest = Path.home() / ".petthermotools_julia_env" / "Manifest.toml"
    if not manifest.exists():
        print(f"[align] {manifest} not found (MAGEMin not installed yet); nothing to align.")
        return 0
    pins = {name: manifest_version(manifest, name) for name in SHARED}
    pins = {k: v for k, v in pins.items() if v}
    if not pins:
        print("[align] MAGEMin environment uses none of the shared packages; nothing to align.")
        return 0
    import juliapkg
    for name, version in pins.items():
        print(f"[align] pinning {name} = {version} in juliacall's environment")
        juliapkg.add(name, SHARED[name], version=f"={version}")
    juliapkg.resolve(force=True)
    print("[align] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
