# -*- coding: utf-8 -*-
"""The P-T grid of the section-level results (Fig 5, Fig 6, Table 1, the Monte
Carlo sweeps and the input-sensitivity rows of Table A.2).

2026-10-27: 550-750 C x 0.2-1.0 GPa at 21 x 21, i.e. 10 C x 0.04 GPa, after
a co-author noted that on the earlier 20 x 20 grid over 300-800 C x 0.1-1.5 GPa
(26 C x 0.074 GPa) the quoted best cells differed by a single cell. Every
accepted region of the paper lies inside the finer window, and the five-phase
stability field closes inside it (it reaches 0.80-0.83 GPa; a 0.2-0.8 GPa
window clipped it). The per-window
inversions (window_pt_run.py) and the scale sweeps keep the full-range coarse
grid, because windows that lose a phase land outside this window.
"""
P_RANGE = (0.2, 1.0)
T_RANGE = (550.0, 750.0)
NP = NT = 21
COARSE = {"p_range": (0.1, 1.5), "t_range": (300.0, 800.0), "n_p": 20, "n_t": 20}   # the grid used until 2026-10-27
