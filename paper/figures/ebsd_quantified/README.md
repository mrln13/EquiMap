# Quantified-tier EBSD validation (section 5.2, the QuantMap contrast)

The count-tier run of section 5.2 lives one level up (`ebsd_confusion_prep.py`
on the h5oina window-integral counts, 6 um/px). This folder holds the
quantified contrast quoted in the same section: the same two maps re-exported
by AZtec QuantMap as per-element wt% CSVs (12 elements Si Al Mg Fe Ca Ti Cr Mn
Na K Ni O; oxygen by stoichiometry; C as a deconvolution element only; 2 x 2
binning, i.e. 12 um bins; files `D:/data/EBSD/serpdi_{1,2}_EBSD/wt%`, decimal-comma
locale, read by `oxford_loader._read_element_csv`).

Run order (THERMOD env python, from this folder; the dataset paths are
hard-coded at the top of each script):

1. `check_wt_bin2.py`: parses the CSVs, caches `serpdi_{1,2}_wt_maps_bin2.npz`
   beside this file (84 / 103 MB, git-ignored, rebuilt on demand), checks grid
   size, closure, stoichiometric O, EBSD registration (zero offset) and the
   per-EBSD-phase means -> `check_wt_bin2.json`.
2. `quant_ebsd_run.py`: section-5.2 pipeline on the wt% maps (sigma and
   min_region scaled to the bin), named in ratio and structural mode on three
   element sets -> `quant_ebsd_{1,2}.json`, `quant_ebsd_{1,2}_labels.npz`
   (keys ebsd, published_bin2, gmm_all12, gmm_noTrace), `confusion/*.csv`,
   `quant_ebsd_summary.txt`.
3. `quant_ebsd_erode.py`: re-names every saved label set from interior-only
   (eroded) means at 0, 1, 2 bins -> `quant_ebsd_erode.json`,
   `quant_ebsd_erode_summary.txt`. Needs the mineral_id carbonate exclusion
   test of 2026-10-09; before it, structural mode named the diopside domain
   "Calcite".
4. `pure_diopside_counts.py`: EBSD-pure interior diopside (EBSD mask eroded
   3 px) scored in counts and in wt% -> `pure_diopside_counts.json`; the
   "83 to 79 vs 97 to 77" margins in the text.

Numbers quoted in section 5.2 (map 1 / map 2): purity 88.7 / 92.7 %
(phase-weighted majority share, `confusion/map{m}_gmmall12_*.csv`),
interior-only structural accuracy 0.878 / 0.899 at 1 bin
(`quant_ebsd_erode.json`, key `gmm_all12` -> `e1/structural/all12`), full-domain
structural top score 0.05 / 0.05, oxide means Si 7-13 wt% and Mg 8-11 wt%
(`check_wt_bin2.json`, `phase_means`).
