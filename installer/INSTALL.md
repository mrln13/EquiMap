# Installing EquiMap (Windows)

EquiMap analyses SEM/EDS maps and runs thermodynamic phase-equilibrium models.
This guide helps setting up a working program with **two double-clicks**.

## What you need

- A Windows 10 or 11 computer.
- An internet connection during installation.
- About **10–15 minutes** the first time (most of it is an automatic download of
  the Julia + MAGEMin modelling engine).
- No administrator rights — everything installs into your own user folder.

## Install (do this once)

1. Download/copy the **EquiMap folder** to your computer (e.g. your Desktop or
   Documents). Keep all the files together (do not move pieces out of it).
2. Open the **`installer`** subfolder and double-click **`install.bat`**.
3. A black window opens and shows progress. Leave it running until it says
   **“Done.”** then press a key to close it.

Done. The installer:

- sets up a private copy of conda in your user folder,
- builds the EquiMap environment,
- downloads the Julia + MAGEMin modelling engine, and
- puts an **EquiMap** shortcut on your Desktop.

> If something goes wrong, the window tells you what and nothing is changed
> system-wide. Just fix the issue and run `install.bat` again.
>
> A full transcript is always saved to **`installer\install_log.txt`** (next to
> `install.bat`, overwritten each run). If you need help, send that file — it
> contains the detailed error messages.

## Run

1. Double-click the **EquiMap** icon on your Desktop (or `installer\EquiMap.bat`
   in the folder).
2. A window asks you to **choose your dataset folder**. Pick the folder that
   holds your stitched mosaic, or an Oxford AZtec / h5oina export. (Have a raw,
   un-stitched folder of Bruker BCF tiles? Open it anyway and use the **Tools**
   menu below to stitch it from inside EquiMap.)
3. EquiMap opens. The first time you point it at an Oxford export it spends a
   few minutes building its cache; this is normal and only happens once.

## Preparing data from inside EquiMap (Tools menu)

You don't need the command line to prepare a dataset. The **Tools** menu at the
top of the EquiMap window can:

- **Stitch mosaic from BCF folder…** — turn a folder of raw Bruker BCF tiles
  into a stitched mosaic (optionally building the EDS cache at the same time).
- **Build / refresh EDS cache…** — (re)build the fast EDS cache for a stitched
  dataset. The status line on the right panel shows whether a cache is ready.
- **Import Oxford dataset…** — convert an Oxford AZtec CSV or `.h5oina` export.
- **EDS cache status…** — check the cache and, if needed, defragment it.

Each opens a small window that shows progress and can be cancelled. These steps
can take a while (stitching and cache building are the slow ones); leave them
running until they report success.

## Notes

- **Modelling** uses the built-in **MAGEMin** engine, which the installer set up
  for you.
- **Perple_X** is an optional, advanced alternative engine. It is a separate
  download from ETH Zurich that cannot be bundled here; you only need it if you
  specifically want Perple_X instead of MAGEMin.
- To **update** EquiMap later, replace the program files and run `install.bat`
  again. It updates the environment in place.
- Everything lives under `%USERPROFILE%\miniforge3` (the conda install), the
  shared Julia folder `%USERPROFILE%\.julia` (Julia + MAGEMin), and your EquiMap
  folder. No PATH, registry, or shell settings are changed.
- **To remove EquiMap**, double-click **`installer\uninstall.bat`**. It deletes
  the Desktop shortcut and the conda environment, and *asks* before removing the
  shared Julia/MAGEMin folder (say no if you use Julia for other things). Then
  delete the EquiMap folder itself.
