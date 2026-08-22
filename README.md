# RADIA Magnet Studio — Strict Correctness Build

An independent Streamlit interface and analysis toolkit for RADIA-based
insertion-device modelling. This repository adds device builders, engineering
error studies, convergence checks, visualization, scientific-data export, and
macOS-oriented launch support around a separately installed RADIA extension.

> **Independent project:** This is not an official ESRF release and is not
> affiliated with or endorsed by ESRF or the RADIA maintainers. RADIA remains a
> separate third-party dependency under its original copyright and license.

## Installation overview

1. Install or build RADIA from the official repository:
   <https://github.com/ochubar/Radia>
2. Confirm that the compiled Python extension is available. The default path
   expected by this project is:

   `~/Desktop/Radia-master/cpp/gcc/radia*.so`

3. On macOS, double-click or run:

   `START_HERE_RADIA_MAGNET_STUDIO.command`

The launcher can also use a custom RADIA location through `RADIA_PYTHONPATH`
and a custom Python interpreter through `PYTHON_BIN`.

Run only:

`START_HERE_RADIA_MAGNET_STUDIO.command`

Expected local RADIA extension:

`~/Desktop/Radia-master/cpp/gcc/radia*.so`

## Correctness fixes in this build

1. **RADIA relaxation convergence gate**
   - `RlxPre -> RlxAuto`
   - requires `AvPrec < precision`
   - requires `Niter < max_iter`
   - otherwise raises a visible error and stops analysis.

2. **Geometry-derived longitudinal field range**
   - finds the true outer z-edges of generated blocks
   - includes APPLE-II row shifts and manufacturing position errors
   - adds configurable fringe-field margin before I1/I2/trajectory analysis.

3. **Explicit K definitions**
   - `Kx_peak = 0.934 * |By|max * lambda_u(cm)`
   - `Ky_peak = 0.934 * |Bx|max * lambda_u(cm)`
   - `K_peak = 0.934 * max(|B_perp|) * lambda_u(cm)`
   - `K_vector_norm = sqrt(Kx_peak^2 + Ky_peak^2)` is reported separately
   - resonance term is explicitly `(Kx_peak^2 + Ky_peak^2)/2`.

4. **Trajectory-derived electron phase error**
   - integrates slippage using x' and y'
   - removes best-fit linear slippage
   - evaluates phase residual at half-period positions
   - reports a separate RMS
   - the old zero-crossing phase is retained only as a diagnostic.

5. **HDF5 export**
   - no silent metric-write exception
   - unsupported metadata are recorded in `export_skipped_items`.

6. **Target B0 calibration modes**
   - Central-period peak B_perp (default)
   - Central 3-period peak B_perp
   - Global peak B_perp
   - default avoids calibrating to accidental fringe-field/end-field overshoot.

## Existing integrated features
- Planar / Helical / Elliptical / APPLE-II prototype / Wiggler
- manufacturing error model + deterministic seed
- ideal-vs-error comparison
- 3D magnet geometry
- 2D / 3D real field sampling
- trajectory, I1/I2, harmonics
- CSV / JSON / HDF5 / PDF / V11-compatible 3D map export

## APPLE-II scope
The APPLE-II backend is a physics-informed four-array research prototype with
real longitudinal row displacement. It is not a manufacturer- or facility-
certified replica of a specific installed device.


## NumPy compatibility
Field integrals no longer depend on the removed legacy `numpy.trapz` alias.
The code uses `numpy.trapezoid` when available, with an equivalent fallback for
older NumPy releases.


## Clear-results / JSON fix
- Numerical backend is unchanged from the NumPy-compatible strict-correctness build.
- The six compressed result cards were changed to two rows of three.
- An exact-value results table now shows values and units.
- Field-range/regime information is shown in a dedicated information panel.
- JSON export recursively converts NumPy arrays and NumPy scalar types to JSON-safe values.
- The complete trajectory and electron-phase arrays are preserved in JSON.
- Export buttons are arranged in a readable 2×2 layout and one failed exporter no longer blocks the others.

## Project scope and validation

This is a research and educational prototype. Results should be independently
validated before use in engineering, facility, medical, or safety-critical
decisions. A real numerical run still requires a compatible RADIA extension on
the user's machine; the included test suite uses a deterministic fake backend
for automated checks where appropriate.

Run the included tests from the repository root with:

```bash
python3 tests/run_all.py
```

## Relationship to RADIA

RADIA is the 3D magnetostatics code originally developed at the European
Synchrotron Radiation Facility and currently maintained in its official
repository: <https://github.com/ochubar/Radia>.

RADIA itself is not bundled in this source repository. Users install it
separately, and this project loads its compiled Python extension at runtime.
The original RADIA copyright and redistribution conditions are reproduced in
[`RADIA_LICENSE.txt`](RADIA_LICENSE.txt).

## Licensing

Original code developed specifically for this repository is available under
the BSD 2-Clause License; see [`LICENSE`](LICENSE). RADIA and all other
third-party dependencies remain governed by their own licenses; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
