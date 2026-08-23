# Changelog

## 3.0.0

- Replaced the temporary two-workspace UI with one continuous page: a successful
  magnetic-field calculation automatically becomes the baseline for the next
  sensitivity, uncertainty, or convergence analysis.
- Isolated advanced-analysis controls in a Streamlit fragment so starting,
  stopping, and refreshing long studies do not erase the magnetic-field results above.
- Removed the separate Study Center and its redundant graphical launcher.
- Unified the original magnetic-field generator/inspector and advanced studies
  under the same Streamlit application and primary launcher.
- Added direct transfer of the last solved device and solver settings into the
  study baseline; study summaries and sensitivity plots render in the same UI.
- Preserved target-B0 calibration in batch workers and records realized parameters.
- Added resumable batch parameter scans and objective ranking.
- Added multi-seed Monte Carlo uncertainty summaries with 95% confidence intervals.
- Added automatic convergence studies for segmentation, sample density, and field margin.
- Added process-isolated parallel evaluation, content-addressed caching, atomic
  checkpoints, cooperative cancellation, retryable failures, Study Center GUI,
  terminal runner, and downloadable study bundles.

## 2.1.0

- Replaced nearest-bin FFT harmonic ratios with a joint sinusoidal fit that is
  resistant to non-integer-window spectral leakage and linear end-field trends.
- Added complete run provenance and optional ideal/error comparison files.
- Added strict transfer-package path, schema, size and checksum validation.
- Added a minimal downstream package reader for collaborators.
- Prevented download buttons from rerunning and clearing the results view.

## 2.0.0

- Added the versioned downstream research-package export.
- Added strict field-array, coordinate, device-parameter and finite-value validation.
- Made JSON standards-compliant by encoding non-finite diagnostics as `null`.
- Preserved nested trajectory and electron-phase results in HDF5.
- Added chunked RADIA sampling and return-value checks.
- Added a dependency-aware self-check runner and launcher RADIA preflight.
- Replaced the temporary build label in the interface with a stable version title.
