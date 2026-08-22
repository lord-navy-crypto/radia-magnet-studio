# Third-Party Notices

This repository contains original interface, modelling, analysis,
visualization, export, and testing code that interoperates with separately
installed third-party software.

## RADIA

RADIA is a separate 3D magnetostatics dependency originally developed at the
European Synchrotron Radiation Facility.

- Official repository: <https://github.com/ochubar/Radia>
- Original copyright and license: [`RADIA_LICENSE.txt`](RADIA_LICENSE.txt)
- Bundling status: RADIA source code and compiled binaries are not included in
  this repository. The application loads a user-installed Python extension at
  runtime.

The name RADIA is used only to identify compatibility with the upstream
software. This repository is not an official ESRF or RADIA-maintainer release.

## Python dependencies

Runtime dependencies are declared in [`requirements.txt`](requirements.txt):

- NumPy
- pandas
- Streamlit
- Plotly
- h5py
- Matplotlib

These packages are not vendored in this repository. They remain subject to
their respective upstream licenses and copyright notices.
