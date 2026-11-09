#!/bin/bash
set -eu
export PATH=/workspace/rankwalk-paper-reproduction/r-env/bin:$PATH
export CONDA_PREFIX=/workspace/rankwalk-paper-reproduction/r-env
export MAKEFLAGS=-j2
export RGL_USE_NULL=TRUE
for p in /workspace/rankwalk-paper-reproduction/r-env/etc/conda/activate.d/*; do source "$p"; done
for p in fdapace funData MFPCA longitudinalData kml kml3d; do R CMD INSTALL --no-docs --no-html /workspace/rankwalk-paper-reproduction/r-sources/$p; done
R CMD INSTALL --no-docs --no-html /workspace/rankwalk-paper-reproduction/TAPIO
