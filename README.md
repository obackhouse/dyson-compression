# Dyson Equation Compression

This module provides functions to compress the spectral representation of a Green's function or self-energy linked by a Dyson equation

<img src="https://render.githubusercontent.com/render/math?math=\Large G(E) = G_0(E) %2B G_0(E) \Sigma(E) + G(E)">

These dynamical quantities can be written in their spectral representation as a set of causal, separable auxiliary poles

<img src="https://render.githubusercontent.com/render/math?math=\Large A_{xy}(E) = \sum_{k} \frac{V_{xk} V_{yk}^\dagger}{E - \epsilon_{k}}">

which may be compressed such that *k* is linear in the number of physical degrees of freedom *x,y*.


## Installation

1. Install `numpy`
2. clone from git: `git clone https://github.com/obackhouse/dyson-compression`
3. Append to `PYTHONPATH`
4. Run tests: `python -m unittest discover dyson-compression/tests`


## Usage

Required inputs are the seperate occupied and virtual moments of either the self-energy or the Green's function, obtained from the uncompressed spectral representations such as

<img src="https://render.githubusercontent.com/render/math?math=\Large T_{xy}^{(n)} = \sum_{k}^\mathrm{occ} V_{xk} V_{yk}^\dagger \epsilon_{k}^{n}">

Use `kernel_se` when starting with moments of the self-energy and `kernel_gf` when starting with moments of the Green's function, both of which can be found in `__init__.py` with docstrings detailing usage.
