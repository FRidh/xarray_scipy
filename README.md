# `xarray_scipy`

This project provides `scipy` functions wrapped for `xarray` and with `dask` support.

## Scope

The goal is to provide a library with `scipy` functions that have been wrapped yet still offer an interface similar to that of the similarly named `scipy` functions. Additional functions
or alternative interfaces are out of scope and will not be accepted.

## Dask support

The aim is `dask` support for all functions using `xr.apply_ufunc`, either with `native` or `parallelized`.
