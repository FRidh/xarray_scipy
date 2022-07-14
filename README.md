# `xarray_scipy`

This project provides `scipy` functions wrapped for `xarray` and with `dask`
support.

## Scope

The goal is to provide a library with `scipy` functions that have been wrapped
yet still offer an interface similar to that of the similarly named `scipy`
functions. Additional functions or alternative interfaces are out of scope and
will not be accepted.

## Dask support

The aim is `dask` support for all functions using `xr.apply_ufunc`, either with
`allowed` or `parallelized`.

## Axes and dimensions

Typically with `numpy` arrays the keywords `axis` and `axes` are used and with
`xarray` the keywords `dim` and `dims`. In this library, `dim` and `dims` are
used to denote the dimensions over which the function is applied, and `newdim`
and `newdims` are used to describe the names of any newly created dimensions.

## Equispaced coordinates

With `xarray` it is possible to not have equispaced coordinates. In this library
only equispaced coordinates are supported, to minimize the difference between
this library and `scipy`.
