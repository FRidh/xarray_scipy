import collections.abc
from typing import List, Tuple, Union
import dask.array.fft
import numpy as np
import scipy.fft  # pylint: disable=no-name-in-module
import scipy.signal
import xarray as xr


def _keep_attrs(keep_attrs):
    if keep_attrs is None:
        keep_attrs = xr.core.options._get_keep_attrs(default=False)
    return keep_attrs


def _compute_delta(arr: xr.DataArray, dim: str):
    return abs((arr.coords[dim].diff(dim=dim).mean(dim=dim)).data)


def _get_length(arr: xr.DataArray, dim: str) -> int:
    """Fast routine to determine the length of a dimension in a :class:`xr.DataArray`."""
    return arr.shape[arr.get_axis_num(dim)]


def _wrap(func_scipy):
    def decorator(func):
        func.__name__ = func_scipy.__name__
        func.__doc__ = func_scipy.__doc__
        return func

    return decorator


@_wrap(scipy.signal.convolve)
def convolve(
    in1: xr.DataArray, in2: xr.DataArray, mode: str = "full", method: str = "auto"
) -> xr.DataArray:
    return _convolve(in1, in2, dims=None, mode=mode, method=method)


@_wrap(scipy.signal.fftconvolve)
def fftconvolve(
    in1: xr.DataArray,
    in2: xr.DataArray,
    dims: Union[None, List[str]] = None,
    mode: str = "full",
) -> xr.DataArray:
    return _convolve(in1, in2, dims=dims, mode=mode, method="fft")


def _convolve(
    in1: xr.DataArray,
    in2: xr.DataArray,
    dims: Union[None, List[str]],
    mode: str = "full",
    method: str = "auto",
) -> xr.DataArray:

    if not (set(in1.dims) == set(in2.dims)):
        raise ValueError(f"in1 and in2 do not have the same dims")

    if dims is None:
        dims = list(set(in1.dims) & set(in2.dims))
    else:
        dims_ = set(dims)
        if not set(in1.dims) >= dims_:
            raise ValueError(f"in1 is missing dims {dims_ - set(in1.dims)}")
        elif not set(in1.dims) >= dims_:
            raise ValueError(f"in2 is missing dims {dims_ - set(in2.dims)}")

    def _compute_output_size(dim):
        if mode == "full":
            return len(in1[dim]) + len(in2[dim]) - 1
        elif mode == "same":
            return len(in1[dim])
        elif mode == "valid":
            return NotImplemented

    output_sizes = {dim: _compute_output_size(dim) for dim in dims}

    in1 = in1.transpose(..., *dims)
    in2 = in2.transpose(..., *dims)

    result = xr.apply_ufunc(
        scipy.signal.fftconvolve,
        in1,
        in2,
        kwargs={
            "mode": mode,
            "axes": np.array(in1.get_axis_num(dims)),
        },
        input_core_dims=[
            dims,
            dims,
        ],
        output_core_dims=[
            dims,
        ],
        exclude_dims=set(dims),
        dask="parallelized",
        dask_gufunc_kwargs={
            "allow_rechunk": True,
            "output_sizes": output_sizes,
        },
    )
    return result


@_wrap(scipy.signal.decimate)
def decimate(
    x: xr.DataArray,
    q: int,
    dim: str,
    n: int = None,
    ftype: str = "iir",
    zero_phase: bool = True,
    keep_attrs=None,
) -> xr.DataArray:
    """Decimate signal.

    This function wraps :func:`scipy.signal.decimate`.
    """

    # TODO: recompute coordinates of dim

    result = xr.apply_ufunc(
        scipy.signal.decimate,
        x,
        kwargs={
            "ftype": ftype,
            "n": n,
            "q": q,
            "zero_phase": zero_phase,
        },
        input_core_dims=[(dim,)],
        exclude_dims=set((dim,)),
        output_core_dims=[
            (dim,),
        ],
        dask="parallelized",
        dask_gufunc_kwargs={
            "allow_rechunk": True,
            "output_sizes": {
                dim: len(x[dim]) // q,
            },
        },
        vectorize=True,
        output_dtypes=[x.dtype],
        keep_attrs=_keep_attrs(keep_attrs),
    )

    return result


def _fft_wrap(
    fft_kind: str, inverse: bool, real: bool = False, hermitian: bool = False
):
    # TODO make proper decorator

    if real and hermitian:
        raise ValueError("FFT has to be real, hermitian or neither but not both.")

    def func(
        a: xr.DataArray,
        dim: str,
        newdim: str,
        n: int = None,
        norm: str = None,
        keep_attrs=None,
    ) -> xr.DataArray:

        with_dask = a.chunks is not None
        lib = dask.array.fft if with_dask else np.fft
        func = getattr(lib, fft_kind)

        kwargs = {
            "n": n,
        }

        if with_dask:
            if norm is not None:
                raise ValueError("norm is not supported with dask arrays")
        else:
            kwargs["norm"] = norm

        result = xr.apply_ufunc(
            func,
            a,
            kwargs=kwargs,
            input_core_dims=((dim,),),
            exclude_dims=set([dim]),
            output_core_dims=((newdim,),),
            vectorize=False,
            dask="allowed" if with_dask else "forbidden",
            keep_attrs=_keep_attrs(keep_attrs),
        )

        # DFT size
        ndim = n if n is not None else _get_length(a, dim)

        # Coordinate spacing along `dim`
        delta = _compute_delta(a, dim)

        # Determine which function to use to compute the coordinates
        # and whether the amount of coordinates needs to be adjusted.
        if real and not inverse:
            func = np.fft.rfftfreq
        elif real and inverse:
            func = np.fft.fftfreq
            if n is None:
                ndim = (ndim - 1) * 2
        elif hermitian and not inverse:
            func = np.fft.fftfreq
            if n is None:
                ndim = (ndim - 1) * 2
        elif hermitian and inverse:
            func = np.fft.rfftfreq
        else:
            func = np.fft.fftfreq

        delta = delta / ndim

        if newdim in result.coords:
            raise ValueError(f"Coordinates already exist for dimension: {newdim}")

        result = result.assign_coords({newdim: func(ndim, delta)})
        return result

    func_scipy = getattr(scipy.fft, fft_kind)  # pylint: disable=no-member
    func = _wrap(func_scipy)(func)
    return func


fft = _fft_wrap("fft", inverse=False)
ifft = _fft_wrap("ifft", inverse=True)

rfft = _fft_wrap("rfft", inverse=False, real=True)
irfft = _fft_wrap("irfft", inverse=True, real=True)

hfft = _fft_wrap("hfft", inverse=False, hermitian=True)
ihfft = _fft_wrap("ihfft", inverse=True, hermitian=True)


def _fftshift_wrap(fftshift_kind: str) -> collections.abc.Callable:
    def func(x: xr.DataArray, dims: Tuple) -> xr.DataArray:

        if not isinstance(dims, collections.abc.Iterable) or isinstance(dims, str):
            raise ValueError("dims needs to be an iterable of strings")

        dims = tuple(dims)

        with_dask = x.chunks is not None
        lib = dask.array.fft if with_dask else np.fft
        func = getattr(lib, fftshift_kind)

        # Shift the values of our array
        result = xr.apply_ufunc(
            func,
            x,
            input_core_dims=(dims,),
            output_core_dims=(dims,),
        )

        # And shift also our coordinates
        for dim in dims:
            result.coords[dim] = func(result.coords[dim])

        return result

    func_scipy = getattr(scipy.fft, fftshift_kind)  # pylint: disable=no-member
    func = _wrap(func_scipy)(func)
    return func


fftshift = _fftshift_wrap("fftshift")
ifftshift = _fftshift_wrap("ifftshift")


@_wrap(scipy.signal.hilbert)
def hilbert(x: xr.DataArray, dim: str, N: int = None, keep_attrs=None) -> xr.DataArray:
    """Hilbert transform.

    Wraps :func:`scipy.signal.hilbert`.

    """

    exclude_dims = []

    if N is not None:
        exclude_dims.append(dim)

    result = xr.apply_ufunc(
        scipy.signal.hilbert,
        x,
        kwargs={
            "N": N,
        },
        input_core_dims=[
            (dim,),
        ],
        exclude_dims=set(exclude_dims),
        output_core_dims=[
            (dim,),
        ],
        dask_gufunc_kwargs={
            "output_sizes": {
                dim: N if N is not None else len(x[dim]),
            },
        },
        dask="parallelized",  # np.asarray is used within scipy.signal.hilbert
        keep_attrs=_keep_attrs(keep_attrs),
    )

    if N is not None and dim in x.coords:

        def linspace(arr, N):
            # np.linspace does not function with datetimes
            arr = np.asarray(arr)
            delta = arr.max() - arr.min()
            result = np.arange(N) * delta / N
            return result

        result = result.assign_coords(
            {dim: x.coords[dim].interp({dim: linspace(x.coords[dim], N)})}
        )

    return result


@_wrap(scipy.signal.peak_widths)
def peak_widths(x: xr.DataArray, peaks: xr.DataArray, dim: str, **kwargs):
    """Calculate the width of each peak in a signal.

    Parameters:
        x: A signal with peaks
        peaks: Indices of peaks in ``x``. The dimension should be called ``peak``.
        dim: Dimension in ``x`` along which to estimate the bandwidth of ``peaks``.

    Returns:
        Four arrays just as :func:`scipy.signal.peak_widths`.

    This function wraps :func:`scipy.signal.peak_widths`.

    """

    # TODO: BROKEN

    def _peak_widths(x, peaks, **kwargs):
        x = x.copy()
        peaks = peaks.copy()
        return scipy.signal.peak_widths(x, peaks, **kwargs)

    result = xr.apply_ufunc(
        _peak_widths,
        x.transpose(..., dim),
        peaks.transpose(..., "peak"),
        input_core_dims=[
            (dim,),
            ("peak",),
        ],
        exclude_dims=set((dim,)),
        output_core_dims=[
            ("peak",),
            ("peak",),
            ("peak",),
            ("peak",),
        ],
        kwargs=kwargs,
        vectorize=True,
    )
    return result


@_wrap(scipy.signal.resample)
def resample(
    x: xr.DataArray,
    num: int,
    dim: str,
    window=None,
    domain: str = "time",
    keep_attrs=None,
) -> xr.DataArray:
    # TODO: support t=None
    result = xr.apply_ufunc(
        scipy.signal.resample,
        x,
        input_core_dims=[[dim]],
        exclude_dims=set((dim,)),
        output_core_dims=[
            (dim,),
        ],
        kwargs={
            "num": num,
            "domain": domain,
            "axis": -1,
        },
        dask="allowed",
        keep_attrs=_keep_attrs(keep_attrs),
    )
    return result


@_wrap(scipy.signal.sosfilt)
def sosfilt(
    sos: np.ndarray, x: xr.DataArray, dim: str, zi: np.ndarray = None
) -> xr.DataArray:

    result = xr.apply_ufunc(
        scipy.signal.sosfilt,
        sos,
        x,
        kwargs={
            "zi": zi,
        },
        input_core_dims=(
            ("section", "coefficient"),
            (dim,),
        ),
        output_core_dims=((dim,),),
        dask="parallelized",
        dask_gufunc_kwargs={
            "allow_rechunk": True,
            "output_sizes": {
                dim: _get_length(x, dim),
            },
        },
        output_dtypes=(x.dtype,),
    )

    return result


@_wrap(scipy.signal.sosfiltfilt)
def sosfiltfilt(
    sos: np.ndarray, x: xr.DataArray, dim: str, padtype: str = "odd", padlen: int = None
) -> xr.DataArray:

    result = xr.apply_ufunc(
        scipy.signal.sosfiltfilt,
        sos,
        x,
        kwargs={
            "padtype": padtype,
            "padlen": padlen,
        },
        input_core_dims=(
            ("section", "coefficient"),
            (dim,),
        ),
        output_core_dims=((dim,),),
        dask="parallelized",
        dask_gufunc_kwargs={
            "allow_rechunk": True,
            "output_sizes": {
                dim: _get_length(x, dim),
            },
        },
        output_dtypes=(x.dtype,),
    )

    return result
