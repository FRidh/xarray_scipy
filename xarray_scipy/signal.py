from typing import List, Union
import dask.array.fft
import numpy as np
import scipy.signal
import xarray as xr


def _keep_attrs(keep_attrs):
    if keep_attrs is None:
        keep_attrs = xr.core.options._get_keep_attrs(default=False)
    return keep_attrs


def _get_length(arr: xr.DataArray, dim: str) -> int:
    """Fast routine to determine the length of a dimension in a :class:`xr.DataArray`."""
    return arr.shape[arr.get_axis_num(dim)]


def convolve(
    in1: xr.DataArray, in2: xr.DataArray, mode: str = "full", method: str = "auto"
) -> xr.DataArray:
    return _convolve(in1, in2, dims=None, mode=mode, method=method)


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


def _fft_wrap(fft_func, inverse):
    # TODO make proper decorator

    def func(
        a: xr.DataArray,
        dim: str,
        newdim: str,
        n: int = None,
        norm: str = None,
        keep_attrs=None,
    ) -> xr.DataArray:

        with_dask = a.chunks is not None
        func = dask.array.fft.fft if with_dask else np.fft.fft

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
        n = n if n is not None else _get_length(a, dim)

        # Coordinate spacing along `dim`
        delta = (a.coords[dim].diff(dim=dim).mean(dim=dim)).data

        # Coordinates for `newdim`
        if inverse:
            delta = n * delta
        result[newdim] = np.fft.fftfreq(n, delta)
        return result

    return func


fft = _fft_wrap(dask.array.fft.fft, inverse=False)
ifft = _fft_wrap(dask.array.fft.ifft, inverse=True)


def hilbert(x: xr.DataArray, dim: str, N: int = None, keep_attrs=None) -> xr.DataArray:
    """Hilbert transform.

    Wraps :func:`scipy.signal.hilbert`.

    """

    result = xr.apply_ufunc(
        scipy.signal.hilbert,
        x,
        kwargs={
            "N": N,
        },
        input_core_dims=[
            (dim,),
        ],
        exclude_dims=set(
            [
                dim,
            ]
        ),
        output_core_dims=[
            (dim,),
        ],
        dask_gufunc_kwargs={
            "output_sizes": {
                dim: N if N is not None else len(x[dim]),
            },
        },
        dask="parallelized",
        keep_attrs=_keep_attrs(keep_attrs),
    )
    return result


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


def resample(x, num, dim: str, window=None, domain="time", keep_attrs=None):
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
