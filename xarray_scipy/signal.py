from typing import List, Union
import numpy as np
import scipy.signal
import xarray as xr


def _keep_attrs(keep_attrs):
    if keep_attrs is None:
        keep_attrs = xr.core.options._get_keep_attrs(default=False)
    return keep_attrs


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
