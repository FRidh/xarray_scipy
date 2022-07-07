import scipy.signal
import xarray as xr


def _keep_attrs(keep_attrs):
    if keep_attrs is None:
        keep_attrs = xr.core.options._get_keep_attrs(default=False)
    return keep_attrs


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
        output_sizes={
            dim: N if N is not None else len(x[dim]),
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
