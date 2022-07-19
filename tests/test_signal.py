import numpy as np
import pandas as pd
import xarray as xr

import pytest
import xarray.testing
import xarray_scipy.signal

from xarray_scipy.signal import (
    _compute_delta,
    _get_length,
    convolve,
    decimate,
    fft,
    ifft,
    rfft,
    irfft,
    hfft,
    ihfft,
    fftshift,
    ifftshift,
    fftconvolve,
    hilbert,
    peak_widths,
)


@pytest.fixture(params=[True, False], ids=["With dask", "Without dask"])
def dask(request):
    return request.param


@pytest.fixture(params=[None, 1, 4])
def nchannels(request):
    return request.param


@pytest.fixture
def signal(dask, nchannels):
    duration = 10.0
    fs = 8000.0
    nsamples = int(fs * duration)
    time = np.arange(nsamples) / fs
    frequency = 100.0

    signal = np.sin(2.0 * np.pi * frequency * time)
    signal = xr.DataArray(
        signal, dims=["time"], coords={"time": time}, attrs={"fs": fs}
    )

    if dask:
        signal = signal.chunk({"time": 4})

    if nchannels is not None:
        signal = xr.concat(
            [signal] * nchannels, dim=pd.Index(np.arange(nchannels), name="channel")
        )

    return signal


class TestConvolve:
    @pytest.fixture(params=["full", "same"], ids=["full", "same"])
    def convolve_mode(self, request):
        return request.param

    def test_fftconvolve(self, signal, convolve_mode):

        result = fftconvolve(signal, signal, mode=convolve_mode)
        if convolve_mode == "full":
            assert len(result.time) == len(signal.time) + len(signal.time) - 1
        elif convolve_mode == "same":
            assert len(result.time) == len(signal.time)

        assert set(signal.dims) == set(result.dims)

    @pytest.fixture(params=[["time"], None])
    def dims(self, request):
        return request.param

    def test_fftconvolve(self, signal, convolve_mode, dims):

        result = fftconvolve(signal, signal, dims=dims, mode=convolve_mode)
        if convolve_mode == "full":
            assert len(result.time) == len(signal.time) + len(signal.time) - 1
        elif convolve_mode == "same":
            assert len(result.time) == len(signal.time)

        assert set(signal.dims) == set(result.dims)


def test_decimate(signal, nchannels):

    q = 4

    result = decimate(signal, q, dim="time")

    assert len(result.time) == len(signal.time) // q

    if nchannels is not None:
        assert len(result.channel) == nchannels


class TestPeakWidths:
    @pytest.fixture
    def signal(self, request):
        SIGNALS = {
            # Horizontal is time, vertical is frequency
            "1_signal_1_width": np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0],
                ]
            ),
            "1_signal_2_width": np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0, 0],
                    [0, 0, 1, 1, 0, 0, 0],
                    [0, 0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0],
                ]
            ),
            "2_signal_1_width": np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0],
                    [1, 0, 0, 1, 0, 0, 0],
                    [0, 1, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 0, 1],
                    [0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0],
                ]
            ),
            "2_signal_2_width": np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0, 0],
                    [1, 0, 1, 1, 0, 0, 0],
                    [1, 1, 0, 1, 1, 0, 0],
                    [0, 1, 1, 0, 1, 1, 0],
                    [0, 0, 1, 1, 0, 1, 1],
                    [0, 0, 0, 1, 1, 0, 1],
                    [0, 0, 0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0],
                ]
            ),
        }
        return SIGNALS[request.param]

    @pytest.mark.parametrize(
        "signal, bandwidth",
        [
            (
                "1_signal_1_width",
                1,
            ),
            (
                "1_signal_2_width",
                2,
            ),
        ],
        indirect=["signal"],
    )
    def test_peak_widths(self, signal, bandwidth):

        print(signal.shape)

        ntime, nfreq = signal.shape[::-1][0:2]
        print(ntime, nfreq)

        times = np.arange(ntime)

        peaks = xr.DataArray(
            np.arange(1, ntime + 1), dims=["time"], coords={"time": times}, name="peaks"
        ).expand_dims({"peak": [0]})

        signal = xr.DataArray(
            signal,
            dims=["frequency", "time"],
            coords={"time": times, "frequency": np.arange(nfreq)},
        )

        output = peak_widths(signal, peaks, dim="frequency")
        widths = output[0]

        assert len(output) == 4
        assert len(widths.time) == ntime
        assert peaks.ndim == widths.ndim
        assert (bandwidth == widths).all()

        assert "peak" in widths.dims
        assert "peak" in widths.coords
        assert "time" in widths.coords
        assert "time" in widths.coords

    @pytest.mark.parametrize(
        "signal, bandwidth",
        [
            (
                "2_signal_1_width",
                1,
            ),
            (
                "2_signal_2_width",
                2,
            ),
        ],
        indirect=["signal"],
    )
    def test_peak_width__multiple_peaks(self, signal, bandwidth):
        """2D array that has more than one peak moving with time."""

        print(signal.shape)
        ntime, nfreq = signal.shape[::-1][0:2]
        print(ntime, nfreq)

        times = np.arange(ntime)

        peaks = np.array(
            [
                np.arange(ntime) + 1,
                np.arange(ntime) + 4,
            ]
        )

        peaks = xr.DataArray(
            peaks,
            dims=["peak", "time"],
            coords={"time": times, "peak": np.arange(2)},
            name="peaks",
        )

        signal = xr.DataArray(
            signal,
            dims=["frequency", "time"],
            coords={"time": times, "frequency": np.arange(nfreq)},
        )

        output = peak_widths(signal, peaks, dim="frequency")
        widths = output[0]

        assert len(output) == 4
        assert len(widths.time) == ntime
        assert peaks.ndim == widths.ndim
        assert (bandwidth == widths).all()

        assert "peak" in widths.dims
        assert "peak" in widths.coords
        assert "time" in widths.coords
        assert "time" in widths.coords

    @pytest.mark.parametrize(
        "signal, tile, bandwidth",
        [
            ("2_signal_1_width", (3, 1, 1), 1),
            ("2_signal_2_width", (3, 1, 1), 2),
        ],
        indirect=["signal"],
    )
    def test_peak_widths__multiple_peaks_3d(self, signal, tile, bandwidth):
        """3D array that has more than one peak moving with time."""
        signal = np.tile(signal, tile)

        print(signal.shape)

        ntime, nfreq = signal.shape[::-1][0:2]
        print(ntime, nfreq)

        times = np.arange(ntime)

        peaks = np.array(
            [
                np.arange(ntime) + 1,
                np.arange(ntime) + 4,
            ]
        )

        extradim = np.arange(3)

        peaks = xr.DataArray(
            peaks,
            dims=["peak", "time"],
            coords={"time": times, "peak": np.arange(2)},
            name="peaks",
        ).expand_dims({"extradim": extradim})

        signal = xr.DataArray(
            signal,
            dims=["extradim", "frequency", "time"],
            coords={"extradim": extradim, "time": times, "frequency": np.arange(nfreq)},
        )

        output = peak_widths(signal, peaks, dim="frequency")
        widths = output[0]

        assert len(output) == 4
        assert len(widths.time) == ntime
        # assert peaks.ndim == widths.ndim
        print(peaks, widths)
        print(bandwidth)
        assert (bandwidth == widths).all()

        assert "peak" in widths.dims
        assert "peak" in widths.coords
        assert "time" in widths.coords
        assert "time" in widths.coords


class TestFFT:
    @staticmethod
    def check_resolution(df, fs, n, relerror=0.1):
        return pytest.approx(df, relerror) == fs / n

    @pytest.fixture(params=[None, 99, 100, 101, 102])
    def n(self, request):
        """FFT size"""
        return request.param

    @pytest.fixture
    def signal(self):
        duration = 10.0
        fs = 8000.0
        nsamples = int(fs * duration)
        f = 100.0
        A = 2.0
        dim = "time"
        signal = A * np.sin(2.0 * np.pi * f * np.arange(nsamples) / fs)
        signal = xr.DataArray(
            signal, dims=[dim], coords={dim: np.arange(nsamples) / fs}
        ).expand_dims("channel")
        return signal

    def test_fft(self, signal, dask, n):
        dim = "time"
        newdim = "frequency"
        if dask:
            signal = signal.chunk({"channel": 1})
        newspectrum = fft(signal, n=n, dim=dim, newdim=newdim)
        assert dim not in newspectrum.dims
        assert newdim in newspectrum.dims
        if n is None:
            n = _get_length(signal, "time")
        assert _get_length(newspectrum, "frequency") == n
        self.check_resolution(
            _compute_delta(newspectrum, newdim), 1.0 / _compute_delta(signal, dim), n
        )

    def test_ifft(self, signal, dask, n):
        dim = "time"
        newdim = "frequency"
        if dask:
            signal = signal.chunk({"channel": 1})
        newspectrum = fft(signal, n=n, dim=dim, newdim=newdim)

        assert _get_length(newspectrum, "frequency") == n or _get_length(signal, "time")
        self.check_resolution(
            _compute_delta(newspectrum, newdim),
            1.0 / _compute_delta(signal, dim),
            n or _get_length(signal, "time"),
        )

        newsignal = ifft(newspectrum, n=n, dim=newdim, newdim=dim)
        assert dim in newsignal.dims
        assert newdim not in newsignal.dims
        if n is None:
            n = _get_length(signal, "time")
        assert _get_length(newsignal, "time") == n
        self.check_resolution(
            _compute_delta(newspectrum, newdim), 1.0 / _compute_delta(newsignal, dim), n
        )

    def test_ifft_full(self, signal, dask):
        n = None
        dim = "time"
        newdim = "frequency"
        if dask:
            signal = signal.chunk({"channel": 1})
        newspectrum = fft(signal, n=n, dim=dim, newdim=newdim)
        newsignal = ifft(newspectrum, n=n, dim=newdim, newdim=dim)
        assert dim in newsignal.dims
        assert newdim not in newsignal.dims
        assert _get_length(signal, "time") == _get_length(newsignal, "time")
        # Input and output signal shall have the same time resolution
        assert pytest.approx(_compute_delta(signal, "time"), 0.0001) == _compute_delta(
            newsignal, "time"
        )

        self.check_resolution(
            _compute_delta(newspectrum, newdim),
            1.0 / _compute_delta(signal, dim),
            _get_length(signal, "time"),
        )
        self.check_resolution(
            _compute_delta(newspectrum, newdim),
            1.0 / _compute_delta(signal, dim),
            _get_length(signal, "time"),
        )

    def test_fft__dask_raises_main_axis(self, signal):
        """For the FFT chunking along the main axis is not permitted. Dask will raise."""
        n = 100
        dim = "time"
        newdim = "frequency"
        signal = signal.chunk({"time": 50})
        with pytest.raises(ValueError):
            result = fft(signal, n=n, dim=dim, newdim=newdim)

    def test_rfft(self, signal, dask, n):
        dim = "time"
        newdim = "frequency"
        if dask:
            signal = signal.chunk({"channel": 1})
        newspectrum = rfft(signal, n=n, dim=dim, newdim=newdim)
        assert dim not in newspectrum.dims
        assert newdim in newspectrum.dims
        if n is None:
            n = _get_length(signal, "time")
        assert _get_length(newspectrum, "frequency") == n // 2 + 1
        self.check_resolution(
            _compute_delta(newspectrum, newdim), 1.0 / _compute_delta(signal, dim), n
        )

    def test_irfft(self, signal, dask, n):
        dim = "time"
        newdim = "frequency"
        if dask:
            signal = signal.chunk({"channel": 1})
        newspectrum = rfft(signal, dim=dim, newdim=newdim)
        newsignal = irfft(newspectrum, n=n, dim=newdim, newdim=dim)
        assert dim in newsignal.dims
        assert newdim not in newsignal.dims
        if n is None:
            n = (_get_length(newspectrum, "frequency") - 1) * 2
        assert _get_length(newsignal, "time") == n

        self.check_resolution(
            _compute_delta(newspectrum, newdim), 1.0 / _compute_delta(signal, dim), n
        )
        self.check_resolution(
            _compute_delta(newspectrum, newdim), 1.0 / _compute_delta(newsignal, dim), n
        )
        assert (
            pytest.approx(_compute_delta(signal, dim), 0.1)
            == _compute_delta(newsignal, dim) * n
        )

    def test_hfft(self, signal, dask, n):
        dim = "time"
        newdim = "frequency"
        if dask:
            signal = signal.chunk({"channel": 1})
        newspectrum = hfft(signal, n=n, dim=dim, newdim=newdim)
        assert dim not in newspectrum.dims
        assert newdim in newspectrum.dims
        if n is None:
            n = _get_length(signal, "time")
            n = (n - 1) * 2
        assert _get_length(newspectrum, "frequency") == n
        self.check_resolution(
            _compute_delta(newspectrum, newdim), 1.0 / _compute_delta(signal, dim), n
        )

    def test_ihfft(self, signal, dask, n):
        dim = "time"
        newdim = "frequency"
        if dask:
            signal = signal.chunk({"channel": 1})
        newspectrum = hfft(signal, dim=dim, newdim=newdim)

        # Check resolution when going from time to frequency
        self.check_resolution(
            _compute_delta(newspectrum, newdim),
            1.0 / _compute_delta(signal, dim),
            _get_length(signal, dim),
        )

        newsignal = ihfft(newspectrum, n=n, dim=newdim, newdim=dim)
        assert dim in newsignal.dims
        assert newdim not in newsignal.dims

        if n is None:
            n = _get_length(signal, dim)
            n = (n - 1) * 2

        # Check resolution when going from time to frequency
        self.check_resolution(
            _compute_delta(newspectrum, newdim), 1.0 / _compute_delta(signal, dim), n
        )
        self.check_resolution(
            _compute_delta(newspectrum, newdim), 1.0 / _compute_delta(newsignal, dim), n
        )


class TestFFTShift:
    @pytest.fixture
    def signal(self):
        duration = 10.0
        fs = 8000.0
        nsamples = int(fs * duration)
        f = 100.0
        A = 2.0
        dim = "time"
        signal = A * np.sin(2.0 * np.pi * f * np.arange(nsamples) / fs)
        signal = xr.DataArray(
            signal, dims=[dim], coords={dim: np.arange(nsamples)}
        ).expand_dims("channel")
        return signal

    @pytest.fixture
    def spectrum(self, signal):
        return fft(signal, dim="time", newdim="frequency")

    def test_fftshift(self, spectrum):
        shifted = fftshift(spectrum, dims=["frequency"])
        assert "frequency" in shifted.dims

    def test_ifftshift(self, spectrum):
        shifted = ifftshift(spectrum, dims=["frequency"])
        assert "frequency" in shifted.dims


class TestHilbert:
    @pytest.fixture
    def amplitude(self):
        return 2.0

    @pytest.fixture
    def dim(self):
        return "time"

    @pytest.fixture
    def signal(self, dim, amplitude):
        duration = 10.0
        fs = 8000.0
        nsamples = int(fs * duration)
        f = 100.0
        signal = amplitude * np.sin(2.0 * np.pi * f * np.arange(nsamples) / fs)
        signal = xr.DataArray(signal, dims=[dim], coords={dim: np.arange(nsamples)})
        return signal

    def test_hilbert(self, signal, amplitude, dim):
        result = hilbert(signal, dim=dim)
        assert np.allclose(np.abs(result).data, np.ones_like(signal) * amplitude)

    def test_hilbert__N(self, signal, dim):
        N = 100
        result = hilbert(signal, N=N, dim=dim)
        assert len(result) == N
        assert dim in result.coords

    def test_hilbert__N_without_coords(self, signal, dim):
        del signal[dim]
        assert dim not in signal.coords
        N = 100
        result = hilbert(signal, N=N, dim=dim)
        assert len(result) == N
        assert dim not in result.coords
