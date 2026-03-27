"""Tests for the noise reduction module."""

import numpy as np
import pytest

from music_editor.noise_reduction import NoiseReducer


SR = 16000  # 16 kHz for fast tests


def _make_sine(freq=440.0, duration=1.0, sr=SR) -> np.ndarray:
    """Generate a pure sine wave."""
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def _make_white_noise(amplitude=0.05, duration=1.0, sr=SR, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(duration * sr)) * amplitude).astype(np.float32)


class TestNoiseReducer:
    def test_init_defaults(self):
        reducer = NoiseReducer(SR)
        assert reducer.sample_rate == SR
        assert reducer.prop_decrease == 1.0
        assert reducer.n_fft == 2048
        assert reducer._noise_profile is None

    def test_set_noise_profile_from_segment(self):
        reducer = NoiseReducer(SR)
        audio = _make_white_noise()
        reducer.set_noise_profile_from_segment(audio, 0.0, 0.5)
        assert reducer._noise_profile is not None
        assert reducer._noise_profile.shape[0] == reducer.n_fft // 2 + 1

    def test_set_noise_profile_from_array(self):
        reducer = NoiseReducer(SR)
        noise = _make_white_noise()
        reducer.set_noise_profile_from_array(noise)
        assert reducer._noise_profile is not None

    def test_detect_and_set_noise_profile(self):
        reducer = NoiseReducer(SR)
        audio = _make_sine() + _make_white_noise()
        reducer.detect_and_set_noise_profile(audio)
        assert reducer._noise_profile is not None

    def test_reduce_mono_returns_same_length(self):
        reducer = NoiseReducer(SR)
        signal = _make_sine() + _make_white_noise()
        result = reducer.reduce(signal)
        assert result.shape == signal.shape

    def test_reduce_stereo_returns_same_shape(self):
        reducer = NoiseReducer(SR)
        mono = _make_sine() + _make_white_noise()
        stereo = np.stack([mono, mono], axis=1)
        result = reducer.reduce(stereo)
        assert result.shape == stereo.shape

    def test_reduce_decreases_noise(self):
        """
        After noise reduction the energy of a noise-only signal should
        decrease (we can't guarantee silence, but the power must go down).
        """
        rng = np.random.default_rng(42)
        noise = (rng.standard_normal(SR) * 0.1).astype(np.float32)
        reducer = NoiseReducer(SR)
        reducer.set_noise_profile_from_array(noise)
        result = reducer.reduce(noise)
        assert np.mean(result ** 2) < np.mean(noise ** 2)

    def test_set_noise_profile_empty_segment_raises(self):
        reducer = NoiseReducer(SR)
        audio = _make_white_noise(duration=1.0)
        with pytest.raises(ValueError):
            reducer.set_noise_profile_from_segment(audio, 5.0, 6.0)  # out of range

    def test_prop_decrease_clipped(self):
        reducer = NoiseReducer(SR, prop_decrease=1.5)
        assert reducer.prop_decrease == 1.0
        reducer2 = NoiseReducer(SR, prop_decrease=-0.5)
        assert reducer2.prop_decrease == 0.0
