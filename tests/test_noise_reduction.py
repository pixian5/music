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


def _make_breathy_noise(duration=1.0, sr=SR, amplitude=0.03, seed=123) -> np.ndarray:
    """
    Create a breath-like noise (high-frequency dominant and unpitched).
    """
    rng = np.random.default_rng(seed)
    n = int(duration * sr)
    white = rng.standard_normal(n).astype(np.float32)
    # Simple high-pass by first-order difference.
    # Prepend first sample so diff-based signal keeps original length n.
    breath = np.concatenate(([white[0]], np.diff(white))).astype(np.float32)
    breath /= np.max(np.abs(breath)) + 1e-8
    return breath * amplitude


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

    def test_suppress_breath_sounds_reduces_breath_like_segment(self):
        reducer = NoiseReducer(SR, breath_reduce_strength=0.5)
        base = _make_sine(freq=220.0, duration=1.0) * 0.06
        breath = np.zeros_like(base)
        start, end = int(0.55 * SR), int(0.75 * SR)
        breath[start:end] = _make_breathy_noise(duration=(end - start) / SR, amplitude=0.08)
        mixed = base + breath

        cleaned = reducer._suppress_breath_sounds(mixed)

        # Use diff energy as high-frequency proxy where breath dominates.
        before_hf = np.mean(np.diff(mixed[start:end]) ** 2)
        after_hf = np.mean(np.diff(cleaned[start:end]) ** 2)
        assert after_hf < before_hf * 0.93

    def test_suppress_breath_sounds_preserves_soft_voiced_signal(self):
        reducer = NoiseReducer(SR, breath_reduce_strength=0.5)
        soft_voice = (_make_sine(freq=220.0, duration=1.0) * 0.04).astype(np.float32)

        cleaned = reducer._suppress_breath_sounds(soft_voice)

        before_rms = float(np.sqrt(np.mean(soft_voice ** 2)))
        after_rms = float(np.sqrt(np.mean(cleaned ** 2)))
        assert after_rms > before_rms * 0.92

    def test_reduce_can_disable_breath_suppression(self):
        reducer = NoiseReducer(SR, breath_reduce_strength=0.7)
        base = _make_sine(freq=220.0, duration=1.0) * 0.06
        breath = np.zeros_like(base)
        start, end = int(0.55 * SR), int(0.75 * SR)
        breath[start:end] = _make_breathy_noise(duration=(end - start) / SR, amplitude=0.08)
        mixed = base + breath
        reducer.set_noise_profile_from_array(np.zeros(SR // 4, dtype=np.float32))

        with_breath = reducer.reduce(mixed, apply_breath_suppression=True)
        without_breath = reducer.reduce(mixed, apply_breath_suppression=False)

        with_hf = np.mean(np.diff(with_breath[start:end]) ** 2)
        without_hf = np.mean(np.diff(without_breath[start:end]) ** 2)
        assert with_hf < without_hf * 0.95
