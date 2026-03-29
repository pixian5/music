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
        assert reducer.breath_suppression == 0.75
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

    def test_breath_suppression_reduces_high_band_more(self):
        # Synthetic "breath-like" segment: strong high-frequency hiss + low tone.
        t = np.linspace(0, 1.0, SR, endpoint=False)
        tone = 0.08 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
        rng = np.random.default_rng(7)
        hiss = (rng.standard_normal(SR).astype(np.float32) * 0.08)
        hiss = hiss - np.convolve(hiss, np.ones(9) / 9.0, mode="same")  # emphasize HF
        audio = (tone + hiss).astype(np.float32)

        reducer = NoiseReducer(SR, breath_suppression=0.8)
        reducer.set_noise_profile_from_array(hiss[: SR // 4])
        out = reducer.reduce(audio)

        # Compare HF/LF energy ratio before vs after.
        def _band_ratio(x: np.ndarray) -> float:
            spec = np.fft.rfft(x)
            freqs = np.fft.rfftfreq(len(x), d=1.0 / SR)
            hi = np.mean(np.abs(spec[freqs >= 3500]))
            lo = np.mean(np.abs(spec[(freqs >= 120) & (freqs <= 1200)])) + 1e-10
            return float(hi / lo)

        assert _band_ratio(out) < _band_ratio(audio) * 0.85

    def test_public_suppress_breath_sounds_api(self):
        reducer = NoiseReducer(SR, breath_suppression=0.6)
        x = _make_sine(freq=220.0, duration=0.2) + _make_white_noise(0.03, duration=0.2)
        y = reducer.suppress_breath_sounds(x)
        assert y.shape == x.shape

    def test_breath_strength_parameter_changes_result(self):
        rng = np.random.default_rng(9)
        hiss = (rng.standard_normal(SR).astype(np.float32) * 0.06)
        hiss = hiss - np.convolve(hiss, np.ones(7) / 7.0, mode="same")
        t = np.linspace(0, 1.0, SR, endpoint=False)
        tone = 0.08 * np.sin(2 * np.pi * 240 * t).astype(np.float32)
        audio = (tone + hiss).astype(np.float32)

        weak = NoiseReducer(
            SR,
            breath_suppression=1.0,
            breath_reduce_strength=0.2,
            breath_method="deep",
            breath_sensitivity=0.8,
            breath_band_focus=0.85,
        ).suppress_breath_sounds(audio)
        strong = NoiseReducer(
            SR,
            breath_suppression=1.0,
            breath_reduce_strength=0.9,
            breath_method="deep",
            breath_sensitivity=0.8,
            breath_band_focus=0.85,
        ).suppress_breath_sounds(audio)

        weak_hi = np.mean(np.abs(np.fft.rfft(weak))[3500:])
        strong_hi = np.mean(np.abs(np.fft.rfft(strong))[3500:])
        assert strong_hi < weak_hi * 0.85

    def test_ultra_method_reduces_quiet_inhale_more_than_deep(self):
        rng = np.random.default_rng(11)
        n = SR
        t = np.linspace(0, 1.0, n, endpoint=False)
        tone = 0.22 * np.sin(2 * np.pi * 210 * t).astype(np.float32)
        breath = (rng.standard_normal(n).astype(np.float32) * 0.022)
        breath = breath - np.convolve(breath, np.ones(9) / 9.0, mode="same")
        audio = (tone + breath).astype(np.float32)

        deep = NoiseReducer(
            SR,
            breath_suppression=1.0,
            breath_reduce_strength=0.85,
            breath_method="deep",
            breath_sensitivity=0.85,
            breath_band_focus=0.9,
        ).suppress_breath_sounds(audio)
        ultra = NoiseReducer(
            SR,
            breath_suppression=1.0,
            breath_reduce_strength=0.85,
            breath_method="ultra",
            breath_sensitivity=0.85,
            breath_band_focus=0.9,
        ).suppress_breath_sounds(audio)

        deep_hi = np.mean(np.abs(np.fft.rfft(deep))[3000:])
        ultra_hi = np.mean(np.abs(np.fft.rfft(ultra))[3000:])
        assert ultra_hi < deep_hi * 0.9

    def test_extreme_method_reduces_residual_hiss_more_than_ultra(self):
        rng = np.random.default_rng(21)
        n = SR
        t = np.linspace(0, 1.0, n, endpoint=False)
        tone = 0.2 * np.sin(2 * np.pi * 200 * t).astype(np.float32)
        breath = (rng.standard_normal(n).astype(np.float32) * 0.028)
        breath = breath - np.convolve(breath, np.ones(11) / 11.0, mode="same")
        audio = (tone + breath).astype(np.float32)

        ultra = NoiseReducer(
            SR,
            breath_suppression=1.0,
            breath_reduce_strength=0.92,
            breath_method="ultra",
            breath_sensitivity=0.9,
            breath_band_focus=0.92,
        ).suppress_breath_sounds(audio)
        extreme = NoiseReducer(
            SR,
            breath_suppression=1.0,
            breath_reduce_strength=0.92,
            breath_method="extreme",
            breath_sensitivity=0.9,
            breath_band_focus=0.92,
        ).suppress_breath_sounds(audio)

        ultra_hi = np.mean(np.abs(np.fft.rfft(ultra))[3000:])
        extreme_hi = np.mean(np.abs(np.fft.rfft(extreme))[3000:])
        assert extreme_hi < ultra_hi * 0.93

    def test_extreme_lowers_inhale_segment_volume(self):
        rng = np.random.default_rng(31)
        n = SR * 2
        t = np.linspace(0, 2.0, n, endpoint=False)
        voice = 0.14 * np.sin(2 * np.pi * 220 * t).astype(np.float32)

        # Build 2 strong inhale-like sections.
        hiss = (rng.standard_normal(n).astype(np.float32) * 0.018)
        hiss = hiss - np.convolve(hiss, np.ones(9) / 9.0, mode="same")
        mask = np.zeros(n, dtype=np.float32)
        mask[int(0.45 * SR): int(0.70 * SR)] = 1.0
        mask[int(1.20 * SR): int(1.45 * SR)] = 1.0
        audio = (voice + hiss * mask * 2.6).astype(np.float32)

        out = NoiseReducer(
            SR,
            breath_suppression=1.0,
            breath_reduce_strength=0.95,
            breath_method="extreme",
            breath_sensitivity=0.9,
            breath_band_focus=0.95,
        ).suppress_breath_sounds(audio)

        # Evaluate high-frequency component in inhale regions (where breath lives).
        hp_in = np.concatenate(([audio[0]], np.diff(audio))).astype(np.float32)
        hp_out = np.concatenate(([out[0]], np.diff(out))).astype(np.float32)
        inhale_rms_in = np.sqrt(np.mean((hp_in[mask > 0]) ** 2))
        inhale_rms_out = np.sqrt(np.mean((hp_out[mask > 0]) ** 2))
        assert inhale_rms_out < inhale_rms_in * 0.52

        inhale_peak_in = np.max(np.abs(audio[mask > 0]))
        inhale_peak_out = np.max(np.abs(out[mask > 0]))
        assert inhale_peak_out < inhale_peak_in * 0.45
