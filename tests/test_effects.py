"""Tests for the audio effects module."""

import numpy as np
import pytest

from music_editor.effects import AudioEffects


SR = 16000


def _sine(freq=440.0, duration=0.5, sr=SR) -> np.ndarray:
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def _stereo(duration=0.5, sr=SR) -> np.ndarray:
    mono = _sine(duration=duration, sr=sr)
    return np.stack([mono, mono * 0.8], axis=1)


class TestNormalize:
    def test_normalize_output_shape_mono(self):
        fx = AudioEffects(SR)
        audio = _sine()
        result = fx.normalize(audio)
        assert result.shape == audio.shape

    def test_normalize_output_shape_stereo(self):
        fx = AudioEffects(SR)
        audio = _stereo()
        result = fx.normalize(audio)
        assert result.shape == audio.shape

    def test_normalize_clipped_to_one(self):
        fx = AudioEffects(SR)
        audio = _sine() * 10  # very loud
        result = fx.normalize(audio)
        assert np.max(np.abs(result)) <= 1.0

    def test_normalize_silence_unchanged(self):
        fx = AudioEffects(SR)
        silence = np.zeros(SR, dtype=np.float32)
        result = fx.normalize(silence)
        np.testing.assert_array_equal(result, silence)

    def test_dynamic_normalize_same_shape(self):
        fx = AudioEffects(SR)
        audio = _sine()
        result = fx.dynamic_normalize(audio)
        assert result.shape == audio.shape


class TestReverb:
    def test_studio_reverb_same_length_mono(self):
        fx = AudioEffects(SR)
        audio = _sine()
        result = fx.studio_reverb(audio)
        assert result.shape == audio.shape

    def test_studio_reverb_same_length_stereo(self):
        fx = AudioEffects(SR)
        audio = _stereo()
        result = fx.studio_reverb(audio)
        assert result.shape == audio.shape

    def test_studio_reverb_not_identical(self):
        fx = AudioEffects(SR)
        audio = _sine()
        result = fx.studio_reverb(audio, wet=0.5)
        # Result should differ from input
        assert not np.allclose(result, audio, atol=1e-3)

    def test_ktv_effect_same_length(self):
        fx = AudioEffects(SR)
        audio = _sine()
        result = fx.ktv_effect(audio)
        assert result.shape == audio.shape


class TestPitch:
    def test_pitch_shift_same_length_mono(self):
        fx = AudioEffects(SR)
        audio = _sine()
        result = fx.pitch_shift(audio, semitones=2.0)
        assert result.shape == audio.shape

    def test_pitch_shift_same_length_stereo(self):
        fx = AudioEffects(SR)
        audio = _stereo()
        result = fx.pitch_shift(audio, semitones=2.0)
        assert result.shape == audio.shape

    def test_male_to_female(self):
        fx = AudioEffects(SR)
        audio = _sine(freq=120.0)  # bass frequency
        result = fx.male_to_female(audio, semitones=5.0)
        assert result.shape == audio.shape

    def test_female_to_male(self):
        fx = AudioEffects(SR)
        audio = _sine(freq=250.0)
        result = fx.female_to_male(audio, semitones=5.0)
        assert result.shape == audio.shape

    def test_zero_shift_close_to_original(self):
        fx = AudioEffects(SR)
        audio = _sine()
        result = fx.pitch_shift(audio, semitones=0.0)
        assert result.shape == audio.shape


class TestStereo:
    def test_widen_mono_returns_stereo(self):
        fx = AudioEffects(SR)
        audio = _sine()
        result = fx.stereo_widen(audio, width=1.5)
        assert result.ndim == 2
        assert result.shape[1] == 2

    def test_widen_stereo_returns_stereo(self):
        fx = AudioEffects(SR)
        audio = _stereo()
        result = fx.stereo_widen(audio, width=2.0)
        assert result.shape == audio.shape

    def test_widen_clipped(self):
        fx = AudioEffects(SR)
        audio = _stereo()
        result = fx.stereo_widen(audio, width=10.0)
        assert np.max(np.abs(result)) <= 1.0


class TestEQ:
    def test_bass_boost_same_shape(self):
        fx = AudioEffects(SR)
        audio = _sine()
        result = fx.bass_boost(audio, gain_db=6.0)
        assert result.shape == audio.shape

    def test_treble_boost_same_shape(self):
        fx = AudioEffects(SR)
        audio = _sine()
        result = fx.treble_boost(audio, gain_db=6.0)
        assert result.shape == audio.shape

    def test_bass_boost_clipped(self):
        fx = AudioEffects(SR)
        audio = _sine() * 0.9
        result = fx.bass_boost(audio, gain_db=24.0)
        assert np.max(np.abs(result)) <= 1.0


class TestFade:
    def test_fade_in_starts_near_zero(self):
        fx = AudioEffects(SR)
        audio = _sine()
        result = fx.fade_in(audio, duration_sec=0.2)
        assert abs(result[0]) < 1e-6

    def test_fade_out_ends_near_zero(self):
        fx = AudioEffects(SR)
        audio = _sine()
        result = fx.fade_out(audio, duration_sec=0.2)
        assert abs(result[-1]) < 1e-6

    def test_fade_in_shape_preserved(self):
        fx = AudioEffects(SR)
        audio = _stereo()
        result = fx.fade_in(audio, duration_sec=0.1)
        assert result.shape == audio.shape

    def test_fade_out_shape_preserved(self):
        fx = AudioEffects(SR)
        audio = _stereo()
        result = fx.fade_out(audio, duration_sec=0.1)
        assert result.shape == audio.shape
