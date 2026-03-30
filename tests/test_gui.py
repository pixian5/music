"""Tests for GUI utility behavior."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("tkinter")

from gui import (
    MusicEditorApp,
    OUTPUT_FORMATS,
    _SPECTROGRAM_MAX_SAMPLES,
    _frame_mask_to_segments,
    _prepare_spectrogram_signal,
    _replace_extension,
    _suggest_output_path,
)


class _DummyVar:
    def __init__(self, value=""):
        self._value = value

    def set(self, value):
        self._value = value

    def get(self):
        return self._value


def test_replace_extension():
    assert _replace_extension("/tmp/a/b_output.mp3", "wav") == "/tmp/a/b_output.wav"


def test_suggest_output_path_uses_selected_format():
    assert _suggest_output_path("/tmp/song.m4a", "flac") == "/tmp/song_output.flac"


def test_browse_input_sets_output_with_selected_format_and_auto_load(monkeypatch):
    load_calls = []
    fake_app = SimpleNamespace(
        _input_path=_DummyVar(),
        _output_path=_DummyVar(),
        _output_format=_DummyVar("ogg"),
        _load=lambda: load_calls.append("called"),
    )

    monkeypatch.setattr(
        "gui.filedialog.askopenfilename",
        lambda **_kwargs: "/tmp/demo/voice.m4a",
    )

    MusicEditorApp._browse_input(fake_app)

    assert fake_app._input_path.get() == "/tmp/demo/voice.m4a"
    assert fake_app._output_path.get() == "/tmp/demo/voice_output.ogg"
    assert load_calls == ["called"]


def test_output_format_selection_updates_output_extension():
    fake_app = SimpleNamespace(
        _output_path=_DummyVar("/tmp/demo/out.flac"),
        _output_format=_DummyVar("wav"),
    )

    MusicEditorApp._on_output_format_selected(fake_app)

    assert fake_app._output_path.get() == "/tmp/demo/out.wav"


def test_output_formats_include_mp3():
    assert "mp3" in OUTPUT_FORMATS


def test_suggest_output_path_defaults_to_mp3_extension():
    assert _suggest_output_path("/tmp/song.wav", "mp3") == "/tmp/song_output.mp3"


def test_frame_mask_to_segments():
    mask = np.array([False, True, True, False, True, False], dtype=bool)
    segments = _frame_mask_to_segments(mask, hop_length=100, sample_rate=1000, total_samples=800)
    assert segments == [(0.1, 0.3), (0.4, 0.5)]


def test_frame_mask_to_segments_empty():
    mask = np.array([], dtype=bool)
    assert _frame_mask_to_segments(mask, hop_length=100, sample_rate=1000, total_samples=800) == []


def test_frame_mask_to_segments_clamps_total_samples():
    mask = np.array([False, False, True, True, True], dtype=bool)
    segments = _frame_mask_to_segments(mask, hop_length=300, sample_rate=1000, total_samples=1000)
    assert segments == [(0.6, 1.0)]


def test_frame_mask_to_segments_exact_total_samples_boundary():
    mask = np.array([False, True, True], dtype=bool)
    segments = _frame_mask_to_segments(mask, hop_length=250, sample_rate=1000, total_samples=750)
    assert segments == [(0.25, 0.75)]


def test_prepare_spectrogram_signal_keeps_short_signal():
    signal = np.arange(1000, dtype=np.float32)
    plot_signal, plot_sr = _prepare_spectrogram_signal(signal, sample_rate=48000)
    assert np.array_equal(plot_signal, signal)
    assert plot_sr == 48000.0


def test_prepare_spectrogram_signal_downsamples_long_signal():
    signal = np.arange(_SPECTROGRAM_MAX_SAMPLES + 1, dtype=np.float32)
    plot_signal, plot_sr = _prepare_spectrogram_signal(signal, sample_rate=48000)
    assert len(plot_signal) <= _SPECTROGRAM_MAX_SAMPLES
    assert plot_signal[0] == signal[0]
    assert plot_signal[1] == signal[2]
    assert plot_sr == pytest.approx(24000.0)


def test_prepare_spectrogram_signal_downsamples_with_larger_stride():
    signal = np.arange(_SPECTROGRAM_MAX_SAMPLES * 5 + 1, dtype=np.float32)
    plot_signal, plot_sr = _prepare_spectrogram_signal(signal, sample_rate=48000)
    assert len(plot_signal) <= _SPECTROGRAM_MAX_SAMPLES
    assert plot_signal[0] == signal[0]
    assert plot_signal[1] == signal[6]
    assert plot_sr == pytest.approx(8000.0)
