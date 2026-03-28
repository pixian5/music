"""Tests for the audio I/O module."""

import importlib
import sys
import numpy as np
import pytest

from music_editor.audio_io import save_audio


def test_audio_io_import_does_not_eager_import_librosa():
    """
    audio_io should not import librosa at module import time to avoid
    GUI startup blocking on platforms where librosa import is slow.
    """
    old_audio_io = sys.modules.get("music_editor.audio_io")
    old_librosa = sys.modules.get("librosa")
    sys.modules.pop("music_editor.audio_io", None)
    sys.modules.pop("librosa", None)

    try:
        importlib.import_module("music_editor.audio_io")
        assert "librosa" not in sys.modules
    finally:
        sys.modules.pop("music_editor.audio_io", None)
        sys.modules.pop("librosa", None)
        if old_audio_io is not None:
            sys.modules["music_editor.audio_io"] = old_audio_io
        if old_librosa is not None:
            sys.modules["librosa"] = old_librosa


def test_save_audio_rejects_unsupported_formats():
    """
    save_audio should raise ValueError for unsupported output formats
    like M4A, MP3, AAC, WMA, etc.
    """
    audio = np.array([0.1, 0.2, 0.3])
    sample_rate = 44100

    unsupported_formats = [
        'output.m4a',
        'output.mp3',
        'output.aac',
        'output.wma',
        'output.mp4',
        'OUTPUT.M4A',  # Test case-insensitive
        '/path/to/鸳鸯戏_output.m4a',  # Test with Chinese characters
    ]

    for filepath in unsupported_formats:
        with pytest.raises(ValueError) as exc_info:
            save_audio(filepath, audio, sample_rate)

        error_message = str(exc_info.value)
        assert "Cannot save to format" in error_message
        assert "Supported output formats are: WAV, FLAC, OGG" in error_message


def test_save_audio_accepts_supported_formats():
    """
    save_audio should accept WAV, FLAC, OGG formats without error.
    This test only checks that the validation passes, not actual file writing.
    """
    import tempfile
    import os

    audio = np.array([0.1, 0.2, 0.3])
    sample_rate = 44100

    supported_formats = ['wav', 'flac', 'ogg']

    for fmt in supported_formats:
        with tempfile.NamedTemporaryFile(suffix=f'.{fmt}', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Should not raise ValueError
            save_audio(tmp_path, audio, sample_rate)
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
