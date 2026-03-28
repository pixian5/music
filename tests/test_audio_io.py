"""Tests for the audio I/O module."""

import importlib
import sys


def test_audio_io_import_does_not_eager_import_librosa():
    """
    audio_io should not import librosa at module import time to avoid
    GUI startup blocking on platforms where librosa import is slow.
    """
    sys.modules.pop("music_editor.audio_io", None)
    sys.modules.pop("librosa", None)

    importlib.import_module("music_editor.audio_io")

    assert "librosa" not in sys.modules
