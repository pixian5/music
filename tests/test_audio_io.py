"""Tests for the audio I/O module."""

import importlib
import sys


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
