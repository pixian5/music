"""
Audio I/O utilities for loading and saving audio files.
"""

import numpy as np
import soundfile as sf
import librosa


def load_audio(filepath: str, sr: int = None, mono: bool = False):
    """
    Load an audio file.

    Parameters
    ----------
    filepath : str
        Path to the audio file (WAV, FLAC, OGG, MP3, etc.).
    sr : int or None
        Target sample rate. If None, the native sample rate is used.
        If an integer, the audio is resampled to this rate.
    mono : bool
        If True, convert to mono. Default is False (preserve channels).

    Returns
    -------
    audio : np.ndarray
        Audio data with shape (samples,) for mono or (samples, channels)
        for multi-channel audio.
    sample_rate : int
        Sample rate of the returned audio.
    """
    if filepath.lower().endswith(".mp3"):
        # librosa handles MP3 via audioread
        audio, native_sr = librosa.load(filepath, sr=sr, mono=mono)
        sample_rate = sr if sr is not None else native_sr
        if not mono and audio.ndim == 1:
            audio = audio[:, np.newaxis]
        return audio, sample_rate

    audio, native_sr = sf.read(filepath, always_2d=True)
    # audio shape: (samples, channels)
    if mono:
        audio = audio.mean(axis=1)
    else:
        audio = audio.squeeze() if audio.shape[1] == 1 else audio

    if sr is not None and sr != native_sr:
        audio = _resample(audio, native_sr, sr)
        sample_rate = sr
    else:
        sample_rate = native_sr

    return audio, sample_rate


def save_audio(filepath: str, audio: np.ndarray, sample_rate: int):
    """
    Save audio data to a file.

    Parameters
    ----------
    filepath : str
        Output file path. The format is inferred from the extension
        (e.g., .wav, .flac, .ogg).
    audio : np.ndarray
        Audio data with shape (samples,) or (samples, channels).
    sample_rate : int
        Sample rate of the audio.
    """
    if audio.ndim == 1:
        audio_out = audio[:, np.newaxis]
    else:
        audio_out = audio

    # Clip to prevent clipping artifacts
    audio_out = np.clip(audio_out, -1.0, 1.0)
    sf.write(filepath, audio_out, sample_rate)


def get_segment(audio: np.ndarray, sample_rate: int,
                start_sec: float, end_sec: float) -> np.ndarray:
    """
    Extract a time segment from audio.

    Parameters
    ----------
    audio : np.ndarray
        Audio data with shape (samples,) or (samples, channels).
    sample_rate : int
        Sample rate.
    start_sec : float
        Start time in seconds.
    end_sec : float
        End time in seconds.

    Returns
    -------
    np.ndarray
        Audio segment.
    """
    start_sample = int(start_sec * sample_rate)
    end_sample = int(end_sec * sample_rate)
    if audio.ndim == 1:
        return audio[start_sample:end_sample]
    return audio[start_sample:end_sample, :]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to a new sample rate."""
    if audio.ndim == 1:
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    # Multi-channel: resample each channel
    channels = [
        librosa.resample(audio[:, ch], orig_sr=orig_sr, target_sr=target_sr)
        for ch in range(audio.shape[1])
    ]
    return np.stack(channels, axis=1)
