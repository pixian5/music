"""
Noise reduction module.

Provides two modes:
1. Automatic noise detection – estimates a noise profile from the
   quietest portions of the audio.
2. User-defined noise profile – the user selects a segment that
   contains only ambient noise; that segment is used to build the
   noise profile which is then subtracted from the whole recording.

The implementation uses spectral subtraction / Wiener filtering via
the `noisereduce` library, supplemented by a custom spectral-gate
fallback so the module works even if noisereduce is unavailable.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import stft, istft

try:
    import noisereduce as nr

    _NOISEREDUCE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _NOISEREDUCE_AVAILABLE = False


class NoiseReducer:
    """
    Ambient-noise remover for audio signals.

    Parameters
    ----------
    sample_rate : int
        Sample rate of the audio that will be processed.
    prop_decrease : float
        Proportion by which the noise is decreased (0–1).
        1.0 = full removal, 0.5 = half removed. Default 1.0.
    n_fft : int
        FFT window size used for spectral analysis. Default 2048.
    hop_length : int or None
        Hop length for STFT. Default n_fft // 4.
    """

    def __init__(
        self,
        sample_rate: int,
        prop_decrease: float = 1.0,
        n_fft: int = 2048,
        hop_length: int | None = None,
    ):
        self.sample_rate = sample_rate
        self.prop_decrease = float(np.clip(prop_decrease, 0.0, 1.0))
        self.n_fft = n_fft
        self.hop_length = hop_length if hop_length is not None else n_fft // 4
        self._noise_profile: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_noise_profile_from_segment(
        self, audio: np.ndarray, start_sec: float, end_sec: float
    ) -> None:
        """
        Build a noise profile from a user-selected time segment.

        Parameters
        ----------
        audio : np.ndarray
            Full audio signal (mono, shape (samples,)).
        start_sec : float
            Start of the noise-only segment in seconds.
        end_sec : float
            End of the noise-only segment in seconds.
        """
        start = int(start_sec * self.sample_rate)
        end = int(end_sec * self.sample_rate)
        segment = _to_mono(audio)[start:end]
        if len(segment) == 0:
            raise ValueError(
                f"Noise segment [{start_sec}s, {end_sec}s] produced an empty array."
            )
        self._noise_profile = self._compute_noise_profile(segment)

    def set_noise_profile_from_array(self, noise_segment: np.ndarray) -> None:
        """
        Build a noise profile from an externally supplied array.

        Parameters
        ----------
        noise_segment : np.ndarray
            1-D array containing only ambient noise (same sample rate as
            the audio that will be processed).
        """
        segment = _to_mono(noise_segment)
        self._noise_profile = self._compute_noise_profile(segment)

    def detect_and_set_noise_profile(self, audio: np.ndarray) -> None:
        """
        Automatically detect ambient noise by analysing the quietest
        frames of the audio and use them as the noise profile.

        Parameters
        ----------
        audio : np.ndarray
            Audio signal (mono or stereo).
        """
        mono = _to_mono(audio)
        profile_segment = self._auto_detect_noise_segment(mono)
        self._noise_profile = self._compute_noise_profile(profile_segment)

    def reduce(self, audio: np.ndarray) -> np.ndarray:
        """
        Remove noise from an audio signal.

        If no noise profile has been set, :meth:`detect_and_set_noise_profile`
        is called automatically.

        Parameters
        ----------
        audio : np.ndarray
            Audio signal with shape (samples,) or (samples, channels).

        Returns
        -------
        np.ndarray
            De-noised audio with the same shape as the input.
        """
        stereo_input = audio.ndim == 2
        if stereo_input:
            channels = [self.reduce(audio[:, ch]) for ch in range(audio.shape[1])]
            return np.stack(channels, axis=1)

        mono = audio.astype(np.float32)
        if self._noise_profile is None:
            self.detect_and_set_noise_profile(mono)

        if _NOISEREDUCE_AVAILABLE:
            return self._reduce_noisereduce(mono)
        return self._reduce_spectral_gate(mono)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_noise_profile(self, segment: np.ndarray) -> np.ndarray:
        """Compute mean noise magnitude spectrum from a segment."""
        _, _, Zxx = stft(
            segment,
            fs=self.sample_rate,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
        )
        return np.mean(np.abs(Zxx), axis=1)  # shape: (n_fft//2+1,)

    def _auto_detect_noise_segment(self, mono: np.ndarray) -> np.ndarray:
        """
        Find the quietest 0.5-second window in the audio and return it
        as the noise reference segment.
        """
        window = max(int(0.5 * self.sample_rate), self.n_fft)
        hop = window // 2
        energies = []
        for start in range(0, len(mono) - window, hop):
            chunk = mono[start: start + window]
            energies.append((np.mean(chunk ** 2), start))

        if not energies:
            return mono[: window]

        _, best_start = min(energies, key=lambda x: x[0])
        return mono[best_start: best_start + window]

    def _reduce_noisereduce(self, audio: np.ndarray) -> np.ndarray:
        """Use the noisereduce library (stationary mode)."""
        # Reconstruct a short noise clip from our profile via inverse FFT
        noise_clip = self._profile_to_clip()
        return nr.reduce_noise(
            y=audio,
            y_noise=noise_clip,
            sr=self.sample_rate,
            prop_decrease=self.prop_decrease,
            stationary=True,
        ).astype(np.float32)

    def _profile_to_clip(self) -> np.ndarray:
        """
        Synthesise a short white-noise clip shaped by the stored noise
        profile so it can be passed to noisereduce.
        """
        rng = np.random.default_rng(0)
        n_frames = 8
        profile = self._noise_profile  # (n_fft//2+1,)
        # Build a complex spectrum: magnitude from profile, random phase
        n_bins = len(profile)
        phase = np.exp(
            1j * rng.uniform(0, 2 * np.pi, (n_bins, n_frames))
        )
        Zxx = profile[:, np.newaxis] * phase
        _, clip = istft(
            Zxx,
            fs=self.sample_rate,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
        )
        return clip.astype(np.float32)

    def _reduce_spectral_gate(self, audio: np.ndarray) -> np.ndarray:
        """
        Fallback spectral-subtraction / Wiener-filter implementation that
        does not depend on the noisereduce package.
        """
        freqs, times, Zxx = stft(
            audio,
            fs=self.sample_rate,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
        )
        mag = np.abs(Zxx)
        phase = np.angle(Zxx)

        noise = self._noise_profile[:, np.newaxis]  # broadcast over time

        # Wiener-style gain
        noise_reduced = mag - self.prop_decrease * noise
        noise_reduced = np.maximum(noise_reduced, 0.0)

        Zxx_clean = noise_reduced * np.exp(1j * phase)
        _, audio_clean = istft(
            Zxx_clean,
            fs=self.sample_rate,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
        )
        # Trim/pad to original length
        n = len(audio)
        if len(audio_clean) >= n:
            return audio_clean[:n].astype(np.float32)
        return np.pad(audio_clean, (0, n - len(audio_clean))).astype(np.float32)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert to mono float32 if necessary."""
    if audio.ndim == 2:
        return audio.mean(axis=1).astype(np.float32)
    return audio.astype(np.float32)
