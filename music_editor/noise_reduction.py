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
from scipy.ndimage import uniform_filter1d

try:
    import noisereduce as nr

    _NOISEREDUCE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _NOISEREDUCE_AVAILABLE = False

_BREATH_LOW_FREQ_CUTOFF = 1200.0
_BREATH_HIGH_FREQ_CUTOFF = 2500.0
_BREATH_MIN_FLATNESS = 0.18
_BREATH_MIN_HIGH_RATIO = 0.08
_BREATH_MAX_LOW_RATIO = 0.92
_BREATH_MAX_PEAKINESS = 140.0


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
        breath_reduce_strength: float = 0.35,
    ):
        self.sample_rate = sample_rate
        self.prop_decrease = float(np.clip(prop_decrease, 0.0, 1.0))
        self.n_fft = n_fft
        self.hop_length = hop_length if hop_length is not None else n_fft // 4
        self.breath_reduce_strength = float(np.clip(breath_reduce_strength, 0.0, 1.0))
        self._noise_profile: np.ndarray | None = None

    def _effective_fft_params(self, n_samples: int) -> tuple[int, int]:
        """
        Pick robust STFT params for very short clips.
        """
        nperseg = int(min(self.n_fft, max(64, n_samples)))
        hop = int(min(self.hop_length, max(16, nperseg // 4)))
        noverlap = nperseg - hop
        return nperseg, noverlap

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

    def reduce(
        self,
        audio: np.ndarray,
        apply_breath_suppression: bool = True,
    ) -> np.ndarray:
        """
        Remove noise from an audio signal.

        If no noise profile has been set, :meth:`detect_and_set_noise_profile`
        is called automatically.

        Parameters
        ----------
        audio : np.ndarray
            Audio signal with shape (samples,) or (samples, channels).
        apply_breath_suppression : bool
            Whether to apply breath-sound suppression after denoising.
            Default True.

        Returns
        -------
        np.ndarray
            De-noised audio with the same shape as the input.
        """
        stereo_input = audio.ndim == 2
        if stereo_input:
            channels = [
                self.reduce(
                    audio[:, ch],
                    apply_breath_suppression=apply_breath_suppression,
                )
                for ch in range(audio.shape[1])
            ]
            return np.stack(channels, axis=1)

        mono = audio.astype(np.float32)
        if self._noise_profile is None:
            self.detect_and_set_noise_profile(mono)

        if _NOISEREDUCE_AVAILABLE:
            reduced = self._reduce_noisereduce(mono)
        else:
            reduced = self._reduce_spectral_gate(mono)
        if not apply_breath_suppression:
            return reduced
        return self._suppress_breath_sounds(reduced)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_noise_profile(self, segment: np.ndarray) -> np.ndarray:
        """Compute robust noise magnitude spectrum from a segment."""
        nperseg, noverlap = self._effective_fft_params(len(segment))
        _, _, zxx = stft(
            segment,
            fs=self.sample_rate,
            nperseg=nperseg,
            noverlap=noverlap,
            window="hann",
        )
        mag = np.abs(zxx)
        # Percentile is more robust than mean when short speech leaks in.
        return np.percentile(mag, 30, axis=1).astype(np.float32)

    def _auto_detect_noise_segment(self, mono: np.ndarray) -> np.ndarray:
        """
        Find several quiet windows and concatenate them as a stronger
        noise reference segment.
        """
        window = max(int(0.4 * self.sample_rate), self.n_fft)
        hop = max(window // 4, 1)
        energies: list[tuple[float, int]] = []
        zcrs: list[tuple[float, int]] = []

        for start in range(0, max(len(mono) - window + 1, 1), hop):
            chunk = mono[start: start + window]
            if len(chunk) < 8:
                continue
            energies.append((np.mean(chunk ** 2), start))
            signs = np.sign(chunk)
            zcr = float(np.mean(np.abs(np.diff(signs)) > 0))
            zcrs.append((zcr, start))

        if not energies:
            return mono[:window]

        # Jointly prefer low-energy and low-ZCR windows (less voiced content).
        energy_map = {pos: e for e, pos in energies}
        zcr_map = {pos: z for z, pos in zcrs}
        e_values = np.array(list(energy_map.values()), dtype=np.float64)
        z_values = np.array(list(zcr_map.values()), dtype=np.float64)
        e_norm = (e_values - e_values.min()) / (np.ptp(e_values) + 1e-12)
        z_norm = (z_values - z_values.min()) / (np.ptp(z_values) + 1e-12)

        positions = list(energy_map.keys())
        score = 0.75 * e_norm + 0.25 * z_norm
        best_count = max(1, min(6, len(positions) // 6))
        best_idx = np.argsort(score)[:best_count]
        segments = [mono[positions[i]: positions[i] + window] for i in best_idx]
        return np.concatenate(segments).astype(np.float32)

    def _reduce_noisereduce(self, audio: np.ndarray) -> np.ndarray:
        """Use the noisereduce library (stationary mode)."""
        nperseg, _ = self._effective_fft_params(len(audio))
        hop = min(self.hop_length, nperseg // 2)
        noise_clip = self._profile_to_clip()
        return nr.reduce_noise(
            y=audio,
            y_noise=noise_clip,
            sr=self.sample_rate,
            prop_decrease=self.prop_decrease,
            stationary=True,
            n_fft=nperseg,
            win_length=nperseg,
            hop_length=hop,
            n_std_thresh_stationary=1.2,
            freq_mask_smooth_hz=300,
            time_mask_smooth_ms=80,
        ).astype(np.float32)

    def _profile_to_clip(self) -> np.ndarray:
        """
        Synthesise a short white-noise clip shaped by the stored noise
        profile so it can be passed to noisereduce.
        """
        rng = np.random.default_rng(0)
        n_frames = 16
        profile = self._noise_profile
        nperseg, noverlap = self._effective_fft_params(max(self.n_fft, len(profile) * 2))
        n_bins = len(profile)
        phase = np.exp(1j * rng.uniform(0, 2 * np.pi, (n_bins, n_frames)))
        zxx = profile[:, np.newaxis] * phase
        _, clip = istft(
            zxx,
            fs=self.sample_rate,
            nperseg=nperseg,
            noverlap=noverlap,
            window="hann",
        )
        return clip.astype(np.float32)

    def _reduce_spectral_gate(self, audio: np.ndarray) -> np.ndarray:
        """
        Fallback spectral gating with adaptive oversubtraction, spectral floor,
        and temporal smoothing for fewer musical-noise artifacts.
        """
        nperseg, noverlap = self._effective_fft_params(len(audio))
        _, _, zxx = stft(
            audio,
            fs=self.sample_rate,
            nperseg=nperseg,
            noverlap=noverlap,
            window="hann",
        )
        mag = np.abs(zxx)
        phase = np.angle(zxx)

        if len(self._noise_profile) != mag.shape[0]:
            # handle short clips where effective FFT differs from configured n_fft
            target_bins = mag.shape[0]
            x_old = np.linspace(0.0, 1.0, len(self._noise_profile), dtype=np.float64)
            x_new = np.linspace(0.0, 1.0, target_bins, dtype=np.float64)
            noise_profile = np.interp(x_new, x_old, self._noise_profile).astype(np.float32)
        else:
            noise_profile = self._noise_profile

        noise = noise_profile[:, np.newaxis]
        snr = np.maximum((mag ** 2 - noise ** 2) / (noise ** 2 + 1e-12), 0.0)

        # More attenuation in low-SNR regions; gentler in clean regions.
        alpha = 1.0 + self.prop_decrease * (1.8 / (1.0 + snr))
        beta = 0.03 + 0.07 * self.prop_decrease  # spectral floor

        clean_mag = mag - alpha * noise
        clean_mag = np.maximum(clean_mag, beta * noise)

        # smooth over time & frequency to reduce rough artifacts
        clean_mag = uniform_filter1d(clean_mag, size=3, axis=0, mode="nearest")
        clean_mag = uniform_filter1d(clean_mag, size=3, axis=1, mode="nearest")
        # decision-directed smoothing to suppress musical noise
        clean_mag = 0.85 * clean_mag + 0.15 * np.maximum(
            np.roll(clean_mag, 1, axis=1), beta * noise
        )

        zxx_clean = clean_mag * np.exp(1j * phase)
        _, audio_clean = istft(
            zxx_clean,
            fs=self.sample_rate,
            nperseg=nperseg,
            noverlap=noverlap,
            window="hann",
        )

        n = len(audio)
        if len(audio_clean) >= n:
            return audio_clean[:n].astype(np.float32)
        return np.pad(audio_clean, (0, n - len(audio_clean))).astype(np.float32)

    def _suppress_breath_sounds(self, audio: np.ndarray) -> np.ndarray:
        """
        Conservatively attenuate likely breath-noise frames while
        preserving soft voiced singing.
        """
        if self.breath_reduce_strength <= 0.0 or len(audio) < 64:
            return audio.astype(np.float32, copy=False)

        nperseg, noverlap = self._effective_fft_params(len(audio))
        freqs, _, zxx = stft(
            audio,
            fs=self.sample_rate,
            nperseg=nperseg,
            noverlap=noverlap,
            window="hann",
        )
        mag = np.abs(zxx)
        if mag.size == 0:
            return audio.astype(np.float32, copy=False)

        eps = 1e-10
        frame_energy = np.mean(mag ** 2, axis=0)
        energy_low = np.percentile(frame_energy, 15)
        energy_high = np.percentile(frame_energy, 85)
        if energy_high <= 0:
            return audio.astype(np.float32, copy=False)

        mag_eps = mag + eps
        mean_mag = np.mean(mag_eps, axis=0)
        flatness = np.exp(np.mean(np.log(mag_eps), axis=0)) / mean_mag
        low_mask = freqs <= _BREATH_LOW_FREQ_CUTOFF
        high_mask = freqs >= _BREATH_HIGH_FREQ_CUTOFF
        if not np.any(low_mask) or not np.any(high_mask):
            return audio.astype(np.float32, copy=False)

        # Add eps for ratio divisions; percentile-based frame_energy itself
        # does not divide and remains stable without it.
        total_energy = np.sum(mag ** 2, axis=0) + eps
        low_ratio = np.sum(mag[low_mask] ** 2, axis=0) / total_energy
        high_ratio = np.sum(mag[high_mask] ** 2, axis=0) / total_energy

        peakiness = np.max(mag_eps, axis=0) / mean_mag

        breath_like = (
            (frame_energy >= energy_low)
            & (frame_energy <= energy_high)
            & (flatness > _BREATH_MIN_FLATNESS)
            & (high_ratio > _BREATH_MIN_HIGH_RATIO)
            & (low_ratio < _BREATH_MAX_LOW_RATIO)
            & (peakiness < _BREATH_MAX_PEAKINESS)
        )
        if not np.any(breath_like):
            return audio.astype(np.float32, copy=False)

        mask = breath_like.astype(np.float32)
        mask = uniform_filter1d(mask, size=3, mode="nearest")
        attenuation = 1.0 - self.breath_reduce_strength * np.clip(mask, 0.0, 1.0)
        zxx_clean = zxx * attenuation[np.newaxis, :]
        _, cleaned = istft(
            zxx_clean,
            fs=self.sample_rate,
            nperseg=nperseg,
            noverlap=noverlap,
            window="hann",
        )

        n = len(audio)
        if len(cleaned) >= n:
            return cleaned[:n].astype(np.float32)
        return np.pad(cleaned, (0, n - len(cleaned))).astype(np.float32)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert to mono float32 if necessary."""
    if audio.ndim == 2:
        return audio.mean(axis=1).astype(np.float32)
    return audio.astype(np.float32)
