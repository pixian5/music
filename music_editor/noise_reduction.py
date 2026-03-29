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
_BREATH_MIN_FLATNESS = 0.16
_BREATH_MIN_HIGH_RATIO = 0.06
_BREATH_MAX_LOW_RATIO = 0.94
_BREATH_MAX_PEAKINESS = 170.0
_BREATH_METHODS = {"attenuate", "high_band", "hybrid", "deep", "ultra", "extreme"}
_BREATH_SENS_ENERGY_LOW_BASE = 20.0
_BREATH_SENS_ENERGY_LOW_SPAN = 10.0
_BREATH_SENS_ENERGY_HIGH_BASE = 80.0
_BREATH_SENS_ENERGY_HIGH_SPAN = 10.0
_BREATH_SENS_FLATNESS_BASE = 1.15
_BREATH_SENS_FLATNESS_SPAN = 0.5
_BREATH_SENS_HIGH_RATIO_BASE = 1.2
_BREATH_SENS_HIGH_RATIO_SPAN = 0.6
_BREATH_SENS_LOW_RATIO_SPAN = 0.08
_BREATH_SENS_PEAKINESS_BASE = 1.15
_BREATH_SENS_PEAKINESS_SPAN = 0.5
_BREATH_SENS_CREST_BASE = 1.18
_BREATH_SENS_CREST_SPAN = 0.42
_BREATH_SENS_CONTRAST_BASE_DB = 5.4
_BREATH_SENS_CONTRAST_SPAN_DB = 2.6
_BREATH_SENS_RISE_BASE_DB = 2.8
_BREATH_SENS_RISE_SPAN_DB = 1.8
_BREATH_HYBRID_FRAME_WEIGHT = 0.35
_BREATH_HYBRID_BAND_WEIGHT = 0.65


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
        breath_suppression: float = 0.75,
        n_fft: int = 2048,
        hop_length: int | None = None,
        breath_reduce_strength: float = 1.0,
        breath_method: str = "extreme",
        breath_sensitivity: float = 1.0,
        breath_band_focus: float = 1.0,
    ):
        self.sample_rate = sample_rate
        self.prop_decrease = float(np.clip(prop_decrease, 0.0, 1.0))
        # Keep backward-compatible argument names while making strength control
        # explicit: final suppression is a product of base suppression and UI/CLI
        # strength slider.
        self.breath_suppression = float(np.clip(breath_suppression, 0.0, 1.0))
        self.n_fft = n_fft
        self.hop_length = hop_length if hop_length is not None else n_fft // 4
        self.breath_reduce_strength = float(np.clip(breath_reduce_strength, 0.0, 1.0))
        if breath_method not in _BREATH_METHODS:
            raise ValueError(
                f"Unsupported breath_method '{breath_method}'. "
                f"Expected one of {sorted(_BREATH_METHODS)}."
            )
        self.breath_method = breath_method
        self.breath_sensitivity = float(np.clip(breath_sensitivity, 0.0, 1.0))
        self.breath_band_focus = float(np.clip(breath_band_focus, 0.0, 1.0))
        self._noise_profile: np.ndarray | None = None

    @property
    def _effective_breath_strength(self) -> float:
        """
        Final breath attenuation intensity.
        """
        return float(np.clip(self.breath_suppression * self.breath_reduce_strength, 0.0, 1.0))

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
        if apply_breath_suppression:
            return self._suppress_breath_noise(reduced)
        return reduced.astype(np.float32)

    def suppress_breath_sounds(self, audio: np.ndarray) -> np.ndarray:
        """
        Public API for breath-sound suppression.

        Kept as a stable entrypoint because external callers may invoke
        this method directly.
        """
        if audio.ndim == 2:
            channels = [
                self._suppress_breath_noise(audio[:, ch].astype(np.float32))
                for ch in range(audio.shape[1])
            ]
            return np.stack(channels, axis=1)
        return self._suppress_breath_noise(audio.astype(np.float32))

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

    def _suppress_breath_noise(self, audio: np.ndarray) -> np.ndarray:
        """
        Suppress breath/inhalation noise (typically broadband with strong
        high-frequency frication energy) while preserving voiced regions.
        """
        strength = self._effective_breath_strength
        if strength <= 1e-6 or len(audio) < 64:
            return audio.astype(np.float32)

        nperseg, noverlap = self._effective_fft_params(len(audio))
        freqs, _, zxx = stft(
            audio,
            fs=self.sample_rate,
            nperseg=nperseg,
            noverlap=noverlap,
            window="hann",
        )
        mag = np.abs(zxx)
        phase = np.angle(zxx)

        # Method-dependent analysis bands. deep/ultra/extreme are intentionally
        # more aggressive.
        very_aggressive = self.breath_method in {"deep", "ultra", "extreme"}
        ultra_mode = self.breath_method == "ultra"
        extreme_mode = self.breath_method == "extreme"
        hi_start = 1700.0 if extreme_mode else (
            2000.0 if ultra_mode else (2400.0 if very_aggressive else 3000.0)
        )
        lo_end = 1800.0 if extreme_mode else (
            1700.0 if ultra_mode else (1500.0 if very_aggressive else 1300.0)
        )
        hi_mask = freqs >= hi_start
        lo_mask = (freqs >= 100.0) & (freqs <= lo_end)
        mid_mask = (freqs >= 1200.0) & (freqs <= 3200.0)
        if not np.any(hi_mask) or not np.any(lo_mask):
            return audio.astype(np.float32)

        hi_mag = mag[hi_mask]
        hi_energy = np.mean(hi_mag, axis=0)
        total_energy = np.mean(mag, axis=0) + 1e-10
        lo_energy = np.mean(mag[lo_mask], axis=0) + 1e-10
        hi_ratio = hi_energy / total_energy
        lo_ratio = lo_energy / total_energy
        frame_energy = 20.0 * np.log10(total_energy + 1e-12)
        # Voiced speech/singing tends to present stronger tonal peaks in low-mid
        # frequencies, while inhale noise is flatter/broader.
        lo_mag = mag[lo_mask]
        lo_crest = np.max(lo_mag + 1e-12, axis=0) / (np.mean(lo_mag + 1e-12, axis=0))
        # Flat/noisy high-band texture tends to indicate breath more than voiced fricatives.
        hi_geo = np.exp(np.mean(np.log(hi_mag + 1e-12), axis=0))
        hi_arith = np.mean(hi_mag + 1e-12, axis=0)
        hi_flatness = hi_geo / hi_arith
        peakiness = np.max(hi_mag + 1e-12, axis=0) / hi_arith

        sens = self.breath_sensitivity
        ratio_q = 0.7 - 0.24 * sens
        flat_q = 0.66 - 0.24 * sens
        ratio_th = max(
            _BREATH_MIN_HIGH_RATIO,
            np.quantile(hi_ratio, np.clip(ratio_q, 0.35, 0.9))
            * (_BREATH_SENS_HIGH_RATIO_BASE - _BREATH_SENS_HIGH_RATIO_SPAN * sens),
        )
        flat_th = max(
            _BREATH_MIN_FLATNESS,
            np.quantile(hi_flatness, np.clip(flat_q, 0.35, 0.9))
            * (_BREATH_SENS_FLATNESS_BASE - _BREATH_SENS_FLATNESS_SPAN * sens),
        )
        low_ratio_th = _BREATH_MAX_LOW_RATIO - _BREATH_SENS_LOW_RATIO_SPAN * sens
        peak_th = _BREATH_MAX_PEAKINESS * (
            _BREATH_SENS_PEAKINESS_BASE - _BREATH_SENS_PEAKINESS_SPAN * sens
        )
        crest_q = 0.63 + 0.18 * (1.0 - sens)
        crest_th = np.quantile(lo_crest, np.clip(crest_q, 0.5, 0.95)) * (
            _BREATH_SENS_CREST_BASE - _BREATH_SENS_CREST_SPAN * sens
        )
        energy_low_percentile = np.clip(
            _BREATH_SENS_ENERGY_LOW_BASE + _BREATH_SENS_ENERGY_LOW_SPAN * sens,
            10.0,
            45.0,
        )
        energy_high_percentile = np.clip(
            _BREATH_SENS_ENERGY_HIGH_BASE - _BREATH_SENS_ENERGY_HIGH_SPAN * sens,
            55.0,
            95.0,
        )
        low_energy_th = np.quantile(frame_energy, energy_low_percentile / 100.0)
        high_energy_th = np.quantile(frame_energy, energy_high_percentile / 100.0)
        quiet_frames = frame_energy <= low_energy_th
        loud_frames = frame_energy >= high_energy_th
        # Inhales are often a local "hiss bump": clearly louder than nearby
        # context, with an audible jump from previous frames and drop after.
        local_window = 7 if extreme_mode else (9 if ultra_mode else 11)
        local_energy = uniform_filter1d(frame_energy, size=local_window, mode="nearest")
        prev_energy = np.roll(local_energy, 1)
        next_energy = np.roll(local_energy, -1)
        prev_energy[0] = local_energy[0]
        next_energy[-1] = local_energy[-1]
        local_contrast_db = frame_energy - local_energy
        rise_db = frame_energy - prev_energy
        fall_db = frame_energy - next_energy
        contrast_th = _BREATH_SENS_CONTRAST_BASE_DB - _BREATH_SENS_CONTRAST_SPAN_DB * sens
        rise_th = _BREATH_SENS_RISE_BASE_DB - _BREATH_SENS_RISE_SPAN_DB * sens
        has_volume_contrast = (
            (local_contrast_db > contrast_th)
            & (rise_db > rise_th)
            & (fall_db > rise_th * (0.75 if extreme_mode else 0.9))
        )
        # Quiet inhale can appear as a very small low-level hump:
        # both sides are very quiet, middle only slightly raised.
        low_level_hump = (
            (frame_energy < (low_energy_th + (1.0 if extreme_mode else 0.6)))
            & (rise_db > rise_th * (0.45 if extreme_mode else 0.55))
            & (fall_db > rise_th * (0.40 if extreme_mode else 0.50))
            & (local_contrast_db > contrast_th * (0.33 if extreme_mode else 0.42))
        )
        quiet_gate = quiet_frames
        if ultra_mode or extreme_mode:
            # Reduce false positives: in aggressive modes, quietness alone is
            # not enough; require stronger high-band evidence when no obvious
            # discontinuity exists.
            quiet_gate = quiet_frames & (
                (hi_ratio > ratio_th * (0.98 if extreme_mode else 1.0))
                & (hi_flatness > flat_th * (0.98 if extreme_mode else 1.0))
            )
        breath_frames = (
            (hi_ratio > ratio_th)
            & (hi_flatness > flat_th)
            & (lo_ratio < low_ratio_th)
            & (peakiness < peak_th)
            & (lo_crest < crest_th)
            & (has_volume_contrast | quiet_gate)
        )
        speech_like_frames = (lo_crest >= crest_th) & (lo_ratio > np.minimum(0.98, low_ratio_th * 1.03))
        if very_aggressive:
            # Deep/ultra: for weak but annoying inhales, allow lower flatness bar on quiet frames.
            relaxed_frames = (
                (hi_ratio > ratio_th * (0.76 if extreme_mode else (0.82 if ultra_mode else 0.88)))
                & (hi_flatness > flat_th * (0.62 if extreme_mode else (0.7 if ultra_mode else 0.78)))
                & (lo_ratio < min(0.96, low_ratio_th * 1.04))
                & (lo_crest < crest_th * (1.03 if extreme_mode else 1.0))
                & (quiet_frames | has_volume_contrast | low_level_hump)
            )
            if extreme_mode:
                # Extra branch for weak but structured quiet inhales.
                quiet_hump_frames = (
                    (hi_ratio > ratio_th * 0.58)
                    & (hi_flatness > flat_th * 0.54)
                    & (lo_ratio < min(0.97, low_ratio_th * 1.06))
                    & (lo_crest < crest_th * 1.05)
                    & low_level_hump
                )
                relaxed_frames = relaxed_frames | quiet_hump_frames
            breath_frames = breath_frames | relaxed_frames
        breath_frames = breath_frames & (~speech_like_frames)
        if not np.any(breath_frames):
            if ultra_mode or extreme_mode:
                has_inhale_signature = (
                    (np.quantile(hi_ratio, 0.8) > ratio_th * (0.84 if extreme_mode else 0.92))
                    and (np.quantile(hi_flatness, 0.7) > flat_th * (0.82 if extreme_mode else 0.9))
                )
                if extreme_mode and not has_inhale_signature:
                    # Secondary signature for weak quiet inhales:
                    # low absolute energy + local hump + some high-band bias.
                    quiet_hump_signature = (
                        (np.quantile(frame_energy, 0.72) <= low_energy_th + 2.6)
                        and (np.quantile(local_contrast_db, 0.85) >= contrast_th * 0.25)
                        and (np.quantile(hi_ratio, 0.72) > ratio_th * 0.55)
                    )
                    has_inhale_signature = quiet_hump_signature
                if not has_inhale_signature:
                    return audio.astype(np.float32)
                surrogate = (
                    (hi_ratio > ratio_th * (0.55 if extreme_mode else 0.65))
                    & quiet_frames
                    & (lo_crest < crest_th * (1.06 if extreme_mode else 1.0))
                )
                if extreme_mode and not np.any(surrogate):
                    # Fallback for difficult material: pick likely inhale frames
                    # by combining high hi-ratio and low frame energy ranking.
                    hi_rank = hi_ratio >= np.quantile(hi_ratio, 0.78)
                    quiet_rank = frame_energy <= np.quantile(frame_energy, 0.62)
                    crest_rank = lo_crest <= np.quantile(lo_crest, 0.52)
                    surrogate = hi_rank & quiet_rank & crest_rank
                if extreme_mode and not np.any(surrogate):
                    # Last-resort fallback: force-select top scored frames so
                    # extreme mode always attempts audible inhale attenuation.
                    hi_norm = (hi_ratio - np.min(hi_ratio)) / (np.ptp(hi_ratio) + 1e-12)
                    en_norm = (frame_energy - np.min(frame_energy)) / (np.ptp(frame_energy) + 1e-12)
                    crest_norm = (lo_crest - np.min(lo_crest)) / (np.ptp(lo_crest) + 1e-12)
                    score = 0.56 * hi_norm + 0.26 * (1.0 - en_norm) + 0.18 * (1.0 - crest_norm)
                    top_k = max(1, int(0.14 * len(score)))
                    top_idx = np.argpartition(score, -top_k)[-top_k:]
                    surrogate = np.zeros_like(score, dtype=bool)
                    surrogate[top_idx] = True
                if np.any(surrogate):
                    breath_frames = surrogate
                else:
                    return audio.astype(np.float32)
            else:
                return audio.astype(np.float32)

        if ultra_mode or extreme_mode:
            original_breath_frames = breath_frames.copy()
            breath_frames = self._refine_breath_frame_mask(
                breath_frames=breath_frames,
                frame_energy_db=frame_energy,
                local_energy_db=local_energy,
                rise_db=rise_db,
                fall_db=fall_db,
                sens=sens,
                extreme_mode=extreme_mode,
                ultra_mode=ultra_mode,
            )
            if not np.any(breath_frames):
                breath_frames = original_breath_frames

        # Smooth frame mask to avoid chattering and attacks.
        breath_gain = np.ones_like(hi_ratio, dtype=np.float32)
        ratio_strength = np.clip((hi_ratio - ratio_th) / (ratio_th + 1e-10), 0.0, 1.5)
        flat_strength = np.clip((hi_flatness - flat_th) / (flat_th + 1e-10), 0.0, 1.2)
        low_ratio_strength = np.clip(
            (low_ratio_th - lo_ratio) / (low_ratio_th + 1e-10),
            0.0,
            1.2,
        )
        energy_strength = np.clip(
            (high_energy_th - frame_energy) / (high_energy_th - low_energy_th + 1e-10),
            0.0,
            1.0,
        )
        energy_strength[loud_frames] *= 0.15  # protect loud sung consonants
        reduction = strength * np.clip(
            0.52 * ratio_strength
            + 0.18 * flat_strength
            + 0.14 * low_ratio_strength
            + 0.16 * energy_strength,
            0.0,
            1.0,
        )
        breath_gain[breath_frames] = 1.0 - reduction[breath_frames]
        if ultra_mode or extreme_mode:
            # Ultra mode: extra attenuation on low-energy hiss-like frames.
            ultra_mask = breath_frames | ((hi_ratio > ratio_th * (0.62 if extreme_mode else 0.75)) & quiet_frames)
            ultra_extra = strength * ((0.28 + 0.42 * sens) if extreme_mode else (0.20 + 0.35 * sens))
            breath_gain[ultra_mask] *= (1.0 - ultra_extra)

        smooth_size = 19 if extreme_mode else (15 if ultra_mode else (11 if self.breath_method == "deep" else 7))
        breath_gain = uniform_filter1d(breath_gain, size=smooth_size, mode="nearest")
        breath_gain = np.clip(breath_gain, 1.0 - strength, 1.0)

        # Apply suppression to upper band with frequency-dependent strength:
        # more attenuation in very high frequencies where inhale hiss dominates.
        gain_2d = np.ones_like(mag, dtype=np.float32)
        hi_freqs = freqs[hi_mask]
        focus = 0.4 + 0.6 * self.breath_band_focus
        denom = 2600.0 if extreme_mode else (3000.0 if ultra_mode else (3600.0 if self.breath_method == "deep" else 4200.0))
        freq_weight = np.clip((hi_freqs - hi_start) / denom, 0.0, 1.0).astype(np.float32)
        weighted_gain = 1.0 - (1.0 - breath_gain[np.newaxis, :]) * (
            (1.0 - focus) + focus * freq_weight[:, np.newaxis]
        )
        gain_2d[hi_mask, :] = np.clip(weighted_gain, 1.0 - strength, 1.0)

        if self.breath_method in {"hybrid", "deep", "ultra", "extreme"} and np.any(mid_mask):
            # Mild attenuation in upper-mid frication band to reduce residual breath.
            mid_floor = (
                0.18
                + 0.25 * float(self.breath_method == "deep")
                + 0.18 * float(ultra_mode)
                + 0.34 * float(extreme_mode)
            )
            mid_gain = 1.0 - (1.0 - breath_gain[np.newaxis, :]) * mid_floor
            gain_2d[mid_mask, :] = np.minimum(gain_2d[mid_mask, :], mid_gain)

        if ultra_mode or extreme_mode:
            # Ultra mode: frame-level ducking to remove small inhale residue.
            global_duck = 1.0 - (1.0 - breath_gain)[np.newaxis, :] * (
                (0.34 + 0.28 * sens) if extreme_mode else (0.22 + 0.22 * sens)
            )
            gain_2d = np.minimum(gain_2d, global_duck.astype(np.float32))

        # For detected breath frames, estimate a residual hiss floor and subtract a portion.
        breath_floor = np.median(hi_mag[:, breath_frames], axis=1, keepdims=True)
        if breath_floor.size > 0:
            floor_base = 0.64 if extreme_mode else (0.52 if ultra_mode else (0.42 if self.breath_method == "deep" else 0.3))
            reduction_floor = strength * floor_base
            hi_mag_clean = np.maximum(
                hi_mag * gain_2d[hi_mask, :] - reduction_floor * breath_floor,
                0.0,
            )
            mag_clean = mag.copy()
            mag_clean[hi_mask, :] = hi_mag_clean
        else:
            mag_clean = mag * gain_2d

        if extreme_mode:
            # Second-stage spectral floor suppression:
            # estimate a broad high-band hiss floor and remove a controlled amount.
            hi_floor = np.quantile(mag_clean[hi_mask, :], 0.35, axis=1, keepdims=True)
            extra_floor = strength * (0.34 + 0.30 * sens)
            mag_clean[hi_mask, :] = np.maximum(
                mag_clean[hi_mask, :] - extra_floor * hi_floor,
                0.0,
            )

        zxx_clean = mag_clean * np.exp(1j * phase)
        _, out = istft(
            zxx_clean,
            fs=self.sample_rate,
            nperseg=nperseg,
            noverlap=noverlap,
            window="hann",
        )
        if len(out) >= len(audio):
            out = out[: len(audio)]
        else:
            out = np.pad(out, (0, len(audio) - len(out)))

        # Additional segment-level volume ducking on detected inhale frames.
        # This directly lowers inhale section loudness (time-domain), which helps
        # in cases where spectral attenuation alone still leaves audible breaths.
        if np.any(breath_frames):
            if extreme_mode:
                segment_factor = 1.30
            elif ultra_mode:
                segment_factor = 1.05
            elif self.breath_method == "deep":
                segment_factor = 0.86
            else:
                segment_factor = 0.60
            out = self._apply_breath_segment_ducking(
                out,
                breath_gain=breath_gain,
                segment_factor=segment_factor * strength,
            )
        return out.astype(np.float32)

    def _apply_breath_segment_ducking(
        self,
        audio: np.ndarray,
        breath_gain: np.ndarray,
        segment_factor: float,
    ) -> np.ndarray:
        """
        Apply time-domain ducking derived from frame-level breath gain.

        `segment_factor` controls how much of frame attenuation is projected
        to sample level (0 = disabled, 1 = full).
        """
        if len(audio) == 0 or breath_gain.size < 2 or segment_factor <= 1e-6:
            return audio.astype(np.float32)

        frame_x = np.linspace(0.0, 1.0, breath_gain.size, dtype=np.float32)
        sample_x = np.linspace(0.0, 1.0, len(audio), dtype=np.float32)
        sample_gain = np.interp(sample_x, frame_x, breath_gain.astype(np.float32))

        inhale_strength = np.clip(1.0 - sample_gain, 0.0, 1.0)
        duck_gain = 1.0 - inhale_strength * float(np.clip(segment_factor, 0.0, 1.45))
        # Make ducking curve steeper on likely inhale regions so residual peak-like
        # pulses are reduced more than surrounding speech/singing frames.
        duck_gain = np.power(duck_gain, 1.70).astype(np.float32)
        smooth = max(int(0.016 * self.sample_rate), 16)
        kernel = np.ones(smooth, dtype=np.float32) / smooth
        duck_gain = np.convolve(duck_gain, kernel, mode="same")
        duck_gain = np.clip(duck_gain, 0.002, 1.0)

        # Peak-aware cap in inhale-heavy samples: if envelope still spikes after
        # ducking, force additional local attenuation to avoid obvious peaks.
        target_peak = 1.0 - 0.9985 * inhale_strength
        abs_audio = np.abs(audio.astype(np.float32))
        allowed = np.maximum(target_peak, 0.008)
        peak_cap = np.minimum(1.0, allowed / (abs_audio + 1e-8))
        duck_gain = np.minimum(duck_gain, peak_cap).astype(np.float32)
        out = (audio.astype(np.float32) * duck_gain).astype(np.float32)

        # Second-stage local peak limiter for inhale-heavy samples: this keeps
        # short "mountain-like" pulses clearly below surrounding voiced peaks.
        inhale_mask = inhale_strength > 0.035
        if np.any(inhale_mask):
            abs_out = np.abs(out)
            non_inhale = abs_out[~inhale_mask]
            if non_inhale.size > 64:
                speech_ref_peak = float(np.quantile(non_inhale, 0.68))
            else:
                speech_ref_peak = float(np.quantile(abs_out, 0.52))
            inhale_peak_cap = max(0.002, speech_ref_peak * 0.15)
            limiter_gain = np.ones_like(out, dtype=np.float32)
            inhale_abs = abs_out[inhale_mask]
            over_ratio = inhale_abs / (inhale_peak_cap + 1e-8)
            limiter_gain_inhale = np.where(
                over_ratio > 1.0,
                np.power(over_ratio, -1.25),
                1.0,
            ).astype(np.float32)
            limiter_gain[inhale_mask] = limiter_gain_inhale
            limiter_gain = np.convolve(limiter_gain, kernel, mode="same")
            out = (out * limiter_gain.astype(np.float32)).astype(np.float32)
            # Final inhale-frame duck: ensure inhale sections are audibly lower
            # than normal singing/speech even when residual wideband energy remains.
            final_duck = 1.0 - 0.992 * inhale_strength
            final_duck = np.clip(final_duck, 0.006, 1.0).astype(np.float32)
            out = (out * final_duck).astype(np.float32)

            # Hard inhale attenuation stage: when inhale confidence is high,
            # force significant dB reduction on those segments.
            if self.breath_method == "extreme":
                max_inhale_duck_db = 34.0
            elif self.breath_method == "ultra":
                max_inhale_duck_db = 28.0
            elif self.breath_method == "deep":
                max_inhale_duck_db = 22.0
            else:
                max_inhale_duck_db = 14.0
            strong_inhale = inhale_strength > 0.12
            if np.any(strong_inhale):
                inhale_duck_db = max_inhale_duck_db * inhale_strength * float(np.clip(segment_factor, 0.0, 1.45))
                inhale_duck_gain = np.power(10.0, -inhale_duck_db / 20.0).astype(np.float32)
                inhale_duck_gain = np.clip(inhale_duck_gain, 0.008, 1.0)
                hard_gain = np.ones_like(out, dtype=np.float32)
                hard_gain[strong_inhale] = inhale_duck_gain[strong_inhale]
                hard_gain = np.convolve(hard_gain, kernel, mode="same")
                out = (out * hard_gain.astype(np.float32)).astype(np.float32)
        return out.astype(np.float32)

    def _refine_breath_frame_mask(
        self,
        breath_frames: np.ndarray,
        frame_energy_db: np.ndarray,
        local_energy_db: np.ndarray,
        rise_db: np.ndarray,
        fall_db: np.ndarray,
        sens: float,
        extreme_mode: bool,
        ultra_mode: bool,
    ) -> np.ndarray:
        """
        Segment-level post filter for inhale candidates.

        Goals:
        1) keep obvious, discontinuous inhale bumps,
        2) reject tiny low-volume syllables inside a sentence,
        3) enforce inhale segments to remain lower than global average loudness.
        """
        if not np.any(breath_frames):
            return breath_frames

        refined = np.zeros_like(breath_frames, dtype=bool)
        n_frames = len(breath_frames)
        idx = np.flatnonzero(breath_frames)
        if idx.size == 0:
            return refined

        starts = [idx[0]]
        ends: list[int] = []
        for i in range(1, idx.size):
            if idx[i] != idx[i - 1] + 1:
                ends.append(idx[i - 1])
                starts.append(idx[i])
        ends.append(idx[-1])

        global_energy_db = 20.0 * np.log10(np.mean(np.power(10.0, frame_energy_db / 20.0)) + 1e-12)
        min_len = 1
        max_len = 16 if extreme_mode else (18 if ultra_mode else 20)
        side_ctx = 2 if extreme_mode else 3
        min_edge_jump = 0.55 - 0.25 * sens
        min_local_contrast = 0.75 - 0.30 * sens
        max_peak_vs_global_db = -0.9 + 0.6 * (1.0 - sens)
        quiet_peak_margin_db = 3.8 - 1.4 * sens
        quiet_edge_margin_db = 5.0 - 1.9 * sens
        quiet_center_lift_db = 0.40 - 0.16 * sens

        for start, end in zip(starts, ends):
            seg_len = end - start + 1
            if seg_len < min_len or seg_len > max_len:
                continue

            left0 = max(0, start - side_ctx)
            left1 = start
            right0 = end + 1
            right1 = min(n_frames, end + 1 + side_ctx)
            if left1 <= left0 or right1 <= right0:
                continue

            seg_peak_db = float(np.max(frame_energy_db[start : end + 1]))
            seg_mean_db = float(np.mean(frame_energy_db[start : end + 1]))
            left_mean_db = float(np.mean(frame_energy_db[left0:left1]))
            right_mean_db = float(np.mean(frame_energy_db[right0:right1]))
            seg_local_contrast = float(np.max(frame_energy_db[start : end + 1] - local_energy_db[start : end + 1]))
            edge_rise = float(np.max(rise_db[start : end + 1]))
            edge_fall = float(np.max(fall_db[start : end + 1]))
            edge_floor_db = max(left_mean_db, right_mean_db)
            center_slice_start = start + seg_len // 3
            center_slice_end = max(center_slice_start + 1, end + 1 - seg_len // 3)
            center_mean_db = float(np.mean(frame_energy_db[center_slice_start:center_slice_end]))
            center_lift_db = center_mean_db - edge_floor_db
            quiet_edge_profile = (
                (left_mean_db <= global_energy_db - quiet_edge_margin_db)
                and (right_mean_db <= global_energy_db - quiet_edge_margin_db)
            )
            quiet_low_hump = (
                (seg_len <= (12 if extreme_mode else 14))
                and (seg_peak_db <= global_energy_db - quiet_peak_margin_db)
                and quiet_edge_profile
                and (center_lift_db >= quiet_center_lift_db)
                and (edge_rise >= min_edge_jump * 0.55)
                and (edge_fall >= min_edge_jump * 0.50)
            )

            # Inhale peak should stay clearly below average loudness of the whole file.
            if seg_peak_db > global_energy_db + max_peak_vs_global_db:
                continue
            # Inhale should be a discontinuous bump compared with neighbors.
            strong_discontinuous = (
                (seg_mean_db - left_mean_db) >= min_edge_jump
                and (seg_mean_db - right_mean_db) >= min_edge_jump
                and (seg_local_contrast >= min_local_contrast)
                and (edge_rise >= min_edge_jump)
                and (edge_fall >= min_edge_jump * 0.9)
            )
            if not (strong_discontinuous or quiet_low_hump):
                continue

            refined[start : end + 1] = True

        return refined


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert to mono float32 if necessary."""
    if audio.ndim == 2:
        return audio.mean(axis=1).astype(np.float32)
    return audio.astype(np.float32)
