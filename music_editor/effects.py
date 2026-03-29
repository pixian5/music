"""
Audio effects module.

Provides common audio effects used in music editing software:

* Volume normalisation     – make the loudness consistent throughout
* Studio reverb            – simulate a recording-studio room
* Male-to-female pitch     – shift pitch up (≈+5 semitones)
* Female-to-male pitch     – shift pitch down (≈-5 semitones)
* KTV / karaoke effect     – reverb + slapback echo + slight chorus
* Stereo widening          – expand the stereo image
* Bass boost               – shelving EQ boost at low frequencies
* Treble boost             – shelving EQ boost at high frequencies
* Fade in / fade out       – linear amplitude ramps
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, lfilter


class AudioEffects:
    """
    Collection of audio effects that can be applied to a NumPy array.

    All methods accept and return audio arrays with shape
    ``(samples,)`` for mono or ``(samples, channels)`` for stereo.
    Sample values are expected to be in the range [-1, 1].

    Parameters
    ----------
    sample_rate : int
        Sample rate of the audio to be processed.
    """

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate

    # ------------------------------------------------------------------
    # Volume / loudness
    # ------------------------------------------------------------------

    def normalize(self, audio: np.ndarray, target_db: float = -18.0, peak_db: float = -1.0) -> np.ndarray:
        """
        Loudness normalization with peak protection.

        Uses RMS target but prevents clipping by enforcing a peak ceiling.
        """
        audio = audio.astype(np.float32)
        rms = _rms(audio)
        if rms < 1e-10:
            return audio.copy()

        target_rms = 10 ** (target_db / 20.0)
        gain_rms = target_rms / rms

        peak = float(np.max(np.abs(audio)))
        peak_limit = 10 ** (peak_db / 20.0)
        gain_peak = peak_limit / max(peak, 1e-10)

        gain = min(gain_rms, gain_peak)
        return np.clip(audio * gain, -1.0, 1.0).astype(np.float32)

    def dynamic_normalize(
        self,
        audio: np.ndarray,
        window_sec: float = 0.2,
        target_db: float = -20.0,
        max_gain_db: float = 18.0,
        attack_sec: float = 0.02,
        release_sec: float = 0.20,
    ) -> np.ndarray:
        """
        Dynamic (time-varying) loudness normalization.

        Envelope-based AGC with attack/release smoothing, generally more
        natural than per-window interpolation.
        """
        x = audio.astype(np.float32)
        n = len(x) if x.ndim == 1 else x.shape[0]
        if n == 0:
            return x

        if x.ndim == 2:
            mono = np.mean(x, axis=1)
        else:
            mono = x

        window = max(int(window_sec * self.sample_rate), 32)
        kernel = np.ones(window, dtype=np.float32) / window
        power = np.convolve(mono.astype(np.float32) ** 2, kernel, mode="same")
        rms_curve = np.sqrt(np.maximum(power, 1e-10))

        target_rms = 10 ** (target_db / 20.0)
        max_gain = 10 ** (max_gain_db / 20.0)
        desired = np.clip(target_rms / rms_curve, 1.0 / max_gain, max_gain)

        attack = np.exp(-1.0 / max(attack_sec * self.sample_rate, 1.0))
        release = np.exp(-1.0 / max(release_sec * self.sample_rate, 1.0))

        gain_curve = np.empty_like(desired)
        g = desired[0]
        for i, gd in enumerate(desired):
            coeff = attack if gd < g else release
            g = coeff * g + (1.0 - coeff) * gd
            gain_curve[i] = g

        if x.ndim == 1:
            return _soft_clip(x * gain_curve)
        return _soft_clip(x * gain_curve[:, np.newaxis])

    # ------------------------------------------------------------------
    # Reverb / room simulation
    # ------------------------------------------------------------------

    def studio_reverb(
        self,
        audio: np.ndarray,
        room_size: float = 0.4,
        damping: float = 0.5,
        wet: float = 0.25,
    ) -> np.ndarray:
        """
        Simulate a recording-studio room using a Schroeder reverberator
        (comb + all-pass filter network).

        Parameters
        ----------
        audio : np.ndarray
            Input audio.
        room_size : float
            Controls reverb tail length (0–1). Default 0.4.
        damping : float
            High-frequency damping (0–1). Default 0.5.
        wet : float
            Mix ratio of wet (reverb) signal (0–1). Default 0.25.

        Returns
        -------
        np.ndarray
            Audio with studio reverb applied.
        """
        wet = float(np.clip(wet, 0.0, 1.0))
        if audio.ndim == 1:
            reverb = _schroeder_reverb(audio.astype(np.float32), self.sample_rate, room_size, damping)
            mixed = (1 - wet) * audio + wet * reverb
            return _soft_clip(mixed)

        channels = []
        for ch in range(audio.shape[1]):
            src = audio[:, ch].astype(np.float32)
            rvb = _schroeder_reverb(src, self.sample_rate, room_size, damping)
            channels.append(_soft_clip((1 - wet) * src + wet * rvb))
        return np.stack(channels, axis=1).astype(np.float32)

    def ktv_effect(
        self,
        audio: np.ndarray,
        reverb_wet: float = 0.35,
        echo_delay_ms: float = 120.0,
        echo_decay: float = 0.4,
        chorus_depth_ms: float = 8.0,
        chorus_rate_hz: float = 0.8,
    ) -> np.ndarray:
        """
        KTV / karaoke effect: reverb + slapback echo + mild chorus.

        Parameters
        ----------
        audio : np.ndarray
            Input audio.
        reverb_wet : float
            Wet mix of the reverb component.
        echo_delay_ms : float
            Echo delay in milliseconds.
        echo_decay : float
            Echo feedback gain (0–1).
        chorus_depth_ms : float
            Chorus modulation depth in milliseconds.
        chorus_rate_hz : float
            Chorus LFO rate in Hz.

        Returns
        -------
        np.ndarray
            Audio with KTV effect applied.
        """
        def _process_channel(ch_audio: np.ndarray) -> np.ndarray:
            reverb = _schroeder_reverb(ch_audio, self.sample_rate, room_size=0.6, damping=0.4)
            with_reverb = (1 - reverb_wet) * ch_audio + reverb_wet * reverb
            with_echo = _slapback_echo(with_reverb, self.sample_rate, echo_delay_ms, echo_decay)
            with_chorus = _chorus(with_echo, self.sample_rate, chorus_depth_ms, chorus_rate_hz)
            return _soft_clip(with_chorus)

        if audio.ndim == 1:
            return _process_channel(audio.astype(np.float32)).astype(np.float32)

        channels = [_process_channel(audio[:, ch].astype(np.float32)) for ch in range(audio.shape[1])]
        return np.stack(channels, axis=1).astype(np.float32)

    # ------------------------------------------------------------------
    # Pitch shifting
    # ------------------------------------------------------------------

    def male_to_female(self, audio: np.ndarray, semitones: float = 5.0) -> np.ndarray:
        """
        Shift pitch upward to simulate a female voice from a male voice.

        Parameters
        ----------
        audio : np.ndarray
            Input audio.
        semitones : float
            Semitones to shift upward. Default 5.

        Returns
        -------
        np.ndarray
            Pitch-shifted audio.
        """
        return self.pitch_shift(audio, semitones=semitones)

    def female_to_male(self, audio: np.ndarray, semitones: float = 5.0) -> np.ndarray:
        """
        Shift pitch downward to simulate a male voice from a female voice.

        Parameters
        ----------
        audio : np.ndarray
            Input audio.
        semitones : float
            Semitones to shift downward. Default 5.

        Returns
        -------
        np.ndarray
            Pitch-shifted audio.
        """
        return self.pitch_shift(audio, semitones=-semitones)

    def pitch_shift(self, audio: np.ndarray, semitones: float = 0.0) -> np.ndarray:
        """
        Shift the pitch of the audio without changing its duration.

        Uses a phase-vocoder approach implemented via librosa.

        Parameters
        ----------
        audio : np.ndarray
            Input audio.
        semitones : float
            Number of semitones to shift (positive = up, negative = down).

        Returns
        -------
        np.ndarray
            Pitch-shifted audio.
        """
        import librosa

        audio = audio.astype(np.float32)
        if audio.ndim == 1:
            return librosa.effects.pitch_shift(
                audio, sr=self.sample_rate, n_steps=semitones
            ).astype(np.float32)
        # Stereo: shift each channel independently
        channels = [
            librosa.effects.pitch_shift(
                audio[:, ch], sr=self.sample_rate, n_steps=semitones
            )
            for ch in range(audio.shape[1])
        ]
        return np.stack(channels, axis=1).astype(np.float32)

    # ------------------------------------------------------------------
    # Stereo effects
    # ------------------------------------------------------------------

    def stereo_widen(
        self, audio: np.ndarray, width: float = 1.5
    ) -> np.ndarray:
        """
        Widen the stereo image using mid-side (M/S) processing.

        If the input is mono, it is first expanded to stereo with a
        subtle Haas delay on one channel.

        Parameters
        ----------
        audio : np.ndarray
            Input audio (mono or stereo).
        width : float
            Stereo width multiplier (1.0 = unchanged, >1 wider, <1 narrower).

        Returns
        -------
        np.ndarray
            Stereo-widened audio with shape (samples, 2).
        """
        if audio.ndim == 1:
            audio = _mono_to_stereo_haas(audio, self.sample_rate)

        left = audio[:, 0].astype(np.float32)
        right = audio[:, 1].astype(np.float32)

        mid = (left + right) * 0.5
        side = (left - right) * 0.5 * float(width)

        new_left = np.clip(mid + side, -1.0, 1.0).astype(np.float32)
        new_right = np.clip(mid - side, -1.0, 1.0).astype(np.float32)
        return np.stack([new_left, new_right], axis=1)

    # ------------------------------------------------------------------
    # Equalisation
    # ------------------------------------------------------------------

    def bass_boost(self, audio: np.ndarray, gain_db: float = 6.0, cutoff_hz: float = 200.0) -> np.ndarray:
        """
        Boost low-frequency content with a low-shelf filter.

        Parameters
        ----------
        audio : np.ndarray
            Input audio.
        gain_db : float
            Gain in dB (positive = boost).
        cutoff_hz : float
            Shelf cutoff frequency in Hz.

        Returns
        -------
        np.ndarray
            Bass-boosted audio.
        """
        return _shelf_filter(audio, self.sample_rate, cutoff_hz, gain_db, "low")

    def treble_boost(self, audio: np.ndarray, gain_db: float = 6.0, cutoff_hz: float = 6000.0) -> np.ndarray:
        """
        Boost high-frequency content with a high-shelf filter.

        Parameters
        ----------
        audio : np.ndarray
            Input audio.
        gain_db : float
            Gain in dB (positive = boost).
        cutoff_hz : float
            Shelf cutoff frequency in Hz.

        Returns
        -------
        np.ndarray
            Treble-boosted audio.
        """
        return _shelf_filter(audio, self.sample_rate, cutoff_hz, gain_db, "high")

    # ------------------------------------------------------------------
    # Fade utilities
    # ------------------------------------------------------------------

    def fade_in(self, audio: np.ndarray, duration_sec: float = 1.0) -> np.ndarray:
        """Apply a linear fade-in at the start of the audio."""
        audio = audio.astype(np.float32)
        n = len(audio) if audio.ndim == 1 else audio.shape[0]
        fade_len = min(int(duration_sec * self.sample_rate), n)
        ramp = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
        if audio.ndim == 1:
            audio[:fade_len] *= ramp
        else:
            audio[:fade_len] *= ramp[:, np.newaxis]
        return audio

    def fade_out(self, audio: np.ndarray, duration_sec: float = 1.0) -> np.ndarray:
        """Apply a linear fade-out at the end of the audio."""
        audio = audio.astype(np.float32)
        n = len(audio) if audio.ndim == 1 else audio.shape[0]
        fade_len = min(int(duration_sec * self.sample_rate), n)
        ramp = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
        if audio.ndim == 1:
            audio[n - fade_len:] *= ramp
        else:
            audio[n - fade_len:] *= ramp[:, np.newaxis]
        return audio


# ===========================================================================
# Private DSP helpers
# ===========================================================================

def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


def _soft_clip(audio: np.ndarray, drive: float = 1.15) -> np.ndarray:
    """
    Gentle soft clipper to avoid hard digital clipping artifacts.
    """
    y = np.tanh(audio.astype(np.float32) * drive) / np.tanh(drive)
    return np.clip(y, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Schroeder reverberator
# ---------------------------------------------------------------------------

def _schroeder_reverb(
    mono: np.ndarray,
    sr: int,
    room_size: float = 0.5,
    damping: float = 0.5,
) -> np.ndarray:
    """
    Simple Schroeder reverberator: four parallel comb filters followed
    by two series all-pass filters.
    """
    room_size = float(np.clip(room_size, 0.01, 0.99))
    damping = float(np.clip(damping, 0.0, 1.0))

    # Comb-filter delay times (in ms), scaled by room_size
    comb_delays_ms = [29.7, 37.1, 41.1, 43.7]
    comb_gains = [0.805, 0.827, 0.783, 0.764]

    allpass_delays_ms = [5.0, 1.7]
    allpass_gain = 0.7

    output = np.zeros(len(mono), dtype=np.float64)
    mono = mono.astype(np.float64)

    for delay_ms, gain in zip(comb_delays_ms, comb_gains):
        delay_samples = int((delay_ms * room_size * 2) * sr / 1000)
        if delay_samples < 1:
            delay_samples = 1
        g = gain * (1 - damping * 0.4)
        output += _comb_filter(mono, delay_samples, g)

    output /= len(comb_delays_ms)

    for delay_ms in allpass_delays_ms:
        delay_samples = int(delay_ms * sr / 1000)
        if delay_samples < 1:
            delay_samples = 1
        output = _allpass_filter(output, delay_samples, allpass_gain)

    # Normalise reverb signal
    peak = np.max(np.abs(output))
    if peak > 1e-10:
        output /= peak

    return output.astype(np.float32)


def _comb_filter(x: np.ndarray, delay: int, gain: float) -> np.ndarray:
    """Feedback comb filter implemented as IIR."""
    # y[n] = x[n] + gain * y[n-delay]
    b = np.zeros(delay + 1, dtype=np.float64)
    a = np.zeros(delay + 1, dtype=np.float64)
    b[0] = 1.0
    a[0] = 1.0
    a[-1] = -gain
    return lfilter(b, a, x).astype(np.float64)


def _allpass_filter(x: np.ndarray, delay: int, gain: float) -> np.ndarray:
    """Schroeder all-pass filter implemented as IIR."""
    # H(z) = (-g + z^-d) / (1 - g z^-d)
    b = np.zeros(delay + 1, dtype=np.float64)
    a = np.zeros(delay + 1, dtype=np.float64)
    b[0] = -gain
    b[-1] = 1.0
    a[0] = 1.0
    a[-1] = -gain
    return lfilter(b, a, x).astype(np.float64)


# ---------------------------------------------------------------------------
# Slapback echo
# ---------------------------------------------------------------------------

def _slapback_echo(
    mono: np.ndarray, sr: int, delay_ms: float = 120.0, decay: float = 0.4
) -> np.ndarray:
    """Single-tap slapback echo."""
    delay_samples = int(delay_ms * sr / 1000)
    out = mono.copy().astype(np.float64)
    if delay_samples < len(mono):
        out[delay_samples:] += decay * mono[: len(mono) - delay_samples]
    return np.clip(out, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Chorus
# ---------------------------------------------------------------------------

def _chorus(
    mono: np.ndarray,
    sr: int,
    depth_ms: float = 8.0,
    rate_hz: float = 0.8,
    wet: float = 0.4,
) -> np.ndarray:
    """
    Mono chorus effect using a modulated delay line.
    """
    mono = mono.astype(np.float64)
    n = len(mono)
    depth_samples = depth_ms * sr / 1000.0

    # Modulation LFO
    t = np.arange(n) / sr
    mod = (np.sin(2 * np.pi * rate_hz * t) * 0.5 + 0.5) * depth_samples
    read_idx = np.arange(n, dtype=np.float64) - mod
    delayed = np.interp(read_idx, np.arange(n, dtype=np.float64), mono, left=0.0, right=0.0)
    out = mono * (1.0 - wet) + delayed * wet
    return _soft_clip(out)


# ---------------------------------------------------------------------------
# Haas stereo expansion from mono
# ---------------------------------------------------------------------------

def _mono_to_stereo_haas(mono: np.ndarray, sr: int, delay_ms: float = 20.0) -> np.ndarray:
    """
    Convert mono to pseudo-stereo using the Haas effect:
    left channel = original, right channel = slightly delayed copy.
    """
    delay_samples = int(delay_ms * sr / 1000)
    right = np.zeros_like(mono)
    if delay_samples < len(mono):
        right[delay_samples:] = mono[: len(mono) - delay_samples]
    return np.stack([mono, right], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Shelving EQ
# ---------------------------------------------------------------------------

def _shelf_filter(
    audio: np.ndarray,
    sr: int,
    cutoff_hz: float,
    gain_db: float,
    shelf_type: str,
) -> np.ndarray:
    """
    Apply a low-shelf or high-shelf EQ filter.

    Uses a simple approach: design a Butterworth filter for the boost/cut
    band and mix it with the dry signal.
    """
    audio = audio.astype(np.float32)
    gain_linear = 10 ** (gain_db / 20.0)
    nyq = sr / 2.0
    cutoff_norm = min(cutoff_hz / nyq, 0.99)

    if shelf_type == "low":
        sos = butter(2, cutoff_norm, btype="low", output="sos")
    else:
        sos = butter(2, cutoff_norm, btype="high", output="sos")

    if audio.ndim == 1:
        band = sosfilt(sos, audio)
        result = audio + (gain_linear - 1.0) * band
        return np.clip(result, -1.0, 1.0).astype(np.float32)

    channels = []
    for ch in range(audio.shape[1]):
        band = sosfilt(sos, audio[:, ch])
        ch_result = audio[:, ch] + (gain_linear - 1.0) * band
        channels.append(np.clip(ch_result, -1.0, 1.0))
    return np.stack(channels, axis=1).astype(np.float32)
