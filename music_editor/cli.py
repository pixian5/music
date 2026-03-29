"""
Command-line interface for the music editor.

Usage examples
--------------
# Auto-detect and remove noise
music-editor denoise input.wav output.wav

# Remove noise using a reference segment (seconds 0-3 are ambient noise)
music-editor denoise input.wav output.wav --noise-start 0 --noise-end 3

# Normalize volume
music-editor normalize input.wav output.wav

# Dynamic loudness levelling
music-editor dynamic-normalize input.wav output.wav

# Studio reverb
music-editor reverb input.wav output.wav --room-size 0.5 --wet 0.3

# KTV effect
music-editor ktv input.wav output.wav

# Male to female voice
music-editor male2female input.wav output.wav --semitones 5

# Female to male voice
music-editor female2male input.wav output.wav --semitones 5

# Stereo widening
music-editor stereo-widen input.wav output.wav --width 1.5

# Pitch shift
music-editor pitch-shift input.wav output.wav --semitones 2

# Bass boost
music-editor bass-boost input.wav output.wav --gain 6

# Treble boost
music-editor treble-boost input.wav output.wav --gain 6

# Fade in
music-editor fade-in input.wav output.wav --duration 2

# Fade out
music-editor fade-out input.wav output.wav --duration 2
"""

import argparse
import sys

from .audio_io import load_audio, save_audio
from .noise_reduction import NoiseReducer
from .effects import AudioEffects


def main(args=None):
    parser = _build_parser()
    ns = parser.parse_args(args)

    if not hasattr(ns, "func"):
        parser.print_help()
        sys.exit(0)

    ns.func(ns)


# ---------------------------------------------------------------------------
# Sub-command implementations
# ---------------------------------------------------------------------------

def _cmd_denoise(ns):
    audio, sr = load_audio(ns.input)
    reducer = NoiseReducer(
        sr,
        prop_decrease=ns.prop_decrease,
    )

    if ns.noise_start is not None and ns.noise_end is not None:
        print(
            f"Using noise reference segment [{ns.noise_start}s – {ns.noise_end}s] …"
        )
        reducer.set_noise_profile_from_segment(audio, ns.noise_start, ns.noise_end)
    else:
        print("Auto-detecting ambient noise profile …")
        reducer.detect_and_set_noise_profile(audio)

    print("Reducing noise …")
    result = reducer.reduce(audio, apply_breath_suppression=False)
    save_audio(ns.output, result, sr)
    print(f"Saved → {ns.output}")


def _cmd_suppress_breath(ns):
    audio, sr = load_audio(ns.input)
    reducer = NoiseReducer(
        sr,
        breath_reduce_strength=ns.strength,
        breath_method=ns.method,
        breath_sensitivity=ns.sensitivity,
        breath_band_focus=ns.band_focus,
    )
    print("Suppressing breath sounds …")
    result = reducer.suppress_breath_sounds(audio)
    save_audio(ns.output, result, sr)
    print(f"Saved → {ns.output}")


def _cmd_normalize(ns):
    audio, sr = load_audio(ns.input)
    fx = AudioEffects(sr)
    result = fx.normalize(audio, target_db=ns.target_db)
    save_audio(ns.output, result, sr)
    print(f"Saved → {ns.output}")


def _cmd_dynamic_normalize(ns):
    audio, sr = load_audio(ns.input)
    fx = AudioEffects(sr)
    result = fx.dynamic_normalize(
        audio,
        window_sec=ns.window,
        target_db=ns.target_db,
        max_gain_db=ns.max_gain,
    )
    save_audio(ns.output, result, sr)
    print(f"Saved → {ns.output}")


def _cmd_reverb(ns):
    audio, sr = load_audio(ns.input)
    fx = AudioEffects(sr)
    result = fx.studio_reverb(audio, room_size=ns.room_size, damping=ns.damping, wet=ns.wet)
    save_audio(ns.output, result, sr)
    print(f"Saved → {ns.output}")


def _cmd_ktv(ns):
    audio, sr = load_audio(ns.input)
    fx = AudioEffects(sr)
    result = fx.ktv_effect(
        audio,
        reverb_wet=ns.reverb_wet,
        echo_delay_ms=ns.echo_delay,
        echo_decay=ns.echo_decay,
    )
    save_audio(ns.output, result, sr)
    print(f"Saved → {ns.output}")


def _cmd_male2female(ns):
    audio, sr = load_audio(ns.input)
    fx = AudioEffects(sr)
    result = fx.male_to_female(audio, semitones=ns.semitones)
    save_audio(ns.output, result, sr)
    print(f"Saved → {ns.output}")


def _cmd_female2male(ns):
    audio, sr = load_audio(ns.input)
    fx = AudioEffects(sr)
    result = fx.female_to_male(audio, semitones=ns.semitones)
    save_audio(ns.output, result, sr)
    print(f"Saved → {ns.output}")


def _cmd_pitch_shift(ns):
    audio, sr = load_audio(ns.input)
    fx = AudioEffects(sr)
    result = fx.pitch_shift(audio, semitones=ns.semitones)
    save_audio(ns.output, result, sr)
    print(f"Saved → {ns.output}")


def _cmd_stereo_widen(ns):
    audio, sr = load_audio(ns.input)
    fx = AudioEffects(sr)
    result = fx.stereo_widen(audio, width=ns.width)
    save_audio(ns.output, result, sr)
    print(f"Saved → {ns.output}")


def _cmd_bass_boost(ns):
    audio, sr = load_audio(ns.input)
    fx = AudioEffects(sr)
    result = fx.bass_boost(audio, gain_db=ns.gain, cutoff_hz=ns.cutoff)
    save_audio(ns.output, result, sr)
    print(f"Saved → {ns.output}")


def _cmd_treble_boost(ns):
    audio, sr = load_audio(ns.input)
    fx = AudioEffects(sr)
    result = fx.treble_boost(audio, gain_db=ns.gain, cutoff_hz=ns.cutoff)
    save_audio(ns.output, result, sr)
    print(f"Saved → {ns.output}")


def _cmd_fade_in(ns):
    audio, sr = load_audio(ns.input)
    fx = AudioEffects(sr)
    result = fx.fade_in(audio, duration_sec=ns.duration)
    save_audio(ns.output, result, sr)
    print(f"Saved → {ns.output}")


def _cmd_fade_out(ns):
    audio, sr = load_audio(ns.input)
    fx = AudioEffects(sr)
    result = fx.fade_out(audio, duration_sec=ns.duration)
    save_audio(ns.output, result, sr)
    print(f"Saved → {ns.output}")


# ---------------------------------------------------------------------------
# Parser builder
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="music-editor",
        description="Music editing software – noise removal and audio effects",
    )
    sub = p.add_subparsers(title="commands")

    # -- denoise
    sp = sub.add_parser("denoise", help="Remove ambient/background noise")
    sp.add_argument("input", help="Input audio file")
    sp.add_argument("output", help="Output audio file")
    sp.add_argument(
        "--noise-start",
        type=float,
        default=None,
        metavar="SEC",
        help="Start of user-selected noise segment (seconds)",
    )
    sp.add_argument(
        "--noise-end",
        type=float,
        default=None,
        metavar="SEC",
        help="End of user-selected noise segment (seconds)",
    )
    sp.add_argument(
        "--prop-decrease",
        type=float,
        default=1.0,
        metavar="0-1",
        help="Proportion of noise to remove (default 1.0)",
    )
    sp.set_defaults(func=_cmd_denoise)

    # -- suppress-breath
    sp = sub.add_parser("suppress-breath", help="Suppress breath sounds only")
    sp.add_argument("input", help="Input audio file")
    sp.add_argument("output", help="Output audio file")
    sp.add_argument(
        "--strength",
        type=float,
        default=0.35,
        metavar="0-1",
        help="Breath suppression strength (default 0.35)",
    )
    sp.add_argument(
        "--method",
        choices=["hybrid", "attenuate", "high_band"],
        default="hybrid",
        help="Breath suppression method",
    )
    sp.add_argument(
        "--sensitivity",
        type=float,
        default=0.5,
        metavar="0-1",
        help="Breath detection sensitivity (default 0.5)",
    )
    sp.add_argument(
        "--band-focus",
        type=float,
        default=0.65,
        metavar="0-1",
        help="How strongly high-band attenuation is focused (default 0.65)",
    )
    sp.set_defaults(func=_cmd_suppress_breath)

    # -- normalize
    sp = sub.add_parser("normalize", help="RMS volume normalisation")
    sp.add_argument("input")
    sp.add_argument("output")
    sp.add_argument("--target-db", type=float, default=-3.0, metavar="dB")
    sp.set_defaults(func=_cmd_normalize)

    # -- dynamic-normalize
    sp = sub.add_parser(
        "dynamic-normalize",
        help="Time-varying loudness levelling (auto-gain)",
    )
    sp.add_argument("input")
    sp.add_argument("output")
    sp.add_argument("--target-db", type=float, default=-6.0, metavar="dB")
    sp.add_argument("--window", type=float, default=0.5, metavar="SEC")
    sp.add_argument("--max-gain", type=float, default=24.0, metavar="dB")
    sp.set_defaults(func=_cmd_dynamic_normalize)

    # -- reverb
    sp = sub.add_parser("reverb", help="Studio recording-room reverb simulation")
    sp.add_argument("input")
    sp.add_argument("output")
    sp.add_argument("--room-size", type=float, default=0.4, metavar="0-1")
    sp.add_argument("--damping", type=float, default=0.5, metavar="0-1")
    sp.add_argument("--wet", type=float, default=0.25, metavar="0-1")
    sp.set_defaults(func=_cmd_reverb)

    # -- ktv
    sp = sub.add_parser("ktv", help="KTV / karaoke effect (reverb + echo + chorus)")
    sp.add_argument("input")
    sp.add_argument("output")
    sp.add_argument("--reverb-wet", type=float, default=0.35, metavar="0-1")
    sp.add_argument("--echo-delay", type=float, default=120.0, metavar="MS")
    sp.add_argument("--echo-decay", type=float, default=0.4, metavar="0-1")
    sp.set_defaults(func=_cmd_ktv)

    # -- male2female
    sp = sub.add_parser("male2female", help="Shift pitch upward (male→female voice)")
    sp.add_argument("input")
    sp.add_argument("output")
    sp.add_argument("--semitones", type=float, default=5.0)
    sp.set_defaults(func=_cmd_male2female)

    # -- female2male
    sp = sub.add_parser("female2male", help="Shift pitch downward (female→male voice)")
    sp.add_argument("input")
    sp.add_argument("output")
    sp.add_argument("--semitones", type=float, default=5.0)
    sp.set_defaults(func=_cmd_female2male)

    # -- pitch-shift
    sp = sub.add_parser("pitch-shift", help="Arbitrary pitch shift (semitones)")
    sp.add_argument("input")
    sp.add_argument("output")
    sp.add_argument("--semitones", type=float, default=0.0)
    sp.set_defaults(func=_cmd_pitch_shift)

    # -- stereo-widen
    sp = sub.add_parser("stereo-widen", help="Stereo image widening")
    sp.add_argument("input")
    sp.add_argument("output")
    sp.add_argument("--width", type=float, default=1.5, metavar="FACTOR")
    sp.set_defaults(func=_cmd_stereo_widen)

    # -- bass-boost
    sp = sub.add_parser("bass-boost", help="Low-shelf bass boost")
    sp.add_argument("input")
    sp.add_argument("output")
    sp.add_argument("--gain", type=float, default=6.0, metavar="dB")
    sp.add_argument("--cutoff", type=float, default=200.0, metavar="HZ")
    sp.set_defaults(func=_cmd_bass_boost)

    # -- treble-boost
    sp = sub.add_parser("treble-boost", help="High-shelf treble boost")
    sp.add_argument("input")
    sp.add_argument("output")
    sp.add_argument("--gain", type=float, default=6.0, metavar="dB")
    sp.add_argument("--cutoff", type=float, default=6000.0, metavar="HZ")
    sp.set_defaults(func=_cmd_treble_boost)

    # -- fade-in
    sp = sub.add_parser("fade-in", help="Apply fade-in at the start")
    sp.add_argument("input")
    sp.add_argument("output")
    sp.add_argument("--duration", type=float, default=1.0, metavar="SEC")
    sp.set_defaults(func=_cmd_fade_in)

    # -- fade-out
    sp = sub.add_parser("fade-out", help="Apply fade-out at the end")
    sp.add_argument("input")
    sp.add_argument("output")
    sp.add_argument("--duration", type=float, default=1.0, metavar="SEC")
    sp.set_defaults(func=_cmd_fade_out)

    return p


if __name__ == "__main__":
    main()
