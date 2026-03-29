"""
Graphical User Interface for the Music Editor.

Run with:
    python gui.py

Requires tkinter (usually bundled with Python) plus the packages in
requirements.txt.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os

from music_editor.audio_io import load_audio, save_audio
from music_editor.noise_reduction import NoiseReducer
from music_editor.effects import AudioEffects

OUTPUT_FORMATS = ("wav", "flac", "ogg", "mp3")


def _replace_extension(path: str, extension: str) -> str:
    base, _ = os.path.splitext(path)
    return f"{base}.{extension.lower()}"


def _suggest_output_path(input_path: str, output_format: str) -> str:
    return _replace_extension(f"{os.path.splitext(input_path)[0]}_output", output_format)


# ---------------------------------------------------------------------------
# Helper: run a function in a background thread and report completion
# ---------------------------------------------------------------------------

class _Worker(threading.Thread):
    def __init__(self, fn, on_done, on_error):
        super().__init__(daemon=True)
        self._fn = fn
        self._on_done = on_done
        self._on_error = on_error

    def run(self):
        try:
            result = self._fn()
            self._on_done(result)
        except Exception as exc:  # noqa: BLE001
            self._on_error(exc)


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------

class MusicEditorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("音乐编辑器 – Music Editor")
        self.resizable(True, True)
        self.minsize(640, 500)

        self._audio = None
        self._sr = None
        self._input_path = tk.StringVar()
        self._output_path = tk.StringVar()
        self._output_format = tk.StringVar(value="mp3")
        self._status = tk.StringVar(value="就绪 / Ready")

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # ── File selection row ──────────────────────────────────────────
        file_frame = ttk.LabelFrame(self, text="文件 / Files", padding=8)
        file_frame.pack(fill="x", padx=10, pady=6)

        ttk.Label(file_frame, text="输入 / Input:").grid(row=0, column=0, sticky="w")
        ttk.Entry(file_frame, textvariable=self._input_path, width=45).grid(
            row=0, column=1, padx=4
        )
        ttk.Button(file_frame, text="浏览…", command=self._browse_input).grid(
            row=0, column=2
        )

        ttk.Label(file_frame, text="输出 / Output:").grid(
            row=1, column=0, sticky="w", pady=(4, 0)
        )
        ttk.Entry(file_frame, textvariable=self._output_path, width=45).grid(
            row=1, column=1, padx=4, pady=(4, 0)
        )
        ttk.Button(file_frame, text="浏览…", command=self._browse_output).grid(
            row=1, column=2, pady=(4, 0)
        )
        output_format_combo = ttk.Combobox(
            file_frame,
            textvariable=self._output_format,
            values=OUTPUT_FORMATS,
            state="readonly",
            width=8,
        )
        output_format_combo.grid(row=1, column=3, padx=(4, 0), pady=(4, 0))
        output_format_combo.bind("<<ComboboxSelected>>", lambda _: self._on_output_format_selected())

        ttk.Button(file_frame, text="加载 / Load", command=self._load).grid(
            row=2, column=1, pady=(6, 0), sticky="w"
        )

        # ── Notebook with tabs for each feature ────────────────────────
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=10, pady=4)

        nb.add(self._build_breath_tab(nb), text="换气音抑制 / Breath")
        nb.add(self._build_noise_tab(nb), text="降噪 / Noise Removal")
        nb.add(self._build_normalize_tab(nb), text="音量 / Volume")
        nb.add(self._build_reverb_tab(nb), text="混响 / Reverb")
        nb.add(self._build_ktv_tab(nb), text="KTV")
        nb.add(self._build_pitch_tab(nb), text="变调 / Pitch")
        nb.add(self._build_stereo_tab(nb), text="立体声 / Stereo")
        nb.add(self._build_eq_tab(nb), text="均衡器 / EQ")
        nb.add(self._build_fade_tab(nb), text="淡入淡出 / Fade")

        # ── Status bar ─────────────────────────────────────────────────
        ttk.Label(self, textvariable=self._status, relief="sunken",
                  anchor="w").pack(fill="x", padx=10, pady=(0, 4))

    # ------------------------------------------------------------------
    # Tab builders
    # ------------------------------------------------------------------

    def _build_noise_tab(self, parent):
        frame = ttk.Frame(parent, padding=10)

        ttk.Label(frame, text="降噪模式 / Noise reduction mode:").grid(
            row=0, column=0, columnspan=2, sticky="w"
        )

        self._noise_mode = tk.StringVar(value="auto")
        ttk.Radiobutton(
            frame, text="自动检测 / Auto-detect", variable=self._noise_mode,
            value="auto", command=self._toggle_noise_seg
        ).grid(row=1, column=0, sticky="w")
        ttk.Radiobutton(
            frame, text="用户选择片段 / User segment", variable=self._noise_mode,
            value="manual", command=self._toggle_noise_seg
        ).grid(row=1, column=1, sticky="w")

        self._noise_seg_frame = ttk.Frame(frame)
        self._noise_seg_frame.grid(row=2, column=0, columnspan=2, pady=4)

        ttk.Label(self._noise_seg_frame, text="噪音片段开始 / Start (s):").grid(
            row=0, column=0
        )
        self._noise_start = tk.DoubleVar(value=0.0)
        ttk.Entry(self._noise_seg_frame, textvariable=self._noise_start, width=8).grid(
            row=0, column=1, padx=4
        )
        ttk.Label(self._noise_seg_frame, text="结束 / End (s):").grid(row=0, column=2)
        self._noise_end = tk.DoubleVar(value=3.0)
        ttk.Entry(self._noise_seg_frame, textvariable=self._noise_end, width=8).grid(
            row=0, column=3, padx=4
        )

        ttk.Label(frame, text="降噪强度 / Reduction strength (0–100):").grid(
            row=3, column=0, sticky="w", pady=(8, 0)
        )
        self._prop_decrease = tk.DoubleVar(value=100.0)
        ttk.Scale(frame, from_=0.0, to=100.0, variable=self._prop_decrease,
                  orient="horizontal", length=220).grid(row=3, column=1, sticky="w",
                                                        pady=(8, 0))

        ttk.Button(frame, text="执行降噪 / Apply Noise Removal",
                   command=self._run_denoise).grid(
            row=4, column=0, columnspan=2, pady=10
        )

        self._toggle_noise_seg()
        return frame

    def _build_breath_tab(self, parent):
        frame = ttk.Frame(parent, padding=10)
        self._breath_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            frame,
            text="启用换气音抑制 / Enable breath suppression",
            variable=self._breath_enabled,
        ).grid(row=0, column=0, columnspan=2, sticky="w")

        ttk.Label(frame, text="抑制方法 / Method:").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        self._breath_method = tk.StringVar(value="extreme")
        breath_method_combo = ttk.Combobox(
            frame,
            textvariable=self._breath_method,
            values=("extreme", "ultra", "deep", "hybrid", "attenuate", "high_band"),
            state="readonly",
            width=16,
        )
        breath_method_combo.grid(row=1, column=1, sticky="w", pady=(8, 0))

        ttk.Label(frame, text="抑制强度 / Strength (0–100):").grid(
            row=2, column=0, sticky="w", pady=(8, 0)
        )
        self._breath_strength = tk.DoubleVar(value=100.0)
        ttk.Scale(
            frame,
            from_=0.0,
            to=100.0,
            variable=self._breath_strength,
            orient="horizontal",
            length=220,
        ).grid(row=2, column=1, sticky="w", pady=(8, 0))

        ttk.Label(frame, text="检测灵敏度 / Sensitivity (0–100):").grid(
            row=3, column=0, sticky="w", pady=(8, 0)
        )
        self._breath_sensitivity = tk.DoubleVar(value=100.0)
        ttk.Scale(
            frame,
            from_=0.0,
            to=100.0,
            variable=self._breath_sensitivity,
            orient="horizontal",
            length=220,
        ).grid(row=3, column=1, sticky="w", pady=(8, 0))

        ttk.Label(frame, text="高频重点 / High-band focus (0–100):").grid(
            row=4, column=0, sticky="w", pady=(8, 0)
        )
        self._breath_band_focus = tk.DoubleVar(value=100.0)
        ttk.Scale(
            frame,
            from_=0.0,
            to=100.0,
            variable=self._breath_band_focus,
            orient="horizontal",
            length=220,
        ).grid(row=4, column=1, sticky="w", pady=(8, 0))

        ttk.Button(frame, text="执行换气音抑制 / Apply Breath Suppression",
                   command=self._run_breath_suppression).grid(
            row=5, column=0, columnspan=2, pady=10
        )
        ttk.Label(
            frame,
            text=(
                "提示：若换气音仍明显，建议 method=deep，strength≥75，"
                "sensitivity≥75，high-band focus≥80；"
                "极难去除时改用 method=ultra 或 method=extreme（最强）。"
            ),
            foreground="#555",
            wraplength=520,
            justify="left",
        ).grid(row=6, column=0, columnspan=2, sticky="w")
        return frame

    def _toggle_noise_seg(self):
        state = "normal" if self._noise_mode.get() == "manual" else "disabled"
        for child in self._noise_seg_frame.winfo_children():
            child.configure(state=state)

    def _build_normalize_tab(self, parent):
        frame = ttk.Frame(parent, padding=10)
        ttk.Label(frame, text="目标 RMS 电平 / Target RMS level (dBFS):").grid(
            row=0, column=0, sticky="w"
        )
        self._norm_db = tk.DoubleVar(value=-3.0)
        ttk.Entry(frame, textvariable=self._norm_db, width=8).grid(
            row=0, column=1, padx=6
        )

        self._dyn_norm = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frame, text="动态均衡 / Dynamic levelling", variable=self._dyn_norm
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=4)

        ttk.Button(frame, text="执行 / Apply", command=self._run_normalize).grid(
            row=2, column=0, columnspan=2, pady=8
        )
        return frame

    def _build_reverb_tab(self, parent):
        frame = ttk.Frame(parent, padding=10)
        params = [
            ("房间大小 / Room size (0–1):", "rev_room", 0.4),
            ("阻尼 / Damping (0–1):", "rev_damp", 0.5),
            ("湿声比 / Wet mix (0–1):", "rev_wet", 0.25),
        ]
        self._rev_vars = {}
        for i, (label, key, default) in enumerate(params):
            ttk.Label(frame, text=label).grid(row=i, column=0, sticky="w", pady=3)
            var = tk.DoubleVar(value=default)
            self._rev_vars[key] = var
            ttk.Scale(frame, from_=0.0, to=1.0, variable=var,
                      orient="horizontal", length=200).grid(row=i, column=1, padx=6)
        ttk.Button(frame, text="执行 / Apply Studio Reverb",
                   command=self._run_reverb).grid(
            row=len(params), column=0, columnspan=2, pady=8
        )
        return frame

    def _build_ktv_tab(self, parent):
        frame = ttk.Frame(parent, padding=10)
        ttk.Label(frame, text="混响湿声 / Reverb wet (0–1):").grid(
            row=0, column=0, sticky="w"
        )
        self._ktv_wet = tk.DoubleVar(value=0.35)
        ttk.Scale(frame, from_=0.0, to=1.0, variable=self._ktv_wet,
                  orient="horizontal", length=200).grid(row=0, column=1)
        ttk.Label(frame, text="回声延迟 / Echo delay (ms):").grid(
            row=1, column=0, sticky="w", pady=4
        )
        self._ktv_echo_ms = tk.DoubleVar(value=120.0)
        ttk.Entry(frame, textvariable=self._ktv_echo_ms, width=8).grid(row=1, column=1, sticky="w")
        ttk.Label(frame, text="回声衰减 / Echo decay (0–1):").grid(
            row=2, column=0, sticky="w"
        )
        self._ktv_decay = tk.DoubleVar(value=0.4)
        ttk.Scale(frame, from_=0.0, to=1.0, variable=self._ktv_decay,
                  orient="horizontal", length=200).grid(row=2, column=1)
        ttk.Button(frame, text="执行 KTV 效果 / Apply KTV Effect",
                   command=self._run_ktv).grid(
            row=3, column=0, columnspan=2, pady=8
        )
        return frame

    def _build_pitch_tab(self, parent):
        frame = ttk.Frame(parent, padding=10)
        ttk.Label(frame, text="变调模式 / Pitch mode:").grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        self._pitch_mode = tk.StringVar(value="custom")
        modes = [
            ("男转女 / Male→Female (+5 st)", "m2f"),
            ("女转男 / Female→Male (−5 st)", "f2m"),
            ("自定义 / Custom", "custom"),
        ]
        for i, (text, val) in enumerate(modes):
            ttk.Radiobutton(frame, text=text, variable=self._pitch_mode,
                            value=val).grid(row=i + 1, column=0, sticky="w")

        ttk.Label(frame, text="自定义半音数 / Custom semitones:").grid(
            row=len(modes) + 1, column=0, sticky="w", pady=(8, 0)
        )
        self._semitones = tk.DoubleVar(value=0.0)
        ttk.Entry(frame, textvariable=self._semitones, width=8).grid(
            row=len(modes) + 1, column=1, pady=(8, 0)
        )

        ttk.Button(frame, text="执行变调 / Apply Pitch Shift",
                   command=self._run_pitch).grid(
            row=len(modes) + 2, column=0, columnspan=2, pady=8
        )
        return frame

    def _build_stereo_tab(self, parent):
        frame = ttk.Frame(parent, padding=10)
        ttk.Label(frame, text="立体声宽度 / Stereo width (1=unchanged, >1 wider):").grid(
            row=0, column=0, sticky="w"
        )
        self._stereo_width = tk.DoubleVar(value=1.5)
        ttk.Scale(frame, from_=0.0, to=3.0, variable=self._stereo_width,
                  orient="horizontal", length=220).grid(row=0, column=1, padx=6)
        ttk.Button(frame, text="执行 / Apply Stereo Widening",
                   command=self._run_stereo).grid(
            row=1, column=0, columnspan=2, pady=8
        )
        return frame

    def _build_eq_tab(self, parent):
        frame = ttk.Frame(parent, padding=10)

        ttk.Label(frame, text="低音增益 / Bass boost (dB):").grid(
            row=0, column=0, sticky="w"
        )
        self._bass_gain = tk.DoubleVar(value=6.0)
        ttk.Scale(frame, from_=0.0, to=24.0, variable=self._bass_gain,
                  orient="horizontal", length=200).grid(row=0, column=1)

        ttk.Label(frame, text="低频截止 / Bass cutoff (Hz):").grid(
            row=1, column=0, sticky="w"
        )
        self._bass_cutoff = tk.DoubleVar(value=200.0)
        ttk.Entry(frame, textvariable=self._bass_cutoff, width=8).grid(row=1, column=1, sticky="w")

        ttk.Button(frame, text="低音增强 / Bass Boost",
                   command=self._run_bass).grid(row=2, column=0, pady=6)

        ttk.Separator(frame, orient="horizontal").grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=4
        )

        ttk.Label(frame, text="高音增益 / Treble boost (dB):").grid(
            row=4, column=0, sticky="w"
        )
        self._treble_gain = tk.DoubleVar(value=6.0)
        ttk.Scale(frame, from_=0.0, to=24.0, variable=self._treble_gain,
                  orient="horizontal", length=200).grid(row=4, column=1)

        ttk.Label(frame, text="高频截止 / Treble cutoff (Hz):").grid(
            row=5, column=0, sticky="w"
        )
        self._treble_cutoff = tk.DoubleVar(value=6000.0)
        ttk.Entry(frame, textvariable=self._treble_cutoff, width=8).grid(row=5, column=1, sticky="w")

        ttk.Button(frame, text="高音增强 / Treble Boost",
                   command=self._run_treble).grid(row=6, column=0, pady=6)

        return frame

    def _build_fade_tab(self, parent):
        frame = ttk.Frame(parent, padding=10)
        ttk.Label(frame, text="淡入时长 / Fade-in duration (s):").grid(
            row=0, column=0, sticky="w"
        )
        self._fade_in_dur = tk.DoubleVar(value=1.0)
        ttk.Entry(frame, textvariable=self._fade_in_dur, width=8).grid(row=0, column=1)
        ttk.Button(frame, text="淡入 / Fade In", command=self._run_fade_in).grid(
            row=0, column=2, padx=8
        )

        ttk.Label(frame, text="淡出时长 / Fade-out duration (s):").grid(
            row=1, column=0, sticky="w", pady=8
        )
        self._fade_out_dur = tk.DoubleVar(value=1.0)
        ttk.Entry(frame, textvariable=self._fade_out_dur, width=8).grid(row=1, column=1)
        ttk.Button(frame, text="淡出 / Fade Out", command=self._run_fade_out).grid(
            row=1, column=2, padx=8
        )

        return frame

    # ------------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------------

    def _browse_input(self):
        path = filedialog.askopenfilename(
            title="选择输入文件 / Select input file",
            filetypes=[("Audio files", "*.wav *.flac *.ogg *.mp3 *.m4a *.mp4 *.aac *.wma"), ("All files", "*.*")],
        )
        if path:
            self._input_path.set(path)
            self._output_path.set(_suggest_output_path(path, self._output_format.get()))
            self._load()

    def _browse_output(self):
        selected_format = self._output_format.get().lower()
        path = filedialog.asksaveasfilename(
            title="选择输出文件 / Select output file",
            defaultextension=f".{selected_format}",
            filetypes=[("WAV", "*.wav"), ("FLAC", "*.flac"), ("OGG", "*.ogg"), ("MP3", "*.mp3")],
        )
        if path:
            self._output_path.set(path)
            chosen_ext = os.path.splitext(path)[1].lstrip(".").lower()
            if chosen_ext in OUTPUT_FORMATS:
                self._output_format.set(chosen_ext)

    def _on_output_format_selected(self):
        output_path = self._output_path.get().strip()
        if output_path:
            self._output_path.set(_replace_extension(output_path, self._output_format.get()))

    def _load(self):
        path = self._input_path.get()
        if not path:
            messagebox.showwarning("提示", "请先选择输入文件 / Please select an input file first.")
            return
        self._set_status("正在加载… / Loading…")
        self._run_async(
            lambda: load_audio(path),
            self._on_loaded,
        )

    def _on_loaded(self, result):
        self._audio, self._sr = result
        n = self._audio.shape[0] if self._audio.ndim == 2 else len(self._audio)
        dur = n / self._sr
        ch = self._audio.shape[1] if self._audio.ndim == 2 else 1
        self._set_status(
            f"已加载 / Loaded: {dur:.1f}s, {self._sr} Hz, {ch} ch"
        )

    # ------------------------------------------------------------------
    # Effect runners
    # ------------------------------------------------------------------

    def _require_audio(self) -> bool:
        if self._audio is None:
            messagebox.showwarning("提示", "请先加载音频文件。\nPlease load an audio file first.")
            return False
        return True

    def _require_output(self) -> bool:
        if not self._output_path.get():
            messagebox.showwarning("提示", "请先指定输出文件。\nPlease specify an output file.")
            return False
        return True

    def _run_denoise(self):
        if not self._require_audio() or not self._require_output():
            return
        audio = self._audio.copy()
        sr = self._sr
        mode = self._noise_mode.get()
        prop = self._prop_decrease.get() / 100.0
        start = self._noise_start.get()
        end = self._noise_end.get()
        out = self._output_path.get()

        def _work():
            reducer = NoiseReducer(
                sr,
                prop_decrease=prop,
            )
            if mode == "manual":
                reducer.set_noise_profile_from_segment(audio, start, end)
            else:
                reducer.detect_and_set_noise_profile(audio)
            result = reducer.reduce(audio, apply_breath_suppression=False)
            save_audio(out, result, sr)

        self._set_status("正在降噪… / Reducing noise…")
        self._run_async(_work, lambda _: self._set_status(f"已保存 / Saved → {out}"))

    def _run_breath_suppression(self):
        if not self._require_audio() or not self._require_output():
            return
        audio = self._audio.copy()
        sr = self._sr
        enabled = self._breath_enabled.get()
        method = self._breath_method.get()
        strength = self._breath_strength.get() / 100.0
        sensitivity = self._breath_sensitivity.get() / 100.0
        band_focus = self._breath_band_focus.get() / 100.0
        out = self._output_path.get()

        def _work():
            reducer = NoiseReducer(
                sr,
                breath_reduce_strength=strength,
                breath_method=method,
                breath_sensitivity=sensitivity,
                breath_band_focus=band_focus,
            )
            result = (
                reducer.suppress_breath_sounds(audio)
                if enabled
                else audio.astype("float32", copy=False)
            )
            save_audio(out, result, sr)

        self._set_status("正在抑制换气音… / Suppressing breath sounds…")
        self._run_async(_work, lambda _: self._set_status(f"已保存 / Saved → {out}"))

    def _run_normalize(self):
        if not self._require_audio() or not self._require_output():
            return
        audio = self._audio.copy()
        sr = self._sr
        db = self._norm_db.get()
        dyn = self._dyn_norm.get()
        out = self._output_path.get()

        def _work():
            fx = AudioEffects(sr)
            if dyn:
                result = fx.dynamic_normalize(audio, target_db=db)
            else:
                result = fx.normalize(audio, target_db=db)
            save_audio(out, result, sr)

        self._set_status("正在处理… / Processing…")
        self._run_async(_work, lambda _: self._set_status(f"已保存 / Saved → {out}"))

    def _run_reverb(self):
        if not self._require_audio() or not self._require_output():
            return
        audio = self._audio.copy()
        sr = self._sr
        room = self._rev_vars["rev_room"].get()
        damp = self._rev_vars["rev_damp"].get()
        wet = self._rev_vars["rev_wet"].get()
        out = self._output_path.get()

        def _work():
            fx = AudioEffects(sr)
            result = fx.studio_reverb(audio, room_size=room, damping=damp, wet=wet)
            save_audio(out, result, sr)

        self._set_status("正在添加混响… / Applying reverb…")
        self._run_async(_work, lambda _: self._set_status(f"已保存 / Saved → {out}"))

    def _run_ktv(self):
        if not self._require_audio() or not self._require_output():
            return
        audio = self._audio.copy()
        sr = self._sr
        wet = self._ktv_wet.get()
        echo_ms = self._ktv_echo_ms.get()
        decay = self._ktv_decay.get()
        out = self._output_path.get()

        def _work():
            fx = AudioEffects(sr)
            result = fx.ktv_effect(audio, reverb_wet=wet, echo_delay_ms=echo_ms, echo_decay=decay)
            save_audio(out, result, sr)

        self._set_status("正在添加KTV效果… / Applying KTV…")
        self._run_async(_work, lambda _: self._set_status(f"已保存 / Saved → {out}"))

    def _run_pitch(self):
        if not self._require_audio() or not self._require_output():
            return
        audio = self._audio.copy()
        sr = self._sr
        mode = self._pitch_mode.get()
        semitones = self._semitones.get()
        out = self._output_path.get()

        def _work():
            fx = AudioEffects(sr)
            if mode == "m2f":
                result = fx.male_to_female(audio)
            elif mode == "f2m":
                result = fx.female_to_male(audio)
            else:
                result = fx.pitch_shift(audio, semitones=semitones)
            save_audio(out, result, sr)

        self._set_status("正在变调… / Shifting pitch…")
        self._run_async(_work, lambda _: self._set_status(f"已保存 / Saved → {out}"))

    def _run_stereo(self):
        if not self._require_audio() or not self._require_output():
            return
        audio = self._audio.copy()
        sr = self._sr
        width = self._stereo_width.get()
        out = self._output_path.get()

        def _work():
            fx = AudioEffects(sr)
            result = fx.stereo_widen(audio, width=width)
            save_audio(out, result, sr)

        self._set_status("正在扩展立体声… / Widening stereo…")
        self._run_async(_work, lambda _: self._set_status(f"已保存 / Saved → {out}"))

    def _run_bass(self):
        if not self._require_audio() or not self._require_output():
            return
        audio = self._audio.copy()
        sr = self._sr
        gain = self._bass_gain.get()
        cutoff = self._bass_cutoff.get()
        out = self._output_path.get()

        def _work():
            fx = AudioEffects(sr)
            result = fx.bass_boost(audio, gain_db=gain, cutoff_hz=cutoff)
            save_audio(out, result, sr)

        self._set_status("正在低音增强… / Boosting bass…")
        self._run_async(_work, lambda _: self._set_status(f"已保存 / Saved → {out}"))

    def _run_treble(self):
        if not self._require_audio() or not self._require_output():
            return
        audio = self._audio.copy()
        sr = self._sr
        gain = self._treble_gain.get()
        cutoff = self._treble_cutoff.get()
        out = self._output_path.get()

        def _work():
            fx = AudioEffects(sr)
            result = fx.treble_boost(audio, gain_db=gain, cutoff_hz=cutoff)
            save_audio(out, result, sr)

        self._set_status("正在高音增强… / Boosting treble…")
        self._run_async(_work, lambda _: self._set_status(f"已保存 / Saved → {out}"))

    def _run_fade_in(self):
        if not self._require_audio() or not self._require_output():
            return
        audio = self._audio.copy()
        sr = self._sr
        dur = self._fade_in_dur.get()
        out = self._output_path.get()

        def _work():
            fx = AudioEffects(sr)
            result = fx.fade_in(audio, duration_sec=dur)
            save_audio(out, result, sr)

        self._set_status("正在淡入… / Applying fade in…")
        self._run_async(_work, lambda _: self._set_status(f"已保存 / Saved → {out}"))

    def _run_fade_out(self):
        if not self._require_audio() or not self._require_output():
            return
        audio = self._audio.copy()
        sr = self._sr
        dur = self._fade_out_dur.get()
        out = self._output_path.get()

        def _work():
            fx = AudioEffects(sr)
            result = fx.fade_out(audio, duration_sec=dur)
            save_audio(out, result, sr)

        self._set_status("正在淡出… / Applying fade out…")
        self._run_async(_work, lambda _: self._set_status(f"已保存 / Saved → {out}"))

    # ------------------------------------------------------------------
    # Async helper
    # ------------------------------------------------------------------

    def _run_async(self, fn, on_done):
        def _error(exc):
            self._set_status(f"错误 / Error: {exc}")
            messagebox.showerror("Error", str(exc))

        _Worker(fn, lambda r: self.after(0, lambda: on_done(r)),
                lambda e: self.after(0, lambda: _error(e))).start()

    def _set_status(self, msg: str):
        self._status.set(msg)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = MusicEditorApp()
    app.mainloop()


if __name__ == "__main__":
    main()
