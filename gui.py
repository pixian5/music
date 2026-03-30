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
import platform
import subprocess
import tempfile
import shutil

import numpy as np
from scipy.signal import stft

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    _MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MATPLOTLIB_AVAILABLE = False

from music_editor.audio_io import load_audio, save_audio
from music_editor.noise_reduction import NoiseReducer
from music_editor.effects import AudioEffects
from music_editor import __version__

OUTPUT_FORMATS = ("wav", "flac", "ogg", "mp3")
_SPECTROGRAM_MAX_NPERSEG = 1024
_SPECTROGRAM_MIN_NPERSEG = 128
_SPECTROGRAM_ABSOLUTE_MIN_NPERSEG = 8
_SPECTROGRAM_MIN_MAGNITUDE = 1e-7
_SPECTROGRAM_MAX_SAMPLES = 200_000


def _replace_extension(path: str, extension: str) -> str:
    base, _ = os.path.splitext(path)
    return f"{base}.{extension.lower()}"


def _suggest_output_path(input_path: str, output_format: str) -> str:
    return _replace_extension(f"{os.path.splitext(input_path)[0]}_output", output_format)


def _frame_mask_to_segments(
    frame_mask: np.ndarray,
    hop_length: int,
    sample_rate: int,
    total_samples: int,
) -> list[tuple[float, float]]:
    if frame_mask.size == 0:
        return []
    idx = np.flatnonzero(frame_mask)
    if idx.size == 0:
        return []
    starts = [int(idx[0])]
    ends: list[int] = []
    for i in range(1, idx.size):
        if idx[i] != idx[i - 1] + 1:
            ends.append(int(idx[i - 1]))
            starts.append(int(idx[i]))
    ends.append(int(idx[-1]))
    segments: list[tuple[float, float]] = []
    max_sample = max(total_samples - 1, 0)
    for start_frame, end_frame in zip(starts, ends):
        start_sample = min(start_frame * hop_length, max_sample)
        end_sample = min((end_frame + 1) * hop_length, total_samples)
        if end_sample <= start_sample:
            continue
        segments.append((start_sample / sample_rate, end_sample / sample_rate))
    return segments


def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
    return audio.mean(axis=1).astype(np.float32) if audio.ndim == 2 else audio.astype(np.float32)


def _prepare_spectrogram_signal(signal: np.ndarray, sample_rate: int) -> tuple[np.ndarray, float]:
    if signal.size <= _SPECTROGRAM_MAX_SAMPLES:
        return signal, float(sample_rate)
    stride = int(np.ceil(signal.size / _SPECTROGRAM_MAX_SAMPLES))
    return signal[::stride], float(sample_rate) / float(stride)


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
        self.title(f"音乐编辑器 v{__version__} – Music Editor")
        self.resizable(True, True)
        self.minsize(640, 500)

        self._audio = None
        self._sr = None
        self._input_path = tk.StringVar()
        self._output_path = tk.StringVar()
        self._output_format = tk.StringVar(value="mp3")
        self._status = tk.StringVar(value="就绪 / Ready")
        self._last_breath_source = None
        self._last_breath_output = None
        self._last_breath_segments = []
        self._selected_breath_segment = None
        self._preview_temp_file = None

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

        self._breath_play_button = ttk.Button(
            frame,
            text="播放选中片段 / Play Selected Segment",
            command=self._play_selected_breath_segment,
            state="disabled",
        )
        self._breath_play_button.grid(row=7, column=0, columnspan=2, sticky="w", pady=(8, 4))
        self._breath_selection_text = tk.StringVar(
            value="未选中片段 / No segment selected"
        )
        ttk.Label(
            frame,
            textvariable=self._breath_selection_text,
            foreground="#2f7d32",
        ).grid(row=8, column=0, columnspan=2, sticky="w")

        self._breath_plot_container = ttk.Frame(frame)
        self._breath_plot_container.grid(row=9, column=0, columnspan=2, sticky="nsew", pady=(8, 0))
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_rowconfigure(9, weight=1)
        if _MATPLOTLIB_AVAILABLE:
            self._breath_figure = Figure(figsize=(7, 3.2), dpi=100)
            self._breath_ax_source = self._breath_figure.add_subplot(211)
            self._breath_ax_output = self._breath_figure.add_subplot(212, sharex=self._breath_ax_source)
            self._breath_canvas = FigureCanvasTkAgg(self._breath_figure, master=self._breath_plot_container)
            self._breath_canvas.get_tk_widget().pack(fill="both", expand=True)
            self._breath_canvas.mpl_connect("button_press_event", self._on_breath_plot_click)
            self._draw_breath_spectrograms()
        else:
            ttk.Label(
                self._breath_plot_container,
                text="未安装 matplotlib，无法显示频谱 / matplotlib not available for spectrogram display",
                foreground="#666",
            ).pack(anchor="w")
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
        mono = _to_mono_float32(self._audio)
        self._last_breath_source = mono
        self._last_breath_output = None
        self._last_breath_segments = []
        self._selected_breath_segment = None
        if hasattr(self, "_breath_play_button"):
            self._breath_play_button.configure(state="disabled")
        if hasattr(self, "_breath_selection_text"):
            self._breath_selection_text.set("未选中片段 / No segment selected")
        if hasattr(self, "_draw_breath_spectrograms"):
            self._draw_breath_spectrograms()
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
            if enabled:
                result, breath_frames = reducer.suppress_breath_sounds_with_frames(audio)
            else:
                result = audio.astype("float32", copy=False)
                breath_frames = np.zeros(0, dtype=bool)
            save_audio(out, result, sr)
            return result, breath_frames, reducer.hop_length

        self._set_status("正在抑制换气音… / Suppressing breath sounds…")
        self._run_async(_work, lambda payload: self._on_breath_suppression_done(payload, out))

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

    def _on_breath_suppression_done(self, payload, out_path: str):
        result, breath_frames, hop_length = payload
        mono_source = (
            self._last_breath_source
            if self._last_breath_source is not None
            else _to_mono_float32(self._audio)
        )
        mono_output = _to_mono_float32(result)
        self._last_breath_source = mono_source
        self._last_breath_output = mono_output
        self._last_breath_segments = _frame_mask_to_segments(
            frame_mask=np.asarray(breath_frames, dtype=bool),
            hop_length=int(max(1, hop_length)),
            sample_rate=int(self._sr),
            total_samples=int(len(mono_source)),
        )
        self._selected_breath_segment = None
        self._breath_play_button.configure(state="disabled")
        if len(self._last_breath_segments) > 0:
            self._breath_selection_text.set(
                f"检测到 {len(self._last_breath_segments)} 段吸气，点击频谱中的绿色区域选择 / "
                f"{len(self._last_breath_segments)} inhale segments detected"
            )
        else:
            self._breath_selection_text.set("未检测到吸气片段 / No inhale segments detected")
        self._draw_breath_spectrograms()
        self._set_status(f"已保存 / Saved → {out_path}")

    def _draw_breath_spectrograms(self):
        if not _MATPLOTLIB_AVAILABLE or not hasattr(self, "_breath_figure"):
            return
        self._breath_ax_source.clear()
        self._breath_ax_output.clear()

        if self._last_breath_source is None or self._sr is None:
            self._breath_ax_source.set_title("源文件频谱 / Source Spectrogram")
            self._breath_ax_output.set_title("输出频谱 / Output Spectrogram")
            self._breath_ax_output.set_xlabel("Time (s)")
            self._breath_figure.tight_layout()
            self._breath_canvas.draw_idle()
            return

        for ax, signal, title in (
            (self._breath_ax_source, self._last_breath_source, "源文件频谱 / Source Spectrogram"),
            (
                self._breath_ax_output,
                self._last_breath_output if self._last_breath_output is not None else self._last_breath_source,
                "输出频谱 / Output Spectrogram",
            ),
        ):
            plot_signal, plot_sr = _prepare_spectrogram_signal(signal, self._sr)
            nperseg = int(
                min(
                    _SPECTROGRAM_MAX_NPERSEG,
                    max(_SPECTROGRAM_ABSOLUTE_MIN_NPERSEG, len(plot_signal)),
                )
            )
            hop = max(1, nperseg // 4)
            noverlap = max(0, nperseg - hop)
            freqs, times, zxx = stft(
                plot_signal,
                fs=plot_sr,
                nperseg=nperseg,
                noverlap=noverlap,
                window="hann",
            )
            mag_db = 20.0 * np.log10(np.abs(zxx) + _SPECTROGRAM_MIN_MAGNITUDE)
            ax.pcolormesh(times, freqs, mag_db, shading="gouraud", cmap="magma")
            ax.set_ylim(0, min(8000, self._sr // 2))
            ax.set_ylabel("Hz")
            ax.set_title(title)
            for idx, (start_t, end_t) in enumerate(self._last_breath_segments):
                color = "#00aa00"
                alpha = 0.22
                if self._selected_breath_segment == idx:
                    color = "#00dd55"
                    alpha = 0.35
                ax.axvspan(start_t, end_t, color=color, alpha=alpha)
        self._breath_ax_output.set_xlabel("Time (s)")
        self._breath_figure.tight_layout()
        self._breath_canvas.draw_idle()

    def _on_breath_plot_click(self, event):
        if (
            (not _MATPLOTLIB_AVAILABLE)
            or not hasattr(self, "_breath_ax_source")
            or not hasattr(self, "_breath_ax_output")
        ):
            return
        if event.inaxes not in {self._breath_ax_source, self._breath_ax_output}:
            return
        if not self._last_breath_segments or event.xdata is None:
            return
        t = float(event.xdata)
        selected = None
        for idx, (start_t, end_t) in enumerate(self._last_breath_segments):
            if start_t <= t <= end_t:
                selected = idx
                break
        if selected is None:
            return
        self._selected_breath_segment = selected
        start_t, end_t = self._last_breath_segments[selected]
        self._breath_play_button.configure(state="normal")
        self._breath_selection_text.set(
            f"已选中片段 / Selected: {start_t:.2f}s - {end_t:.2f}s"
        )
        self._draw_breath_spectrograms()

    def _play_selected_breath_segment(self):
        if self._last_breath_output is None or self._selected_breath_segment is None:
            return
        start_t, end_t = self._last_breath_segments[self._selected_breath_segment]
        start = max(0, int(start_t * self._sr))
        end = min(len(self._last_breath_output), int(end_t * self._sr))
        if end <= start:
            return
        self._cleanup_preview_temp_file()
        segment = self._last_breath_output[start:end]
        with tempfile.NamedTemporaryFile(prefix="music_breath_", suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        save_audio(tmp_path, segment.astype(np.float32), self._sr)
        self._preview_temp_file = tmp_path
        self._play_audio_file(tmp_path)

    def _play_audio_file(self, path: str):
        try:
            system = platform.system()
            if system == "Windows":
                os.startfile(path)  # type: ignore[attr-defined]
            elif system == "Darwin":
                subprocess.run(["open", path], check=False)
            else:
                player = None
                for candidate in ("xdg-open", "ffplay", "aplay", "paplay"):
                    if shutil.which(candidate):
                        player = candidate
                        break
                if player is None:
                    raise RuntimeError(
                        "No suitable audio player found. Please install one of: "
                        "xdg-open, ffplay, aplay, paplay."
                    )
                cmd = [player, path]
                if player == "ffplay":
                    cmd = [player, "-nodisp", "-autoexit", path]
                subprocess.run(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
        except OSError as exc:
            self._set_status(f"播放失败 / Playback failed: {exc}")
            messagebox.showerror("Playback Error", str(exc))
        except subprocess.SubprocessError as exc:
            self._set_status(f"播放器错误 / Player error: {exc}")
            messagebox.showerror("Playback Error", str(exc))
        except RuntimeError as exc:
            self._set_status(f"播放失败 / Playback failed: {exc}")
            messagebox.showerror("Playback Error", str(exc))

    def _cleanup_preview_temp_file(self):
        if self._preview_temp_file and os.path.exists(self._preview_temp_file):
            try:
                os.remove(self._preview_temp_file)
            except OSError:
                pass
        self._preview_temp_file = None

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

    def destroy(self):
        self._cleanup_preview_temp_file()
        super().destroy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = MusicEditorApp()
    app.after(60, app.lift)
    app.after(80, app.focus_force)
    if platform.system() == "Darwin":
        # macOS: try to bring Tk window to foreground for `python gui.py`.
        pid = os.getpid()
        osa = (
            'tell application "System Events" to set frontmost of '
            f'(first process whose unix id is {pid}) to true'
        )
        try:
            subprocess.run(
                ["osascript", "-e", osa],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except Exception:
            # Best-effort only: GUI should still open if this fails.
            pass
    app.mainloop()


if __name__ == "__main__":
    main()
