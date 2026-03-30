# 音乐编辑器 / Music Editor

一款支持**噪音消除**和**多种声音效果**的音频编辑软件。
A music editing application with noise removal and a rich set of audio effects.

---

## 功能 / Features

### 噪音消除 / Noise Removal
- **自动检测** – 自动识别音频中最安静的片段作为环境噪音基准  
  **Auto-detect** – automatically finds the quietest frames as the noise reference.
- **用户选取片段** – 用户指定某段纯噪音区间（秒），程序以此为基准在整段音频中去除类似噪音  
  **User segment** – user specifies a time range containing only ambient noise; that profile is subtracted from the entire recording.
- **换气音抑制（唱歌）** – 在降噪后使用频谱平坦度、频带能量比例、帧能量与峰值特征，自适应削弱换气噪声；支持 `deep / ultra` 强抑制模式用于明显吸气声  
  **Breath suppression (singing)** – after denoise, adaptively attenuates likely breath sounds using spectral flatness, band-energy ratio, frame energy and peakiness features; supports stronger `deep / ultra` modes for obvious inhales.

### 声音效果 / Audio Effects
| 效果 | 说明 |
|------|------|
| 音量归一化 / Normalize | RMS 均一化，让整体音量稳定在目标电平 |
| 动态均衡 / Dynamic Normalize | 时变增益，让全曲音量保持一致（类似 Auto-Gain）|
| 录音棚混响 / Studio Reverb | Schroeder 混响器，模拟录音棚房间声 |
| KTV 效果 / KTV Effect | 混响 + 回声 + 和声，模拟 KTV 麦克风效果 |
| 男转女 / Male→Female | 升调约 +5 半音，模拟女性音色 |
| 女转男 / Female→Male | 降调约 −5 半音，模拟男性音色 |
| 自定义变调 / Pitch Shift | 任意半音数的音高偏移（相位声码器）|
| 立体声扩展 / Stereo Widen | 中侧（M/S）处理，拓宽立体声声场 |
| 低音增强 / Bass Boost | 低频搁架滤波器增益 |
| 高音增强 / Treble Boost | 高频搁架滤波器增益 |
| 淡入淡出 / Fade In/Out | 线性振幅斜坡 |

---

## 安装 / Installation

```bash
pip install -r requirements.txt
pip install -e .
```

---

## 使用方法 / Usage

### 图形界面 / Graphical Interface (GUI)

```bash
python gui.py
```

> macOS 前台启动说明 / macOS foreground launch  
> 请直接在前台运行 `python gui.py`（不要加 `&`）。程序会尝试自动置顶窗口；若仍未置顶，请点击 Dock 中的 Python 图标切回前台。

### 命令行 / Command-Line Interface (CLI)

```bash
# 自动降噪 / Auto noise removal
music-editor denoise input.wav output.mp3

# 用户选取噪音片段（前3秒为纯噪音）/ User noise segment (first 3s is ambient noise)
music-editor denoise input.wav output.mp3 --noise-start 0 --noise-end 3

# 换气音抑制（独立命令）/ Breath suppression (standalone command)
music-editor suppress-breath input.wav output.mp3 --strength 0.85 --sensitivity 0.85 --band-focus 0.9 --method ultra

# 音量归一化 / Volume normalization
music-editor normalize input.wav output.wav

# 动态音量均衡 / Dynamic volume levelling
music-editor dynamic-normalize input.wav output.wav

# 录音棚混响 / Studio reverb
music-editor reverb input.wav output.wav --room-size 0.5 --wet 0.3

# KTV 效果 / KTV effect
music-editor ktv input.wav output.wav

# 男转女 / Male to female
music-editor male2female input.wav output.wav --semitones 5

# 女转男 / Female to male
music-editor female2male input.wav output.wav --semitones 5

# 自定义变调 / Custom pitch shift
music-editor pitch-shift input.wav output.wav --semitones 2

# 立体声扩展 / Stereo widening
music-editor stereo-widen input.wav output.wav --width 1.5

# 低音增强 / Bass boost
music-editor bass-boost input.wav output.wav --gain 6

# 高音增强 / Treble boost
music-editor treble-boost input.wav output.wav --gain 6

# 淡入 / Fade in
music-editor fade-in input.wav output.wav --duration 2

# 淡出 / Fade out
music-editor fade-out input.wav output.wav --duration 2
```

支持的格式 / Supported formats: WAV, FLAC, OGG, MP3, M4A, AAC, WMA (read), WAV, FLAC, OGG, MP3 (write).

“唱歌换气音”功能位置：在图形界面最前面的 **换气音抑制 / Breath** 标签页，提供独立开关、抑制方法（ultra / deep / hybrid / attenuate / high_band）、抑制强度、检测灵敏度和高频重点的细粒度控制。  
建议参数（换气音明显时）：`method=deep, strength>=75, sensitivity>=75, high-band focus>=80`；若仍明显，改为 `method=ultra` 并将其余参数提高到 `85+`（GUI 为 0–100 刻度）。
该标签页会显示源文件与输出文件的频谱（上下两幅图），并用绿色标注识别到的吸气片段。可在频谱图上点击绿色片段进行选择，然后点击“播放选中片段”按钮试听该片段。  
Where to find “breath suppression for singing”: the first GUI tab is **换气音抑制 / Breath**, with independent toggle, method selection (ultra / deep / hybrid / attenuate / high_band), strength, sensitivity, and high-band focus controls.  
Suggested stronger settings (for obvious inhales): `method=deep, strength>=75, sensitivity>=75, high-band focus>=80`; if still obvious, switch to `method=ultra` and raise others to `85+` (GUI uses a 0–100 scale).
This tab now shows source/output spectrograms (top/bottom) and highlights detected inhale segments in green. Click a green segment on the spectrogram, then press “Play Selected Segment” to preview it.

---

## 项目结构 / Project Structure

```
music_editor/
├── __init__.py          # Package entry point
├── audio_io.py          # Audio loading and saving utilities
├── noise_reduction.py   # Noise detection and spectral subtraction
└── effects.py           # Audio effects (normalise, reverb, pitch, etc.)
gui.py                   # Tkinter graphical interface
tests/
├── test_noise_reduction.py
└── test_effects.py
requirements.txt
setup.py
```

## 运行测试 / Running Tests

```bash
pip install pytest
pytest tests/ -v
```
