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
- **换气音抑制（唱歌）** – 在降噪后使用频谱平坦度、频带能量比例与峰值特征，保守削弱换气噪声，并尽量保留轻声人声  
  **Breath suppression (singing)** – after denoise, conservatively attenuates likely breath sounds using spectral flatness, band-energy ratio and peakiness features, while preserving soft vocals.

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

### 命令行 / Command-Line Interface (CLI)

```bash
# 自动降噪 / Auto noise removal
music-editor denoise input.wav output.mp3

# 用户选取噪音片段（前3秒为纯噪音）/ User noise segment (first 3s is ambient noise)
music-editor denoise input.wav output.mp3 --noise-start 0 --noise-end 3

# 调整换气音抑制强度 / Tune breath suppression strength
music-editor denoise input.wav output.mp3 --breath-reduce-strength 0.5

# 仅降环境噪音，不抑制换气音 / Disable breath suppression
music-editor denoise input.wav output.mp3 --no-breath-remove

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

“唱歌换气音”功能位置：在图形界面 **降噪 / Noise Removal** 标签页，可通过“抑制换气音 / Suppress breath sounds”独立开关控制，并可用“换气音抑制强度 / Breath strength”细调。
Where to find “breath suppression for singing”: in the GUI **降噪 / Noise Removal** tab. It is controlled by an independent “Suppress breath sounds” toggle and a dedicated breath-strength control.

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
