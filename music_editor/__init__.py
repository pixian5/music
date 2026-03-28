"""
Music Editor - 音乐编辑软件

A music editing tool that supports:
- Noise detection and removal (ambient noise reduction)
- Volume normalization
- Studio reverb simulation
- Voice pitch shifting (male-to-female / female-to-male)
- KTV effect (reverb + echo)
- Stereo widening effect
- And more common audio effects
"""

from .audio_io import load_audio, save_audio
from .noise_reduction import NoiseReducer
from .effects import AudioEffects

__all__ = [
    "load_audio",
    "save_audio",
    "NoiseReducer",
    "AudioEffects",
]

__version__ = "1.0.0"
