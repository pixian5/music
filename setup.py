from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="music-editor",
    version="1.0.0",
    description="Music editing software with noise removal and audio effects",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.23.0",
        "scipy>=1.9.0",
        "soundfile>=0.12.1",
        "librosa>=0.10.0",
        "noisereduce>=3.0.0",
    ],
    entry_points={
        "console_scripts": [
            "music-editor=music_editor.cli:main",
        ],
    },
)
