# Audio Transcription with Speaker Diarization

A single-file Python tool that transcribes audio files and identifies different speakers using OpenAI Whisper and Pyannote Audio.

## Features

- Automatic transcription (100+ languages)
- Speaker identification (who said what)
- Multi-format support (WAV, MP3, M4A, FLAC, AAC, OGG)
- GPU acceleration (optional)
- Simple GUI file picker

## Requirements

- Python 3.8+
- FFmpeg
- 4GB RAM minimum (8GB+ recommended)
- ~10GB storage for models
- NVIDIA GPU (optional, for faster processing)

## Installation

### 1. Install FFmpeg

**Windows:**
```powershell
choco install ffmpeg
```

**Linux:**
```bash
sudo apt update && sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

**For GPU support (CUDA 11.8):**
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install openai-whisper pyannote.audio transformers huggingface-hub
```

**For CPU only:**
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install openai-whisper pyannote.audio transformers huggingface-hub
```

### 3. Setup Hugging Face Token

1. Create account at [huggingface.co](https://huggingface.co)
2. Get token from [Settings → Access Tokens](https://huggingface.co/settings/tokens)
3. Accept terms for:
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
4. Create file named `token` in project directory
5. Paste your token (single line, no quotes)

## Usage

```bash
python generate-transcript-modern.py
```

1. Select your audio file in the file picker
2. Enter number of speakers
3. Wait for processing
4. Find output: `<filename>_transcription.txt`

## Output Format

```
[SPEAKER_00]:
Hello everyone, welcome to the meeting.

[SPEAKER_01]:
Thanks for having me.
```

## Platform Compatibility

| Platform | Status | GPU Support |
|----------|--------|-------------|
| Windows 10/11 | ✅ | NVIDIA CUDA |
| Linux | ✅ | NVIDIA CUDA |
| macOS | ✅ | Apple Silicon (M1/M2/M3) |

## Troubleshooting

**"token file not found"**
- Create a file named `token` with your Hugging Face token

**"FFmpeg not found"**
- Install FFmpeg and ensure it's in your PATH
- Verify: `ffmpeg -version`

**"Failed to load pyannote model"**
- Check your Hugging Face token is valid
- Accept model terms on Hugging Face website

**"CUDA out of memory"**
- Close other GPU applications
- Edit line 164: change `"large-v3"` to `"medium"`

**"No module named 'tkinter'" (Linux)**
```bash
sudo apt install python3-tk
```

## License

Uses open-source components:
- [OpenAI Whisper](https://github.com/openai/whisper) - MIT License
- [Pyannote Audio](https://github.com/pyannote/pyannote-audio) - MIT License
- [PyTorch](https://pytorch.org/) - BSD-style License
