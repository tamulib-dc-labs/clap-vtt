# Instrumentalist

Generate descriptive WebVTT caption files from audio or video recordings using [CLAP](https://github.com/LAION-AI/CLAP) (Contrastive Language-Audio Pretraining). Instead of transcribing speech, it classifies what is *happening* in the audio — instruments playing, crowd noise, announcer speech, silence — based on natural-language labels you define.

## Requirements

- Python 3.14+
- [uv](https://docs.astral.sh/uv/)
- `ffmpeg` on your PATH

Python dependencies are declared in `pyproject.toml` and installed automatically by `uv`.

On first run, the CLAP model (`laion/larger_clap_music_and_speech`, ~900MB) is downloaded and cached locally. Subsequent
runs use the cache.

On Apple Silicon the MPS GPU backend is used automatically. On machines with an NVIDIA GPU, CUDA is used.

## Usage

```bash
# Process all files in a folder
uv run generate_vtt.py /path/to/videos

# Process a single file
uv run generate_vtt.py /path/to/video.mp4
```

A `.vtt` file is written as a sibling to each video in the same path.

## Options

| Flag | Description |
|---|---|
| `--labels <file.json>` | Use a custom label set instead of the built-in defaults (see below) |
| `--skip-existing` | Skip any video that already has a `.vtt` file — safe to use on reruns |
| `--dry-run` | Print the VTT to the terminal instead of writing files |

## Customizing Labels

CLAP is a zero-shot model. The label text itself drives classification. There is no training step. The more specific and
acoustically descriptive your labels, the better the results.

### Editing the default labels

Open `generate_vtt.py` and modify the `DEFAULT_LABELS` list near the top of the file. Each entry is a plain English 
description of a sound:

```python
DEFAULT_LABELS = [
    "full marching band playing — brass and percussion together",
    "drum line and percussion cadence — no brass",
    ...
    "woodwind section — flutes clarinets and saxophones",
]
```

### Using a separate label file

For content that needs a different vocabulary (a different era of recordings, a different genre), create a JSON file 
containing a list of label strings:

```json
[
    "acoustic guitar strumming — Spanish folk or flamenco style",
    "Spanish vocal singing with guitar accompaniment",
    "radio announcer speaking in Spanish",
    "radio static and broadcast interference",
    "silence or dead air"
]
```

Then pass it with `--labels`:

```bash
uv run generate_vtt.py /path/to/videos --labels labels_spanish_radio.json
```

This repository includes `labels_spanish_radio.json` as an example for Spanish-language folk radio recordings.

### Tips for writing effective labels

- **Be descriptive, not brief.** `"drum line and percussion cadence — no brass"` outperforms `"drums"`.
- **Contrast matters.** If two labels sound acoustically similar, CLAP will struggle to separate them. Make the descriptions as distinct as the sounds themselves.
- **Cover edge cases explicitly.** Radio static, silence, and crowd noise will be forced into the nearest music label if you don't give them their own entry.
- **Iterate.** Run on one file with `--dry-run`, review the output, and adjust labels that are misfiring before processing a full batch.

### Further reading

- [CLAP paper (ICASSP 2023)](https://arxiv.org/abs/2211.06687) — explains how contrastive language-audio pretraining works and what kinds of descriptions it responds to
- [LAION-AI/CLAP on GitHub](https://github.com/LAION-AI/CLAP) — model details, training data, and label vocabulary guidance
- [laion/larger_clap_music_and_speech on Hugging Face](https://huggingface.co/laion/larger_clap_music_and_speech) — the specific model used here, with usage examples
