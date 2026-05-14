#!/usr/bin/env python3
"""
Marching Band VTT Generator
============================
Processes a folder of marching band videos (.mp4) and generates WebVTT
caption files describing musical events, transitions, and instrumentation.
 
Uses LAION CLAP for zero-shot audio classification — no song identification,
just descriptive labels you define.
 
REQUIREMENTS
------------
    pip install librosa transformers torch torchaudio soundfile numpy
 
On Apple Silicon (M1/M2/M3), torch will use the MPS backend automatically.
First run will download the CLAP model (~900MB) and cache it locally.
 
USAGE
-----
    # Process all .mp4 files in a folder
    python generate_vtt.py /path/to/videos
 
    # Skip files that already have a .vtt
    python generate_vtt.py /path/to/videos --skip-existing
 
    # Use a different label set
    python generate_vtt.py /path/to/videos --labels my_labels.json
 
    # Dry run — print VTT to terminal, don't write files
    python generate_vtt.py /path/to/videos --dry-run
 
    # Process a single file
    python generate_vtt.py /path/to/video.mp4
"""
 
import os
import sys
import json
import argparse
import subprocess
import tempfile
from pathlib import Path
from collections import Counter
 
import numpy as np
import librosa
 
# ---------------------------------------------------------------------------
# LABEL SET — customize these for your content
# The more specific your labels, the better CLAP performs.
# ---------------------------------------------------------------------------
DEFAULT_LABELS = [
    "full marching band playing — brass and percussion together",
    "brass section only — trumpets trombones and tubas",
    "drum line and percussion cadence — no brass",
    "trumpet fanfare or solo trumpet melody",
    "low brass feature — trombones and tubas",
    "crowd noise and stadium ambience",
    "silence or very quiet ambient noise",
    "band tuning or warming up before performance",
    "applause and crowd cheering",
    "announcer or public address system speaking",
    "transition or brief pause between musical pieces",
    "cymbal crash and percussion accent",
]
 
# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------
WINDOW_SECONDS    = 4      # CLAP analysis window length
HOP_SECONDS       = 0.5    # step between windows (smaller = smoother boundaries)
MIN_SEG_SECONDS   = 1.5    # collapse segments shorter than this into neighbors
CLAP_MODEL        = "laion/larger_clap_music_and_speech"
SAMPLE_RATE       = 48000  # CLAP expects 48kHz
 
 
# ---------------------------------------------------------------------------
# CLAP CLASSIFIER
# ---------------------------------------------------------------------------
class ClapClassifier:
    def __init__(self, labels, model_name=CLAP_MODEL):
        print(f"Loading CLAP model: {model_name}")
        print("(First run downloads ~900MB — subsequent runs use cache)\n")
        from transformers import ClapModel, ClapProcessor
        import torch
 
        self.device = (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        print(f"Using device: {self.device}")
 
        self.model = ClapModel.from_pretrained(model_name).to(self.device)
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.labels = labels
        self.model.eval()
 
        # Pre-compute text embeddings once (reused for every window)
        import torch
        with torch.no_grad():
            text_inputs = self.processor(
                text=labels, return_tensors="pt", padding=True
            ).to(self.device)
            raw = self.model.get_text_features(**text_inputs)
            self.text_embeds = raw.pooler_output if hasattr(raw, "pooler_output") else raw
            self.text_embeds = self.text_embeds / self.text_embeds.norm(dim=-1, keepdim=True)
        print(f"Loaded {len(labels)} labels.\n")
 
    def classify(self, audio_chunk):
        """Return the best-matching label for a numpy audio chunk."""
        import torch
        with torch.no_grad():
            inputs = self.processor(
                audio=audio_chunk,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            raw = self.model.get_audio_features(**inputs)
            audio_embeds = raw.pooler_output if hasattr(raw, "pooler_output") else raw
            audio_embeds = audio_embeds / audio_embeds.norm(dim=-1, keepdim=True)
            similarity = (audio_embeds @ self.text_embeds.T).squeeze(0)
            best_idx = int(similarity.argmax())
        return self.labels[best_idx], float(similarity[best_idx])
 
 
# ---------------------------------------------------------------------------
# AUDIO EXTRACTION
# ---------------------------------------------------------------------------
def extract_audio(video_path: Path, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Extract mono audio from video using ffmpeg, return as numpy array."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
 
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", str(target_sr),
        "-f", "wav", tmp_path
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()}")
 
    audio, sr = librosa.load(tmp_path, sr=target_sr, mono=True)
    os.unlink(tmp_path)
    return audio, sr
 
 
# ---------------------------------------------------------------------------
# SEGMENTATION
# ---------------------------------------------------------------------------
def classify_windows(audio, sr, classifier):
    """Slide a window across the audio and classify each chunk."""
    duration = len(audio) / sr
    window_samples = int(WINDOW_SECONDS * sr)
    hop_samples = int(HOP_SECONDS * sr)
 
    frame_labels = {}  # time_pos -> list of labels from overlapping windows
 
    t_sample = 0
    while t_sample < len(audio):
        chunk = audio[t_sample: t_sample + window_samples]
        if len(chunk) < sr * 0.5:
            break
 
        # Pad short chunks at the end
        if len(chunk) < window_samples:
            chunk = np.pad(chunk, (0, window_samples - len(chunk)))
 
        label, score = classifier.classify(chunk)
        t_sec = t_sample / sr
 
        # Record label for every hop position this window covers
        pos = t_sec
        while pos < min(t_sec + WINDOW_SECONDS, duration):
            key = round(pos, 1)
            frame_labels.setdefault(key, []).append(label)
            pos += HOP_SECONDS
 
        t_sample += hop_samples
 
    return frame_labels, duration
 
 
def majority_vote_timeline(frame_labels):
    """Assign each time position the most-voted label."""
    timeline = []
    for pos in sorted(frame_labels.keys()):
        votes = frame_labels[pos]
        winner = Counter(votes).most_common(1)[0][0]
        timeline.append((pos, winner))
    return timeline
 
 
def collapse_segments(timeline, duration, min_dur=MIN_SEG_SECONDS):
    """Merge consecutive same-label frames into non-overlapping segments."""
    if not timeline:
        return []
 
    segments = []
    cur_label = timeline[0][1]
    cur_start = timeline[0][0]
    prev_t = timeline[0][0]
 
    for t, label in timeline[1:]:
        if label != cur_label:
            seg_end = prev_t + HOP_SECONDS
            if seg_end - cur_start >= min_dur:
                segments.append((cur_start, seg_end, cur_label))
            cur_label = label
            cur_start = t
        prev_t = t
 
    segments.append((cur_start, duration, cur_label))
    return segments
 
 
# ---------------------------------------------------------------------------
# VTT FORMATTING
# ---------------------------------------------------------------------------
def fmt_time(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"
 
 
MAX_CUE_SECONDS = 10.0


def segments_to_vtt(segments, video_name: str) -> str:
    lines = ["WEBVTT", ""]
    for start, end, label in segments:
        # Split long cues so seekers/fast-forwarders always land inside an active cue
        cue_start = start
        while cue_start < end:
            cue_end = min(cue_start + MAX_CUE_SECONDS, end)
            lines.append(f"{fmt_time(cue_start)} --> {fmt_time(cue_end)}")
            lines.append(f"[{label}]")
            lines.append("")
            cue_start = cue_end
    return "\n".join(lines)
 
 
# ---------------------------------------------------------------------------
# MAIN PROCESSING
# ---------------------------------------------------------------------------
def process_video(video_path: Path, classifier: ClapClassifier, dry_run=False) -> str:
    print(f"\n{'='*60}")
    print(f"Processing: {video_path.name}")
    print(f"{'='*60}")
 
    audio, sr = extract_audio(video_path)
    duration = len(audio) / sr
    print(f"Duration: {duration:.1f}s  |  Sample rate: {sr}Hz")
 
    print("Classifying audio windows...")
    frame_labels, duration = classify_windows(audio, sr, classifier)
 
    timeline = majority_vote_timeline(frame_labels)
    segments = collapse_segments(timeline, duration)
 
    print(f"Found {len(segments)} segment(s):")
    for s, e, lbl in segments:
        print(f"  {fmt_time(s)} --> {fmt_time(e)}  |  {lbl}")
 
    vtt_content = segments_to_vtt(segments, video_path.name)
 
    if dry_run:
        print("\n--- VTT OUTPUT (dry run) ---")
        print(vtt_content)
    else:
        out_path = video_path.with_suffix(".vtt")
        out_path.write_text(vtt_content, encoding="utf-8")
        print(f"\nSaved: {out_path.name}")
 
    return vtt_content
 
 
def main():
    parser = argparse.ArgumentParser(
        description="Generate descriptive WebVTT files for marching band videos using CLAP."
    )
    parser.add_argument("input", help="Video file or folder of videos")
    parser.add_argument("--labels", help="JSON file with list of labels (overrides defaults)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip videos that already have a .vtt file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print VTT output, do not write files")
    args = parser.parse_args()
 
    # Resolve input path(s)
    input_path = Path(args.input)
    if input_path.is_file():
        videos = [input_path]
    elif input_path.is_dir():
        videos = sorted(input_path.rglob("*.mp4"))
        if not videos:
            print(f"No .mp4 files found in {input_path} (searched recursively)")
            sys.exit(1)
    else:
        print(f"Input not found: {input_path}")
        sys.exit(1)
 
    # Load labels
    if args.labels:
        with open(args.labels) as f:
            labels = json.load(f)
        print(f"Loaded {len(labels)} custom labels from {args.labels}")
    else:
        labels = DEFAULT_LABELS
 
    # Filter already-done videos
    if args.skip_existing:
        before = len(videos)
        videos = [v for v in videos if not v.with_suffix(".vtt").exists()]
        skipped = before - len(videos)
        if skipped:
            print(f"Skipping {skipped} video(s) with existing VTTs.")
 
    if not videos:
        print("Nothing to process.")
        sys.exit(0)
 
    print(f"\nVideos to process: {len(videos)}")
    for v in videos:
        print(f"  {v.name}")
 
    # Load CLAP once, reuse for all videos
    classifier = ClapClassifier(labels)
 
    for video in videos:
        try:
            process_video(video, classifier, dry_run=args.dry_run)
        except Exception as e:
            print(f"\nERROR processing {video.name}: {e}")
            continue
 
    print("\nDone.")
 
 
if __name__ == "__main__":
    main()
