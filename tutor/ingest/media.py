"""ffmpeg helpers: probe duration, sample frames, extract audio."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def _require(binary: str) -> str:
    path = shutil.which(binary)
    if not path:
        raise RuntimeError(f"'{binary}' not found on PATH; install ffmpeg to ingest video")
    return path


def probe_duration_ms(video: str | Path) -> int:
    out = subprocess.run(
        [_require("ffprobe"), "-v", "error", "-show_entries", "format=duration", "-of", "json", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout
    return int(float(json.loads(out)["format"]["duration"]) * 1000)


def sample_frames(video: str | Path, out_dir: str | Path, fps: float = 1.0) -> list[tuple[int, Path]]:
    """Write one JPEG every 1/fps seconds; returns [(timestamp_ms, path)]."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("f_*.jpg"):
        old.unlink()
    subprocess.run(
        [_require("ffmpeg"), "-v", "error", "-y", "-i", str(video), "-vf", f"fps={fps}",
         "-q:v", "3", str(out_dir / "f_%06d.jpg")],
        check=True,
    )
    frames = sorted(out_dir.glob("f_*.jpg"))
    step = 1000.0 / fps
    return [(int(round((i) * step)), p) for i, p in enumerate(frames)]


def extract_audio(video: str | Path, out_wav: str | Path, sr: int = 16000) -> Path:
    out_wav = Path(out_wav)
    subprocess.run(
        [_require("ffmpeg"), "-v", "error", "-y", "-i", str(video), "-ac", "1", "-ar", str(sr), str(out_wav)],
        check=True,
    )
    return out_wav
