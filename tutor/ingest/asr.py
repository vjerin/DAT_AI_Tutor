"""Word-timed transcripts from caption files (JSON/SRT/VTT) or faster-whisper."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Word:
    text: str
    start_ms: int
    end_ms: int


def _clock_to_ms(s: str) -> int:
    s = s.strip().replace(",", ".")
    parts = s.split(":")
    secs = float(parts[-1])
    mins = int(parts[-2]) if len(parts) >= 2 else 0
    hrs = int(parts[-3]) if len(parts) >= 3 else 0
    return int(round((hrs * 3600 + mins * 60 + secs) * 1000))


def _spread(text: str, start: int, end: int) -> list[Word]:
    """Spread a caption cue's words evenly across its time span."""
    toks = text.split()
    if not toks:
        return []
    step = (end - start) / len(toks)
    return [Word(t, int(start + i * step), int(start + (i + 1) * step)) for i, t in enumerate(toks)]


CUE_RE = re.compile(
    r"(\d{1,2}:\d{2}(?::\d{2})?[.,]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}(?::\d{2})?[.,]\d{1,3})[^\n]*\n(.*?)(?:\n\s*\n|\Z)",
    re.S,
)


def load_transcript(path: str | Path) -> list[Word]:
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        data = json.loads(raw)
        if isinstance(data, dict):
            data = data.get("words") or data.get("segments") or []
        words: list[Word] = []
        for item in data:
            if "words" in item:                 # whisper-style segments with words
                for w in item["words"]:
                    words.append(Word(w["word"].strip(), int(w["start"] * 1000), int(w["end"] * 1000)))
            elif "word" in item or "w" in item:  # flat word list (seconds or ms)
                txt = item.get("word", item.get("w"))
                start, end = item.get("start_ms"), item.get("end_ms")
                if start is None:
                    start, end = int(item["start"] * 1000), int(item["end"] * 1000)
                words.append(Word(str(txt).strip(), int(start), int(end)))
            elif "text" in item:                # segment-level
                start = item.get("start_ms", int(item.get("start", 0) * 1000))
                end = item.get("end_ms", int(item.get("end", 0) * 1000))
                words.extend(_spread(item["text"], int(start), int(end)))
        return [w for w in words if w.text]
    if path.suffix in (".srt", ".vtt"):
        words = []
        for a, b, body in CUE_RE.findall(raw):
            body = re.sub(r"<[^>]+>", "", body).replace("\n", " ").strip()
            words.extend(_spread(body, _clock_to_ms(a), _clock_to_ms(b)))
        return words
    raise ValueError(f"unsupported transcript format: {path.suffix}")


class FasterWhisperASR:  # pragma: no cover - heavy optional dependency
    def __init__(self, model: str = "large-v3", device: str = "auto", compute_type: str = "auto"):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model, device=device, compute_type=compute_type)

    def transcribe(self, audio: str | Path, language: str | None = "en") -> list[Word]:
        segments, _ = self.model.transcribe(str(audio), language=language, word_timestamps=True, vad_filter=True)
        out = []
        for seg in segments:
            for w in seg.words or []:
                out.append(Word(w.word.strip(), int(w.start * 1000), int(w.end * 1000)))
        return out


def words_to_json(words: list[Word]) -> list[dict]:
    return [{"word": w.text, "start_ms": w.start_ms, "end_ms": w.end_ms} for w in words]
