# Grounded multimodal AI tutor

CS5903 (DAT) Project 4. A course tutor that answers only from public lecture material, cites the exact moment in
the lecture video (or the slide / code frame) behind each sentence, declines questions the material does not cover,
does not serve superseded content as current, and keeps learner memory under explicit rules. An evaluation
harness measures all of this, including prompt-injection attacks.

* Report: [docs/Project_Report.pdf](docs/Project_Report.pdf)
* Architecture decisions: [docs/Architecture_Decision_Records.md](docs/Architecture_Decision_Records.md)
* Raw evaluation output: [evaluation/results/](evaluation/results/)
* Final tag: `cs5903-final-2026`

**Status:** milestones M1–M8 are implemented and tested with the CPU-only configuration. The GPU model adapters
(BGE-M3, cross-encoder reranker, CLIP, faster-whisper, PaddleOCR, NLI verifier, Prompt Guard, LLM generation)
are implemented but were not executed; see the report, section 8.

## Architecture in one paragraph

`tutor/ingest` turns each lecture (MP4 + captions, optional slide PDF) into evidence units: transcript segments
bounded by slide changes and one unit per slide or code frame (OCR text). Units carry source ID, version, content
hash and time span and are stored in SQLite (`tutor/store.py`). `tutor/index` builds BM25, dense and visual
indexes. For a question, `tutor/retrieve` rewrites and classifies it, searches the three channels, fuses them with
Reciprocal Rank Fusion and reranks. `tutor/generate` writes a cited answer, `tutor/verify` checks each sentence
against its citations and decides answer / qualified / abstain. `tutor/memory` keeps session state apart from a
durable learner model. `tutor/api` serves a REST API and a small web client; `tutor/eval` is the harness.

## Prerequisites

* Python 3.10–3.13 (tested on 3.13.18 and 3.10.12)
* `ffmpeg` and `tesseract` on the PATH
  * macOS: `brew install ffmpeg tesseract`
  * Ubuntu: `sudo apt install ffmpeg tesseract-ocr`
* No GPU, API key or model download is needed for the default configuration.

## Setup

```bash
git clone <repository-url> grounded-ai-tutor
cd grounded-ai-tutor
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements-lock.txt # exact versions used for the reported results
cp .env.example .env                 # optional; only needed for LLM-backed configs
```

`requirements.txt` has version ranges; `requirements-lock.txt` pins the versions used for the report
(Python 3.13). On Python 3.10 use `pip install -r requirements.txt`.

There is no external database to migrate: `tutor.db` and `memory.db` (SQLite) are created inside the workspace
folder on first run. Deleting the workspace folder resets everything.

## Run

The demo course is committed under `data/demo_course` (three synthetic lectures, lecture 1 in two versions, and
`eval.jsonl` with 42 questions). To regenerate it: `python scripts/make_demo_course.py data/demo_course`.

```bash
# 1. ingest version 1, then version 2 of the course
python -m tutor.cli ingest --workspace workspace --manifest data/demo_course/manifest_v1.yaml
python -m tutor.cli ingest --workspace workspace --manifest data/demo_course/manifest_v2.yaml

# 2. ask from the command line
python -m tutor.cli ask --workspace workspace "In the table on the convergence slide, how many steps did learning rate 0.01 take?"

# 3. web client at http://127.0.0.1:8080 (Ctrl+C to stop)
python -m tutor.cli serve --workspace workspace
```

Expected output of step 2:

```
[answered | class=visual | conf=0.97 | ... ms]
- Convergence experiment: Learning rate Steps to converge — 0.01 310.  [Gradient descent basics, v2, 0:25-0:42, slide 3]
  gd101/lec01/v2/slide003 -> /media/gd101/lec01/v2#t=25
```

An off-corpus question (`"How do I deploy a model with Kubernetes?"`) returns `[abstained ...]`.

In the web client, clicking a citation seeks the video to that second and shows the slide; the panel below lists
what the tutor remembers and lets you correct or delete each item.

## Test

```bash
python -m pytest -q            # 45 tests, about 20 s
```

| File | Covers |
|---|---|
| `tests/test_acceptance.py` | Acceptance tests AT-M1-1 … AT-M8-1, the API end-to-end workflow and the held-out lecture |
| `tests/test_robustness.py` | Empty / non-English / SQL-like / very long questions, missing index, invalid API input, no-op re-ingest |
| `tests/test_components.py` | Tokenizer, BM25, RRF, segmentation, captions parsing, OCR helpers, versioning, cache, guard, verifier, metrics |
| `tests/test_memory.py` | 18 scripted learner-memory scenarios |

## Reproduce the reported results

```bash
bash evaluation/reproduce.sh      # about 1 minute on a laptop CPU
```

It ingests the course, runs the baseline and the submitted system, the configuration sweep, the chunking
comparison, the red team with and without defences, the memory scenarios and three repeat runs, and writes
everything to `evaluation/results/`. See [evaluation/README.md](evaluation/README.md).

## Configurations

| Config | Use |
|---|---|
| `configs/default.yaml` | Submitted system (CPU only) |
| `configs/baseline.yaml` | Working baseline: text-only BM25 RAG, no reranker, no verification, never abstains |
| `configs/bm25_only.yaml`, `dense_only.yaml`, `hybrid_no_rerank.yaml`, `transcript_only.yaml` | Retrieval and modality ablations |
| `configs/fixed30.yaml` | Fixed 30 s segments instead of slide-bounded ones |
| `configs/no_defence.yaml` | Injection defences off |
| `configs/calibrated.yaml` | Learned abstention calibrator (`python -m tutor.cli calibrate ...`) |
| `configs/llm_api.yaml` | LLM generation through Ollama / vLLM / OpenAI-compatible / Anthropic endpoints |
| `configs/thorough.yaml` | GPU model stack (not executed for this submission) |

Override any value with `--set key=value`, e.g. `--set retrieval.final_k=8`.

Environment variables (`.env.example`): `TUTOR_WORKSPACE`, `TUTOR_CONFIG`, and `OPENAI_API_KEY` /
`ANTHROPIC_API_KEY` for the LLM configs only.

## Using a real course

Copy `examples/ocw_manifest.example.yaml`, point it at downloaded lecture MP4s and caption files (check the licence;
MIT OpenCourseWare publishes both under CC BY-NC-SA), then run `ingest` with that manifest.

## Operations

* Logs go to stderr (`INFO` level); the API prints each request.
* Reset: stop the server and delete the workspace folder.
* `tesseract not found` / `ffmpeg not found`: install them (see Prerequisites).
* `no index: run ingest first`: run the ingest commands for that workspace.
* LLM configs fail with connection errors if no server is running at `llm.base_url`.

## Limitations

* Slide detection assumes the slide fills most of the frame; picture-in-picture video needs cropping.
* The default generator is extractive and does not synthesise across passages.
* Pattern-based injection screening does not stop plain false statements planted in the material.
* Results come from a small synthetic course; they show the system works, not how well it generalises.

Full discussion: report sections 5 and 8.

## Layout

```
tutor/        source (ingest, index, retrieve, generate, verify, memory, api, client, eval, train)
configs/      experiment configurations
data/         synthetic demo course and evaluation set
evaluation/   reproduction script, repeat-run and environment scripts, raw results
tests/        acceptance, robustness, component and memory tests
docs/         project report and ADR log
scripts/      demo course generator
examples/     manifest template for a real course
```
