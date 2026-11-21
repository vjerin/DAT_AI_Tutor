# Architecture Decision Records — Project 4, Grounded Multimodal AI Tutor

Records are listed in the order the decisions were taken. Superseded records are kept.

---

## ADR-001 — Methodology record: retrieval-first tutor with an offline evaluation harness

* **Status / scope:** Accepted.
  Scope: all milestones M1–M8.
* **Context & drivers:** the tutor must answer only from approved material, cite exact moments, work on visual
  content and be measurable. The team had no GPU allocation and a fixed deadline, and every claim had to be
  reproducible by an evaluator on a clean checkout.
* **Options considered:**
  1. Fine-tune a language model on course content (knowledge in weights).
  2. Long-context prompting: put the whole course into the model context.
  3. Retrieval-augmented generation with explicit, versioned evidence units and a separate verification step.
* **Decision & rationale:** option 3. Retrieval keeps knowledge outside the model, so it can be cited, replaced
  per version and inspected (Lewis et al., 2020). Options 1 and 2 cannot produce timestamp citations or handle
  versions. The method was built so that the whole pipeline runs on CPU with substitutable components, and the
  evaluation harness was built alongside the system rather than after it.
* **Consequences:** quality depends on retrieval; answer quality of the CPU configuration is limited by the
  extractive generator (ADR-006). Every milestone has an automated acceptance test.
* **Evidence:** `tutor/tutor.py`, `tests/test_acceptance.py`, `evaluation/results/`.
* **Revisit trigger:** if a GPU allocation becomes available, re-run the evaluation with `configs/thorough.yaml`
  and revisit ADR-005 and ADR-006.

---

## ADR-002 — Evidence unit as the core data model, segmented on slide boundaries

* **Status / scope:** Accepted. M1, M3, M4.
* **Context & drivers:** citations, versioning, caching and evaluation all need a stable unit of evidence with a
  time span. Speech and slides must line up so that a cited moment shows the slide being discussed.
* **Options considered:**
  1. Fixed time windows (e.g. 30 s) over the transcript.
  2. Sentence-level units.
  3. Slide-bounded transcript segments (split further at sentence ends above 90 s) plus one unit per slide frame.
* **Decision & rationale:** option 3. Slide boundaries are natural topic boundaries and give transcript and frame
  units the same time span. Sentence units were too small to answer from; fixed windows cut across slides.
  Measured: slide-bounded vs fixed 30 s gave correctness 0.793 vs 0.759 on the full set
  (`evaluation/results/chunking/sweep.md`). The difference is small on this workload; the main reason for the
  choice is the shared time span.
* **Consequences:** depends on reliable slide-change detection (ADR-004). Courses without slides fall back to
  fixed windows automatically.
* **Evidence:** `tutor/ingest/segment.py`, `tutor/schema.py` (`EvidenceUnit`), `configs/fixed30.yaml`.
* **Revisit trigger:** if fixed or semantic chunking beats slide-bounded segments by more than 5 points of
  correctness on a real course.

---

## ADR-003 — Hybrid retrieval with Reciprocal Rank Fusion and a reranker

* **Status / scope:** Accepted. M2.
* **Context & drivers:** lecture questions mix exact terms (function names, error text) and paraphrased concepts.
  The scores of BM25, cosine similarity and image–text similarity are not comparable.
* **Options considered:**
  1. Dense retrieval only.
  2. Weighted sum of normalised scores.
  3. Learned fusion (learning to rank).
  4. Reciprocal Rank Fusion (k = 60) with optional per-query-class weights, followed by reranking.
* **Decision & rationale:** option 4. RRF needs no score calibration or training data and performed better than
  more complex fusion methods in Cormack et al. (2009). A learned fusion needs labelled data we do not have.
  Measured on our workload, the reranker matters more than the fusion: removing it lowered correctness from
  0.793 to 0.690 (`evaluation/results/sweep/sweep.md`). BM25-only, dense-only and hybrid gave the same
  correctness on this small corpus, so the benefit of hybrid retrieval is **not demonstrated** here.
* **Consequences:** RRF discards scores, so abstention needs its own signals (ADR-007).
* **Evidence:** `tutor/retrieve/fuse.py`, `tutor/retrieve/pipeline.py`, `tests/test_components.py::test_rrf_matches_formula`.
* **Revisit trigger:** on a larger corpus, if any single channel matches hybrid on recall@5 and correctness, drop
  the extra channels to save latency.

---

## ADR-004 — Separate visual channel; OCR text as the default visual representation

* **Status / scope:** Accepted. M1, M4.
* **Context & drivers:** some answers exist only on screen. In mixed retrieval, transcript text tends to outrank
  visual evidence (Jeong et al., 2025). No GPU was available for image embeddings.
* **Options considered:**
  1. Transcript only.
  2. VLM captions of every frame.
  3. CLIP / ColPali-style image embeddings.
  4. OCR text of de-duplicated slide frames, indexed in a separate visual channel, with image embeddings as a
     configurable alternative.
* **Decision & rationale:** option 4. OCR reads slide text, tables and code well, runs on CPU and is cheap after
  de-duplicating frames. A separate channel reserves candidate slots for visual units. Slide changes are found by
  the fraction of changed pixels between 1 fps thumbnails; an earlier perceptual-hash threshold merged two
  different slides in testing and was replaced. Tesseract runs in layout mode plus a sparse pass to recover
  white-on-dark titles. Measured: transcript-only correctness 0.483 vs 0.793 with the visual channel.
* **Consequences:** diagrams without text are not understood unless VLM captions are switched on; picture-in-
  picture lecture video breaks pixel-change detection.
* **Evidence:** `tutor/ingest/slides.py`, `tutor/ingest/ocr.py`, `configs/transcript_only.yaml`,
  `tests/test_acceptance.py::test_at_m4_2_transcript_only_cannot_answer_visual`.
* **Revisit trigger:** if the share of visual questions answered with the right frame falls below 0.7 on a real
  course, enable captions or image embeddings.

---

## ADR-005 — CPU-first default stack with pluggable model adapters

* **Status / scope:** Accepted. All components.
* **Context & drivers:** no GPU, no paid API budget, and the evaluator must reproduce results on a laptop.
* **Options considered:** (1) API-first (hosted LLM and embeddings); (2) GPU open-model stack only; (3) CPU
  default with adapters for GPU and API models selected by configuration.
* **Decision & rationale:** option 3. Defaults: BM25, TF-IDF+SVD dense vectors, lexical reranker, extractive
  generator, lexical verifier. Adapters exist for sentence-transformers, cross-encoders, CLIP, faster-whisper,
  PaddleOCR, NLI, Prompt Guard and OpenAI-compatible / Anthropic HTTP endpoints.
* **Consequences:** reproducible in about one minute on CPU. The adapters were **not executed** for this
  submission and may need small fixes for library versions.
* **Evidence:** `tutor/index/encoders.py`, `tutor/retrieve/rerank.py`, `tutor/generate/llm.py`, `configs/`.
* **Revisit trigger:** availability of a GPU or API budget.

---

## ADR-006 — Extractive generator as the default; LLM generation through plain HTTP

* **Status / scope:** Accepted. M3.
* **Context & drivers:** answers must be cited per sentence and must not contain unsupported claims; no LLM was
  available for the reported runs.
* **Options considered:** (1) local LLM via vendor SDK; (2) extractive answers built from evidence sentences and
  slide lines; (3) both, with the LLM behind a small HTTP client and a JSON output contract.
* **Decision & rationale:** option 3, with extractive as default. Extractive answers are deterministic and cannot
  invent facts, which makes the harness repeatable. Slide lines are prefixed with the slide title and table header
  so a table row reads as an answer. The LLM path validates that every citation refers to evidence actually given.
* **Consequences:** extractive answers do not synthesise across passages; on the held-out lecture correctness was
  0.50. This is the main quality limitation of the submission.
* **Evidence:** `tutor/generate/generator.py`, `tests/test_components.py::test_llm_generator_validates_citations`,
  `evaluation/results/solution_heldout/report.md`.
* **Revisit trigger:** first LLM-backed evaluation run; adopt it as default if correctness rises by more than
  10 points without lowering citation hit rate.

---

## ADR-007 — Abstention from evidence-coverage features, with an optional calibrator

* **Status / scope:** Accepted. M6.
* **Context & drivers:** unsupported questions must be declined; extractive answers are always "supported" by
  their own source, so support checks alone do not detect off-topic questions.
* **Options considered:** (1) threshold on the top retrieval score; (2) ask the generator whether it can answer;
  (3) features describing how much of the question the cited evidence covers (IDF-weighted), share of question
  terms absent from the whole corpus, reranker score, channel agreement and support rate; combined by a fixed
  formula, or by a logistic-regression calibrator fitted on dev data.
* **Decision & rationale:** option 3 with the fixed formula as default. The first version (weighted mostly on
  support) answered 3 of 8 unanswerable questions; adding coverage and out-of-vocabulary features fixed this.
  Measured: correct abstention 1.00 (10/10) vs 0.60 for the baseline. The learned calibrator was worse on the
  11-item dev split, so it is not the default.
* **Consequences:** the formula was tuned on the dev/test items, so held-out behaviour matters: there it kept
  abstention at 1.00 but refused 2 of 9 answerable questions.
* **Evidence:** `tutor/verify/abstain.py`, `tutor/train/calibrator.py`, `evaluation/results/baseline_vs_solution.md`.
* **Revisit trigger:** a dev set of 200+ labelled questions; then fit and compare the calibrator again.

---

## ADR-008 — Content-hash versioning and a version-aware answer cache

* **Status / scope:** Accepted. M1, M6.
* **Context & drivers:** course material changes; stale evidence must not be presented as current.
* **Options considered:** (1) overwrite the index on update; (2) time-based cache expiry; (3) content-hash
  versions that supersede, never overwrite, plus a cache with reverse indexes from source and memory item to
  cache entries.
* **Decision & rationale:** option 3. Old versions stay available for audit and labelled "superseded"; retrieval
  filters to current units. Cached abstentions are dropped on any update because new content may answer them.
* **Consequences:** storage grows with versions. Measured: 0 stale citations after the v2 update.
* **Evidence:** `tutor/store.py`, `tutor/cache.py`, `tests/test_acceptance.py::test_at_m6_2_stale_evidence_not_served`,
  `test_at_m6_3_cache_invalidated_on_update`.
* **Revisit trigger:** if corpus size makes keeping all versions in one index too slow.

---

## ADR-009 — Rule-based memory policy, separate from the model

* **Status / scope:** Accepted. M5.
* **Context & drivers:** learner data is personal; it must be selective, correctable and deletable.
* **Options considered:** (1) replay all past conversations; (2) model-managed memory as in MemGPT (Packer et al.,
  2023); (3) an extractor that proposes operations and a rule engine that decides: goals only when stated,
  misconceptions only after two observations, sensitive categories rejected, expiry per type; BKT for mastery.
* **Decision & rationale:** option 3. Rules are testable and explainable to a learner; model-managed memory has no
  policy layer. Deletion scrubs session text and invalidates cached answers; the audit log never holds content.
* **Consequences:** the rule-based extractor misses paraphrased statements. Measured: 18/18 scenarios pass.
* **Evidence:** `tutor/memory/`, `tutor/eval/memory_tests.py`, `evaluation/results/memtest.txt`.
* **Revisit trigger:** if real learner logs show the rules missing more than 20% of stated goals.

---

## ADR-010 — Layered injection defences and a targeted red team

* **Status / scope:** Accepted. M7.
* **Context & drivers:** transcripts, slide text and OCR are untrusted input.
* **Options considered:** (1) prompt instructions only; (2) a classifier on every chunk; (3) layers: pattern
  screening at ingestion with quarantine, instruction/data separation in the prompt, output scrubbing of unknown
  links, citation validation.
* **Decision & rationale:** option 3, with a model classifier available as an adapter. The red team first used
  generic payloads, which the extractive generator never quoted (0 successes even without defences), so payloads
  were changed to mention the question topic, as a targeted attacker would.
* **Consequences:** measured: instruction-style attacks 0/12 with defences vs 10/12 without; knowledge poisoning
  3/3 in both. Quarantine removes the poisoned unit and, in this worst-case setup, the evidence with it
  (correct answers 4/15 vs 10/15).
* **Evidence:** `tutor/verify/guard.py`, `tutor/eval/redteam.py`, `evaluation/results/redteam_*`.
* **Revisit trigger:** any successful instruction-style attack in a red-team run.

---

## ADR-011 — SQLite and in-memory vector indexes

* **Status / scope:** Accepted. M1, M5.
* **Context & drivers:** the evaluator must run the system without installing services.
* **Options considered:** (1) PostgreSQL + Qdrant/OpenSearch; (2) SQLite + numpy flat search.
* **Decision & rationale:** option 2. No setup, single-file reset, adequate for a few courses (index 0.25 MB here).
* **Consequences:** single-process; no concurrent writers; brute-force vector search.
* **Evidence:** `tutor/store.py`, `tutor/index/manager.py`.
* **Revisit trigger:** more than about 100k evidence units or multiple API workers.

---

## ADR-012 — Evaluation methodology

* **Status / scope:** Accepted. M7, M8.
* **Context & drivers:** results must be reproducible, cover every milestone and include an unseen case.
* **Options considered:** (1) public benchmarks only; (2) a generated course with known ground truth plus a
  held-out lecture; (3) a real OCW course with hand-labelled questions.
* **Decision & rationale:** option 2 for this submission, because it gives exact gold time windows, a controlled
  version change and repeatable media. Gold evidence is stored as time windows so it survives re-segmentation.
  The judge is lexical (gold tokens and numbers in the answer); metrics have 95% bootstrap intervals; comparisons
  use paired bootstrap tests. The held-out lecture (11 questions) was written after the heuristics were frozen.
* **Consequences:** synthetic speech and clean slides make retrieval easier than on real lectures. Option 3 is the
  first item of the way-forward plan.
* **Evidence:** `scripts/make_demo_course.py`, `tutor/eval/`, `evaluation/reproduce.sh`.
* **Revisit trigger:** availability of a hand-labelled real-course set.
