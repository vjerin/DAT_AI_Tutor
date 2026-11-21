# Evaluation report: `hybrid_no_rerank`

Items: **42** · judge: lexical · latency p50/p95: 2.4 / 2.8 ms · index size: 0.25 MB
Query-classifier accuracy: 0.875 · abstention AURC: 0.2413

## Overall (mean [95% bootstrap CI])

| metric | value | n |
|---|---|---|
| recall@5 | 1.00 [1.00, 1.00] | 32 |
| mrr | 0.89 [0.80, 0.96] | 32 |
| correct | 0.69 [0.52, 0.86] | 29 |
| citation_hit | 0.96 [0.89, 1.00] | 28 |
| timestamp_hit@10s | 0.96 [0.89, 1.00] | 28 |
| temporal_iou | 0.92 [0.85, 0.97] | 28 |
| support_rate | 1.00 [1.00, 1.00] | 28 |
| refused | 0.12 [0.03, 0.25] | 32 |
| abstained_ok | 1.00 [1.00, 1.00] | 10 |
| visual_evidence_returned | 0.83 [0.58, 1.00] | 12 |
| stale_citation | 0.00 | 1 |

## By question class

| class | n | recall@5 | correct | citation_hit | timestamp_hit@10s | refused | abstained_ok | visual_evidence_returned | stale_citation | attack_success |
|---|---|---|---|---|---|---|---|---|---|---|
| conceptual | 8 | 1.00 | 0.75 | 0.88 | 0.88 | 0.00 | - | - | - | - |
| cross_lecture | 1 | 1.00 | 0.00 | - | - | 1.00 | - | - | - | - |
| exact_term | 5 | 1.00 | 0.40 | 1.00 | 1.00 | 0.00 | - | - | - | - |
| near_miss | 4 | - | - | - | - | - | 1.00 | - | - | - |
| procedural | 3 | 1.00 | 0.67 | 1.00 | 1.00 | 0.33 | - | - | - | - |
| temporal | 3 | 1.00 | - | 1.00 | 1.00 | 0.00 | - | - | - | - |
| unanswerable | 6 | - | - | - | - | - | 1.00 | - | - | - |
| visual | 12 | 1.00 | 0.83 | 1.00 | 1.00 | 0.17 | - | 0.83 | 0.00 | - |

## Stage latency (ms)

| stage | p50 | p95 |
|---|---|---|
| rewrite_classify | 0.01 | 0.02 |
| retrieve | 0.79 | 0.97 |
| fuse_rerank | 0.08 | 0.12 |
| memory_read | 0.0 | 0.01 |
| generate | 0.97 | 1.33 |
| verify | 0.17 | 0.31 |
| total | 2.12 | 2.5 |

## Failures (9)

- **c3** (conceptual, answered): Why does momentum help optimisation?  
  answer: Lecture 1: Gradient Descent: Optimisation for Machine Learning. zero_grad, forward, backward, step: Momentum smooths noisy updates. Momentum makes those updates smoother. · cited: gd101/lec01/v2/slide001, gd101/lec02/v1/slide005, gd101/lec02/v1/seg0005
- **e1** (exact_term, answered): Why do we call optimizer.zero_grad() at the start of each iteration?  
  answer: Training loop (PyTorch): optimizer.zero_grad() · cited: gd101/lec02/v1/slide002
- **e3** (exact_term, answered): What causes the error 'Trying to backward through the graph a second time'?  
  answer: A common error: RuntimeError: Trying to backward through. A common error: the graph a second time. · cited: gd101/lec02/v1/slide004
- **e4** (exact_term, answered): What does optimizer.step() do?  
  answer: Training loop (PyTorch): optimizer.step() · cited: gd101/lec02/v1/slide002
- **v7** (visual, abstained): How many epochs does the loop in the code on the slide run for?  
  answer: (none) · cited: -
- **x1** (cross_lecture, abstained): How does the momentum update differ from stepping along the raw gradient?  
  answer: (none) · cited: -
- **h1** (conceptual, answered): What is overfitting?  
  answer: Overfitting: Training loss keeps falling. · cited: gd101/lec03/v1/slide002
- **h4** (procedural, abstained): How does early stopping decide when to stop training?  
  answer: (none) · cited: -
- **h5** (visual, abstained): What validation accuracy did lambda 1e-4 reach in the table on the slide?  
  answer: (none) · cited: -
