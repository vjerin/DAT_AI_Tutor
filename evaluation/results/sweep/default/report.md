# Evaluation report: `default`

Items: **42** · judge: lexical · latency p50/p95: 3.1 / 3.9 ms · index size: 0.25 MB
Query-classifier accuracy: 0.875 · abstention AURC: 0.0884

## Overall (mean [95% bootstrap CI])

| metric | value | n |
|---|---|---|
| recall@5 | 1.00 [1.00, 1.00] | 32 |
| mrr | 0.95 [0.89, 1.00] | 32 |
| correct | 0.79 [0.62, 0.93] | 29 |
| citation_hit | 0.93 [0.83, 1.00] | 30 |
| timestamp_hit@10s | 0.93 [0.83, 1.00] | 30 |
| temporal_iou | 0.87 [0.77, 0.94] | 30 |
| support_rate | 1.00 [1.00, 1.00] | 30 |
| refused | 0.06 [0.00, 0.16] | 32 |
| abstained_ok | 1.00 [1.00, 1.00] | 10 |
| visual_evidence_returned | 0.75 [0.50, 1.00] | 12 |
| stale_citation | 0.00 | 1 |

## By question class

| class | n | recall@5 | correct | citation_hit | timestamp_hit@10s | refused | abstained_ok | visual_evidence_returned | stale_citation | attack_success |
|---|---|---|---|---|---|---|---|---|---|---|
| conceptual | 8 | 1.00 | 0.75 | 0.88 | 0.88 | 0.00 | - | - | - | - |
| cross_lecture | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | - | - | - | - |
| exact_term | 5 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | - | - | - | - |
| near_miss | 4 | - | - | - | - | - | 1.00 | - | - | - |
| procedural | 3 | 1.00 | 0.67 | 1.00 | 1.00 | 0.33 | - | - | - | - |
| temporal | 3 | 1.00 | - | 1.00 | 1.00 | 0.00 | - | - | - | - |
| unanswerable | 6 | - | - | - | - | - | 1.00 | - | - | - |
| visual | 12 | 1.00 | 0.75 | 0.91 | 0.91 | 0.08 | - | 0.75 | 0.00 | - |

## Stage latency (ms)

| stage | p50 | p95 |
|---|---|---|
| rewrite_classify | 0.01 | 0.03 |
| retrieve | 0.78 | 0.84 |
| fuse_rerank | 0.81 | 1.17 |
| memory_read | 0.0 | 0.01 |
| generate | 0.97 | 1.36 |
| verify | 0.2 | 0.39 |
| total | 2.8 | 3.6 |

## Failures (6)

- **c3** (conceptual, answered): Why does momentum help optimisation?  
  answer: Lecture 1: Gradient Descent: Optimisation for Machine Learning. zero_grad, forward, backward, step: Momentum smooths noisy updates. Momentum makes those updates smoother. · cited: gd101/lec01/v2/slide001, gd101/lec02/v1/slide005, gd101/lec02/v1/seg0005
- **v7** (visual, qualified): How many epochs does the loop in the code on the slide run for?  
  answer: The fix is to run one backward pass per forward pass, or keep the graph if you really need it. · cited: gd101/lec02/v1/seg0004
- **h1** (conceptual, answered): What is overfitting?  
  answer: Overfitting: Training loss keeps falling. · cited: gd101/lec03/v1/slide002
- **h4** (procedural, abstained): How does early stopping decide when to stop training?  
  answer: (none) · cited: -
- **h5** (visual, abstained): What validation accuracy did lambda 1e-4 reach in the table on the slide?  
  answer: (none) · cited: -
- **h8** (visual, qualified): What is the maximum number of epochs in the early stopping loop shown?  
  answer: Early stopping ends training when the validation loss has not improved for a number of epochs in a row. · cited: gd101/lec03/v1/seg0005
