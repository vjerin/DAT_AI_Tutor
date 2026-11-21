# Evaluation report: `default`

Items: **11** · judge: lexical · latency p50/p95: 3.2 / 6.9 ms · index size: 0.25 MB
Query-classifier accuracy: 0.7778 · abstention AURC: 0.4994

## Overall (mean [95% bootstrap CI])

| metric | value | n |
|---|---|---|
| recall@5 | 1.00 [1.00, 1.00] | 9 |
| mrr | 1.00 [1.00, 1.00] | 9 |
| correct | 0.50 [0.12, 0.88] | 8 |
| citation_hit | 1.00 [1.00, 1.00] | 7 |
| timestamp_hit@10s | 1.00 [1.00, 1.00] | 7 |
| temporal_iou | 0.91 [0.88, 0.94] | 7 |
| support_rate | 1.00 [1.00, 1.00] | 7 |
| refused | 0.22 [0.00, 0.44] | 9 |
| abstained_ok | 1.00 [1.00, 1.00] | 2 |
| visual_evidence_returned | 0.50 [0.00, 1.00] | 4 |

## By question class

| class | n | recall@5 | correct | citation_hit | timestamp_hit@10s | refused | abstained_ok | visual_evidence_returned | stale_citation | attack_success |
|---|---|---|---|---|---|---|---|---|---|---|
| conceptual | 2 | 1.00 | 0.50 | 1.00 | 1.00 | 0.00 | - | - | - | - |
| exact_term | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | - | - | - | - |
| near_miss | 1 | - | - | - | - | - | 1.00 | - | - | - |
| procedural | 1 | 1.00 | 0.00 | - | - | 1.00 | - | - | - | - |
| temporal | 1 | 1.00 | - | 1.00 | 1.00 | 0.00 | - | - | - | - |
| unanswerable | 1 | - | - | - | - | - | 1.00 | - | - | - |
| visual | 4 | 1.00 | 0.50 | 1.00 | 1.00 | 0.25 | - | 0.50 | - | - |

## Stage latency (ms)

| stage | p50 | p95 |
|---|---|---|
| rewrite_classify | 0.02 | 0.02 |
| retrieve | 0.79 | 4.46 |
| fuse_rerank | 0.84 | 1.07 |
| memory_read | 0.0 | 0.01 |
| generate | 1.01 | 1.3 |
| verify | 0.17 | 0.3 |
| total | 2.91 | 6.54 |

## Failures (4)

- **h1** (conceptual, answered): What is overfitting?  
  answer: Overfitting: Training loss keeps falling. · cited: gd101/lec03/v1/slide002
- **h4** (procedural, abstained): How does early stopping decide when to stop training?  
  answer: (none) · cited: -
- **h5** (visual, abstained): What validation accuracy did lambda 1e-4 reach in the table on the slide?  
  answer: (none) · cited: -
- **h8** (visual, qualified): What is the maximum number of epochs in the early stopping loop shown?  
  answer: Early stopping ends training when the validation loss has not improved for a number of epochs in a row. · cited: gd101/lec03/v1/seg0005
