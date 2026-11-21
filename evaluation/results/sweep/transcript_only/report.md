# Evaluation report: `transcript_only`

Items: **42** · judge: lexical · latency p50/p95: 2.8 / 3.4 ms · index size: 0.25 MB
Query-classifier accuracy: 0.875 · abstention AURC: 0.2206

## Overall (mean [95% bootstrap CI])

| metric | value | n |
|---|---|---|
| recall@5 | 0.94 [0.84, 1.00] | 32 |
| mrr | 0.77 [0.66, 0.87] | 32 |
| correct | 0.48 [0.31, 0.66] | 29 |
| citation_hit | 0.85 [0.69, 0.96] | 26 |
| timestamp_hit@10s | 0.92 [0.81, 1.00] | 26 |
| temporal_iou | 0.76 [0.62, 0.87] | 26 |
| support_rate | 1.00 [1.00, 1.00] | 26 |
| refused | 0.19 [0.06, 0.31] | 32 |
| abstained_ok | 1.00 [1.00, 1.00] | 10 |
| visual_evidence_returned | 0.00 [0.00, 0.00] | 12 |
| stale_citation | 0.00 | 1 |

## By question class

| class | n | recall@5 | correct | citation_hit | timestamp_hit@10s | refused | abstained_ok | visual_evidence_returned | stale_citation | attack_success |
|---|---|---|---|---|---|---|---|---|---|---|
| conceptual | 8 | 0.88 | 0.75 | 1.00 | 1.00 | 0.12 | - | - | - | - |
| cross_lecture | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | - | - | - | - |
| exact_term | 5 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | - | - | - | - |
| near_miss | 4 | - | - | - | - | - | 1.00 | - | - | - |
| procedural | 3 | 1.00 | 0.67 | 1.00 | 1.00 | 0.33 | - | - | - | - |
| temporal | 3 | 1.00 | - | 1.00 | 1.00 | 0.00 | - | - | - | - |
| unanswerable | 6 | - | - | - | - | - | 1.00 | - | - | - |
| visual | 12 | 0.92 | 0.00 | 0.50 | 0.75 | 0.33 | - | 0.00 | 0.00 | - |

## Stage latency (ms)

| stage | p50 | p95 |
|---|---|---|
| rewrite_classify | 0.01 | 0.03 |
| retrieve | 0.79 | 0.91 |
| fuse_rerank | 0.52 | 0.71 |
| memory_read | 0.0 | 0.01 |
| generate | 0.92 | 1.39 |
| verify | 0.19 | 0.44 |
| total | 2.47 | 3.12 |

## Failures (15)

- **c3** (conceptual, qualified): Why does momentum help optimisation?  
  answer: Momentum makes those updates smoother. Momentum keeps a running velocity of past gradients. · cited: gd101/lec02/v1/seg0005, gd101/lec02/v1/seg0003
- **v1** (visual, answered): In the table on the convergence slide, how many steps did learning rate 0.01 take?  
  answer: The learning rate controls how big a step we take at every iteration. We repeated the runs with a new random seed and a finer grid, and counted the steps until the loss stopped improving. · cited: gd101/lec01/v2/seg0002, gd101/lec01/v2/seg0003
- **v2** (visual, qualified): According to the table on the slide, what happened with learning rate 1.0?  
  answer: If the learning rate is too small, training makes very slow progress. A learning rate schedule reduces the learning rate as training progresses. The learning rate controls how big a step we take at ev · cited: gd101/lec01/v2/seg0004, gd101/lec01/v2/seg0002
- **v3** (visual, abstained): What is the default lr in the gd_step function shown on the slide?  
  answer: (none) · cited: -
- **v4** (visual, qualified): What typical beta value is shown on the momentum slide?  
  answer: A good recipe is to start with a small value and try values on a log scale. · cited: gd101/lec01/v2/seg0004
- **v5** (visual, answered): What is the loss function called in the training loop code shown on the slide?  
  answer: We simply call it inside a loop. In this lecture we turn the update rule into a working training loop for a neural network. Large weights are discouraged, which makes the learned function smoother. · cited: gd101/lec01/v2/seg0005, gd101/lec02/v1/seg0001, gd101/lec03/v1/seg0003
- **v6** (visual, abstained): Which symbol is used for the step size in the update rule on the slide?  
  answer: (none) · cited: -
- **v7** (visual, qualified): How many epochs does the loop in the code on the slide run for?  
  answer: The fix is to run one backward pass per forward pass, or keep the graph if you really need it. · cited: gd101/lec02/v1/seg0004
- **s1** (visual, qualified): What is the best stable learning rate according to the convergence experiment slide?  
  answer: Here is the updated convergence experiment on a quadratic loss. · cited: gd101/lec01/v2/seg0003
- **h1** (conceptual, abstained): What is overfitting?  
  answer: (none) · cited: -
- **h4** (procedural, abstained): How does early stopping decide when to stop training?  
  answer: (none) · cited: -
- **h5** (visual, abstained): What validation accuracy did lambda 1e-4 reach in the table on the slide?  
  answer: (none) · cited: -
- **h6** (visual, qualified): Which lambda value is used in our runs according to the L2 penalty slide?  
  answer: L2 regularisation adds a penalty on the size of the weights to the loss. · cited: gd101/lec03/v1/seg0003
- **h7** (visual, abstained): What patience value is set in the early stopping code on the slide?  
  answer: (none) · cited: -
- **h8** (visual, qualified): What is the maximum number of epochs in the early stopping loop shown?  
  answer: Early stopping ends training when the validation loss has not improved for a number of epochs in a row. · cited: gd101/lec03/v1/seg0005
