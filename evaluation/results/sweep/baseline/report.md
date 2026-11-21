# Evaluation report: `baseline`

Items: **42** · judge: lexical · latency p50/p95: 1.1 / 1.7 ms · index size: 0.25 MB
Query-classifier accuracy: 0.875 · abstention AURC: 0.2267

## Overall (mean [95% bootstrap CI])

| metric | value | n |
|---|---|---|
| recall@5 | 0.94 [0.84, 1.00] | 32 |
| mrr | 0.80 [0.69, 0.91] | 32 |
| correct | 0.52 [0.34, 0.69] | 29 |
| citation_hit | 0.86 [0.75, 0.96] | 28 |
| timestamp_hit@10s | 0.93 [0.82, 1.00] | 28 |
| temporal_iou | 0.77 [0.66, 0.87] | 28 |
| support_rate | 1.00 [1.00, 1.00] | 32 |
| refused | 0.12 [0.03, 0.25] | 32 |
| abstained_ok | 0.60 [0.30, 0.90] | 10 |
| visual_evidence_returned | 0.00 [0.00, 0.00] | 12 |
| stale_citation | 0.00 | 1 |

## By question class

| class | n | recall@5 | correct | citation_hit | timestamp_hit@10s | refused | abstained_ok | visual_evidence_returned | stale_citation | attack_success |
|---|---|---|---|---|---|---|---|---|---|---|
| conceptual | 8 | 0.88 | 0.75 | 1.00 | 1.00 | 0.12 | - | - | - | - |
| cross_lecture | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | - | - | - | - |
| exact_term | 5 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | - | - | - | - |
| near_miss | 4 | - | - | - | - | - | 0.50 | - | - | - |
| procedural | 3 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | - | - | - | - |
| temporal | 3 | 1.00 | - | 1.00 | 1.00 | 0.00 | - | - | - | - |
| unanswerable | 6 | - | - | - | - | - | 0.67 | - | - | - |
| visual | 12 | 0.92 | 0.00 | 0.56 | 0.78 | 0.25 | - | 0.00 | 0.00 | - |

## Stage latency (ms)

| stage | p50 | p95 |
|---|---|---|
| rewrite_classify | 0.01 | 0.02 |
| retrieve | 0.05 | 0.07 |
| fuse_rerank | 0.02 | 0.03 |
| memory_read | 0.0 | 0.0 |
| generate | 0.73 | 1.28 |
| verify | 0.12 | 0.27 |
| total | 0.98 | 1.53 |

## Failures (18)

- **c3** (conceptual, answered): Why does momentum help optimisation?  
  answer: Momentum makes those updates smoother. Momentum keeps a running velocity of past gradients. · cited: gd101/lec02/v1/seg0005, gd101/lec02/v1/seg0003
- **v1** (visual, answered): In the table on the convergence slide, how many steps did learning rate 0.01 take?  
  answer: The learning rate controls how big a step we take at every iteration. We repeated the runs with a new random seed and a finer grid, and counted the steps until the loss stopped improving. · cited: gd101/lec01/v2/seg0002, gd101/lec01/v2/seg0003
- **v2** (visual, answered): According to the table on the slide, what happened with learning rate 1.0?  
  answer: If the learning rate is too small, training makes very slow progress. A learning rate schedule reduces the learning rate as training progresses. The learning rate controls how big a step we take at ev · cited: gd101/lec01/v2/seg0004, gd101/lec01/v2/seg0002
- **v3** (visual, abstained): What is the default lr in the gd_step function shown on the slide?  
  answer: (none) · cited: -
- **v4** (visual, abstained): What typical beta value is shown on the momentum slide?  
  answer: (none) · cited: -
- **v5** (visual, answered): What is the loss function called in the training loop code shown on the slide?  
  answer: We simply call it inside a loop. In this lecture we turn the update rule into a working training loop for a neural network. Large weights are discouraged, which makes the learned function smoother. · cited: gd101/lec01/v2/seg0005, gd101/lec02/v1/seg0001, gd101/lec03/v1/seg0003
- **v6** (visual, answered): Which symbol is used for the step size in the update rule on the slide?  
  answer: In this lecture we turn the update rule into a working training loop for a neural network. · cited: gd101/lec02/v1/seg0001
- **v7** (visual, answered): How many epochs does the loop in the code on the slide run for?  
  answer: The fix is to run one backward pass per forward pass, or keep the graph if you really need it. · cited: gd101/lec02/v1/seg0004
- **s1** (visual, answered): What is the best stable learning rate according to the convergence experiment slide?  
  answer: Here is the updated convergence experiment on a quadratic loss. · cited: gd101/lec01/v2/seg0003
- **u1** (unanswerable, answered): What is the default beta2 value of the Adam optimizer?  
  answer: Every iteration starts with optimizer.zero_grad because PyTorch accumulates gradients by default, so old gradients must be cleared. · cited: gd101/lec02/v1/seg0002
- **u4** (unanswerable, answered): Explain how dropout regularisation works.  
  answer: L2 regularisation adds a penalty on the size of the weights to the loss. · cited: gd101/lec03/v1/seg0003
- **n1** (near_miss, answered): What batch size is used in the training loop?  
  answer: In this lecture we turn the update rule into a working training loop for a neural network. · cited: gd101/lec02/v1/seg0001
- **n3** (near_miss, answered): Which learning rate schedule does the lecture recommend for transformers?  
  answer: A learning rate schedule reduces the learning rate as training progresses. · cited: gd101/lec01/v2/seg0004
- **h1** (conceptual, abstained): What is overfitting?  
  answer: (none) · cited: -
- **h5** (visual, abstained): What validation accuracy did lambda 1e-4 reach in the table on the slide?  
  answer: (none) · cited: -
- **h6** (visual, answered): Which lambda value is used in our runs according to the L2 penalty slide?  
  answer: L2 regularisation adds a penalty on the size of the weights to the loss. · cited: gd101/lec03/v1/seg0003
- **h7** (visual, answered): What patience value is set in the early stopping code on the slide?  
  answer: Early stopping ends training when the validation loss has not improved for a number of epochs in a row. · cited: gd101/lec03/v1/seg0005
- **h8** (visual, answered): What is the maximum number of epochs in the early stopping loop shown?  
  answer: Early stopping ends training when the validation loss has not improved for a number of epochs in a row. · cited: gd101/lec03/v1/seg0005
