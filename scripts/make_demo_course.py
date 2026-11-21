"""Generates the synthetic demo course: slide images, MP4 lectures, word-timed
transcripts, two versions of lecture 1, a held-out lecture 3 and the eval set.

python scripts/make_demo_course.py data/demo_course
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
WORD_S = 0.36     # seconds per spoken word
GAP_S = 1.5       # pause at each slide change


def _font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


LEC01 = {
    "id": "lec01", "title": "Gradient descent basics",
    "slides": [
        {"title": "Lecture 1: Gradient Descent", "lines": ["Optimisation for Machine Learning", "Demo course"],
         "say": "Welcome to the first lecture of the course. Today we study gradient descent, the workhorse algorithm for training machine learning models."},
        {"title": "The update rule", "lines": ["theta <- theta - eta * grad L(theta)", "eta = step size", "grad L points uphill"],
         "say": "Gradient descent repeatedly moves the parameters against the gradient of the loss. The gradient points in the direction of steepest increase, so we step the other way. The learning rate controls how big a step we take at every iteration."},
        {"title": "Convergence experiment", "table": [["Learning rate", "Steps to converge"], ["0.001", "2400"], ["0.01", "310"], ["0.1", "45"], ["1.0", "diverges"]],
         "lines": ["Best stable setting: 0.1"],
         "say": "Here is a small experiment on a quadratic loss. We ran gradient descent with several learning rates and counted the steps until the loss stopped improving. Notice the pattern in the table rather than any single number."},
        {"title": "Choosing the learning rate", "lines": ["Too small: very slow progress", "Too large: oscillation or divergence", "Use a schedule: step decay or cosine"],
         "say": "If the learning rate is too small, training makes very slow progress. If it is too large, the loss can oscillate or even diverge. A good recipe is to start with a small value and try values on a log scale. A learning rate schedule reduces the learning rate as training progresses."},
        {"title": "Gradient descent in code", "code": ["def gd_step(theta, grad, lr=0.01):", "    return theta - lr * grad", "", "for epoch in range(100):", "    theta = gd_step(theta, compute_grad(theta))"],
         "say": "Finally, the algorithm in code. The helper function takes the current parameters and the gradient and returns the updated parameters. We simply call it inside a loop."},
    ],
}

LEC01_V2_CHANGES = {
    2: {"lines": ["Best stable setting: 0.05"],
        "say": "Here is the updated convergence experiment on a quadratic loss. We repeated the runs with a new random seed and a finer grid, and counted the steps until the loss stopped improving. Notice the pattern in the table rather than any single number."},
}

LEC02 = {
    "id": "lec02", "title": "Training neural networks in code",
    "slides": [
        {"title": "Lecture 2: Training in Code", "lines": ["From the update rule to a training loop"],
         "say": "In this lecture we turn the update rule into a working training loop for a neural network."},
        {"title": "Training loop (PyTorch)", "code": ["for x, y in loader:", "    optimizer.zero_grad()", "    loss = criterion(model(x), y)", "    loss.backward()", "    optimizer.step()"],
         "say": "Every iteration starts with optimizer.zero_grad because PyTorch accumulates gradients by default, so old gradients must be cleared. Then we compute the loss. Calling loss.backward computes the gradients of the loss with respect to every parameter. Finally optimizer.step updates the parameters using those gradients."},
        {"title": "Momentum", "lines": ["v <- beta * v + grad", "theta <- theta - eta * v", "Typical beta: 0.9"],
         "say": "Momentum keeps a running velocity of past gradients. Instead of stepping along the raw gradient, we step along the velocity. This smooths the updates and dampens oscillations in narrow valleys."},
        {"title": "A common error", "lines": ["RuntimeError: Trying to backward through", "the graph a second time", "Fix: one backward per forward pass", "or pass retain_graph=True"],
         "say": "A frequent error is the runtime error about trying to backward through the graph a second time. It usually means you called backward twice on the same graph. The fix is to run one backward pass per forward pass, or keep the graph if you really need it."},
        {"title": "Summary", "lines": ["zero_grad, forward, backward, step", "Momentum smooths noisy updates"],
         "say": "To summarise, each training step clears gradients, runs the forward pass, computes gradients and updates the parameters. Momentum makes those updates smoother."},
    ],
}


LEC03 = {
    "id": "lec03", "title": "Overfitting and regularisation",
    "slides": [
        {"title": "Lecture 3: Regularisation", "lines": ["Keeping models general"],
         "say": "This lecture is about models that fit the training data too well and how to keep them general."},
        {"title": "Overfitting", "lines": ["Training loss keeps falling", "Validation loss starts rising", "Gap = generalisation error"],
         "say": "A model overfits when it keeps improving on the training data but gets worse on data it has not seen. We watch the validation loss to detect this."},
        {"title": "L2 penalty", "lines": ["L_total = L_data + lambda * ||w||^2", "lambda = 1e-4 in our runs", "Also called weight decay"],
         "say": "L2 regularisation adds a penalty on the size of the weights to the loss. Large weights are discouraged, which makes the learned function smoother. In optimisers this is usually implemented as weight decay."},
        {"title": "Effect of lambda", "table": [["lambda", "Val. accuracy"], ["0", "81.2%"], ["1e-4", "86.5%"], ["1e-2", "79.0%"]],
         "lines": ["Too much penalty underfits"],
         "say": "We compared several penalty strengths on the same network. A moderate penalty helped, while a very strong one made the model underfit."},
        {"title": "Early stopping", "code": ["patience = 5", "best, wait = inf, 0", "for epoch in range(200):", "    val = evaluate(model)",
                                            "    if val < best: best, wait = val, 0", "    else: wait += 1", "    if wait >= patience: break"],
         "say": "Early stopping ends training when the validation loss has not improved for a number of epochs in a row. We keep the best model seen so far."},
    ],
}

def render_slide(slide: dict, path: Path) -> None:
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 110], fill=(31, 56, 100))
    d.text((60, 30), slide["title"], font=_font(FONT_B, 52), fill=(255, 255, 255))
    y = 160
    if "table" in slide:
        f = _font(FONT, 38)
        for r, row in enumerate(slide["table"]):
            for c, cell in enumerate(row):
                x = 120 + c * 460
                d.rectangle([x - 20, y - 8, x + 420, y + 52], outline=(120, 120, 120), width=2,
                            fill=(225, 232, 245) if r == 0 else None)
                d.text((x, y), cell, font=_font(FONT_B, 38) if r == 0 else f, fill=(20, 20, 20))
            y += 62
        y += 30
    if "code" in slide:
        f = _font(MONO, 40)
        d.rectangle([60, y - 20, W - 60, y + 62 * len(slide["code"]) + 10], fill=(245, 245, 245), outline=(200, 200, 200))
        for line in slide["code"]:
            d.text((90, y), line, font=f, fill=(20, 20, 20))
            y += 62
    for line in slide.get("lines", []):
        d.text((90, y), line, font=_font(FONT, 44), fill=(20, 20, 20))
        y += 74
    img.save(path)


def lecture_assets(lec: dict, out: Path, version_changes: dict | None = None) -> dict:
    """Render slides, write a word-level transcript and assemble the MP4. Returns slide windows (ms)."""
    out.mkdir(parents=True, exist_ok=True)
    slides = [dict(s) for s in lec["slides"]]
    for idx, ch in (version_changes or {}).items():
        slides[idx] = {**slides[idx], **ch}
    t = 0.0
    words, concat, windows = [], [], []
    for i, s in enumerate(slides, start=1):
        img = out / f"slide{i:02d}.png"
        render_slide(s, img)
        start = t
        t += GAP_S / 2
        for wtok in s["say"].split():
            words.append({"word": wtok, "start_ms": int(t * 1000), "end_ms": int((t + WORD_S * 0.9) * 1000)})
            t += WORD_S
        t += GAP_S / 2
        dur = t - start
        concat.append(f"file '{img.name}'\nduration {dur:.3f}")
        windows.append((int(start * 1000), int(t * 1000)))
    concat.append(f"file 'slide{len(slides):02d}.png'")     # concat demuxer needs the last file repeated
    (out / "slides.txt").write_text("\n".join(concat) + "\n")
    (out / "transcript.json").write_text(json.dumps({"words": words}))
    total = t
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", "slides.txt",
                    "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", f"{total:.3f}",
                    "-vf", "fps=5,format=yuv420p", "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
                    "-c:a", "aac", "-shortest", "-movflags", "+faststart", "lecture.mp4"], cwd=out, check=True)
    return {"windows": windows, "duration_ms": int(total * 1000)}


def eval_items(w1: list, w2: list, w3: list) -> list[dict]:
    L1, L2, L3 = "gd101/lec01", "gd101/lec02", "gd101/lec03"
    win = lambda w, i: list(w[i - 1])  # noqa: E731  (1-based slide index)
    items = [
        # conceptual
        ("c1", "conceptual", "What does the learning rate control in gradient descent?", "how big a step we take at every iteration", L1, win(w1, 2)),
        ("c2", "conceptual", "What happens if the learning rate is too large?", "the loss can oscillate or even diverge", L1, win(w1, 4)),
        ("c3", "conceptual", "Why does momentum help optimisation?", "it smooths the updates and dampens oscillations", L2, win(w2, 3)),
        ("c4", "conceptual", "What is a learning rate schedule?", "it reduces the learning rate as training progresses", L1, win(w1, 4)),
        ("c5", "conceptual", "Which direction does the gradient point in?", "the direction of steepest increase", L1, win(w1, 2)),
        ("c6", "conceptual", "What is the workhorse algorithm for training machine learning models?", "gradient descent", L1, win(w1, 1)),
        # exact term / code
        ("e1", "exact_term", "Why do we call optimizer.zero_grad() at the start of each iteration?", "PyTorch accumulates gradients by default", L2, win(w2, 2)),
        ("e2", "exact_term", "What does loss.backward() compute?", "the gradients of the loss with respect to every parameter", L2, win(w2, 2)),
        ("e3", "exact_term", "What causes the error 'Trying to backward through the graph a second time'?", "calling backward twice on the same graph", L2, win(w2, 4)),
        ("e4", "exact_term", "What does optimizer.step() do?", "updates the parameters using those gradients", L2, win(w2, 2)),
        # procedural
        ("p1", "procedural", "How do I fix the backward through the graph a second time error?", "one backward pass per forward pass", L2, win(w2, 4)),
        ("p2", "procedural", "How should I search for a good learning rate?", "start with a small value and try values on a log scale", L1, win(w1, 4)),
        # visual-only (answers appear only on the slides)
        ("v1", "visual", "In the table on the convergence slide, how many steps did learning rate 0.01 take?", "310", L1, win(w1, 3)),
        ("v2", "visual", "According to the table on the slide, what happened with learning rate 1.0?", "diverges", L1, win(w1, 3)),
        ("v3", "visual", "What is the default lr in the gd_step function shown on the slide?", "0.01", L1, win(w1, 5)),
        ("v4", "visual", "What typical beta value is shown on the momentum slide?", "0.9", L2, win(w2, 3)),
        ("v5", "visual", "What is the loss function called in the training loop code shown on the slide?", "criterion", L2, win(w2, 2)),
        ("v6", "visual", "Which symbol is used for the step size in the update rule on the slide?", "eta", L1, win(w1, 2)),
        ("v7", "visual", "How many epochs does the loop in the code on the slide run for?", "100", L1, win(w1, 5)),
        # cross-lecture
        ("x1", "cross_lecture", "How does the momentum update differ from stepping along the raw gradient?", "we step along the velocity instead of the raw gradient", L2, win(w2, 3)),
        # temporal
        ("t1", "temporal", "When does the lecture explain the convergence experiment?", None, L1, win(w1, 3)),
        ("t2", "temporal", "At what point in the lectures is momentum introduced?", None, L2, win(w2, 3)),
    ]
    out = []
    for k, (qid, cls, q, gold, src, w) in enumerate(items):
        out.append({"qid": qid, "split": "dev" if k % 3 == 0 else "test", "class": cls, "question": q,
                    "answerable": True, "gold_answer": gold, "gold_source": src, "gold_window_ms": w})
    # stale-version item: only correct after v2 is ingested
    out.append({"qid": "s1", "split": "test", "class": "visual", "question": "What is the best stable learning rate according to the convergence experiment slide?",
                "answerable": True, "gold_answer": "0.05", "gold_source": L1, "gold_window_ms": win(w1, 3), "expected_version": 2})
    unans = [
        ("u1", "What is the default beta2 value of the Adam optimizer?"),
        ("u2", "How do transformers use self-attention?"),
        ("u3", "What is the capital of Australia?"),
        ("u4", "Explain how dropout regularisation works."),
        ("u5", "How do I deploy a model with Kubernetes?"),
        ("n1", "What batch size is used in the training loop?"),
        ("n2", "What momentum value works best for ResNets on ImageNet?"),
        ("n3", "Which learning rate schedule does the lecture recommend for transformers?"),
    ]
    for k, (qid, q) in enumerate(unans):
        out.append({"qid": qid, "split": "dev" if k % 3 == 0 else "test", "class": "unanswerable" if qid[0] == "u" else "near_miss",
                    "question": q, "answerable": False})
    # held-out lecture: written after the heuristics were frozen, never used for tuning
    heldout = [
        ("h1", "conceptual", "What is overfitting?", "keeps improving on the training data but gets worse on data it has not seen", 2),
        ("h2", "conceptual", "What does L2 regularisation add to the loss?", "a penalty on the size of the weights", 3),
        ("h3", "exact_term", "In optimisers, what is the L2 penalty usually implemented as?", "weight decay", 3),
        ("h4", "procedural", "How does early stopping decide when to stop training?", "when the validation loss has not improved for a number of epochs", 5),
        ("h5", "visual", "What validation accuracy did lambda 1e-4 reach in the table on the slide?", "86.5%", 4),
        ("h6", "visual", "Which lambda value is used in our runs according to the L2 penalty slide?", "1e-4", 3),
        ("h7", "visual", "What patience value is set in the early stopping code on the slide?", "5", 5),
        ("h8", "visual", "What is the maximum number of epochs in the early stopping loop shown?", "200", 5),
        ("h9", "temporal", "When is early stopping explained?", None, 5),
    ]
    for qid, cls, q, gold, slide in heldout:
        out.append({"qid": qid, "split": "heldout", "class": cls, "question": q, "answerable": True,
                    "gold_answer": gold, "gold_source": L3, "gold_window_ms": win(w3, slide)})
    out.append({"qid": "h10", "split": "heldout", "class": "unanswerable", "answerable": False,
                "question": "How does batch normalisation reduce internal covariate shift?"})
    out.append({"qid": "h11", "split": "heldout", "class": "near_miss", "answerable": False,
                "question": "What dropout rate is used in the regularisation lecture?"})
    return out


def build(root: str | Path) -> Path:
    root = Path(root)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    a1 = lecture_assets(LEC01, root / "lec01_v1")
    a1v2 = lecture_assets(LEC01, root / "lec01_v2", LEC01_V2_CHANGES)
    a2 = lecture_assets(LEC02, root / "lec02")
    a3 = lecture_assets(LEC03, root / "lec03")
    meta = {"course": "gd101", "title": "Optimisation for Machine Learning (demo)", "licence": "CC BY 4.0 (synthetic demo)"}

    def manifest(lec1_dir):
        return {**meta, "lectures": [
            {"id": "lec01", "title": LEC01["title"], "video": f"{lec1_dir}/lecture.mp4", "transcript": f"{lec1_dir}/transcript.json"},
            {"id": "lec02", "title": LEC02["title"], "video": "lec02/lecture.mp4", "transcript": "lec02/transcript.json"},
            {"id": "lec03", "title": LEC03["title"], "video": "lec03/lecture.mp4", "transcript": "lec03/transcript.json"},
        ]}
    import yaml
    (root / "manifest_v1.yaml").write_text(yaml.safe_dump(manifest("lec01_v1"), sort_keys=False))
    (root / "manifest_v2.yaml").write_text(yaml.safe_dump(manifest("lec01_v2"), sort_keys=False))
    items = eval_items(a1v2["windows"], a2["windows"], a3["windows"])
    (root / "eval.jsonl").write_text("\n".join(json.dumps(i) for i in items) + "\n")
    (root / "slide_windows.json").write_text(json.dumps({"lec01_v1": a1, "lec01_v2": a1v2, "lec02": a2, "lec03": a3}, indent=1))
    return root


if __name__ == "__main__":
    print(build(sys.argv[1] if len(sys.argv) > 1 else "data/demo_course"))
