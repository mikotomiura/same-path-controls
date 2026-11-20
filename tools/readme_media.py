"""The figures of the README: the design, and one known-gap capture read on four time lines (a still and an animation).

Reads only released series (series/v7s/not-kupa-derived/F_K.csv, a capture at constant planned gaps; no person's
record) and stops unless the file is the bytes its manifest names. Writes docs/media/design.png, docs/media/clock.png
and docs/media/clock.gif. The figures are illustrations: no number of the article comes from them.

Usage: uv run --frozen python tools/readme_media.py
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "media"
SERIES = ROOT / "series" / "v7s"
FILE = "not-kupa-derived/F_K.csv"
UNIT = "c003"            # Firefox, constant planned gap of 3 ms, hook installed
N = 60                   # presses shown
ROWS = (("plan", "t_ms", "plan (no input system)"),
        ("hook", "t_ms", "hook (performance counter)"),
        ("page", "h_ms", "page, handler's time (performance.now)"),
        ("page", "t_ms", "page, event's time (event.timeStamp)"))
COLOURS = ("#5c5c5c", "#2f6f9f", "#3f8f4f", "#d66e28")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.hashsalt": "readme",
                     "figure.facecolor": "white", "axes.facecolor": "white"})


def series_bytes() -> bytes:
    """The file, checked against the manifest's SHA-256."""
    b = (SERIES / FILE).read_bytes()
    want = json.loads((SERIES / "manifest.json").read_text(encoding="utf-8"))["files"][FILE]["sha256"]
    if hashlib.sha256(b).hexdigest() != want:
        raise SystemExit(f"{SERIES / FILE} is not the bytes its manifest names")
    return b


def time_lines(b: bytes) -> list[np.ndarray]:
    rows = [r for r in csv.DictReader(io.StringIO(b.decode("utf-8"))) if r["unit"] == UNIT]
    out = []
    for layer, col, _ in ROWS:
        t = np.array([float(r[col]) for r in rows if r["layer"] == layer])
        if t.size < N or t[0] != 0.0:
            raise SystemExit(f"{UNIT}/{layer}/{col}: {t.size} presses from {t[:1]}")
        out.append(t[:N])
    return out


def save_png(fig, path: Path) -> None:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, metadata={"Software": None})
    plt.close(fig)
    path.write_bytes(buf.getvalue())


def design() -> None:
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.1)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#f4f6f8"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec="#333333",
                                    lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)

    def arrow(a, b, dashed=False):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, lw=1.1, color="#333333",
                                     linestyle="--" if dashed else "-"))

    box(0.3, 3.0, 2.3, 0.9, "human record H0\n(the dataset's logger)", "#eef3f8")
    box(0.3, 1.6, 2.3, 0.9, "machine plan\n(no input system)", "#fbefe6")
    box(3.3, 2.3, 1.6, 0.9, "replay tool\n(SendInput)")
    box(5.5, 2.3, 1.4, 0.9, "OS input\nstream")
    box(7.5, 3.0, 2.2, 0.8, "hook layer")
    box(7.5, 1.7, 2.2, 0.8, "page layer\n(Chrome, Firefox)")
    box(0.3, 0.1, 9.4, 1.0, "Each detector judges each series on each layer apart.\n"
        "Human side: H0 vs. its replay H1.   Machine side: the plan vs. its record.", "#ffffff")
    arrow((2.6, 3.45), (3.3, 2.95))
    arrow((2.6, 2.05), (3.3, 2.55))
    arrow((4.9, 2.75), (5.5, 2.75))
    arrow((6.9, 2.85), (7.5, 3.35))
    arrow((6.9, 2.65), (7.5, 2.1))
    arrow((1.45, 1.6), (1.45, 1.1), dashed=True)
    ax.plot([0.3, 0.12, 0.12], [3.45, 3.45, 0.6], color="#333333", lw=1.1, ls="--")
    arrow((0.12, 0.6), (0.3, 0.6), dashed=True)
    arrow((8.6, 1.7), (8.6, 1.1))
    ax.plot([9.7, 9.88, 9.88], [3.4, 3.4, 0.6], color="#333333", lw=1.1)
    arrow((9.88, 0.6), (9.7, 0.6))
    ax.text(2.75, 3.38, "gaps", fontsize=8, color="#555555")
    ax.text(2.72, 1.9, "schedule", fontsize=8, color="#555555")
    save_png(fig, OUT / "design.png")


def clock_axes(ax, lines, upto: int, t_max: float) -> None:
    ax.set_xlim(-3, t_max)
    ax.set_ylim(-0.6, len(ROWS) - 0.4)
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([lab for _, _, lab in ROWS][::-1])
    ax.set_xlabel("time from the first press (ms)")
    for k, (t, c) in enumerate(zip(lines, COLOURS)):
        y = len(ROWS) - 1 - k
        ax.vlines(t[:upto], y - 0.32, y + 0.32, color=c, lw=1.2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def clock(lines: list[np.ndarray]) -> None:
    t_max = float(max(t[-1] for t in lines)) + 8
    fig, ax = plt.subplots(figsize=(8.6, 2.9))
    clock_axes(ax, lines, N, t_max)
    ax.set_title(f"The same {N} key presses, planned 3 ms apart, on four time lines (Firefox)", fontsize=10)
    fig.tight_layout()
    save_png(fig, OUT / "clock.png")

    frames = []
    for upto in list(range(1, N + 1)) + [N] * 12:
        fig, ax = plt.subplots(figsize=(8.6, 2.9), dpi=90)
        clock_axes(ax, lines, upto, t_max)
        ax.set_title(f"press {upto} of {N}, planned 3 ms apart (Firefox)", fontsize=10)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=90)
        plt.close(fig)
        frames.append(Image.open(io.BytesIO(buf.getvalue())).convert("RGB").quantize(colors=64, dither=0))
    frames[0].save(OUT / "clock.gif", save_all=True, append_images=frames[1:], duration=90, loop=0, optimize=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lines = time_lines(series_bytes())
    design()
    clock(lines)
    for p in sorted(OUT.glob("*")):
        print(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}  {p.stat().st_size}")


if __name__ == "__main__":
    sys.exit(main())
