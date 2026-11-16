"""Task figures as screenshot strips: two rows of three frames (B, D, A) from recorded traces.

Each figure shows one recorded episode of one task family as a pair of rows that share the
frame at B. Row 1 is the action as first proposed (or the Paused mode); row 2 is the paired
condition (late binding of the same decision, or Live). Every number is read from the traces.

    PYTHONPATH=src .venv/bin/python studies/fig_task_strips.py [notification trading jump code livepage]
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from chronotope.scenario import manifest_from_dict  # noqa: E402
from chronotope.trace import Trace  # noqa: E402

OUT = ROOT / "paper_chronotope_iclr2027" / "figures"
INK, MUTED, GRID = "#1a1a1a", "#5a5a5a", "#c8c8c2"
FROZEN_C, LATE_C, PAUSED_C, LIVE_C, RED = "#b45f06", "#1baf7a", "#2b5fa8", "#b45f06", "#d62828"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix"})


# ------------------------------------------------------------------ trace access
class Ep:
    """One trace: manifest, observations, the committing decision, commit, receipt."""

    def __init__(self, path: Path):
        self.path = path
        self.run = path.parent
        self.t = Trace.load(path)
        h = self.t.header()
        self.m = manifest_from_dict(h["manifest"])
        trailer = self.t.first("scenario_trailer")
        if trailer is not None and hasattr(self.m, "with_trailer"):
            self.m = self.m.with_trailer(trailer)   # live page: measured geometry and event times
        self.h0 = h["harness_origin_ns"]
        self.obs = self.t.all("observation")
        self.dec = self.t.committing_decision()
        self.inv = next(i for i in self.t.all("decision_invocation") if i["decision_id"] == self.dec["decision_id"])
        self.commit = self.t.first("commit")
        self.receipt = self.t.first("receipt")
        self.corr = next((c for c in self.t.all("coordinate_correction")
                          if c["decision_id"] == self.dec["decision_id"]), None)
        self.ob_b = next(o for o in self.obs if o["observation_id"] == self.dec["observation_id"])
        # newest observation published at or before commitment (harness clock)
        self.ob_d = max((o for o in self.obs if o["publication_time_ns"] <= self.commit["commitment_time_ns"]),
                        key=lambda o: o["publication_time_ns"])
        self.tA = self.receipt["task_time_at_receipt_ns"]

    def s(self, harness_ns: int) -> float:
        return (harness_ns - self.h0) / 1e9

    @property
    def tB(self):
        return self.s(self.dec["begin_time_ns"])

    @property
    def tD(self):
        return self.s(self.commit["commitment_time_ns"])

    @property
    def tA_s(self):
        return self.s(self.receipt["receipt_time_ns"])

    def payload(self, ob) -> Image.Image:
        data = (self.run / "payloads" / f"{ob['payload_hash']}.bin").read_bytes()
        w, h = self.m.canvas_w_px, self.m.canvas_h_px
        if data[:4] == b"\x89PNG":
            return Image.open(io.BytesIO(data)).convert("RGB")
        mode = "RGB" if len(data) == w * h * 3 else "L"
        return Image.frombytes(mode, (w, h), data).convert("RGB")

    def frame_at(self, task_ns: int) -> Image.Image:
        """Screen at a task time: rendered by the generator, or the nearest recorded screenshot
        for the live page (which has no offline renderer)."""
        if self.m.scenario_id == "live_page":
            later = [o for o in self.obs if o["source_time_ns"] >= task_ns]
            ob = min(later, key=lambda o: o["source_time_ns"]) if later else self.obs[-1]
            return self.payload(ob)
        data = self.m.render(task_ns)
        w, h = self.m.canvas_w_px, self.m.canvas_h_px
        mode = "RGB" if len(data) == w * h * 3 else "L"
        return Image.frombytes(mode, (w, h), data).convert("RGB")

    def frames(self):
        return self.payload(self.ob_b), self.payload(self.ob_d), self.frame_at(self.tA)

    def error_px(self, action) -> float | None:
        """Distance from the click to the target's centre at receipt (the same quantity as the
        hero figure; every task here stores its target as a box)."""
        if "x" not in action:
            return None
        # the target as the click finds it: on the live page the click itself advances the state at
        # exactly tA (the page reports completion), so score against the instant before receipt
        t = self.tA - 1 if self.m.scenario_id == "live_page" else self.tA
        cx, cy = self.m.target_box_at(t).center()
        return ((cx - action["x"]) ** 2 + (cy - action["y"]) ** 2) ** 0.5


# ------------------------------------------------------------------ drawing
def crosshair(ax, x, y, color, r=9, lw=2.0):
    ax.add_patch(plt.Circle((x, y), r, fill=False, ec=color, lw=lw, zorder=6))
    ax.plot([x - r - 5, x + r + 5], [y, y], color=color, lw=lw * 0.7, zorder=6)
    ax.plot([x, x], [y - r - 5, y + r + 5], color=color, lw=lw * 0.7, zorder=6)


def strip(title, rows, cols, out, note=None, fw=7.2):
    """rows: list of dict(label, color, frames=[img,img,img], marks=[[(x,y,c)]*3], captions=[str*3],
    verdict=(text, color)). cols: three column titles."""
    n = len(rows)
    img_w, img_h = rows[0]["frames"][0].size
    aspect = img_h / img_w
    left, gap, top_h, right = 1.15, 0.12, 0.42, 0.05
    col_w = (fw - left - right - 2 * gap) / 3
    row_h = col_w * aspect + 0.46
    fh = top_h + n * row_h - 0.08 + (0.16 if note else 0)
    fig = plt.figure(figsize=(fw, fh), dpi=300)
    fig.patch.set_facecolor("white")
    fig.text(0.01, 1 - 0.13 / fh, title, ha="left", va="top", fontsize=9.5, weight="bold", color=INK)
    for j, c in enumerate(cols):
        x0 = (left + j * (col_w + gap)) / fw
        fig.text(x0 + col_w / 2 / fw, 1 - (top_h - 0.02) / fh, c, ha="center", va="bottom", fontsize=7.6, color=INK)
    for i, r in enumerate(rows):
        y_top = fh - top_h - i * row_h
        y_img = (y_top - col_w * aspect - 0.02) / fh
        fig.text(0.012, (y_top - col_w * aspect / 2) / fh, r["label"], ha="left", va="center",
                 fontsize=7.8, color=r["color"], weight="bold", linespacing=1.15)
        for j in range(3):
            x0 = (left + j * (col_w + gap)) / fw
            ax = fig.add_axes([x0, y_img, col_w / fw, col_w * aspect / fh])
            ax.imshow(r["frames"][j], interpolation="nearest")
            ax.set_xlim(0, img_w)
            ax.set_ylim(img_h, 0)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(GRID)
            for (x, y, c) in r["marks"][j]:
                crosshair(ax, x, y, c, r=max(7, img_w // 36), lw=1.8)
            for (x0b, y0b, x1b, y1b, c) in r.get("boxes", [[], [], []])[j]:
                ax.add_patch(Rectangle((x0b, y0b), x1b - x0b, y1b - y0b, fill=False, ec=c, lw=1.4,
                                       ls=(0, (3, 2)), zorder=5))
            cap = r["captions"][j]
            col = r["verdict"][1] if j == 2 else MUTED
            fig.text(x0 + col_w / 2 / fw, y_img - 0.035 / fh, cap, ha="center", va="top", fontsize=6.6,
                     color=col, linespacing=1.15, weight="bold" if j == 2 else "normal")
    if note:
        fig.text(0.01, 0.02 / fh, note, ha="left", va="bottom", fontsize=6.2, color=MUTED, style="italic")
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(out)


def col_titles(e: Ep, a_label: str):
    return [f"$B$: the frame the model saw ($t$ = {e.tB:.2f} s)",
            f"$D$: newest frame at commitment ($t$ = {e.tD:.2f} s)",
            f"$A$: {a_label} ($t$ = {e.tA_s:.2f} s)"]


def click_rows_same_decision(e: Ep, first_label, second_label, hit_word="HIT", miss_word="MISS"):
    """Two rows from one late-bind trace: the proposed (initial) coordinate and the late-bound one."""
    px, py = e.corr["proposed_action"]["x"], e.corr["proposed_action"]["y"]
    cx, cy = e.commit["action"]["x"], e.commit["action"]["y"]
    fb, fd, fa = e.frames()
    err_p = e.error_px(e.corr["proposed_action"])
    ok_p = e.m.action_valid_at(e.tA, e.corr["proposed_action"])
    err_c = e.error_px(e.commit["action"])
    ok_c = e.receipt["valid_at_receipt"]
    delib = e.s(e.inv["end_time_ns"]) - e.tB
    bind_ms = e.corr["track_ns"] / 1e6 if "track_ns" in e.corr else (e.corr["binder_end_ns"] - e.corr["binder_start_ns"]) / 1e6
    moved = e.corr.get("moved_px", 0.0)

    def fmt(ok, err, word_ok, word_bad):
        w = word_ok if ok else word_bad
        return f"{w}, {err:.0f} px off" if err is not None else w

    rows = [dict(label=first_label, color=FROZEN_C, frames=[fb, fd, fa],
                 marks=[[(px, py, FROZEN_C)], [(px, py, FROZEN_C)], [(px, py, FROZEN_C)]],
                 captions=[f"model proposes click({px}, {py})\nafter {delib:.1f} s of deliberation",
                           f"target moved {moved:.0f} px; the\ncoordinate is still ({px}, {py})",
                           fmt(ok_p, err_p, hit_word, miss_word)],
                 verdict=("", LATE_C if ok_p else RED)),
            dict(label=second_label, color=LATE_C, frames=[fb, fd, fa],
                 marks=[[(px, py, FROZEN_C)], [(cx, cy, LATE_C)], [(cx, cy, LATE_C)]],
                 captions=["same model call, same proposal",
                           f"binder re-locates the patch in this\nframe: ({cx}, {cy}), {bind_ms:.0f} ms before $D$",
                           fmt(ok_c, err_c, hit_word, miss_word)],
                 verdict=("", LATE_C if ok_c else RED))]
    return rows


# ------------------------------------------------------------------ the five tasks
def fig_notification(seed=1000, w_ns=4_000_000_000):
    run = ROOT / "runs" / "r1_qwen38_notification_100"
    live = Ep(run / f"eval_w{w_ns}_s{seed}_live.jsonl")
    paused = Ep(run / f"eval_w{w_ns}_s{seed}_paused.jsonl")
    rows = []
    for e, lab, col in ((paused, "Paused\n(clock stops\nwhile the\nmodel thinks)", PAUSED_C),
                        (live, "Live\n(clock runs)", LIVE_C)):
        x, y = e.commit["action"]["x"], e.commit["action"]["y"]
        fb, fd, fa = e.frames()
        ok = e.receipt["valid_at_receipt"]
        err = e.error_px(e.commit["action"])
        delib = e.s(e.inv["end_time_ns"]) - e.tB
        tD_task = e.commit["task_time_at_commit_ns"] / 1e9
        tA_task = e.tA / 1e9
        close = (e.m.event_time_ns + e.m.response_window_ns) / 1e9
        if e is paused:
            c2 = f"task clock still at {tD_task:.2f} s;\nthe card is where it was"
            c3 = f"HIT, {err:.0f} px off (task clock {tA_task:.2f} s)" if ok else "MISS"
        else:
            c2 = f"task clock at {tD_task:.2f} s; the card\nclosed at {close:.1f} s, before $D$"
            c3 = (f"HIT, {err:.0f} px off" if ok else
                  f"MISS: nothing to click\n(outdated at commitment, late)")
        rows.append(dict(label=lab, color=col, frames=[fb, fd, fa],
                         marks=[[(x, y, col)], [(x, y, col)], [(x, y, col)]],
                         captions=[f"card visible; model proposes\nclick({x}, {y}) after {delib:.1f} s", c2, c3],
                         verdict=("", LATE_C if ok else RED)))
    strip(f"Notification, {w_ns/1e9:.0f}-s window, seed {seed}: the same model call in Paused and in Live",
          rows, col_titles(live, "the click lands"), OUT / "task_notification_strip.png",
          note="Times are harness time from episode start; the Paused row's task clock stops at $B$ and resumes after $A$.")


def fig_trading(seed=200, regime="r6680"):
    run = ROOT / "runs" / "b1_qwen38_nothink_100"
    live = Ep(run / f"prl_{regime}_s{seed}_live.jsonl")
    paused = Ep(run / f"prl_{regime}_s{seed}_paused.jsonl")
    rows = []
    rev = live.m.reversal_time_ns() / 1e9
    for e, lab, col in ((paused, "Paused", PAUSED_C), (live, "Live", LIVE_C)):
        x, y = e.commit["action"]["x"], e.commit["action"]["y"]
        fb, fd, fa = e.frames()
        ok = e.receipt["valid_at_receipt"]
        delib = e.s(e.inv["end_time_ns"]) - e.tB
        tD_task = e.commit["task_time_at_commit_ns"] / 1e9
        if e is paused:
            c2 = f"task clock still at {tD_task:.2f} s;\nthe signal is unchanged"
            c3 = "VALID ORDER, return +1" if ok else "INVALID ORDER, return $-$1"
        else:
            c2 = f"task clock at {tD_task:.2f} s; the signal\nreversed at {rev:.2f} s, before $D$"
            c3 = "VALID ORDER, return +1" if ok else "INVALID ORDER, return $-$1\n(outdated at commitment)"
        rows.append(dict(label=lab, color=col, frames=[fb, fd, fa],
                         marks=[[(x, y, col)], [(x, y, col)], [(x, y, col)]],
                         captions=[f"signal on; model proposes the\norder click({x}, {y}) after {delib:.1f} s", c2, c3],
                         verdict=("", LATE_C if ok else RED)))
    strip(f"Trading, reversal {rev - live.m.event_time_ns/1e9:.2f} s after the signal, seed {seed}: the same order in Paused and in Live",
          rows, col_titles(live, "the order arrives"), OUT / "task_trading_strip.png")


def fig_jump(seed=1000, cell="jump2s"):
    e = Ep(ROOT / "runs" / "c2_qwen38_nothink_100" / f"live_{cell}_late_bind_s{seed}.jsonl")
    rows = click_rows_same_decision(e, "initial\ncoordinate", "late binding\n(same decision)")
    jump = (e.m.event_time_ns + e.m.jump_delay_ns) / 1e9
    strip(f"Balloon Jump, seed {seed}: the balloon jumps {e.m.jump_dx_px} px at $t$ = {jump:.1f} s, during deliberation",
          rows, col_titles(e, "the click lands"), OUT / "task_jump_strip.png")


def fig_code(seed=1000, cell="ch2"):
    e = Ep(ROOT / "runs" / "c4_qwen38_nothink_100" / f"live_{cell}_late_bind_s{seed}.jsonl")
    fb, fd, fa = e.frames()
    v_b = e.corr["proposed_action"]["value"]
    v_d = e.commit["action"]["value"]
    code_a = e.m.code_at(e.tA)
    ok_p = e.m.action_valid_at(e.tA, e.corr["proposed_action"])
    ok_c = e.receipt["valid_at_receipt"]
    delib = e.s(e.inv["end_time_ns"]) - e.tB
    dec_ms = (e.corr["binder_end_ns"] - e.corr["binder_start_ns"]) / 1e6
    change = e.m.change_time_ns() / 1e9
    pb = e.m.panel_box()
    box = [(pb.x0 - 2, pb.y0 - 2, pb.x1 + 2, pb.y1 + 2, RED)]
    rows = [dict(label="initial\nvalue", color=FROZEN_C, frames=[fb, fd, fa], marks=[[], [], []],
                 boxes=[[], box, box],
                 captions=[f"panel shows {v_b}; model proposes\ntype(“{v_b}”) after {delib:.1f} s",
                           f"panel now shows {code_a}; the\nvalue to type is still “{v_b}”",
                           (f"CORRECT" if ok_p else f"WRONG: typed {v_b}, panel shows {code_a}")],
                 verdict=("", LATE_C if ok_p else RED)),
            dict(label="late binding\n(same decision)", color=LATE_C, frames=[fb, fd, fa], marks=[[], [], []],
                 boxes=[[], box, box],
                 captions=["same model call, same proposal",
                           f"decoder re-reads the panel in this\nframe: “{v_d}”, {dec_ms:.1f} ms before $D$",
                           (f"CORRECT: typed {v_d}, panel shows {code_a}" if ok_c else f"WRONG: typed {v_d}")],
                 verdict=("", LATE_C if ok_c else RED))]
    strip(f"Code Panel, seed {seed}: the code is replaced at $t$ = {change:.1f} s, during deliberation",
          rows, col_titles(e, "the typed value arrives"), OUT / "task_code_strip.png",
          note="Dashed box: the code panel after the replacement.")


def fig_livepage(seed=1000):
    e = Ep(ROOT / "runs" / "g2_live_qwen38_nothink_100" / f"live_step1_late_bind_s{seed}.jsonl")
    rows = click_rows_same_decision(e, "initial\ncoordinate", "late binding\n(same decision)")
    banner = (e.m.event_time_ns + e.m.banner_delay_ns) / 1e9
    strip(f"Live Chromium page, seed {seed}: a notice banner is inserted at $t$ = {banner:.1f} s, during deliberation",
          rows, col_titles(e, "the click reaches the browser"), OUT / "task_livepage_strip.png",
          note="Frames are the 15-Hz screenshots taken through the observation boundary; the $A$ frame is the first screenshot at or after receipt.")


FIGS = dict(notification=fig_notification, trading=fig_trading, jump=fig_jump, code=fig_code, livepage=fig_livepage)

if __name__ == "__main__":
    names = sys.argv[1:] or list(FIGS)
    for n in names:
        FIGS[n]()
