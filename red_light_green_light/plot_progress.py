#!/usr/bin/env python3
"""Графики обучения из CSV-логов stable-baselines3.

    python plot_progress.py runs/walk runs/game --out progress.png

Рисует три панели: средняя награда за эпизод, состав исходов эпизодов
и уровень сложности curriculum. Если передать несколько папок, они
склеиваются в одну шкалу — видно переход между этапами.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# --- палитра ---------------------------------------------------------------
LIGHT = dict(page="#f9f9f7", surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e",
             muted="#898781", grid="#e1e0d9", axis="#c3c2b7", line="#2a78d6")
DARK = dict(page="#0d0d0d", surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7",
            muted="#898781", grid="#2c2c2a", axis="#383835", line="#3987e5")

# Исходы — это состояния, поэтому берём статусные цвета, а не палитру серий.
# Цвет никогда не единственный признак: рядом всегда легенда и подписи.
OUTCOMES = [
    ("finished",   "дошёл до финиша", "#0ca30c"),
    ("timeout",    "не успел",        "#898781"),
    ("fell",       "упал",            "#fab219"),
    ("eliminated", "вылетел",         "#d03b3b"),
]
CURRICULUM_COLOR = "#eb6834"


def read_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"пустой лог: {path}")
    out = {}
    for key in rows[0]:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[key]))
            except (TypeError, ValueError):
                vals.append(np.nan)
        out[key] = np.asarray(vals, dtype=float)
    return out


def load_stages(dirs: list[Path]):
    """Склеивает логи этапов в общую шкалу шагов."""
    stages, offset = [], 0.0
    for d in dirs:
        path = d / "logs" / "progress.csv" if d.is_dir() else d
        if not path.exists():
            raise SystemExit(f"нет файла с логами: {path}")
        data = read_csv(path)
        steps = data.get("time/total_timesteps")
        if steps is None:
            raise SystemExit(f"в {path} нет колонки time/total_timesteps")
        data["_x"] = steps + offset
        offset = float(np.nanmax(data["_x"]))
        stages.append(data)
    return stages


def style_axis(ax, c, ylabel: str):
    ax.set_facecolor(c["surface"])
    ax.grid(True, axis="y", color=c["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(c["axis"])
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=c["muted"], labelsize=9, length=3, width=0.8)
    ax.set_ylabel(ylabel, color=c["ink2"], fontsize=10)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", help="папки этапов, например runs/walk runs/game")
    ap.add_argument("--out", default="progress.png")
    ap.add_argument("--dark", action="store_true")
    ap.add_argument("--title", default="Обучение агента «Красный свет — зелёный свет»")
    args = ap.parse_args()

    c = DARK if args.dark else LIGHT
    stages = load_stages([Path(d) for d in args.dirs])
    has_curriculum = any(not np.all(np.isnan(s.get("curriculum/difficulty", np.array([np.nan]))))
                         for s in stages)

    n_panels = 3 if has_curriculum else 2
    heights = [2.6, 2.2, 0.9][:n_panels]
    fig_h = sum(heights) + 1.9
    fig, axes = plt.subplots(n_panels, 1, figsize=(9.5, fig_h),
                             height_ratios=heights, sharex=True)
    fig.patch.set_facecolor(c["page"])
    axes = np.atleast_1d(axes)
    # Поля считаем в долях от высоты рисунка, чтобы заголовок, легенда
    # и подпись оси не наезжали друг на друга при 2 и при 3 панелях.
    fig.subplots_adjust(left=0.095, right=0.795,
                        top=1 - 1.15 / fig_h, bottom=0.62 / fig_h,
                        hspace=0.46)

    to_m = 1e6

    # ---- панель 1: награда ------------------------------------------------
    ax = axes[0]
    style_axis(ax, c, "награда за эпизод")
    for s in stages:
        y = s.get("rollout/ep_rew_mean")
        if y is None:
            continue
        ax.plot(s["_x"] / to_m, y, color=c["line"], linewidth=2.0,
                solid_capstyle="round", zorder=3)
    last = stages[-1].get("rollout/ep_rew_mean")
    if last is not None and not np.all(np.isnan(last)):
        xs, ys = stages[-1]["_x"][-1] / to_m, last[-1]
        ax.scatter([xs], [ys], s=36, color=c["line"], zorder=4,
                   edgecolors=c["surface"], linewidths=2)
        ax.annotate(f"{ys:,.0f}".replace(",", " "), (xs, ys),
                    textcoords="offset points", xytext=(-6, 10),
                    ha="right", color=c["ink"], fontsize=10, fontweight="bold")

    # ---- панель 2: исходы -------------------------------------------------
    ax = axes[1]
    style_axis(ax, c, "исходы эпизодов")
    for s in stages:
        x = s["_x"] / to_m
        series = []
        for key, _, _ in OUTCOMES:
            v = s.get(f"outcome/{key}")
            series.append(np.nan_to_num(v, nan=0.0) if v is not None else np.zeros_like(x))
        total = np.sum(series, axis=0)
        total[total == 0] = 1.0
        series = [v / total * 100.0 for v in series]
        ax.stackplot(x, *series, colors=[col for _, _, col in OUTCOMES],
                     edgecolor=c["surface"], linewidth=2.0, zorder=2)
        # прямые подписи у правого края — чтобы цвет не был единственным признаком
        if s is stages[-1]:
            bottom = 0.0
            for (key, label, _), v in zip(OUTCOMES, series):
                share = v[-1]
                if share >= 9.0:
                    ax.text(x[-1], bottom + share / 2, f" {label} {share:.0f}%",
                            va="center", ha="left", fontsize=9, color=c["ink2"],
                            clip_on=False)
                bottom += share
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 50, 100])
    ax.set_yticklabels(["0", "50", "100%"])
    ax.legend(handles=[Patch(facecolor=col, label=label) for _, label, col in OUTCOMES],
              loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=4, frameon=False,
              fontsize=9, labelcolor=c["ink2"], handlelength=1.1, handleheight=1.1,
              columnspacing=1.8, borderpad=0.0, handletextpad=0.6)

    # ---- панель 3: сложность ---------------------------------------------
    if has_curriculum:
        ax = axes[2]
        style_axis(ax, c, "сложность")
        for s in stages:
            y = s.get("curriculum/difficulty")
            if y is None or np.all(np.isnan(y)):
                continue
            ax.plot(s["_x"] / to_m, y, color=CURRICULUM_COLOR, linewidth=2.0,
                    solid_capstyle="round", zorder=3)
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["легко", "как в игре"])

    # ---- границы этапов ---------------------------------------------------
    for s in stages[:-1]:
        xb = float(np.nanmax(s["_x"])) / to_m
        for ax in axes:
            ax.axvline(xb, color=c["muted"], linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
        axes[0].annotate("включили светофор", (xb, 1.0),
                         xycoords=("data", "axes fraction"),
                         textcoords="offset points", xytext=(6, -12),
                         color=c["muted"], fontsize=9)

    axes[-1].set_xlabel("шагов симуляции, млн", color=c["ink2"], fontsize=10)

    fig.text(0.095, 1 - 0.30 / fig_h, args.title, color=c["ink"],
             fontsize=15, fontweight="bold", ha="left", va="top")
    total_steps = float(np.nanmax(stages[-1]["_x"]))
    fig.text(0.095, 1 - 0.62 / fig_h,
             f"всего {total_steps / to_m:.1f} млн шагов · PPO · MuJoCo",
             color=c["muted"], fontsize=10, ha="left", va="top")

    fig.savefig(args.out, dpi=160, facecolor=c["page"])
    print(f"график сохранён: {args.out}")


if __name__ == "__main__":
    main()
