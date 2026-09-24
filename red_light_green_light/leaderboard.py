#!/usr/bin/env python3
"""Общая таблица класса по результатам обучения.

    python leaderboard.py --runs runs --out runs/таблица

Собирает итоги из runs/*/итог.json и делает две вещи: текстовую таблицу
(её удобно вывести на экран) и картинку со столбиками (её удобно показать
на проекторе).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAGE, SURFACE = "#f9f9f7", "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, BAR = "#e1e0d9", "#c3c2b7", "#2a78d6"
FINISH_M = 25.0


def collect(runs_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(runs_dir.glob("*/итог.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"  пропускаю битый файл: {path}")
            continue
        row["папка"] = path.parent.name
        rows.append(row)
    rows.sort(key=lambda r: (-r.get("медианная_дистанция", 0),
                             -r.get("финишей", 0)))
    # Два ученика вполне могут назвать существо одинаково — в таблице
    # такие строки различаем по автору, иначе непонятно, чьё какое.
    счёт: dict[str, int] = {}
    for r in rows:
        счёт[r.get("имя", "?")] = счёт.get(r.get("имя", "?"), 0) + 1
    for r in rows:
        if счёт.get(r.get("имя", "?"), 0) > 1 and r.get("автор"):
            r["имя"] = f"{r['имя']} ({r['автор'].split(',')[0]})"
    return rows


def write_table(rows: list[dict], out: Path) -> None:
    lines = ["# Результаты класса", "",
             "| Место | Существо | Автор | Ног | Финишей | Медиана, м |",
             "|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        lines.append(
            f"| {i} | {r.get('имя', '?')} | {r.get('автор', '')} | "
            f"{r.get('ног', '?')} | {r.get('финишей', 0):.0%} | "
            f"{r.get('медианная_дистанция', 0):.1f} |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def draw_chart(rows: list[dict], out: Path) -> None:
    if not rows:
        return
    names = [f"{r.get('имя', '?')}" for r in rows][::-1]
    authors = [r.get("автор", "") for r in rows][::-1]
    values = [float(r.get("медианная_дистанция", 0.0)) for r in rows][::-1]

    height = max(2.4, 0.42 * len(rows) + 1.7)
    fig, ax = plt.subplots(figsize=(9.0, height))
    fig.patch.set_facecolor(PAGE)
    ax.set_facecolor(SURFACE)

    y = np.arange(len(names))
    # Тонкие столбики с зазором между ними; цвет один на всех — сравниваем
    # величину, а не разные сущности.
    ax.barh(y, values, height=0.62, color=BAR, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n}\n{a}" if a else n for n, a in zip(names, authors)],
                       fontsize=9, color=INK2)

    top = max(max(values), FINISH_M) * 1.18
    ax.set_xlim(0, top)
    ax.axvline(FINISH_M, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate("финиш", (FINISH_M, 1.0), xycoords=("data", "axes fraction"),
                textcoords="offset points", xytext=(5, -11),
                color=MUTED, fontsize=9)

    for yi, v in zip(y, values):
        ax.text(v + top * 0.012, yi, f"{v:.1f} м", va="center", ha="left",
                fontsize=9, color=INK2)

    ax.grid(True, axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=3, width=0.8)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("сколько метров прошло существо (медиана по эпизодам)",
                  color=INK2, fontsize=10)

    fig.subplots_adjust(left=0.21, right=0.965, top=1 - 1.05 / height,
                        bottom=0.62 / height)
    fig.text(0.21, 1 - 0.30 / height, "Кто дальше прошёл", color=INK,
             fontsize=15, fontweight="bold", ha="left", va="top")
    fig.text(0.21, 1 - 0.62 / height,
             f"{len(rows)} существ · дистанция до финиша {FINISH_M:.0f} м",
             color=MUTED, fontsize=10, ha="left", va="top")
    fig.savefig(out, dpi=160, facecolor=PAGE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs", help="папка с результатами")
    ap.add_argument("--out", default=None,
                    help="куда класть таблицу (без расширения)")
    args = ap.parse_args()

    runs_dir = Path(args.runs)
    rows = collect(runs_dir)
    if not rows:
        raise SystemExit(f"в {runs_dir} нет ни одного итог.json — "
                         f"сначала обучи хотя бы одно существо")

    base = Path(args.out) if args.out else runs_dir / "таблица"
    base.parent.mkdir(parents=True, exist_ok=True)
    write_table(rows, base.with_suffix(".md"))
    draw_chart(rows, base.with_suffix(".png"))

    print(f"\n  {'место':>5}  {'существо':<16} {'автор':<16} {'финишей':>8} {'медиана':>9}")
    print(f"  {'-' * 60}")
    for i, r in enumerate(rows, 1):
        print(f"  {i:>5}  {r.get('имя', '?'):<16} {r.get('автор', ''):<16} "
              f"{r.get('финишей', 0):>7.0%} {r.get('медианная_дистанция', 0):>8.1f} м")
    print(f"\n  таблица:  {base.with_suffix('.md')}")
    print(f"  картинка: {base.with_suffix('.png')}\n")


if __name__ == "__main__":
    main()
