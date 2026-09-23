#!/usr/bin/env python3
"""Эволюция тела: существо меняется от поколения к поколению.

    python evolve.py creatures/муравей.yaml --generations 30

Это другая история, чем обучение с подкреплением. Там тело задано раз и
навсегда, а учится нейросеть. Здесь наоборот: мозг простой и неизменный —
каждый сустав просто качается туда-сюда, — зато меняется само тело.
Кто уехал дальше, тот оставляет потомков; потомок чуть-чуть отличается
от родителя. Ноги удлиняются, появляются новые, тело перестраивается.

Считается это быстро: живой мозг учить не надо, поэтому одно существо
проверяется за доли секунды и целый прогон укладывается в минуты.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent))
import creature as creature_mod
from creature import Creature
from arena import build_xml


@dataclass
class Result:
    distance: float      # сколько метров проехал вдоль коридора
    fell: bool
    seconds: float       # сколько продержался


class Simulator:
    """Проверка существа с простой качающейся походкой.

    Модели MuJoCo кешируются: у потомков тело часто совпадает с родительским,
    и пересобирать сцену каждый раз незачем.
    """

    FRAME_SKIP = 5

    def __init__(self, seconds: float = 12.0, drift_penalty: float = 0.3):
        self.seconds = seconds
        self.drift_penalty = drift_penalty
        self._cache: dict[str, mujoco.MjModel] = {}

    def model_for(self, beast: Creature) -> mujoco.MjModel:
        xml = build_xml(beast)
        model = self._cache.get(xml)
        if model is None:
            model = mujoco.MjModel.from_xml_string(xml)
            if len(self._cache) > 256:
                self._cache.clear()
            self._cache[xml] = model
        return model

    def run(self, beast: Creature, rng=None, render=None) -> Result:
        model = self.model_for(beast)
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        data.qpos[2] = beast.height
        mujoco.mj_forward(model, data)

        gait = np.asarray(beast.gait_vector(rng), dtype=float).reshape(-1, 3)
        amp, phase, offset = gait[:, 0], gait[:, 1], gait[:, 2]
        omega = 2.0 * math.pi * beast.gait_freq
        dt = model.opt.timestep * self.FRAME_SKIP
        steps = int(self.seconds / dt)

        low, high = 0.35 * beast.height, 1.8 * beast.height
        torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")

        fell, t = False, 0.0
        for i in range(steps):
            t = i * dt
            data.ctrl[:] = np.clip(amp * np.sin(omega * t + phase) + offset, -1, 1)
            mujoco.mj_step(model, data, nstep=self.FRAME_SKIP)
            if render is not None:
                render(model, data, t)
            z = float(data.qpos[2])
            upright = float(data.xmat[torso].reshape(3, 3)[2, 2])
            if not (low < z < high) or upright < 0.0:
                fell = True
                break

        x, y = float(data.qpos[0]), float(data.qpos[1])
        return Result(distance=x - self.drift_penalty * abs(y), fell=fell,
                      seconds=t)


def fitness(result: Result) -> float:
    """Упал — засчитываем только то, что успел проехать до падения."""
    return result.distance


# --------------------------------------------------------------------------- #
def evolve(base: Creature, args) -> tuple[list[dict], list[Creature]]:
    rng = random.Random(args.seed)
    sim = Simulator(seconds=args.seconds)

    # Стартовое поколение: одно и то же тело с разными случайными походками.
    population = [base.with_random_gait(rng) for _ in range(args.population)]
    history, champions = [], []
    best_ever, best_fit = None, -1e9

    for gen in range(1, args.generations + 1):
        scored = []
        for ind in population:
            res = sim.run(ind, rng)
            scored.append((fitness(res), res, ind))
        scored.sort(key=lambda s: -s[0])

        top_fit, top_res, top_ind = scored[0]
        if top_fit > best_fit:
            best_fit, best_ever = top_fit, top_ind
        champions.append(top_ind)
        history.append({
            "поколение": gen,
            "лучший": round(top_fit, 3),
            "средний": round(float(np.mean([s[0] for s in scored])), 3),
            "ног": len(top_ind.legs),
            "суставов": top_ind.n_joints,
            "упал": bool(top_res.fell),
        })
        if gen == 1 or gen % args.report == 0 or gen == args.generations:
            print(f"  поколение {gen:3d}:  лучший {top_fit:6.2f} м   "
                  f"средний {history[-1]['средний']:6.2f} м   "
                  f"ног {len(top_ind.legs)}  суставов {top_ind.n_joints}",
                  flush=True)

        if gen == args.generations:
            break

        # Отбор: выживает верхушка, остальные — её потомки.
        survivors = [s[2] for s in scored[:max(2, args.survivors)]]
        children = list(survivors)
        # Пара существ со случайной походкой в каждом поколении — чтобы
        # популяция не сходилась в одну точку и не застревала.
        for _ in range(args.fresh):
            if len(children) < args.population:
                children.append(rng.choice(survivors).with_random_gait(rng))
        while len(children) < args.population:
            parent = rng.choice(survivors)
            # Сила мутации у каждого потомка своя: часть детей — почти копии
            # родителя (доводка), часть — резко другие (поиск нового).
            strength = args.strength * math.exp(rng.normalvariate(0.0, 0.45))
            children.append(creature_mod.mutate(parent, rng, min(2.0, strength)))
        population = children

    return history, champions


# --------------------------------------------------------------------------- #
def draw_history(history: list[dict], out: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PAGE, SURFACE = "#f9f9f7", "#fcfcfb"
    INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    BEST, MEAN = "#2a78d6", "#eb6834"

    gens = [h["поколение"] for h in history]
    best = [h["лучший"] for h in history]
    mean = [h["средний"] for h in history]
    legs = [h["суставов"] for h in history]

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9.0, 5.6), height_ratios=[2.6, 1.0],
                                  sharex=True)
    fig.patch.set_facecolor(PAGE)
    fig.subplots_adjust(left=0.115, right=0.80, top=0.80, bottom=0.12, hspace=0.30)

    for a in (ax, ax2):
        a.set_facecolor(SURFACE)
        a.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(AXIS)
        a.tick_params(colors=MUTED, labelsize=9, length=3, width=0.8)

    ax.plot(gens, best, color=BEST, linewidth=2.0, zorder=3, solid_capstyle="round")
    ax.plot(gens, mean, color=MEAN, linewidth=2.0, zorder=2, solid_capstyle="round")
    ax.set_ylabel("метров за попытку", color=INK2, fontsize=10)
    ax.text(gens[-1], best[-1], f"  лучший {best[-1]:.1f} м", va="center",
            ha="left", fontsize=9, color=INK2, clip_on=False)
    ax.text(gens[-1], mean[-1], f"  средний {mean[-1]:.1f} м", va="center",
            ha="left", fontsize=9, color=INK2, clip_on=False)

    ax2.step(gens, legs, where="mid", color=MUTED, linewidth=2.0, zorder=3)
    ax2.set_ylabel("суставов\nу лучшего", color=INK2, fontsize=10)
    ax2.set_ylim(min(legs) - 0.6, max(legs) + 0.6)
    ax2.set_yticks(sorted(set(legs)))
    ax2.set_xlabel("поколение", color=INK2, fontsize=10)

    fig.text(0.115, 0.955, title, color=INK, fontsize=15, fontweight="bold",
             ha="left", va="top")
    fig.text(0.115, 0.895, f"{len(history)} поколений · простая качающаяся походка",
             color=MUTED, fontsize=10, ha="left", va="top")
    fig.savefig(out, dpi=160, facecolor=PAGE)
    plt.close(fig)


def record_champions(champions: list[Creature], history: list[dict],
                     sim: Simulator, out: Path, args) -> None:
    """Видео: лучший из нескольких поколений подряд, чтобы видеть изменения."""
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    from record import load_fonts

    picks = sorted(set([0, len(champions) // 5, 2 * len(champions) // 5,
                        3 * len(champions) // 5, 4 * len(champions) // 5,
                        len(champions) - 1]))
    fonts, _ = load_fonts(args.video_height)
    frames: list[np.ndarray] = []

    for idx in picks:
        beast = champions[idx]
        model = sim.model_for(beast)
        renderer = mujoco.Renderer(model, height=args.video_height,
                                   width=args.video_width)
        grabbed: list[np.ndarray] = []

        def grab(m, d, t, _r=renderer, _g=grabbed):
            if len(_g) < args.video_seconds * 20:
                _r.update_scene(d, camera="track")
                _g.append(_r.render())

        sim.run(beast, render=grab)
        renderer.close()

        info = history[idx]
        for frame in grabbed:
            img = Image.fromarray(frame)
            d = ImageDraw.Draw(img, "RGBA")
            pad = int(16 * args.video_height / 480)
            text = f"поколение {info['поколение']}"
            sub = f"{info['ног']} ног · {info['лучший']:.1f} м"
            w = max(d.textlength(text, font=fonts["big"]),
                    d.textlength(sub, font=fonts["mid"]))
            d.rounded_rectangle([pad, pad, pad * 2 + w, pad + int(86 * args.video_height / 480)],
                                radius=int(10 * args.video_height / 480),
                                fill=(15, 15, 20, 205))
            d.text((pad * 1.5, pad * 1.4), text, font=fonts["big"], fill=(245, 245, 250))
            d.text((pad * 1.5, pad * 1.4 + int(40 * args.video_height / 480)), sub,
                   font=fonts["mid"], fill=(190, 190, 200))
            frames.append(np.asarray(img))

    if frames:
        imageio.mimwrite(str(out), frames, fps=20, quality=7, macro_block_size=None)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", help="стартовое существо, например creatures/муравей.yaml")
    ap.add_argument("--generations", type=int, default=30)
    ap.add_argument("--population", type=int, default=24)
    ap.add_argument("--survivors", type=int, default=6,
                    help="сколько лучших оставляем родителями")
    ap.add_argument("--strength", type=float, default=1.0,
                    help="насколько сильно потомки отличаются от родителей")
    ap.add_argument("--seconds", type=float, default=12.0,
                    help="сколько секунд даём существу на попытку")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fresh", type=int, default=1,
                   help="сколько существ в поколении получают случайную походку")
    ap.add_argument("--report", type=int, default=5, help="как часто печатать строку")
    ap.add_argument("--out", default=None, help="папка результата")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--video-width", type=int, default=720)
    ap.add_argument("--video-height", type=int, default=404)
    ap.add_argument("--video-seconds", type=float, default=6.0)
    args = ap.parse_args()

    try:
        base = creature_mod.load(args.file)
    except creature_mod.CreatureError as exc:
        print(f"\nВ описании существа ошибка:\n\n{exc}\n")
        raise SystemExit(1)

    out = Path(args.out) if args.out else Path("evolution") / base.slug
    out.mkdir(parents=True, exist_ok=True)

    print(f"\nэволюция «{base.name}»: {args.generations} поколений "
          f"по {args.population} существ")
    print(f"каждое проверяется {args.seconds:g} секунд\n")
    started = time.time()
    history, champions = evolve(base, args)
    elapsed = time.time() - started

    # Лучший — тот, у кого рекорд за весь прогон.
    best_idx = max(range(len(history)), key=lambda i: history[i]["лучший"])
    champion = champions[best_idx]
    champion.name = f"{base.name}-потомок"
    champion.author = f"эволюция, поколение {history[best_idx]['поколение']}"
    champion.save(out / "лучший.yaml")
    for i, info in enumerate(history):
        if i in (0, len(history) - 1) or info["поколение"] % max(1, args.report) == 0:
            champions[i].save(out / f"поколение_{info['поколение']:03d}.yaml")

    (out / "история.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    draw_history(history, out / "график.png", f"Эволюция «{base.name}»")

    if not args.no_video:
        print("\n  снимаю видео...", flush=True)
        record_champions(champions, history, Simulator(seconds=args.video_seconds),
                         out / "видео.mp4", args)

    first, last = history[0]["лучший"], history[best_idx]["лучший"]
    print(f"\n  было {first:.2f} м  ->  стало {last:.2f} м   "
          f"(за {elapsed:.0f} секунд, {len(history) * args.population} проверок)")
    print(f"  ног: {history[0]['ног']} -> {history[best_idx]['ног']}, "
          f"суставов: {history[0]['суставов']} -> {history[best_idx]['суставов']}")
    print(f"\n  лучший потомок: {out / 'лучший.yaml'}")
    print(f"  его можно обучить нейросетью:")
    print(f"      python train.py --stage walk --creature {out / 'лучший.yaml'} "
          f"--out runs/потомок\n")


if __name__ == "__main__":
    main()
