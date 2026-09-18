#!/usr/bin/env python3
"""Запись видео с обученным агентом.

    python record.py --model runs/game --out demo.mp4 --episodes 3

Поверх картинки рисуется HUD: состояние светофора, скорость агента
относительно порога вылета и полоса прогресса до финиша.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rlgl_env import RedLightGreenLightEnv, GREEN

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]

LABELS_RU = {"green": "ЗЕЛЁНЫЙ", "red": "КРАСНЫЙ", "grace": "ЗАМРИ!",
             "finished": "ФИНИШ", "eliminated": "ВЫЛЕТЕЛ", "fell": "УПАЛ",
             "timeout": "НЕ УСПЕЛ", "speed": "скорость", "episode": "попытка"}
LABELS_EN = {"green": "GREEN", "red": "RED", "grace": "FREEZE!",
             "finished": "FINISH", "eliminated": "ELIMINATED", "fell": "FELL",
             "timeout": "TIMEOUT", "speed": "speed", "episode": "run"}


def load_fonts(height: int):
    """Возвращает (шрифты, словарь подписей). Кириллица — только для TrueType."""
    scale = height / 480.0
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            return ({
                "big": ImageFont.truetype(path, int(34 * scale)),
                "mid": ImageFont.truetype(path, int(20 * scale)),
                "small": ImageFont.truetype(path, int(15 * scale)),
            }, LABELS_RU)
    default = ImageFont.load_default()
    return ({"big": default, "mid": default, "small": default}, LABELS_EN)


def draw_hud(frame: np.ndarray, env: RedLightGreenLightEnv, fonts, labels,
             episode: int, outcome: str | None) -> np.ndarray:
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img, "RGBA")
    W, H = img.size
    s = H / 480.0
    pad = int(16 * s)

    # --- плашка светофора -------------------------------------------------
    if env.light == GREEN:
        key, color = "green", (40, 200, 70)
    elif env.t < env.grace_until:
        key, color = "grace", (245, 175, 25)
    else:
        key, color = "red", (235, 45, 45)

    text = labels[key]
    tw = d.textlength(text, font=fonts["big"])
    box_w, box_h = tw + 2 * pad, int(52 * s)
    d.rounded_rectangle([pad, pad, pad + box_w, pad + box_h], radius=int(10 * s),
                        fill=color + (230,))
    d.text((pad * 2, pad + box_h / 2), text, font=fonts["big"],
           fill=(255, 255, 255), anchor="lm")

    # --- номер попытки ----------------------------------------------------
    d.text((W - pad, pad + box_h / 2), f"{labels['episode']} {episode}",
           font=fonts["mid"], fill=(235, 235, 240), anchor="rm")

    # --- измеритель скорости ---------------------------------------------
    speed = float(np.linalg.norm(env.data.qvel[:2]))
    thr = env.move_threshold
    bar_w, bar_h = int(220 * s), int(14 * s)
    bx, by = pad, pad + box_h + int(14 * s)
    d.rounded_rectangle([bx, by, bx + bar_w, by + bar_h], radius=bar_h // 2,
                        fill=(255, 255, 255, 45))
    fill_w = int(bar_w * min(speed / max(thr * 2, 1e-6), 1.0))
    over = env.light != GREEN and env.t >= env.grace_until and speed > thr
    d.rounded_rectangle([bx, by, bx + max(fill_w, bar_h), by + bar_h], radius=bar_h // 2,
                        fill=(235, 60, 60, 240) if over else (110, 190, 255, 240))
    # отметка порога вылета
    mx = bx + bar_w // 2
    d.line([mx, by - int(4 * s), mx, by + bar_h + int(4 * s)],
           fill=(255, 255, 255, 200), width=max(1, int(2 * s)))
    d.text((bx + bar_w + int(10 * s), by + bar_h / 2),
           f"{labels['speed']} {speed:4.2f} / {thr:4.2f} м/с",
           font=fonts["small"], fill=(225, 225, 230), anchor="lm")

    # --- прогресс до финиша ----------------------------------------------
    pb_w, pb_h = W - 2 * pad, int(10 * s)
    px, py = pad, H - pad - pb_h
    d.rounded_rectangle([px, py, px + pb_w, py + pb_h], radius=pb_h // 2,
                        fill=(255, 255, 255, 40))
    frac = float(np.clip(env.data.qpos[0] / env.finish_x, 0.0, 1.0))
    if frac > 0:
        d.rounded_rectangle([px, py, px + max(int(pb_w * frac), pb_h), py + pb_h],
                            radius=pb_h // 2, fill=(245, 205, 60, 240))
    d.text((px, py - int(6 * s)),
           f"{env.data.qpos[0]:5.1f} / {env.finish_x:.0f} м",
           font=fonts["small"], fill=(225, 225, 230), anchor="ls")

    # --- итог эпизода -----------------------------------------------------
    if outcome:
        banner = labels.get(outcome, outcome)
        col = {"finished": (60, 210, 90), "eliminated": (235, 45, 45),
               "fell": (245, 175, 25), "timeout": (170, 170, 180)}.get(outcome, (200, 200, 200))
        bw = d.textlength(banner, font=fonts["big"])
        cx, cy = W / 2, H / 2
        d.rounded_rectangle([cx - bw / 2 - pad * 1.5, cy - int(34 * s),
                             cx + bw / 2 + pad * 1.5, cy + int(34 * s)],
                            radius=int(12 * s), fill=(15, 15, 20, 205))
        d.text((cx, cy), banner, font=fonts["big"], fill=col, anchor="mm")

    return np.asarray(img)


def resolve_paths(model_arg: str):
    p = Path(model_arg)
    if p.is_dir():
        return p / "final.zip", p / "vecnormalize.pkl"
    return p, p.parent / "vecnormalize.pkl"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="папка этапа (runs/game) или .zip")
    ap.add_argument("--out", default="demo.mp4")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--camera", default="track", choices=("track", "side"))
    ap.add_argument("--difficulty", type=float, default=1.0)
    ap.add_argument("--no-lights", action="store_true", help="снять забег без светофора")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--stochastic", action="store_true",
                    help="не усреднять действия — видно разброс поведения")
    args = ap.parse_args()

    model_path, vecnorm_path = resolve_paths(args.model)
    if not model_path.exists():
        raise SystemExit(f"нет модели: {model_path}")

    lights = not args.no_lights
    env = RedLightGreenLightEnv(lights=lights, difficulty=args.difficulty,
                                render_width=args.width, render_height=args.height,
                                camera=args.camera)
    model = PPO.load(str(model_path), device="cpu")

    normalizer = None
    if vecnorm_path.exists():
        dummy = DummyVecEnv([lambda: RedLightGreenLightEnv(lights=lights)])
        normalizer = VecNormalize.load(str(vecnorm_path), dummy)
        normalizer.training = False
        normalizer.norm_reward = False
    else:
        print(f"внимание: {vecnorm_path} не найден, качество будет хуже")

    fonts, labels = load_fonts(args.height)
    frames, stats = [], []

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset(seed=args.seed + ep)
        total, outcome = 0.0, None
        while outcome is None:
            o = normalizer.normalize_obs(obs) if normalizer else obs
            action, _ = model.predict(o, deterministic=not args.stochastic)
            obs, reward, term, trunc, info = env.step(action)
            total += reward
            frames.append(draw_hud(env.render(), env, fonts, labels, ep, None))
            if term or trunc:
                outcome = info.get("outcome", "timeout")
        # стоп-кадр с итогом
        final = draw_hud(env.render(), env, fonts, labels, ep, outcome)
        frames.extend([final] * int(args.fps * 1.2))
        stats.append((ep, outcome, float(env.data.qpos[0]), total))
        print(f"попытка {ep}: {outcome:11s} пройдено {env.data.qpos[0]:5.1f} м, "
              f"награда {total:8.1f}")

    env.close()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(out), frames, fps=args.fps, quality=7,
                     macro_block_size=None)

    reached = sum(1 for _, o, _, _ in stats if o == "finished")
    mean_x = np.mean([x for _, _, x, _ in stats])
    print(f"\nвидео: {out}  ({len(frames)} кадров, {len(frames)/args.fps:.1f} с)")
    print(f"дошли до финиша: {reached}/{len(stats)}, средняя дистанция {mean_x:.1f} м")


if __name__ == "__main__":
    main()
