#!/usr/bin/env python3
"""Оценка модели: прогоняет эпизоды и считает статистику исходов.

    python evaluate.py --model runs/game --episodes 50

Без графики и видео, поэтому работает быстро — удобно сравнивать
чекпоинты между собой и проверять, не развалилась ли политика.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rlgl_env import RedLightGreenLightEnv

LABELS = {"finished": "дошёл до финиша", "eliminated": "вылетел на красный",
          "fell": "упал", "timeout": "не успел"}


def resolve(model_arg: str, vecnorm_arg: str | None):
    p = Path(model_arg)
    if not p.is_dir():
        vn = Path(vecnorm_arg) if vecnorm_arg else p.parent / "vecnormalize.pkl"
        return p, vn
    if (p / "best_model.zip").exists():
        vn = p / "vecnormalize_best.pkl"
        return p / "best_model.zip", vn if vn.exists() else p / "vecnormalize.pkl"
    return p / "final.zip", p / "vecnormalize.pkl"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="папка этапа или конкретный .zip")
    ap.add_argument("--vecnormalize", default=None,
                   help="статистика нормализации, если модель задана файлом")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--no-lights", action="store_true")
    ap.add_argument("--difficulty", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--quiet", action="store_true", help="только итоговая строка")
    args = ap.parse_args()

    model_path, vecnorm_path = resolve(args.model, args.vecnormalize)
    if not model_path.exists():
        raise SystemExit(f"нет модели: {model_path}")

    lights = not args.no_lights
    env = RedLightGreenLightEnv(lights=lights, difficulty=args.difficulty)
    model = PPO.load(str(model_path), device="cpu")

    normalizer = None
    if vecnorm_path.exists():
        dummy = DummyVecEnv([lambda: RedLightGreenLightEnv(lights=lights)])
        normalizer = VecNormalize.load(str(vecnorm_path), dummy)
        normalizer.training = False

    outcomes, rewards, distances = Counter(), [], []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        total = 0.0
        while True:
            o = normalizer.normalize_obs(obs) if normalizer else obs
            action, _ = model.predict(o, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            total += r
            if term or trunc:
                outcomes[info.get("outcome", "timeout")] += 1
                rewards.append(total)
                distances.append(float(env.data.qpos[0]))
                break
    env.close()

    n = args.episodes
    finished = outcomes["finished"] / n
    if args.quiet:
        print(f"{model_path}: финишей {finished:.0%}, награда {np.mean(rewards):7.1f}")
        return

    print(f"\nмодель: {model_path}")
    print(f"эпизодов: {n}, светофор: {'да' if lights else 'нет'}, "
          f"сложность: {args.difficulty}")
    print("\nисходы:")
    for key in ("finished", "eliminated", "fell", "timeout"):
        cnt = outcomes[key]
        bar = "#" * int(30 * cnt / n)
        print(f"  {LABELS[key]:20s} {cnt / n:5.0%} {bar}")
    print(f"\nнаграда:   среднее {np.mean(rewards):7.1f}   "
          f"медиана {np.median(rewards):7.1f}")
    print(f"дистанция: среднее {np.mean(distances):7.1f} м   "
          f"медиана {np.median(distances):7.1f} м")


if __name__ == "__main__":
    main()
