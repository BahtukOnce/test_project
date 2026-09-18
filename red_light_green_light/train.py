#!/usr/bin/env python3
"""Обучение агента PPO в два этапа.

  Этап 1 (walk) — светофора нет, агент учится просто бежать вперёд.
  Этап 2 (game) — включаем светофор и постепенно ужесточаем правила.

Пример:
    python train.py --stage walk --steps 3_000_000 --out runs/walk
    python train.py --stage game --steps 3_000_000 --out runs/game --init runs/walk
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,
                                              VecNormalize)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rlgl_env import RedLightGreenLightEnv


# --------------------------------------------------------------------------- #
# Коллбэки
# --------------------------------------------------------------------------- #
class DifficultyCurriculum(BaseCallback):
    """Плавно поднимает сложность: зелёный короче, красный строже.

    Учить ходьбе и правилам одновременно почти безнадёжно — агент вылетает
    раньше, чем успевает сделать шаг. Поэтому сначала судья добрый.
    """

    def __init__(self, start: float, end: float, ramp_fraction: float,
                 total_steps: int, verbose: int = 0):
        super().__init__(verbose)
        self.start, self.end = start, end
        self.ramp = max(ramp_fraction, 1e-6)
        self.total_steps = total_steps
        self.current = start

    def _on_rollout_start(self) -> None:
        progress = self.num_timesteps / max(self.total_steps, 1)
        t = min(1.0, progress / self.ramp)
        self.current = self.start + (self.end - self.start) * t
        self.training_env.env_method("set_difficulty", self.current)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self.logger.record("curriculum/difficulty", self.current)


class OutcomeLogger(BaseCallback):
    """Считает, чем кончаются эпизоды: финиш / вылет / падение / таймаут."""

    KINDS = ("finished", "eliminated", "fell", "timeout")

    def __init__(self, window: int = 200, verbose: int = 0):
        super().__init__(verbose)
        self.window = window
        self.recent: list[str] = []
        self.best_progress = 0.0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            outcome = info.get("outcome")
            if outcome:
                self.recent.append(outcome)
                self.best_progress = max(self.best_progress, info.get("progress", 0.0))
        if len(self.recent) > self.window:
            self.recent = self.recent[-self.window:]
        return True

    def _on_rollout_end(self) -> None:
        if not self.recent:
            return
        n = len(self.recent)
        for kind in self.KINDS:
            self.logger.record(f"outcome/{kind}", self.recent.count(kind) / n)
        self.logger.record("outcome/best_progress", self.best_progress)


# --------------------------------------------------------------------------- #
# Сборка окружения
# --------------------------------------------------------------------------- #
def make_vec_env(n_envs: int, lights: bool, difficulty: float, seed: int,
                 episode_seconds: float):
    def factory(rank: int):
        def _init():
            env = RedLightGreenLightEnv(lights=lights, difficulty=difficulty,
                                        episode_seconds=episode_seconds)
            env.reset(seed=seed + rank)
            return Monitor(env)
        return _init

    envs = [factory(i) for i in range(n_envs)]
    return SubprocVecEnv(envs) if n_envs > 1 else DummyVecEnv(envs)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=("walk", "game"), default="game",
                   help="walk — учим ходить, game — учим играть со светофором")
    p.add_argument("--steps", type=int, default=3_000_000)
    p.add_argument("--n-envs", type=int, default=max(1, (os.cpu_count() or 4)),
                   help="сколько арен крутить параллельно")
    p.add_argument("--out", default=None, help="куда класть модель (по умолчанию runs/<stage>)")
    p.add_argument("--init", default=None,
                   help="папка предыдущего этапа: берём оттуда веса и статистику нормализации")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episode-seconds", type=float, default=45.0)
    p.add_argument("--difficulty-start", type=float, default=0.0)
    p.add_argument("--difficulty-end", type=float, default=1.0)
    p.add_argument("--curriculum-frac", type=float, default=0.6,
                   help="за какую долю обучения доходим до полной сложности")
    p.add_argument("--checkpoint-every", type=int, default=250_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--tensorboard", action="store_true",
                   help="писать логи ещё и в tensorboard (нужен пакет tensorboard)")
    args = p.parse_args()

    out = Path(args.out or f"runs/{args.stage}")
    out.mkdir(parents=True, exist_ok=True)

    lights = args.stage == "game"
    start_diff = args.difficulty_start if lights else 0.0

    print(f"[{args.stage}] арен: {args.n_envs}, шагов: {args.steps:,}, "
          f"светофор: {'да' if lights else 'нет'}")

    venv = make_vec_env(args.n_envs, lights, start_diff, args.seed, args.episode_seconds)

    vecnorm_path = Path(args.init) / "vecnormalize.pkl" if args.init else None
    if vecnorm_path and vecnorm_path.exists():
        venv = VecNormalize.load(str(vecnorm_path), venv)
        venv.training = True
        venv.norm_reward = True
        print(f"  статистика нормализации подхвачена из {vecnorm_path}")
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    log_formats = ["stdout", "csv"]
    tb_dir = None
    if args.tensorboard:
        try:
            import tensorboard  # noqa: F401
            tb_dir = str(out / "tb")
            log_formats.append("tensorboard")
        except ImportError:
            print("  tensorboard не установлен — пишу только CSV")

    n_steps = max(256, 8192 // args.n_envs)
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))

    if args.init:
        model_path = Path(args.init) / "final.zip"
        if not model_path.exists():
            raise SystemExit(f"не нашёл модель предыдущего этапа: {model_path}")
        model = PPO.load(str(model_path), env=venv, device="cpu",
                         tensorboard_log=tb_dir)
        model.learning_rate = args.lr
        model._setup_lr_schedule()
        print(f"  веса подхвачены из {model_path}")
    else:
        model = PPO("MlpPolicy", venv, verbose=1, seed=args.seed, device="cpu",
                    n_steps=n_steps, batch_size=256, n_epochs=10,
                    learning_rate=args.lr, gamma=0.99, gae_lambda=0.95,
                    clip_range=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
                    policy_kwargs=policy_kwargs, tensorboard_log=tb_dir)

    model.set_logger(configure(str(out / "logs"), log_formats))

    callbacks = [OutcomeLogger()]
    if lights:
        callbacks.append(DifficultyCurriculum(args.difficulty_start, args.difficulty_end,
                                              args.curriculum_frac, args.steps))
    if args.checkpoint_every > 0:
        callbacks.append(CheckpointCallback(
            save_freq=max(1, args.checkpoint_every // args.n_envs),
            save_path=str(out / "checkpoints"), name_prefix="ppo"))

    try:
        model.learn(total_timesteps=args.steps, callback=callbacks,
                    progress_bar=False, reset_num_timesteps=True)
    except KeyboardInterrupt:
        print("\nпрервано вручную — сохраняю что есть")

    model.save(str(out / "final.zip"))
    venv.save(str(out / "vecnormalize.pkl"))
    venv.close()
    print(f"[{args.stage}] готово -> {out/'final.zip'}")


if __name__ == "__main__":
    main()
