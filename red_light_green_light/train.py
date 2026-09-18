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
from stable_baselines3.common.callbacks import (BaseCallback, CheckpointCallback,
                                                EvalCallback)
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


class SaveNormalizerOnBest(BaseCallback):
    """Кладёт статистику нормализации рядом с лучшей моделью.

    EvalCallback сохраняет только веса, а без совпадающей статистики
    нормализации наблюдений модель на инференсе работает заметно хуже.
    """

    def __init__(self, eval_env, save_path: Path, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.save_path = save_path

    def _on_step(self) -> bool:
        if isinstance(self.eval_env, VecNormalize):
            self.eval_env.save(str(self.save_path / "vecnormalize_best.pkl"))
        return True


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
def make_lr(initial: float, schedule: str):
    """Постоянный или линейно затухающий learning rate.

    Затухание заметно снижает шанс развалить уже найденную политику
    на поздних шагах обучения.
    """
    if schedule == "constant":
        return initial
    return lambda progress_remaining: progress_remaining * initial


def resolve_stage_files(d: Path) -> tuple[Path, Path]:
    """Берём лучшую модель этапа, если она есть, иначе последнюю."""
    if (d / "best_model.zip").exists():
        vn = d / "vecnormalize_best.pkl"
        return d / "best_model.zip", vn if vn.exists() else d / "vecnormalize.pkl"
    return d / "final.zip", d / "vecnormalize.pkl"


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
    p.add_argument("--curriculum-frac", type=float, default=0.7,
                   help="за какую долю обучения доходим до полной сложности")
    p.add_argument("--checkpoint-every", type=int, default=250_000)
    p.add_argument("--eval-every", type=int, default=100_000,
                   help="как часто честно оценивать модель и обновлять рекорд")
    p.add_argument("--eval-episodes", type=int, default=12)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-schedule", choices=("linear", "constant"), default="linear",
                   help="linear — затухание к нулю, устойчивее на длинных прогонах")
    p.add_argument("--target-kl", type=float, default=0.02,
                   help="обрывать обновление, если политика уходит слишком далеко")
    p.add_argument("--ent-coef", type=float, default=None,
                   help="вес энтропии; по умолчанию 0 для walk и 0.004 для game")
    p.add_argument("--reset-std", type=float, default=0.35,
                   help="вернуть разброс действий при старте с готовых весов "
                        "(0 — не трогать). Без этого агент не нащупает "
                        "принципиально новое поведение")
    p.add_argument("--tensorboard", action="store_true",
                   help="писать логи ещё и в tensorboard (нужен пакет tensorboard)")
    args = p.parse_args()

    out = Path(args.out or f"runs/{args.stage}")
    out.mkdir(parents=True, exist_ok=True)

    lights = args.stage == "game"
    start_diff = args.difficulty_start if lights else 0.0
    # На втором этапе нужна энтропия: агент приходит со сложившейся походкой
    # и низким разбросом действий, а от него требуется поведение, которого
    # в его репертуаре нет вообще.
    ent_coef = args.ent_coef if args.ent_coef is not None else (0.004 if lights else 0.0)

    print(f"[{args.stage}] арен: {args.n_envs}, шагов: {args.steps:,}, "
          f"правила светофора: {'соблюдаем' if lights else 'игнорируем'}, "
          f"энтропия: {ent_coef}")

    venv = make_vec_env(args.n_envs, lights, start_diff, args.seed, args.episode_seconds)

    init_model_path, vecnorm_path = (resolve_stage_files(Path(args.init))
                                     if args.init else (None, None))
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
        if not init_model_path.exists():
            raise SystemExit(f"не нашёл модель предыдущего этапа: {init_model_path}")
        model = PPO.load(str(init_model_path), env=venv, device="cpu",
                         tensorboard_log=tb_dir)
        model.learning_rate = make_lr(args.lr, args.lr_schedule)
        model.target_kl = args.target_kl
        model.ent_coef = ent_coef
        model._setup_lr_schedule()
        print(f"  веса подхвачены из {init_model_path}")
        if args.reset_std > 0 and hasattr(model.policy, "log_std"):
            import torch
            was = float(torch.exp(model.policy.log_std.data).mean())
            with torch.no_grad():
                model.policy.log_std.data.fill_(float(np.log(args.reset_std)))
            print(f"  разброс действий восстановлен: {was:.3f} -> {args.reset_std:.3f}")
    else:
        model = PPO("MlpPolicy", venv, verbose=1, seed=args.seed, device="cpu",
                    n_steps=n_steps, batch_size=256, n_epochs=10,
                    learning_rate=make_lr(args.lr, args.lr_schedule),
                    target_kl=args.target_kl, gamma=0.99, gae_lambda=0.95,
                    clip_range=0.2, ent_coef=ent_coef, vf_coef=0.5, max_grad_norm=0.5,
                    policy_kwargs=policy_kwargs, tensorboard_log=tb_dir)

    model.set_logger(configure(str(out / "logs"), log_formats))

    # Оценочная среда всегда на финальной сложности: иначе рекорд,
    # поставленный на лёгких правилах, не побить никогда.
    eval_env = make_vec_env(1, lights, args.difficulty_end, args.seed + 9000,
                            args.episode_seconds)
    eval_env = VecNormalize(eval_env, training=False, norm_reward=False, clip_obs=10.0)

    callbacks = [OutcomeLogger(),
                 EvalCallback(eval_env,
                              best_model_save_path=str(out),
                              log_path=str(out / "eval"),
                              eval_freq=max(1, args.eval_every // args.n_envs),
                              n_eval_episodes=args.eval_episodes,
                              deterministic=True, render=False,
                              callback_on_new_best=SaveNormalizerOnBest(eval_env, out),
                              verbose=1)]
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
    eval_env.close()
    best = out / "best_model.zip"
    print(f"[{args.stage}] готово -> "
          f"{best if best.exists() else out / 'final.zip'}")


if __name__ == "__main__":
    main()
