#!/usr/bin/env python3
"""Быстрая проверка, что на этой машине всё работает.

Проверяет по порядку: импорты, сборку сцены, API среды, рендер без монитора,
кодирование видео и один короткий цикл обучения. Занимает меньше минуты.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
sys.path.insert(0, str(Path(__file__).resolve().parent))

OK, FAIL = "  [ ok ]", "  [ПРОВАЛ]"
failures = []


def check(name: str):
    def wrap(fn):
        print(f"{name} ...", flush=True)
        t0 = time.time()
        try:
            detail = fn()
            print(f"{OK} {detail or ''}  ({time.time() - t0:.1f} c)")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"{FAIL} {type(exc).__name__}: {exc}")
        return fn
    return wrap


@check("1/6 Импорт зависимостей")
def _imports():
    import mujoco, gymnasium, stable_baselines3, torch, imageio, PIL
    return (f"mujoco {mujoco.__version__}, gymnasium {gymnasium.__version__}, "
            f"sb3 {stable_baselines3.__version__}, torch {torch.__version__}")


@check("2/6 Сборка сцены MuJoCo")
def _model():
    import mujoco
    from arena import build_xml
    m = mujoco.MjModel.from_xml_string(build_xml())
    return f"{m.nq} координат, {m.nu} моторов"


@check("3/6 API среды (gymnasium)")
def _env_api():
    from gymnasium.utils.env_checker import check_env
    from rlgl_env import RedLightGreenLightEnv
    import warnings
    env = RedLightGreenLightEnv()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check_env(env, skip_render_check=True)
    env.close()
    return f"наблюдение {env.observation_space.shape}, действие {env.action_space.shape}"


@check("4/6 Рендер без монитора")
def _render():
    from rlgl_env import RedLightGreenLightEnv
    env = RedLightGreenLightEnv(render_width=320, render_height=240)
    env.reset(seed=0)
    frame = env.render()
    env.close()
    if frame.std() < 1.0:
        raise RuntimeError("кадр пустой — проверь MUJOCO_GL и libOSMesa")
    return f"кадр {frame.shape}, MUJOCO_GL={os.environ.get('MUJOCO_GL')}"


@check("5/6 Кодирование видео")
def _video():
    import numpy as np
    import imageio.v2 as imageio
    from rlgl_env import RedLightGreenLightEnv
    env = RedLightGreenLightEnv(render_width=320, render_height=240)
    env.reset(seed=0)
    frames = []
    for _ in range(10):
        env.step(env.action_space.sample())
        frames.append(env.render())
    env.close()
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as fh:
        path = fh.name
    imageio.mimwrite(path, frames, fps=10, macro_block_size=None)
    size = Path(path).stat().st_size
    Path(path).unlink()
    return f"{size // 1024} КБ на 10 кадров"


@check("6/6 Короткий цикл обучения")
def _train():
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from rlgl_env import RedLightGreenLightEnv
    venv = VecNormalize(DummyVecEnv([lambda: RedLightGreenLightEnv()]))
    model = PPO("MlpPolicy", venv, n_steps=64, batch_size=32, verbose=0, device="cpu")
    t0 = time.time()
    model.learn(total_timesteps=256)
    fps = 256 / max(time.time() - t0, 1e-6)
    venv.close()
    return f"~{fps:.0f} шагов/с на одной арене (в обучении умножь на число арен)"


print()
if failures:
    print(f"Провалено проверок: {len(failures)}")
    for name, exc in failures:
        print(f"  - {name}: {exc}")
    sys.exit(1)
print("Всё работает. Можно запускать train.py")
