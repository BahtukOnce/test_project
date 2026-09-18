"""Среда "Красный свет — зелёный свет" на MuJoCo.

Агент — четвероногий робот, который должен добежать до финиша,
но замирать на красный. Никакой готовой походки нет: политика
управляет напрямую моментами на 8 суставах и учится всему с нуля.

Устройство награды (подробности в README):
  зелёный : + скорость вдоль коридора
  красный : + за неподвижность, вылет из эпизода за движение выше порога
  всегда  : + бонус за "живость", - штраф за время и за усилия на моторах
  финиш   : + крупный бонус

Штраф за время принципиален: без него "простоять весь эпизод не двигаясь"
приносит больше награды, чем добежать до финиша, и агент залипает в этой
стратегии навсегда.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from arena import build_xml, FINISH_X

GREEN, RED = 1, 0

# Цвета лампы-светофора
_RGBA_GREEN = np.array([0.10, 0.90, 0.20, 1.0])
_RGBA_RED = np.array([0.95, 0.15, 0.15, 1.0])
_RGBA_AMBER = np.array([0.98, 0.70, 0.10, 1.0])   # кукла разворачивается


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


class RedLightGreenLightEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    # --- параметры, которые двигает curriculum ---------------------------
    # (значение при difficulty=0  ->  значение при difficulty=1)
    GREEN_MIN = (4.5, 1.8)      # длительность зелёного, с
    GREEN_MAX = (7.0, 3.5)
    RED_MIN = (1.0, 2.2)        # длительность красного, с
    RED_MAX = (1.8, 4.0)
    GRACE = (1.10, 0.35)        # сколько секунд можно тормозить после красного
    MOVE_THRESHOLD = (1.60, 0.30)  # м/с, выше — "ты пошевелился"

    # --- веса награды ----------------------------------------------------
    W_FORWARD = 1.0
    W_HEALTHY = 0.6
    W_CTRL = 0.1
    W_TIME = 0.3     # штраф за каждый шаг: иначе выгоднее простоять весь эпизод
    W_STILL = 0.3
    W_BRAKE = 2.0
    MAX_FORWARD_SPEED = 4.0   # не поощряем баллистические выбросы скорости
    R_FINISH = 300.0
    R_ELIMINATED = -100.0
    R_FELL = -20.0

    HEALTHY_Z = (0.26, 1.35)   # низ — лёг на брюхо, верх — улетел кувырком
    MIN_UPRIGHT = 0.0          # косинус наклона: ниже нуля — корпус вверх ногами
    FRAME_SKIP = 5              # 0.01 c * 5 = 20 Гц управления

    def __init__(self,
                 lights: bool = True,
                 difficulty: float = 1.0,
                 episode_seconds: float = 45.0,
                 finish_x: float = FINISH_X,
                 render_width: int = 640,
                 render_height: int = 480,
                 camera: str = "track",
                 seed: int | None = None):
        super().__init__()
        self.lights = lights
        self.difficulty = float(np.clip(difficulty, 0.0, 1.0))
        self.finish_x = finish_x

        self.model = mujoco.MjModel.from_xml_string(build_xml(finish_x=finish_x))
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep * self.FRAME_SKIP
        self.max_steps = int(episode_seconds / self.dt)

        self._light_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "light")
        self._torso_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self._init_qpos = self.model.key_qpos[0].copy() if self.model.nkey else None

        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.model.nu,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(32,), dtype=np.float64)

        self._renderer = None
        self._render_wh = (render_width, render_height)
        self._camera = camera
        if seed is not None:
            self.reset(seed=seed)

    # ------------------------------------------------------------------ #
    # Curriculum
    # ------------------------------------------------------------------ #
    def set_difficulty(self, difficulty: float) -> float:
        """Вызывается коллбэком во время обучения: 0 = легко, 1 = как в игре."""
        self.difficulty = float(np.clip(difficulty, 0.0, 1.0))
        return self.difficulty

    @property
    def move_threshold(self) -> float:
        return _lerp(*self.MOVE_THRESHOLD, self.difficulty)

    @property
    def grace_seconds(self) -> float:
        return _lerp(*self.GRACE, self.difficulty)

    def _sample_phase_length(self, light: int) -> float:
        if light == GREEN:
            lo = _lerp(*self.GREEN_MIN, self.difficulty)
            hi = _lerp(*self.GREEN_MAX, self.difficulty)
        else:
            lo = _lerp(*self.RED_MIN, self.difficulty)
            hi = _lerp(*self.RED_MAX, self.difficulty)
        return float(self.np_random.uniform(lo, hi))

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        qpos = self.data.qpos.copy()
        qpos[:3] = [0.0, 0.0, 0.75]
        qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        qpos[7:] += self.np_random.uniform(-0.1, 0.1, size=self.model.nq - 7)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = self.np_random.normal(scale=0.1, size=self.model.nv)
        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.t = 0.0
        self.light = GREEN
        self.phase_ends_at = self._sample_phase_length(GREEN) if self.lights else 1e9
        self.grace_until = 0.0
        self.prev_x = float(self.data.qpos[0])
        self.prev_speed = 0.0
        self.eliminated = False
        self.finished = False
        self._sync_visuals()
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data, nstep=self.FRAME_SKIP)

        self.step_count += 1
        self.t += self.dt

        was_green = self.light == GREEN
        in_grace_before = self.t < self.grace_until
        self._advance_light()

        x = float(self.data.qpos[0])
        z = float(self.data.qpos[2])
        speed_xy = float(np.linalg.norm(self.data.qvel[:2]))
        forward_speed = float(np.clip((x - self.prev_x) / self.dt,
                                      -self.MAX_FORWARD_SPEED, self.MAX_FORWARD_SPEED))
        joint_motion = float(np.linalg.norm(self.data.qvel[6:]))

        reward = (self.W_HEALTHY - self.W_TIME
                  - self.W_CTRL * float(np.sum(np.square(action))))
        terminated = False
        info = {}

        if was_green:
            reward += self.W_FORWARD * forward_speed
        elif in_grace_before:
            # Кукла ещё разворачивается — награждаем за торможение.
            reward += self.W_BRAKE * max(0.0, self.prev_speed - speed_xy)
        else:
            thr = self.move_threshold
            if speed_xy > thr:
                reward += self.R_ELIMINATED
                terminated = True
                self.eliminated = True
                info["outcome"] = "eliminated"
            else:
                stillness = 1.0 - speed_xy / thr
                reward += self.W_STILL * stillness * (1.0 / (1.0 + 0.2 * joint_motion))

        # Порядок важен: движение на красный отменяет всё (агент выбыл),
        # но пересечённый финиш побеждает падение — иначе прыжок через
        # черту засчитывается как падение вместо победы.
        if x >= self.finish_x and not terminated:
            reward += self.R_FINISH
            terminated = True
            self.finished = True
            info["outcome"] = "finished"

        upright = float(self.data.xmat[self._torso_bid].reshape(3, 3)[2, 2])
        fell = (not (self.HEALTHY_Z[0] < z < self.HEALTHY_Z[1])
                or upright < self.MIN_UPRIGHT)
        if fell and not terminated:
            reward += self.R_FELL
            terminated = True
            info["outcome"] = "fell"

        truncated = (not terminated) and self.step_count >= self.max_steps
        if truncated:
            info["outcome"] = "timeout"

        self.prev_x = x
        self.prev_speed = speed_xy
        self._sync_visuals()

        info.update(light="green" if self.light == GREEN else "red",
                    x=x, speed=speed_xy, difficulty=self.difficulty,
                    progress=x / self.finish_x)
        return self._obs(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------ #
    # Внутреннее
    # ------------------------------------------------------------------ #
    def _advance_light(self):
        if not self.lights:
            return
        if self.t >= self.phase_ends_at:
            self.light = RED if self.light == GREEN else GREEN
            self.phase_ends_at = self.t + self._sample_phase_length(self.light)
            if self.light == RED:
                self.grace_until = self.t + self.grace_seconds

    def _obs(self) -> np.ndarray:
        qpos, qvel = self.data.qpos, self.data.qvel
        time_to_switch = np.clip(self.phase_ends_at - self.t, 0.0, 8.0) / 8.0
        grace_left = np.clip(self.grace_until - self.t, 0.0, 2.0) / 2.0
        return np.concatenate([
            [qpos[2]],                                    # высота корпуса
            qpos[3:7],                                    # ориентация (кватернион)
            qpos[7:],                                     # углы 8 суставов
            np.clip(qvel[:3], -10, 10),                   # линейная скорость
            np.clip(qvel[3:6], -10, 10),                  # угловая скорость
            np.clip(qvel[6:], -20, 20),                   # скорости суставов
            [float(self.light == GREEN)],                 # какой сейчас свет
            [time_to_switch],                             # скоро ли переключится
            [grace_left],                                 # сколько осталось тормозить
            [self.move_threshold / 2.0],                  # насколько строгий судья
            [(self.finish_x - qpos[0]) / self.finish_x],  # сколько до финиша
        ]).astype(np.float64)

    def _sync_visuals(self):
        """Цвет лампы и разворот куклы — только для картинки."""
        if self.light == GREEN:
            rgba = _RGBA_GREEN
        elif self.t < self.grace_until:
            rgba = _RGBA_AMBER
        else:
            rgba = _RGBA_RED
        self.model.geom_rgba[self._light_gid] = rgba
        # На зелёный кукла отвернулась (поворот на 180 вокруг Z), на красный — смотрит.
        if self.light == GREEN:
            self.data.mocap_quat[0] = [0.0, 0.0, 0.0, 1.0]
        else:
            self.data.mocap_quat[0] = [1.0, 0.0, 0.0, 0.0]

    # ------------------------------------------------------------------ #
    # Рендер
    # ------------------------------------------------------------------ #
    def render(self):
        if self._renderer is None:
            w, h = self._render_wh
            self._renderer = mujoco.Renderer(self.model, height=h, width=w)
        mujoco.mj_forward(self.model, self.data)
        self._renderer.update_scene(self.data, camera=self._camera)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
