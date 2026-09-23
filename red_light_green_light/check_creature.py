#!/usr/bin/env python3
"""Проверка существа до обучения: ошибки, картинка и проба на устойчивость.

    python check_creature.py creatures/пухлик.yaml

Делает три вещи:
  1. читает файл и понятно объясняет, что в нём не так;
  2. рисует существо и сохраняет картинку рядом с файлом;
  3. ставит его на пол и смотрит, устоит ли оно само по себе.

Обучать существо, которое разваливается за две секунды без единой команды,
почти бесполезно — лучше сразу поправить.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent))
import creature as creature_mod
from arena import build_xml


def preview_camera(beast, model) -> mujoco.MjvCamera:
    reach = beast.size + max((2 * l.thigh + l.shin for l in beast.legs), default=0.6)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, beast.height * 0.6]
    cam.distance = max(2.2, 3.4 * reach)
    cam.azimuth = 125.0
    cam.elevation = -18.0
    return cam


def settle(model, data, seconds: float) -> None:
    """Даём существу опуститься на пол без единой команды моторам."""
    data.ctrl[:] = 0.0
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", help="файл существа, например creatures/пухлик.yaml")
    ap.add_argument("--out", default=None, help="куда сохранить картинку")
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--seconds", type=float, default=2.0,
                    help="сколько секунд проверять устойчивость")
    args = ap.parse_args()

    try:
        beast = creature_mod.load(args.file)
    except creature_mod.CreatureError as exc:
        print(f"\nВ описании существа ошибка:\n\n{exc}\n")
        raise SystemExit(1)

    print(f"\n  {beast.name}" + (f"  ({beast.author})" if beast.author else ""))
    print(f"  {'-' * 46}")
    print(f"  туловище      {beast.shape}, размер {beast.size:g}")
    print(f"  ног           {len(beast.legs)}")
    if beast.tail:
        print(f"  хвост         {beast.tail.segments} сегментов")
    print(f"  суставов      {beast.n_joints}   (столько моторов у нейросети)")
    print(f"  рост          {beast.height:.2f} м")
    print(f"  сила моторов  {beast.motor_power:g}")

    model = mujoco.MjModel.from_xml_string(build_xml(beast))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[2] = beast.height
    mujoco.mj_forward(model, data)

    # --- проба на устойчивость -------------------------------------------
    settle(model, data, args.seconds)
    z = float(data.qpos[2])
    upright = float(data.xmat[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "torso")].reshape(3, 3)[2, 2])
    drift = float(np.linalg.norm(data.qpos[:2]))
    low, high = 0.35 * beast.height, 1.8 * beast.height

    print(f"\n  Проба: поставили на пол и {args.seconds:g} секунды не трогали")
    print(f"  {'-' * 46}")
    print(f"  высота корпуса стала {z:.2f} м (допустимо от {low:.2f} до {high:.2f})")
    print(f"  наклон {upright:+.2f} (1.00 — стоит ровно, меньше 0 — перевернулось)")
    print(f"  сползло на {drift:.2f} м")

    if upright < 0.0:
        verdict = ("ПЕРЕВЕРНУЛОСЬ. Обучать почти бесполезно.\n"
                   "  Попробуй: короче ноги, толще туловище, ноги пошире по углам.")
    elif z < low:
        verdict = ("ЛЕЖИТ НА ПОЛУ. Ноги не держат вес.\n"
                   "  Попробуй: увеличить «подъём», укоротить голень "
                   "или уменьшить размер туловища.")
    elif z > high:
        verdict = "ВИСИТ СЛИШКОМ ВЫСОКО — проверь «высота_старта»."
    else:
        verdict = "СТОИТ. Можно обучать."
    print(f"\n  {verdict}\n")

    # --- картинка ---------------------------------------------------------
    out = Path(args.out) if args.out else Path(args.file).with_suffix(".png")
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    renderer.update_scene(data, camera=preview_camera(beast, model))
    frame = renderer.render()
    renderer.close()
    import imageio.v2 as imageio
    imageio.imwrite(str(out), frame)
    print(f"  картинка: {out}\n")


if __name__ == "__main__":
    main()
