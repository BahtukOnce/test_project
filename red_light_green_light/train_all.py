#!/usr/bin/env python3
"""Обучение всех существ класса подряд.

    python train_all.py creatures/*.yaml
    python train_all.py creatures/*.yaml --steps 1500000 --jobs 2 --resume

Для каждого существа: обучение, оценка, видео, график. В конце — общая
таблица класса. Если одно существо сломалось, остальные всё равно
досчитаются: ошибка попадает в лог и в итоговую сводку.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import creature as creature_mod

STEPS_PER_SEC_PER_ARENA = 310   # замер: 4 арены на 4 ядрах дают ~1250 шагов/с

PYTHON = str(HERE.parent / ".venv" / "bin" / "python")
if not Path(PYTHON).exists():
    PYTHON = sys.executable


def run_step(cmd: list[str], log, name: str) -> bool:
    log.write(f"\n$ {' '.join(cmd)}\n")
    log.flush()
    result = subprocess.run(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        log.write(f"\n!! шаг «{name}» завершился с кодом {result.returncode}\n")
        log.flush()
    return result.returncode == 0


def process(path: Path, args, n_envs: int) -> dict:
    """Полный цикл для одного существа. Возвращает краткий итог."""
    started = time.time()
    try:
        beast = creature_mod.load(path)
    except creature_mod.CreatureError as exc:
        return {"файл": path.name, "статус": "ошибка в описании",
                "сообщение": str(exc).splitlines()[0], "минут": 0.0}

    out = Path(args.runs) / beast.slug
    out.mkdir(parents=True, exist_ok=True)
    result_json = out / "итог.json"

    if args.resume and result_json.exists():
        return {"файл": path.name, "имя": beast.name, "статус": "уже готово",
                "минут": 0.0}

    with (out / "лог.txt").open("w", encoding="utf-8") as log:
        log.write(f"существо: {beast.name} ({beast.author})\n"
                  f"файл: {path}\nсуставов: {beast.n_joints}\n")

        common = ["--creature", str(path), "--n-envs", str(n_envs)]
        ok = run_step([PYTHON, "train.py", "--stage", "walk",
                       "--steps", str(args.steps), "--out", str(out / "walk"),
                       "--checkpoint-every", "0", *common], log, "обучение бегу")
        model_dir = out / "walk"

        if ok and args.stage == "both":
            ok = run_step([PYTHON, "train.py", "--stage", "game",
                           "--steps", str(args.steps), "--init", str(out / "walk"),
                           "--out", str(out / "game"), "--checkpoint-every", "0",
                           *common], log, "обучение светофору")
            model_dir = out / "game"

        if not ok:
            return {"файл": path.name, "имя": beast.name, "статус": "обучение упало",
                    "минут": (time.time() - started) / 60}

        lights = ["--no-lights"] if args.stage == "walk" else []
        run_step([PYTHON, "evaluate.py", "--model", str(model_dir),
                  "--creature", str(path), "--episodes", str(args.episodes),
                  "--json", str(result_json), *lights], log, "оценка")
        run_step([PYTHON, "record.py", "--model", str(model_dir),
                  "--creature", str(path), "--out", str(out / "видео.mp4"),
                  "--episodes", "3", "--width", "720", "--height", "404",
                  *lights], log, "видео")
        стадии = [str(out / "walk")] + ([str(out / "game")] if args.stage == "both" else [])
        run_step([PYTHON, "plot_progress.py", *стадии,
                  "--out", str(out / "график.png"),
                  "--title", f"Как учился «{beast.name}»"], log, "график")

    return {"файл": path.name, "имя": beast.name,
            "статус": "готово" if result_json.exists() else "нет результата",
            "минут": (time.time() - started) / 60}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="файлы существ, например creatures/*.yaml")
    ap.add_argument("--steps", type=int, default=1_500_000,
                    help="шагов обучения на существо (1.5 млн ~ 20-25 мин на 4 ядрах)")
    ap.add_argument("--stage", choices=("walk", "both"), default="walk",
                    help="walk — только добежать до финиша; both — ещё и светофор")
    ap.add_argument("--jobs", type=int, default=1,
                    help="сколько существ учить одновременно")
    ap.add_argument("--episodes", type=int, default=30, help="эпизодов на оценку")
    ap.add_argument("--runs", default="runs", help="куда складывать результаты")
    ap.add_argument("--resume", action="store_true",
                    help="пропускать существ, у которых уже есть итог")
    args = ap.parse_args()

    paths = [Path(f) for f in args.files if not Path(f).name.startswith("ШАБЛОН")]
    paths = [p for p in paths if p.suffix in (".yaml", ".yml")]
    if not paths:
        raise SystemExit("не нашёл ни одного файла существа")

    cpu = os.cpu_count() or 4
    n_envs = max(1, cpu // max(1, args.jobs))
    # Замер на этой машине: около 310 шагов в секунду на одну арену.
    per_creature_min = args.steps / (STEPS_PER_SEC_PER_ARENA * n_envs) / 60
    if args.stage == "both":
        per_creature_min *= 2
    total_min = per_creature_min * len(paths) / max(1, args.jobs)

    print(f"\nсуществ: {len(paths)}, шагов на каждое: {args.steps:,}")
    print(f"ядер: {cpu}, одновременно: {args.jobs}, арен на существо: {n_envs}")
    print(f"примерно {per_creature_min:.0f} мин на существо, "
          f"всего ~{total_min / 60:.1f} ч\n")

    results = []
    if args.jobs > 1:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(process, p, args, n_envs) for p in paths]
            for i, fut in enumerate(futures, 1):
                r = fut.result()
                results.append(r)
                print(f"[{i}/{len(paths)}] {r.get('имя', r['файл']):<18} "
                      f"{r['статус']:<18} {r['минут']:5.1f} мин")
    else:
        for i, p in enumerate(paths, 1):
            print(f"[{i}/{len(paths)}] {p.name} ...", flush=True)
            r = process(p, args, n_envs)
            results.append(r)
            print(f"[{i}/{len(paths)}] {r.get('имя', r['файл']):<18} "
                  f"{r['статус']:<18} {r['минут']:5.1f} мин", flush=True)

    print("\nитог:")
    for r in results:
        line = f"  {r.get('имя', r['файл']):<18} {r['статус']}"
        if "сообщение" in r:
            line += f" — {r['сообщение']}"
        print(line)

    try:
        subprocess.run([PYTHON, "leaderboard.py", "--runs", args.runs], cwd=HERE)
    except Exception as exc:                                  # noqa: BLE001
        print(f"таблицу построить не удалось: {exc}")


if __name__ == "__main__":
    main()
