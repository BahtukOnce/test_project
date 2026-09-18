#!/usr/bin/env bash
# Установка на чистый Linux-сервер (Ubuntu/Debian). Видеокарта не нужна.
set -euo pipefail
cd "$(dirname "$0")"

SUDO=""
[ "$(id -u)" -ne 0 ] && SUDO="sudo"

echo ">>> Системные библиотеки для рендера без монитора (OSMesa)"
$SUDO apt-get update -qq
$SUDO apt-get install -y -qq python3-venv libosmesa6 libosmesa6-dev libgl1-mesa-dri

echo ">>> Виртуальное окружение"
python3 -m venv .venv
./.venv/bin/pip install -q --upgrade pip

echo ">>> PyTorch (CPU-сборка, без CUDA)"
./.venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu

echo ">>> Остальные зависимости"
./.venv/bin/pip install -q -r requirements.txt

echo ">>> Проверка"
MUJOCO_GL=osmesa ./.venv/bin/python selfcheck.py

cat <<'MSG'

Готово. Дальше:

  ./.venv/bin/python train.py --stage walk --steps 4000000 --out runs/walk
  ./.venv/bin/python train.py --stage game --steps 4000000 --out runs/game --init runs/walk
  ./.venv/bin/python record.py --model runs/game --out demo.mp4

MSG
