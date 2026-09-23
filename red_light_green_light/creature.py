"""Существо, собранное учеником.

Ученик правит файл в папке creatures/ на понятных ключах, а этот модуль
превращает его в тело для MuJoCo. Здесь же проверка с человеческими
сообщениями об ошибках и мутации тела — их использует эволюция.

Система координат: существо бежит в сторону +X.
Угол ноги: 0 — вперёд, 90 — влево, 180 — назад, 270 — вправо.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml


class CreatureError(Exception):
    """Ошибка в описании существа — текст показывается ученику как есть."""


# --------------------------------------------------------------------------- #
# Словарь понятных названий
# --------------------------------------------------------------------------- #
DIRECTIONS = {
    "спереди": 0.0, "вперёд": 0.0, "вперед": 0.0,
    "спереди-слева": 45.0, "слева-спереди": 45.0,
    "слева": 90.0,
    "сзади-слева": 135.0, "слева-сзади": 135.0,
    "сзади": 180.0, "назад": 180.0,
    "сзади-справа": 225.0, "справа-сзади": 225.0,
    "справа": 270.0,
    "спереди-справа": 315.0, "справа-спереди": 315.0,
}

COLORS = {
    "оранжевый": (0.82, 0.55, 0.28), "красный": (0.90, 0.35, 0.30),
    "синий": (0.30, 0.50, 0.85), "зелёный": (0.35, 0.72, 0.42),
    "зеленый": (0.35, 0.72, 0.42), "жёлтый": (0.93, 0.78, 0.25),
    "желтый": (0.93, 0.78, 0.25), "фиолетовый": (0.62, 0.44, 0.84),
    "розовый": (0.93, 0.55, 0.72), "бирюзовый": (0.25, 0.75, 0.72),
    "серый": (0.60, 0.62, 0.66), "белый": (0.90, 0.90, 0.92),
}

SHAPES = ("шар", "капсула")

# Границы разумного. Выход за них — ошибка с подсказкой, а не падение MuJoCo.
LIMITS = {
    "размер":         (0.10, 0.60),
    "длина":          (0.10, 1.20),
    "бедро":          (0.08, 0.80),
    "голень":         (0.08, 1.20),
    "толщина":        (0.02, 0.20),
    "подъём":         (10.0, 80.0),
    "сила_моторов":   (20.0, 400.0),
    "длина_сегмента": (0.08, 0.60),
}
MAX_LEGS = 12
MAX_TAIL = 6


def _check_number(value, key: str, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CreatureError(
            f"{where}: «{key}» должно быть числом, а сейчас там {value!r}.")
    value = float(value)
    lo, hi = LIMITS[key]
    if not (lo <= value <= hi):
        raise CreatureError(
            f"{where}: «{key}» = {value:g}, а должно быть от {lo:g} до {hi:g}.\n"
            f"  Слишком маленькие части проваливаются сквозь пол, "
            f"слишком большие — не сдвинуть моторами.")
    return value


# --------------------------------------------------------------------------- #
# Части тела
# --------------------------------------------------------------------------- #
@dataclass
class Leg:
    angle: float        # градусы, 0 = вперёд
    thigh: float        # бедро
    shin: float         # голень
    thickness: float
    lift: float         # «подъём»: на сколько градусов голень смотрит вниз
    # Походка для эволюции: по три числа на каждый из двух суставов —
    # размах, сдвиг по времени и среднее положение. В файл ученика не
    # пишется; обучение с подкреплением её не использует.
    gait: tuple = ()

    @property
    def foot_drop(self) -> float:
        """На сколько метров стопа опускается ниже бедра."""
        return self.shin * math.sin(math.radians(self.lift))


@dataclass
class Tail:
    segments: int
    segment_length: float
    thickness: float
    gait: tuple = ()


@dataclass
class Creature:
    name: str
    author: str = ""
    color: str = "оранжевый"
    shape: str = "шар"
    size: float = 0.25
    length: float = 0.5           # только для капсулы
    legs: list[Leg] = field(default_factory=list)
    tail: Tail | None = None
    motor_power: float = 150.0
    start_height: float | None = None   # None — посчитать по геометрии
    gait_freq: float = 1.6              # взмахов в секунду, для эволюции

    # ------------------------------------------------------------------ #
    @property
    def n_joints(self) -> int:
        return 2 * len(self.legs) + (self.tail.segments if self.tail else 0)

    @property
    def height(self) -> float:
        """Высота, на которой существо начинает эпизод."""
        if self.start_height is not None:
            return self.start_height
        drop = max((leg.foot_drop + leg.thickness for leg in self.legs),
                   default=0.0)
        body_bottom = self.size if self.shape == "шар" else self.size
        return round(max(drop, body_bottom) + 0.04, 3)

    def gait_vector(self, rng=None) -> list[float]:
        """Параметры походки в том же порядке, что и моторы."""
        out: list[float] = []
        for leg in self.legs:
            g = leg.gait if len(leg.gait) == 6 else random_gait(rng, 2)
            out += list(g)
        if self.tail:
            n = self.tail.segments
            g = self.tail.gait if len(self.tail.gait) == 3 * n else random_gait(rng, n)
            out += list(g)
        return out

    def with_random_gait(self, rng) -> "Creature":
        """Копия с новой случайной походкой (тело не меняется)."""
        legs = [replace(l, gait=random_gait(rng, 2)) for l in self.legs]
        tail = (replace(self.tail, gait=random_gait(rng, self.tail.segments))
                if self.tail else None)
        return replace(self, legs=legs, tail=tail,
                       gait_freq=rng.uniform(0.8, 3.0))

    @property
    def slug(self) -> str:
        """Безопасное имя для папки: «Пухлик 2.0!» -> «Пухлик_2_0»."""
        s = re.sub(r"[^\w\-]+", "_", self.name, flags=re.UNICODE).strip("_")
        return s or "существо"

    @property
    def rgba(self) -> str:
        r, g, b = COLORS.get(self.color, COLORS["оранжевый"])
        return f"{r} {g} {b} 1"

    # ------------------------------------------------------------------ #
    # Сборка тела для MuJoCo
    # ------------------------------------------------------------------ #
    def _leg_xml(self, index: int, leg: Leg) -> str:
        rad = math.radians(leg.angle)
        dx, dy = math.cos(rad), math.sin(rad)
        # У вытянутого туловища передние ноги растут спереди, задние сзади.
        # У шарообразного все ноги выходят из центра — как у муравья.
        ox = dx * self.length / 2 if self.shape == "капсула" else 0.0
        # Ось сгиба голени перпендикулярна ноге и лежит горизонтально;
        # при положительном угле стопа уходит вниз.
        ax, ay = -math.sin(rad), math.cos(rad)
        lo, hi = leg.lift - 20.0, leg.lift + 20.0
        t, s = leg.thigh, leg.shin
        n = f"нога{index}"
        return f"""
      <body name="{n}" pos="{ox:.4f} 0 0">
        <geom name="{n}_основание" type="capsule" size="{leg.thickness:.4f}"
              fromto="0 0 0 {t * dx:.4f} {t * dy:.4f} 0"/>
        <body name="{n}_бедро" pos="{t * dx:.4f} {t * dy:.4f} 0">
          <joint name="{n}_сустав1" type="hinge" axis="0 0 1" pos="0 0 0" range="-30 30"/>
          <geom name="{n}_бедро_geom" type="capsule" size="{leg.thickness:.4f}"
                fromto="0 0 0 {t * dx:.4f} {t * dy:.4f} 0"/>
          <body name="{n}_голень" pos="{t * dx:.4f} {t * dy:.4f} 0">
            <joint name="{n}_сустав2" type="hinge" pos="0 0 0"
                   axis="{ax:.4f} {ay:.4f} 0" range="{lo:.1f} {hi:.1f}"/>
            <geom name="{n}_голень_geom" type="capsule" size="{leg.thickness:.4f}"
                  fromto="0 0 0 {s * dx:.4f} {s * dy:.4f} 0"/>
          </body>
        </body>
      </body>"""

    def _tail_xml(self) -> str:
        if not self.tail or self.tail.segments <= 0:
            return ""
        L, th = self.tail.segment_length, self.tail.thickness
        # Хвост растёт назад от туловища, каждый сегмент на своём суставе.
        xml, closing = "", ""
        start = -(self.size if self.shape == "шар" else self.length / 2 + self.size)
        for i in range(self.tail.segments):
            pos = f"{start:.4f} 0 0" if i == 0 else f"{-L:.4f} 0 0"
            axis = "0 0 1" if i % 2 == 0 else "0 1 0"   # виляет то вбок, то вверх-вниз
            xml += f"""
      <body name="хвост{i + 1}" pos="{pos}">
        <joint name="хвост{i + 1}_сустав" type="hinge" pos="0 0 0"
               axis="{axis}" range="-45 45"/>
        <geom name="хвост{i + 1}_geom" type="capsule" size="{th:.4f}"
              fromto="0 0 0 {-L:.4f} 0 0"/>"""
            closing = "</body>" + closing
        return xml + "\n      " + closing

    def _torso_geom(self) -> str:
        if self.shape == "капсула":
            half = self.length / 2
            return (f'<geom name="туловище" type="capsule" size="{self.size:.4f}"\n'
                    f'            fromto="{-half:.4f} 0 0 {half:.4f} 0 0" '
                    f'rgba="{self.rgba}"/>')
        return (f'<geom name="туловище" type="sphere" size="{self.size:.4f}" '
                f'rgba="{self.rgba}"/>')

    def body_xml(self) -> str:
        """Кусок XML с телом существа — вставляется в сцену."""
        legs = "".join(self._leg_xml(i + 1, leg) for i, leg in enumerate(self.legs))
        return f"""
    <body name="torso" pos="0 0 {self.height}">
      <camera name="track" mode="trackcom" pos="-0.5 -5.0 1.4"
              xyaxes="0.9950 -0.0995 0 0.0267 0.2671 0.9633"/>
      <camera name="side" mode="trackcom" pos="0 -8.0 3.0"
              xyaxes="1 0 0 0 0.3511 0.9363"/>
      {self._torso_geom()}
      <joint armature="0" damping="0" limited="false" margin="0.01" name="root"
             pos="0 0 0" type="free"/>{legs}{self._tail_xml()}
    </body>"""

    def actuators_xml(self) -> str:
        names = []
        for i in range(len(self.legs)):
            names += [f"нога{i + 1}_сустав1", f"нога{i + 1}_сустав2"]
        if self.tail:
            names += [f"хвост{i + 1}_сустав" for i in range(self.tail.segments)]
        return "\n".join(
            f'    <motor joint="{n}" gear="{self.motor_power:g}"/>' for n in names)

    # ------------------------------------------------------------------ #
    # Чтение и запись файла
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        d: dict = {"имя": self.name}
        if self.author:
            d["автор"] = self.author
        d["цвет"] = self.color
        torso: dict = {"форма": self.shape, "размер": round(self.size, 4)}
        if self.shape == "капсула":
            torso["длина"] = round(self.length, 4)
        d["туловище"] = torso
        d["ноги"] = [{"угол": round(l.angle, 1), "бедро": round(l.thigh, 4),
                      "голень": round(l.shin, 4), "толщина": round(l.thickness, 4),
                      "подъём": round(l.lift, 1)} for l in self.legs]
        if self.tail and self.tail.segments > 0:
            d["хвост"] = {"сегментов": self.tail.segments,
                          "длина_сегмента": round(self.tail.segment_length, 4),
                          "толщина": round(self.tail.thickness, 4)}
        d["сила_моторов"] = round(self.motor_power, 1)
        if self.start_height is not None:
            d["высота_старта"] = round(self.start_height, 4)
        return d

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            yaml.safe_dump(self.to_dict(), allow_unicode=True, sort_keys=False),
            encoding="utf-8")


# --------------------------------------------------------------------------- #
# Чтение описания
# --------------------------------------------------------------------------- #
def random_gait(rng, n: int) -> tuple:
    """n троек (размах, сдвиг по времени, среднее положение)."""
    out = []
    for _ in range(n):
        out += [rng.uniform(0.2, 1.0), rng.uniform(0.0, 2 * math.pi),
                rng.uniform(-0.3, 0.3)]
    return tuple(out)


def _mutate_gait(rng, gait: tuple, n: int, strength: float) -> tuple:
    """Подкручивает походку; при смене числа суставов дотягивает длину."""
    if len(gait) != 3 * n:
        gait = (tuple(gait) + random_gait(rng, n))[:3 * n]
    out = []
    for i in range(0, len(gait), 3):
        amp, phase, off = gait[i:i + 3]
        out += [min(1.0, max(0.0, amp + rng.normalvariate(0, 0.15 * strength))),
                (phase + rng.normalvariate(0, 0.5 * strength)) % (2 * math.pi),
                min(0.6, max(-0.6, off + rng.normalvariate(0, 0.1 * strength)))]
    return tuple(out)


def _clamp(value: float, key: str) -> float:
    lo, hi = LIMITS[key]
    return min(max(value, lo), hi)


def _jiggle(rng, value: float, key: str, strength: float) -> float:
    """Умножаем на случайный множитель около единицы и держим в рамках."""
    return _clamp(value * math.exp(rng.normalvariate(0.0, 0.25 * strength)), key)


def mutate(base: Creature, rng, strength: float = 1.0) -> Creature:
    """Потомок с небольшими изменениями тела.

    Меняются и числа (длины, толщина, сила моторов), и строение: нога
    может добавиться, исчезнуть или переехать на другой угол. Именно
    из-за строения потомки бывают не похожи на родителя.
    """
    legs = []
    for leg in base.legs:
        # Иногда нога просто исчезает.
        if len(base.legs) > 1 and rng.random() < 0.06 * strength:
            continue
        angle = leg.angle
        if rng.random() < 0.25 * strength:
            angle = (angle + rng.normalvariate(0.0, 18.0 * strength)) % 360.0
        legs.append(Leg(angle=angle,
                        thigh=_jiggle(rng, leg.thigh, "бедро", strength),
                        shin=_jiggle(rng, leg.shin, "голень", strength),
                        thickness=_jiggle(rng, leg.thickness, "толщина", strength),
                        lift=_clamp(leg.lift + rng.normalvariate(0.0, 6.0 * strength),
                                    "подъём"),
                        gait=_mutate_gait(rng, leg.gait, 2, strength)))

    # Иногда вырастает новая нога — копия случайной существующей, сдвинутая
    # по углу. Так у потомка может стать пять ног вместо четырёх.
    if legs and len(legs) < MAX_LEGS and rng.random() < 0.10 * strength:
        donor = rng.choice(legs)
        legs.append(replace(donor,
                            angle=(donor.angle + rng.normalvariate(0.0, 40.0)) % 360.0,
                            gait=_mutate_gait(rng, donor.gait, 2, 2.0 * strength)))
    if not legs:
        legs = [replace(base.legs[0])] if base.legs else []

    tail = base.tail
    if tail is not None:
        segments = tail.segments
        if rng.random() < 0.12 * strength:
            segments = max(0, min(MAX_TAIL, segments + rng.choice([-1, 1])))
        tail = (Tail(segments=segments,
                     segment_length=_jiggle(rng, tail.segment_length,
                                            "длина_сегмента", strength),
                     thickness=_jiggle(rng, tail.thickness, "толщина", strength),
                     gait=_mutate_gait(rng, tail.gait, segments, strength))
                if segments > 0 else None)
    elif rng.random() < 0.05 * strength:
        tail = Tail(segments=1, segment_length=0.2, thickness=0.05,
                    gait=random_gait(rng, 1))

    child = Creature(
        name=base.name, author=base.author, color=base.color,
        shape=base.shape,
        size=_jiggle(rng, base.size, "размер", strength),
        length=_jiggle(rng, base.length, "длина", strength),
        legs=legs, tail=tail,
        motor_power=_jiggle(rng, base.motor_power, "сила_моторов", strength),
        start_height=None,      # рост пересчитываем по новой геометрии
        gait_freq=min(4.0, max(0.4, base.gait_freq
                               + rng.normalvariate(0.0, 0.25 * strength))))

    if child.n_joints == 0:     # выродившееся тело возвращаем к родителю
        return base
    return child


def _leg_from_dict(raw: dict, index: int) -> Leg:
    where = f"нога №{index}"
    if not isinstance(raw, dict):
        raise CreatureError(f"{where}: описание ноги должно быть списком свойств "
                            f"(«где», «бедро», «голень»), а не {raw!r}.")

    unknown = set(raw) - {"где", "угол", "бедро", "голень", "толщина", "подъём", "подъем"}
    if unknown:
        raise CreatureError(
            f"{where}: непонятные свойства {sorted(unknown)}.\n"
            f"  У ноги бывают: где (или угол), бедро, голень, толщина, подъём.")

    if "угол" in raw:
        angle = raw["угол"]
        if isinstance(angle, bool) or not isinstance(angle, (int, float)):
            raise CreatureError(f"{where}: «угол» должен быть числом градусов "
                                f"(0 — вперёд, 90 — влево).")
        angle = float(angle) % 360.0
    elif "где" in raw:
        key = str(raw["где"]).strip().lower()
        if key not in DIRECTIONS:
            raise CreatureError(
                f"{where}: не знаю направление «{raw['где']}».\n"
                f"  Можно: {', '.join(sorted(set(DIRECTIONS)))}.\n"
                f"  Или задай число градусов: угол: 45")
        angle = DIRECTIONS[key]
    else:
        raise CreatureError(f"{where}: не сказано, куда смотрит нога. "
                            f"Добавь «где: спереди-слева» или «угол: 45».")

    for required in ("бедро", "голень"):
        if required not in raw:
            raise CreatureError(f"{where}: не хватает свойства «{required}».")

    lift = raw.get("подъём", raw.get("подъем", 50.0))
    return Leg(angle=angle,
               thigh=_check_number(raw["бедро"], "бедро", where),
               shin=_check_number(raw["голень"], "голень", where),
               thickness=_check_number(raw.get("толщина", 0.08), "толщина", where),
               lift=_check_number(lift, "подъём", where))


def from_dict(raw: dict, source: str = "описание") -> Creature:
    if not isinstance(raw, dict):
        raise CreatureError(f"{source}: файл должен начинаться со свойств вида "
                            f"«имя: Пухлик», а получилось {type(raw).__name__}.")

    unknown = set(raw) - {"имя", "автор", "цвет", "туловище", "ноги", "хвост",
                          "сила_моторов", "высота_старта"}
    if unknown:
        raise CreatureError(
            f"{source}: непонятные разделы {sorted(unknown)}.\n"
            f"  Бывают: имя, автор, цвет, туловище, ноги, хвост, сила_моторов.")

    name = str(raw.get("имя", "")).strip()
    if not name:
        raise CreatureError(f"{source}: у существа должно быть имя. "
                            f"Добавь первой строкой «имя: Пухлик».")

    color = str(raw.get("цвет", "оранжевый")).strip().lower()
    if color not in COLORS:
        raise CreatureError(f"{source}: не знаю цвет «{color}».\n"
                            f"  Можно: {', '.join(sorted(COLORS))}.")

    torso = raw.get("туловище", {})
    if not isinstance(torso, dict):
        raise CreatureError(f"{source}: «туловище» описывается свойствами "
                            f"«форма» и «размер».")
    shape = str(torso.get("форма", "шар")).strip().lower()
    if shape not in SHAPES:
        raise CreatureError(f"{source}: форма туловища бывает "
                            f"{' или '.join(SHAPES)}, а не «{shape}».")
    size = _check_number(torso.get("размер", 0.25), "размер", f"{source}: туловище")
    length = _check_number(torso.get("длина", 0.5), "длина", f"{source}: туловище")

    raw_legs = raw.get("ноги", [])
    if raw_legs is None:
        raw_legs = []
    if not isinstance(raw_legs, list):
        raise CreatureError(f"{source}: «ноги» — это список. Каждая нога "
                            f"начинается с «- где: ...».")
    if len(raw_legs) > MAX_LEGS:
        raise CreatureError(f"{source}: {len(raw_legs)} ног — это слишком много, "
                            f"не больше {MAX_LEGS}.")
    legs = [_leg_from_dict(item, i + 1) for i, item in enumerate(raw_legs)]

    tail = None
    raw_tail = raw.get("хвост")
    if raw_tail:
        if not isinstance(raw_tail, dict):
            raise CreatureError(f"{source}: «хвост» описывается свойствами "
                                f"«сегментов» и «длина_сегмента».")
        segments = raw_tail.get("сегментов", 0)
        if isinstance(segments, bool) or not isinstance(segments, int):
            raise CreatureError(f"{source}: «сегментов» — целое число.")
        if not (0 <= segments <= MAX_TAIL):
            raise CreatureError(f"{source}: сегментов хвоста от 0 до {MAX_TAIL}, "
                                f"а не {segments}.")
        if segments:
            tail = Tail(segments=segments,
                        segment_length=_check_number(
                            raw_tail.get("длина_сегмента", 0.25),
                            "длина_сегмента", f"{source}: хвост"),
                        thickness=_check_number(raw_tail.get("толщина", 0.07),
                                                "толщина", f"{source}: хвост"))

    creature = Creature(
        name=name, author=str(raw.get("автор", "")).strip(), color=color,
        shape=shape, size=size, length=length, legs=legs, tail=tail,
        motor_power=_check_number(raw.get("сила_моторов", 150.0),
                                  "сила_моторов", source),
        start_height=(float(raw["высота_старта"]) if "высота_старта" in raw else None))

    if creature.n_joints == 0:
        raise CreatureError(
            f"{source}: у «{name}» нет ни одного сустава — двигаться нечем.\n"
            f"  Добавь хотя бы одну ногу или хвост из нескольких сегментов.")
    return creature


def load(path: str | Path) -> Creature:
    path = Path(path)
    if not path.exists():
        raise CreatureError(f"Нет файла {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise CreatureError(
            f"{path.name}: файл сломан — YAML не читается.\n"
            f"  Чаще всего это лишний пробел в начале строки или пропущенное "
            f"двоеточие.\n  Подробности: {exc}") from exc
    return from_dict(raw or {}, source=path.name)
