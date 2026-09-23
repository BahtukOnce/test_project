"""Сборка MuJoCo-сцены: коридор, финиш, кукла-ведущий и тело агента.

Тело приходит из описания существа (creature.py) — того самого файла,
который правит ученик. Ходить существо изначально не умеет: политика
учится управлять его суставами с нуля.
"""

FINISH_X = 25.0          # где находится финишная черта, метры
CORRIDOR_HALF_W = 3.0    # полуширина коридора
DOLL_X = FINISH_X + 2.0  # кукла стоит за финишем


def build_xml(creature, finish_x: float = FINISH_X,
              half_width: float = CORRIDOR_HALF_W) -> str:
    """Полная сцена: коридор с финишем плюс тело переданного существа."""
    doll_x = finish_x + 2.0
    mid_x = finish_x / 2.0
    floor_half_len = finish_x / 2.0 + 12.0
    body = creature.body_xml()
    actuators = creature.actuators_xml()

    return f"""
<mujoco model="red_light_green_light">
  <compiler angle="degree" inertiafromgeom="true"/>
  <option integrator="RK4" timestep="0.01"/>

  <!-- Размер буфера офлайн-рендера. По умолчанию MuJoCo даёт 640x480,
       и запись видео в большем разрешении падает с ошибкой. -->
  <visual>
    <global offwidth="1920" offheight="1080"/>
    <quality shadowsize="4096"/>
  </visual>

  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom conaffinity="0" condim="3" density="5.0" friction="1 0.5 0.5"
          margin="0.01" rgba="0.82 0.55 0.28 1"/>
    <motor ctrllimited="true" ctrlrange="-1 1" gear="150"/>
  </default>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.26 0.42 0.62" rgb2="0.05 0.07 0.12"
             width="512" height="512"/>
    <texture name="grid_tex" type="2d" builtin="checker" rgb1="0.21 0.24 0.29"
             rgb2="0.26 0.30 0.36" width="512" height="512"/>
    <material name="grid" texture="grid_tex" texrepeat="{int(floor_half_len)} 6" reflectance="0.05"/>
  </asset>

  <worldbody>
    <light pos="{mid_x} 0 14" dir="0 0 -1" diffuse="0.85 0.85 0.85" specular="0.1 0.1 0.1"
           directional="true" castshadow="true"/>

    <geom name="floor" type="plane" material="grid" pos="{mid_x} 0 0"
          size="{floor_half_len} {half_width + 1.0} 0.1" conaffinity="1" condim="3"
          friction="1 0.5 0.5" rgba="0.55 0.58 0.62 1"/>

    <!-- Стены коридора разделены на две части. Высокая невидимая
         (rgba с нулевой прозрачностью) удерживает агента, но не
         загораживает камеру. Низкий бортик рядом — только для вида. -->
    <geom name="wall_left" type="box" pos="{mid_x} {half_width} 0.8"
          size="{floor_half_len} 0.15 0.8" conaffinity="1" rgba="0 0 0 0"/>
    <geom name="wall_right" type="box" pos="{mid_x} {-half_width} 0.8"
          size="{floor_half_len} 0.15 0.8" conaffinity="1" rgba="0 0 0 0"/>
    <geom name="curb_left" type="box" pos="{mid_x} {half_width} 0.14"
          size="{floor_half_len} 0.16 0.14" contype="0" conaffinity="0"
          rgba="0.30 0.33 0.40 1"/>
    <geom name="curb_right" type="box" pos="{mid_x} {-half_width} 0.14"
          size="{floor_half_len} 0.16 0.14" contype="0" conaffinity="0"
          rgba="0.30 0.33 0.40 1"/>

    <geom name="start_line" type="box" pos="0 0 0.011" size="0.1 {half_width} 0.01"
          contype="0" conaffinity="0" rgba="0.85 0.85 0.85 1"/>
    <geom name="finish_line" type="box" pos="{finish_x} 0 0.011" size="0.18 {half_width} 0.01"
          contype="0" conaffinity="0" rgba="0.95 0.95 0.95 1"/>

    <!-- Лампа-светофор: цвет меняется из кода через model.geom_rgba -->
    <geom name="light" type="sphere" pos="{doll_x} 0 3.4" size="0.45"
          contype="0" conaffinity="0" rgba="0.1 0.9 0.2 1"/>

    <!-- Кукла-ведущий. Это mocap-тело: разворачиваем его на 180 градусов,
         когда загорается красный, — так видно, что она "смотрит" на агента. -->
    <body name="doll" mocap="true" pos="{doll_x} 0 0">
      <geom name="doll_dress" type="cylinder" pos="0 0 0.85" size="0.42 0.85"
            contype="0" conaffinity="0" rgba="0.95 0.72 0.18 1"/>
      <geom name="doll_head" type="sphere" pos="0 0 2.05" size="0.38"
            contype="0" conaffinity="0" rgba="0.94 0.82 0.70 1"/>
      <geom name="doll_face" type="box" pos="-0.34 0 2.10" size="0.06 0.26 0.12"
            contype="0" conaffinity="0" rgba="0.12 0.12 0.14 1"/>
    </body>

{body}
  </worldbody>

  <actuator>
{actuators}
  </actuator>
</mujoco>
"""
