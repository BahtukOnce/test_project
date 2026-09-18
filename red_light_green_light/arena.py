"""Сборка MuJoCo-сцены: коридор, финиш, кукла-ведущий и четвероногий агент.

Тело агента — классический "ant" (4 ноги, 8 моторов). Он не умеет ходить
изначально: политика учится управлять суставами с нуля.
"""

FINISH_X = 25.0          # где находится финишная черта, метры
CORRIDOR_HALF_W = 3.0    # полуширина коридора
DOLL_X = FINISH_X + 2.0  # кукла стоит за финишем


def _leg(name: str, hip: str, ankle: str, dx: float, dy: float,
         ankle_axis: str, ankle_range: str) -> str:
    """Одна нога: бедро (поворот вокруг Z) + голень (сгиб)."""
    return f"""
      <body name="{name}" pos="0 0 0">
        <geom fromto="0 0 0 {0.2 * dx} {0.2 * dy} 0" name="{name}_aux_geom" size="0.08" type="capsule"/>
        <body name="{name}_aux" pos="{0.2 * dx} {0.2 * dy} 0">
          <joint axis="0 0 1" name="{hip}" pos="0 0 0" range="-30 30" type="hinge"/>
          <geom fromto="0 0 0 {0.2 * dx} {0.2 * dy} 0" name="{name}_upper_geom" size="0.08" type="capsule"/>
          <body pos="{0.2 * dx} {0.2 * dy} 0">
            <joint axis="{ankle_axis}" name="{ankle}" pos="0 0 0" range="{ankle_range}" type="hinge"/>
            <geom fromto="0 0 0 {0.4 * dx} {0.4 * dy} 0" name="{name}_lower_geom" size="0.08" type="capsule"/>
          </body>
        </body>
      </body>"""


def build_xml(finish_x: float = FINISH_X,
              half_width: float = CORRIDOR_HALF_W) -> str:
    doll_x = finish_x + 2.0
    mid_x = finish_x / 2.0
    floor_half_len = finish_x / 2.0 + 12.0

    legs = (
        _leg("leg_fl", "hip_1", "ankle_1", +1, +1, "-1 1 0", "30 70") +
        _leg("leg_fr", "hip_2", "ankle_2", -1, +1, "1 1 0", "-70 -30") +
        _leg("leg_bl", "hip_3", "ankle_3", -1, -1, "-1 1 0", "-70 -30") +
        _leg("leg_br", "hip_4", "ankle_4", +1, -1, "1 1 0", "30 70")
    )

    return f"""
<mujoco model="red_light_green_light">
  <compiler angle="degree" inertiafromgeom="true"/>
  <option integrator="RK4" timestep="0.01"/>

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

    <geom name="wall_left" type="box" pos="{mid_x} {half_width} 0.8"
          size="{floor_half_len} 0.15 0.8" conaffinity="1" rgba="0.30 0.33 0.40 1"/>
    <geom name="wall_right" type="box" pos="{mid_x} {-half_width} 0.8"
          size="{floor_half_len} 0.15 0.8" conaffinity="1" rgba="0.30 0.33 0.40 1"/>

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

    <body name="torso" pos="0 0 0.75">
      <camera name="track" mode="trackcom" pos="-1.0 -5.0 2.2" xyaxes="1 0 0 0 0.42 0.91"/>
      <camera name="side" mode="trackcom" pos="0 -7.0 1.6" xyaxes="1 0 0 0 0.2 0.98"/>
      <geom name="torso_geom" type="sphere" size="0.25" rgba="0.90 0.35 0.30 1"/>
      <joint armature="0" damping="0" limited="false" margin="0.01" name="root"
             pos="0 0 0" type="free"/>{legs}
    </body>
  </worldbody>

  <actuator>
    <motor joint="hip_1"/>
    <motor joint="ankle_1"/>
    <motor joint="hip_2"/>
    <motor joint="ankle_2"/>
    <motor joint="hip_3"/>
    <motor joint="ankle_3"/>
    <motor joint="hip_4"/>
    <motor joint="ankle_4"/>
  </actuator>
</mujoco>
"""
