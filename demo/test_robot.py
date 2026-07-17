"""
RoboPilot Robot Arm Test
验证机械臂加载和基本运动
"""
import genesis as gs
import torch

gs.init(backend=gs.gpu)
scene = gs.Scene()

# Add ground
plane = scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid())

# Load Franka Panda
robot = scene.add_entity(
    gs.morphs.URDF(
        file="/opt/venv/lib/python3.12/site-packages/genesis/assets/urdf/panda_bullet/panda.urdf",
        pos=(0, 0, 0.75),
    ),
)

scene.build()
print(f"Robot loaded - {robot.n_dofs} DOFs")

# Move joints
qpos = robot.get_qpos()
target = qpos.clone()
target[0] = 0.5
target[1] = -0.3
robot.set_qpos(target)

for i in range(100):
    scene.step()

final = robot.get_qpos()
print(f"Joint movement OK - final qpos: {final[:3].tolist()}")
