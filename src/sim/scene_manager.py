"""
Module 10: Scene Manager
Genesis scene — Franka Panda + Tabletop Manipulation.
"""
import genesis as gs
import numpy as np
import os
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any


@dataclass
class SceneConfig:
    robot_pos: Tuple[float, float, float] = (0, 0, 0.75)
    gravity: Tuple[float, float, float] = (0, 0, -9.8)
    dt: float = 0.01
    show_viewer: bool = False
    camera_res: Tuple[int, int] = (640, 480)
    camera_pos: Tuple[float, float, float] = (0.5, 2.0, 2.0)
    camera_lookat: Tuple[float, float, float] = (0.3, 0, 0.3)
    camera_fov: int = 45


# ── Asset Paths ──────────────────────────────────────────────
# Use panda_no_tendon.xml — the standard panda.xml has tendon-driven finger
# actuators approximated in Genesis, which makes control_dofs_position not
# reliably drive the gripper. The no-tendon variant has independent prismatic
# joint actuators that respond to position targets correctly.
FRANKA_MJCF = "/opt/venv/lib/python3.12/site-packages/genesis/assets/xml/franka_emika_panda/panda_no_tendon.xml"
FRANKA_URDF = "/opt/venv/lib/python3.12/site-packages/genesis/assets/urdf/panda_bullet/panda.urdf"


class SceneManager:
    def __init__(self, config: Optional[SceneConfig] = None):
        self.config = config or SceneConfig()
        self.scene = None
        self.robot = None
        self.camera = None
        self.table = None
        self.objects: Dict[str, Any] = {}
        self.frames: List[np.ndarray] = []

    # ── Genesis Init ─────────────────────────────────────────

    def init_genesis(self):
        gs.init(backend=gs.gpu)
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.config.dt,
                gravity=self.config.gravity,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.5, 0, 2),
                camera_lookat=(0, 0, 0.5),
            ),
            show_viewer=self.config.show_viewer,
        )
        return self

    # ── Scene Entities ───────────────────────────────────────

    def add_ground(self):
        self.scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid())
        return self

    def add_table(self, size=(0.8, 1.0, 0.05), pos=(0.3, 0, 0.35)):
        self.table = self.scene.add_entity(
            gs.morphs.Box(size=size, pos=pos),
            material=gs.materials.Rigid(),
        )
        return self

    def add_robot(self, use_mjcf: bool = True):
        if use_mjcf:
            self.robot = self.scene.add_entity(
                gs.morphs.MJCF(file=FRANKA_MJCF),
            )
        else:
            self.robot = self.scene.add_entity(
                gs.morphs.URDF(file=FRANKA_URDF, pos=self.config.robot_pos),
            )
        return self

    def add_cup(self, name="red_cup", color="red", pos=(0.2, 0.15, 0.43)):
        cup = self.scene.add_entity(
            gs.morphs.Cylinder(radius=0.03, height=0.08, pos=pos),
            material=gs.materials.Rigid(),
        )
        self.objects[name] = cup
        return self

    def add_box(self, name="blue_box", color="blue", pos=(-0.2, 0.15, 0.42)):
        box = self.scene.add_entity(
            gs.morphs.Box(size=(0.1, 0.1, 0.07), pos=pos),
            material=gs.materials.Rigid(),
        )
        self.objects[name] = box
        return self

    def add_sphere(self, name="apple", radius=0.03, pos=(0.2, -0.15, 0.41)):
        sphere = self.scene.add_entity(
            gs.morphs.Sphere(radius=radius, pos=pos),
            material=gs.materials.Rigid(),
        )
        self.objects[name] = sphere
        return self

    def add_bottle(self, name="bottle", pos=(-0.2, -0.15, 0.44)):
        bottle = self.scene.add_entity(
            gs.morphs.Cylinder(radius=0.02, height=0.15, pos=pos),
            material=gs.materials.Rigid(),
        )
        self.objects[name] = bottle
        return self

    def add_camera(self, res=None, pos=None, lookat=None, fov=None):
        self.camera = self.scene.add_camera(
            res=res or self.config.camera_res,
            pos=pos or self.config.camera_pos,
            lookat=lookat or self.config.camera_lookat,
            up=(0.0, 0.0, 1.0),
            fov=fov or self.config.camera_fov,
        )
        return self

    # ── Build & Step ─────────────────────────────────────────

    def build(self):
        self.scene.build()
        # Set robot to a "ready" pose so PD control starts from a reasonable
        # configuration (otherwise all-zeros home makes IK and tracking hard).
        self._go_to_ready_pose()
        return self

    def _go_to_ready_pose(self):
        """Drive Franka to a bent 'ready' pose via PD control."""
        import torch
        if self.robot is None or self.robot.n_dofs < 7:
            return
        import numpy as _np
        ready = torch.tensor(
            [0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0, 0.04, 0.04][: self.robot.n_dofs],
            dtype=torch.float32, device=self.robot.get_qpos().device,
        )
        self.robot.control_dofs_position(ready)
        for _ in range(200):
            self.scene.step()

    def step(self, n=1):
        for _ in range(n):
            self.scene.step()

    def settle(self, steps=100):
        self._go_to_ready_pose()
        self.step(steps)

    # ── Camera ───────────────────────────────────────────────

    def render_rgb(self) -> np.ndarray:
        return self.camera.render()[0]

    def render_depth(self) -> np.ndarray:
        return self.camera.render(depth=True)[1]

    # ── Frame Recording ──────────────────────────────────────

    def capture_frame(self):
        self.frames.append(self.render_rgb())

    def save_video(self, path: str, fps: int = 30) -> str:
        if not self.frames:
            return ""
        import imageio
        writer = imageio.get_writer(path, fps=fps)
        try:
            for frame in self.frames:
                writer.append_data(frame)
        finally:
            writer.close()
        print(f"Video saved: {path} ({len(self.frames)} frames, {fps} FPS)")
        return path

    def save_frame(self, path: str):
        from PIL import Image
        Image.fromarray(self.render_rgb()).save(path)

    def clear_frames(self):
        self.frames.clear()

    # ── State ────────────────────────────────────────────────

    def get_state(self) -> dict:
        state = {}
        if self.robot:
            state["robot_qpos"] = self.robot.get_qpos().tolist()
            state["robot_pos"] = self.robot.get_pos().tolist()
        for name, obj in self.objects.items():
            state[f"{name}_pos"] = obj.get_pos().tolist()
        return state


# ── Default Scene: Tabletop with 4 objects ───────────────────

def create_tabletop_scene(show_viewer=False) -> SceneManager:
    config = SceneConfig(show_viewer=show_viewer)
    return (
        SceneManager(config)
        .init_genesis()
        .add_ground()
        .add_table()
        .add_robot()
        .add_cup("red_cup", pos=(0.2, 0.15, 0.43))
        .add_box("blue_box", pos=(-0.2, 0.15, 0.42))
        .add_sphere("apple", radius=0.03, pos=(0.2, -0.15, 0.41))
        .add_bottle("bottle", pos=(-0.2, -0.15, 0.44))
        .add_camera()
        .build()
    )
