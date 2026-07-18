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
    # Camera at ~40° top-down angle, centered on the workspace.
    # Covers robot arm, table surface, and all objects in one frame.
    camera_pos: Tuple[float, float, float] = (0.3, -1.2, 1.6)
    camera_lookat: Tuple[float, float, float] = (0.3, 0, 0.05)
    camera_fov: int = 55


# ── Asset Paths ──────────────────────────────────────────────
# panda_no_tendon.xml: proven working with teleport+PD strategy.
_GENESIS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
    "../../venv/lib/python3.12/site-packages/genesis/assets")
FRANKA_MJCF = os.path.join(_GENESIS_DIR, "xml/franka_emika_panda/panda_no_tendon.xml")
FRANKA_URDF = os.path.join(_GENESIS_DIR, "urdf/panda_bullet/panda.urdf")


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
        # Only init if not already initialized (allows CPU fallback in tests)
        try:
            gs.init(backend=gs.amdgpu)
        except Exception:
            try:
                gs.init(backend=gs.gpu)
            except Exception:
                pass  # already initialized — continue with current backend
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

    def add_area_light(self, pos, size=0.3, color=(1, 1, 1), intensity=20.0, double_sided=True):
        """Add an area mesh light (emissive surface) to the scene. RayTracer compatible.

        pos: light center position [x, y, z]
        size: light panel size (square, in meters)
        color: RGB tuple (0-1 range)
        intensity: light brightness
        double_sided: emit light from both sides
        """
        light_morph = gs.morphs.Box(
            size=(size, size, 0.01),
            pos=pos,
        )
        self.scene.add_mesh_light(
            morph=light_morph, color=color,
            intensity=intensity, double_sided=double_sided,
        )
        return self

    # ── Build & Step ─────────────────────────────────────────

    def build(self):
        self.scene.build()
        return self

    def step(self, n=1):
        for _ in range(n):
            self.scene.step()

    def settle(self, steps=100):
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


# ── Default Scene: Tabletop with 8 objects ───────────────────
# Table sits on the ground (z=0) to avoid falling through physics.
# Robot base is raised to z=0.35 so the arm can reach the table.
# Objects sit directly on the table surface (z=0.05 + half height).
# Objects are spread across the table to give VLM spatial reasoning tasks.

TABLE_HEIGHT = 0.05  # table thickness
TABLE_TOP = TABLE_HEIGHT  # table top z (bottom at z=0)


def create_tabletop_scene(show_viewer=False) -> SceneManager:
    config = SceneConfig(show_viewer=show_viewer, robot_pos=(0, 0, 0.35))
    scene = (
        SceneManager(config)
        .init_genesis()
        .add_ground()
        .add_table(size=(0.8, 1.0, 0.05), pos=(0.5, 0, 0.025))
        .add_robot(use_mjcf=True)
    )

    # 8 objects on Kinematic table. Table top at z=0.05.
    # Suction gripper works best with cubes and simple shapes.
    table_top = 0.05
    scene.add_box("red_cube", pos=(0.6, 0.0, table_top + 0.02))
    scene.add_box("blue_cube", pos=(0.55, 0.2, table_top + 0.02))
    scene.add_cup("green_cup", pos=(0.7, -0.1, table_top + 0.04))
    scene.add_bottle("yellow_bottle", pos=(0.6, -0.2, table_top + 0.075))
    scene.add_box("white_cube", pos=(0.65, 0.1, table_top + 0.02))
    scene.add_box("orange_cube", pos=(0.75, 0.15, table_top + 0.02))
    scene.add_sphere("red_apple", radius=0.03, pos=(0.55, -0.1, table_top + 0.03))
    scene.add_sphere("blue_ball", radius=0.02, pos=(0.7, -0.15, table_top + 0.02))

    scene.add_camera()
    scene.build()
    return scene
