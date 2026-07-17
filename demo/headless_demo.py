"""
Headless Genesis Demo — Tabletop Manipulation
Franka Panda + 4 objects + Camera → PNG + MP4

Usage:
    python demo/headless_demo.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.sim.scene_manager import create_tabletop_scene
from src.control.primitives import RobotPrimitives
import time


def main():
    print("=" * 60)
    print("  RoboPilot — Tabletop Manipulation Demo")
    print("  Franka Panda + 4 Objects + Headless Camera")
    print("=" * 60)

    # ── Phase 1: Build Scene ─────────────────────────────────
    print("\n[1/5] Building tabletop scene...")
    start = time.time()

    scene = create_tabletop_scene(show_viewer=False)

    print(f"  Scene built in {time.time() - start:.1f}s")
    print(f"  Robot: Franka Panda (MJCF)")
    print(f"  Objects: red_cup, blue_box, apple, bottle")

    # ── Phase 2: Settle & Capture Initial ────────────────────
    print("\n[2/5] Settling physics & capturing initial frame...")
    scene.settle(100)
    scene.save_frame("demo/output/01_initial.png")
    print("  Saved: 01_initial.png")

    # ── Phase 3: Pick & Place with Primitives ────────────────
    print("\n[3/5] Running Pick & Place (Primitives)...")

    robot = scene.robot
    primitives = RobotPrimitives(robot, scene.scene)

    # Get object positions
    cup_pos = scene.objects["red_cup"].get_pos().tolist()
    box_pos = scene.objects["blue_box"].get_pos().tolist()
    print(f"  Red cup position: {[f'{x:.2f}' for x in cup_pos]}")
    print(f"  Blue box position: {[f'{x:.2f}' for x in box_pos]}")

    # Primitive 1: Open gripper
    print("  [1/5] Open gripper")
    primitives.open_gripper()
    scene.capture_frame()

    # Primitive 2: Move above cup
    print("  [2/5] Move above red cup")
    primitives.move_above_object(cup_pos, height=0.15)
    scene.capture_frame()

    # Primitive 3: Pick cup
    print("  [3/5] Pick red cup")
    primitives.pick(cup_pos)
    scene.capture_frame()

    # Primitive 4: Move above box
    print("  [4/5] Move above blue box")
    primitives.move_above_object(box_pos, height=0.15)
    scene.capture_frame()

    # Primitive 5: Place in box
    print("  [5/5] Place in blue box")
    primitives.place(box_pos)
    scene.capture_frame()

    # ── Phase 4: Save Outputs ────────────────────────────────
    print("\n[4/5] Saving outputs...")
    scene.save_frame("demo/output/02_final.png")
    scene.save_video("demo/output/tabletop_demo.mp4", fps=30)

    # ── Phase 5: Summary ─────────────────────────────────────
    print("\n[5/5] Summary")
    state = scene.get_state()
    print(f"  Robot position: {[f'{x:.3f}' for x in state.get('robot_pos', [])]}")
    print(f"  Red cup position: {[f'{x:.3f}' for x in state.get('red_cup_pos', [])]}")
    print(f"  Blue box position: {[f'{x:.3f}' for x in state.get('blue_box_pos', [])]}")
    print(f"  Total frames: {len(scene.frames)}")
    print(f"\n  Output files:")
    print(f"    demo/output/01_initial.png")
    print(f"    demo/output/02_final.png")
    print(f"    demo/output/tabletop_demo.mp4")
    print("\n  Tabletop demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    os.makedirs("demo/output", exist_ok=True)
    main()
