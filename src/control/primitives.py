"""
Robot Primitives — 5 basic actions for tabletop manipulation.

Following Genesis World tutorial:
- plan_path() for collision-free motion planning
- control_dofs_force() for gripper grasping
- set_dofs_kp/kv/force_range for PD controller tuning
- inverse_kinematics() with quat=[0,1,0,0] for top-down orientation

Primitives: Move, Open Gripper, Close Gripper, Pick, Place.
"""
import numpy as np
import torch
from typing import List, Optional


class RobotPrimitives:
    def __init__(self, robot, scene, objects=None):
        """
        robot: gs.RigidEntity (Franka Panda)
        scene: gs.Scene (used for scene.step)
        objects: optional {name: entity} dict from SceneManager
        """
        self.robot = robot
        self.scene = scene
        self.objects = objects or {}
        self.n_dofs = robot.n_dofs
        self.n_arm = min(7, self.n_dofs)  # Franka arm joints
        self.gripper_dofs = list(range(self.n_arm, self.n_dofs))  # gripper DOF indices
        self.arm_dofs = list(range(self.n_arm))

        # End-effector link for IK target
        self.ee_link = robot.get_link("hand")

        # Configure PD gains (from Genesis tutorial)
        robot.set_dofs_kp(
            kp=np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
            dofs_idx_local=list(range(self.n_dofs)),
        )
        robot.set_dofs_kv(
            kv=np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
            dofs_idx_local=list(range(self.n_dofs)),
        )
        robot.set_dofs_force_range(
            lower=np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            upper=np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
            dofs_idx_local=list(range(self.n_dofs)),
        )

    # ── IK ───────────────────────────────────────────────────

    # Top-down orientation: gripper points down (180° rotation about X)
    TOP_DOWN_QUAT = np.array([0, 1, 0, 0])

    def solve_ik(self, x: float, y: float, z: float, quat=None) -> List[float]:
        """Solve IK for target position. Uses top-down orientation by default."""
        if quat is None:
            quat = self.TOP_DOWN_QUAT
        qpos = self.robot.inverse_kinematics(
            link=self.ee_link,
            pos=np.array([x, y, z]),
            quat=np.array(quat),
        )
        return qpos.tolist()

    # ── Motion Planning ──────────────────────────────────────

    def plan_and_execute(self, qpos_goal, num_waypoints=200):
        """Plan collision-free path and execute it."""
        path = self.robot.plan_path(
            qpos_goal=qpos_goal,
            num_waypoints=num_waypoints,
        )
        for waypoint in path:
            self.robot.control_dofs_position(waypoint)
            self.scene.step()
        # Let PD controller settle
        for _ in range(100):
            self.scene.step()

    # ── Primitive 1: Move ────────────────────────────────────

    def move_to_xyz(self, x: float, y: float, z: float, quat=None):
        """Plan and execute motion to target XYZ position."""
        qpos = self.solve_ik(x, y, z, quat)
        # Set gripper to open
        qpos[-2:] = 0.04
        self.plan_and_execute(qpos)

    def move_above_object(self, object_pos: List[float], height: float = 0.12):
        """Move hand above object for pre-grasp."""
        self.move_to_xyz(object_pos[0], object_pos[1], object_pos[2] + height)

    # ── Primitive 2: Open Gripper ────────────────────────────

    def open_gripper(self, width: float = 0.04, steps: int = 100):
        """Open gripper via PD position control."""
        target = torch.tensor([width] * len(self.gripper_dofs),
                              dtype=torch.float32, device=self.robot.get_qpos().device)
        self.robot.control_dofs_position(target, dofs_idx_local=self.gripper_dofs)
        self.scene.step(steps)

    # ── Primitive 3: Close Gripper (force control) ───────────

    def close_gripper_force(self, force_per_finger: float = 1.0, steps: int = 200):
        """Close gripper using force control (from Genesis tutorial).

        Uses control_dofs_force() — applies inward force per finger.
        More robust than position control for grasping.
        """
        force = torch.tensor([-force_per_finger, -force_per_finger],
                             dtype=torch.float32, device=self.robot.get_qpos().device)
        self.robot.control_dofs_force(force, dofs_idx_local=self.gripper_dofs)
        self.scene.step(steps)

    # ── Primitive 4: Pick (tutorial-based) ────────────────────

    def pick(self, object_pos: List[float], num_waypoints: int = 200,
             grasp_force: float = 1.0, approach_height: float = 0.12,
             grasp_height: float = 0.02):
        """Pick sequence following Genesis tutorial pattern:
          1. Plan path to pre-grasp (above object)
          2. Plan path to grasp height (collision-free descent)
          3. Force-control grasp
          4. Plan path to lift
        """
        # Phase 1: Plan path to pre-grasp
        pre_qpos = self.solve_ik(object_pos[0], object_pos[1], object_pos[2] + approach_height)
        pre_qpos_arr = np.array(pre_qpos)
        pre_qpos_arr[-2:] = 0.04  # open gripper
        self.plan_and_execute(pre_qpos_arr.tolist(), num_waypoints)

        # Phase 2: Plan path to grasp height (collision-free)
        grasp_qpos = self.solve_ik(object_pos[0], object_pos[1], object_pos[2] + grasp_height)
        grasp_arr = np.array(grasp_qpos)
        grasp_arr[-2:] = 0.04  # keep gripper open during descent
        self.plan_and_execute(grasp_arr.tolist(), num_waypoints)

        # Phase 3: Force-control grasp (from tutorial)
        self.close_gripper_force(grasp_force, steps=200)

        # Phase 4: Plan path to lift
        lift_qpos = self.solve_ik(object_pos[0], object_pos[1], object_pos[2] + 0.20)
        lift_list = lift_qpos if isinstance(lift_qpos, list) else lift_qpos.tolist()
        self.plan_and_execute(lift_list, num_waypoints)

    # ── Primitive 5: Place ───────────────────────────────────

    def place(self, target_pos: List[float], num_waypoints: int = 200):
        """Place sequence: plan to above → descend → open gripper."""
        # Move above target
        place_qpos = self.solve_ik(target_pos[0], target_pos[1], target_pos[2] + 0.12)
        self.plan_and_execute(place_qpos, num_waypoints)

        # Descend
        descend_qpos = self.solve_ik(target_pos[0], target_pos[1], target_pos[2] + 0.02)
        self.robot.control_dofs_position(descend_qpos[:-2], dofs_idx_local=self.arm_dofs)
        self.scene.step(200)

        # Open gripper to release
        self.open_gripper(0.04, steps=200)

        # Lift away
        lift_qpos = self.solve_ik(target_pos[0], target_pos[1], target_pos[2] + 0.15)
        self.plan_and_execute(lift_qpos, num_waypoints)

    # ── Pick & Place ─────────────────────────────────────────

    def pick_and_place(self, object_pos, target_pos):
        self.pick(object_pos)
        self.place(target_pos)
