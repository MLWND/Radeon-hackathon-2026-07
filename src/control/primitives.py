"""
Robot Primitives — 5 basic actions for tabletop manipulation.
Uses Genesis control_dofs_position (PD controller) for real physics interaction.

Why this matters:
  - set_qpos() is a state teleport (no forces, gripper can't grasp objects)
  - control_dofs_position() drives the robot toward the target via PD controller
  - inverse_kinematics() is Genesis' built-in, GPU-accelerated IK solver

Primitives: Move, Open Gripper, Close Gripper, Pick, Place.
"""
import numpy as np
import torch
from typing import List, Optional


class RobotPrimitives:
    def __init__(self, robot, scene, objects=None):
        """
        robot: gs.RigidEntity
        scene: gs.Scene (used for scene.step)
        objects: optional {name: entity} dict from SceneManager — enables
                 self.objects[name].get_pos() and proper recovery updates
        """
        self.robot = robot
        self.scene = scene
        self.objects = objects or {}  # Genesis entities by name (e.g. {"red_cup": entity})
        self.n_dofs = robot.n_dofs
        self.n_arm = min(7, self.n_dofs)  # Franka arm joints
        self.gripper_dofs = list(range(self.n_arm, self.n_dofs))  # gripper DOF indices
        self.arm_dofs = list(range(self.n_arm))

        # End-effector link ("hand") for IK target
        self.ee_link = self._find_link("hand", "link7", "panda_hand", "panda_link7")
        # Use the link's position in robot.links list for get_links_pos() indexing,
        # NOT link.idx (which is Genesis' internal index, may differ).
        self.ee_link_idx = self._find_link_list_index(self.ee_link)

        # Cache computed IK targets to avoid recomputation
        self._ik_cache = {}

    def _find_link(self, *candidates):
        names = {ln.name: ln for ln in self.robot.links}
        for c in candidates:
            if c in names:
                return names[c]
        return list(names.values())[-1]  # last link as fallback

    def _find_link_list_index(self, link):
        """Find the index of a link in robot.links list (for get_links_pos indexing)."""
        for i, ln in enumerate(self.robot.links):
            if ln is link:
                return i
        return len(self.robot.links) - 1

    # ── State accessors ─────────────────────────────────────

    def _get_arm_qpos(self) -> np.ndarray:
        return self.robot.get_qpos()[:self.n_arm].cpu().numpy()

    def _set_arm_qpos_raw(self, joints):
        """Teleport (use only for FK/IK probing, never in execution)."""
        qpos = self.robot.get_qpos().clone()
        for i, val in enumerate(list(joints)[:self.n_arm]):
            qpos[i] = float(val)
        self.robot.set_qpos(qpos)

    def _get_gripper_qpos(self) -> np.ndarray:
        return self.robot.get_qpos()[self.n_arm:self.n_dofs].cpu().numpy()

    # ── PD-control execution ────────────────────────────────
    # control_dofs_position(target, dofs_idx_local) tells the
    # built-in controller to drive the selected DOFs to target.

    def _q_limit_tensors(self):
        """Get joint limit tensors (cached)."""
        lims = torch.tensor(np.asarray(self.robot.q_limit), dtype=torch.float32, device=self.robot.get_qpos().device)
        return lims[0], lims[1]  # (lower, upper) (n_dofs,)

    def control_arm_to(self, arm_qpos, steps: int = 300):
        """Drive arm joints toward target via PD controller."""
        target = torch.tensor(list(arm_qpos), dtype=torch.float32, device=self.robot.get_qpos().device)
        lo, hi = self._q_limit_tensors()
        target = torch.clamp(target, lo[:self.n_arm], hi[:self.n_arm])
        self.robot.control_dofs_position(target, dofs_idx_local=self.arm_dofs)
        self.scene.step(steps)

    def control_gripper_to(self, width: float, steps: int = 100):
        """Drive gripper DOFs toward target width."""
        target = torch.tensor([width] * len(self.gripper_dofs), dtype=torch.float32, device=self.robot.get_qpos().device)
        lo, hi = self._q_limit_tensors()
        gripper_idx = torch.tensor(self.gripper_dofs, dtype=torch.long, device=target.device)
        target = torch.clamp(target, lo[gripper_idx], hi[gripper_idx])
        self.robot.control_dofs_position(target, dofs_idx_local=self.gripper_dofs)
        self.scene.step(steps)

    # ── IK via Genesis built-in ──────────────────────────────

    # "Ready" pose init for IK — a slightly bent Franka arm pose.
    # Used only when the arm is at home (all zeros); otherwise current state is used.
    READY_INIT_QPOS = [0.0, 0.0, 0.0, -0.3, 0.0, 0.3, 0.0]

    def _solve_ik(self, x: float, y: float, z: float, quat: Optional[List[float]] = None) -> List[float]:
        """Solve IK for target position (x, y, z).

        Tries Genesis inverse_kinematics first, then verifies the result with FK.
        Falls back to Jacobian IK if Genesis IK result has large FK error (> 2cm).
        """
        device = self.robot.get_qpos().device
        target = np.array([x, y, z])
        init_qpos = self.robot.get_qpos().clone()

        # If arm is near home (all zeros), use ready pose for IK init
        arm_vals = init_qpos[:self.n_arm].abs().sum().item()
        if arm_vals < 0.1:
            for i, val in enumerate(self.READY_INIT_QPOS[:self.n_arm]):
                init_qpos[i] = val

        # Try Genesis IK
        try:
            target_t = torch.tensor([x, y, z], dtype=torch.float32, device=device)
            kwargs = dict(
                link=self.ee_link,
                pos=target_t,
                init_qpos=init_qpos,
                max_samples=500,
                max_solver_iters=100,
                damping=0.01,
                pos_tol=0.0001,
                rot_tol=0.01,
                respect_joint_limit=True,
                return_error=False,
            )
            if quat is not None:
                kwargs["quat"] = torch.tensor(quat, dtype=torch.float32, device=device)
            result = self.robot.inverse_kinematics(**kwargs)
            joints = result[:self.n_arm].tolist()

            # Verify with FK — if error is small, use Genesis IK result
            ee_pos = self._fk_safe(joints)
            error = np.linalg.norm(target - ee_pos)
            if error < 0.02:
                return joints
            # Genesis IK converged but FK mismatch — fall through to Jacobian
        except Exception:
            pass

        # Jacobian IK — FK is always consistent since it uses get_links_pos()
        return self._solve_ik_jacobian(x, y, z)

    def _solve_ik_jacobian(self, x, y, z) -> List[float]:
        """Jacobian-based fallback IK with state-safe FK."""
        target = np.array([x, y, z])
        joints = self._get_arm_qpos().copy()
        for _ in range(100):
            ee_pos = self._fk_safe(joints)
            error = target - ee_pos
            if np.linalg.norm(error) < 0.003:
                break
            J = self._jacobian(joints)
            lam = 0.05
            JJT = J @ J.T + lam**2 * np.eye(3)
            delta = J.T @ np.linalg.solve(JJT, error)
            delta = np.clip(delta, -0.03, 0.03)
            joints = joints + delta
        return joints.tolist()

    def _fk_safe(self, joints):
        """State-safe forward kinematics for Jacobian fallback only."""
        saved = self.robot.get_qpos().clone()
        self._set_arm_qpos_raw(joints)
        links_pos = self.robot.get_links_pos()
        ee_pos = links_pos[self.ee_link_idx].detach().cpu().numpy()
        self.robot.set_qpos(saved)
        return ee_pos

    def _jacobian(self, joints, eps=0.005):
        J = np.zeros((3, self.n_arm))
        for i in range(self.n_arm):
            q_plus = joints.copy(); q_plus[i] += eps
            p_plus = self._fk_safe(q_plus)
            q_minus = joints.copy(); q_minus[i] -= eps
            p_minus = self._fk_safe(q_minus)
            J[:, i] = (p_plus - p_minus) / (2 * eps)
        return J

    # ── Primitive 1: Move ────────────────────────────────────

    def teleport_arm_to(self, target_joints: List[float]):
        """Teleport arm to target joints via set_qpos (instant, no physics interaction).

        Use for large movements where PD convergence is too slow.
        Not suitable for grasp/release (gripper needs PD forces).
        """
        qpos = self.robot.get_qpos().clone()
        for i, val in enumerate(list(target_joints)[:self.n_arm]):
            qpos[i] = float(val)
        self.robot.set_qpos(qpos)

    def move_to(self, target_joints: List[float], steps: int = 300):
        self.control_arm_to(target_joints, steps)

    def move_above_object(self, object_pos: List[float], height: float = 0.10):
        """Move hand to be directly above the object by `height` (default 10 cm).
        Uses teleport for fast, precise positioning.
        """
        joints = self._solve_ik(object_pos[0], object_pos[1], object_pos[2] + height)
        self.teleport_arm_to(joints)

    def move_to_object(self, object_pos: List[float], offset_z: float = 0.0):
        """Move hand to be at object height + offset_z for gripper-finger alignment.
        Uses teleport for fast, precise positioning.
        """
        joints = self._solve_ik(object_pos[0], object_pos[1], object_pos[2] + offset_z)
        self.teleport_arm_to(joints)

    # ── Primitive 2: Open Gripper ────────────────────────────

    def open_gripper(self, width: float = 0.04, steps: int = 100):
        self.control_gripper_to(width, steps)

    # ── Primitive 3: Close Gripper ───────────────────────────

    def close_gripper(self, width: float = 0.0, steps: int = 100, grasp_target_diameter: Optional[float] = None):
        """Gripper close. If grasp_target_diameter given, close fingers to that width
        so they press against the object's sides (good for position-control grips).
        Otherwise close fully (good for force-controlled grips).
        """
        if grasp_target_diameter is not None:
            width = max(width, grasp_target_diameter * 0.95)  # slight squeeze
        self.control_gripper_to(width, steps)

    # ── Primitive 4: Pick ────────────────────────────────────

    # Finger length below the hand link (TCP) — used to offset approach height
    # so fingers reach the object center without the hand overlapping.
    FINGER_LENGTH = 0.05

    def pick(self, object_pos: List[float], steps: int = 500, grasp_diameter: float = 0.034):
        """Pick sequence: open → teleport above → PD descend → close → PD lift.

        Strategy (tested on GPU with Genesis 1.2.2):
          1. Open gripper
          2. Teleport above object (z+0.10) — safe, no collision with object
          3. PD descend to grasp height (z+0.02) — slow approach avoids pushing object
          4. Close gripper with PD control
          5. PD lift (z+0.20)

        grasp_diameter: closed-target width so fingers press against object sides.
        """
        lo, hi = self._q_limit_tensors()
        gripper_idx = torch.tensor(self.gripper_dofs, dtype=torch.long, device=lo.device)
        device = lo.device

        # Open gripper
        self.open_gripper(0.04, steps=100)

        # Teleport above object (safe height, no collision)
        above_joints = self._solve_ik(object_pos[0], object_pos[1], object_pos[2] + 0.10)
        self.teleport_arm_to(above_joints)

        # PD descend to grasp height
        grasp_joints = self._solve_ik(object_pos[0], object_pos[1], object_pos[2] + 0.02)
        arm_target = torch.clamp(
            torch.tensor(grasp_joints, dtype=torch.float32, device=device),
            lo[:self.n_arm], hi[:self.n_arm],
        )
        gripper_open = torch.tensor(
            [0.04] * len(self.gripper_dofs), dtype=torch.float32, device=device,
        )
        for _ in range(steps):
            self.robot.control_dofs_position(arm_target, dofs_idx_local=self.arm_dofs)
            self.robot.control_dofs_position(gripper_open, dofs_idx_local=self.gripper_dofs)
            self.scene.step(1)

        # Close gripper while holding arm position
        gripper_close_target = torch.clamp(
            torch.tensor([grasp_diameter] * len(self.gripper_dofs), dtype=torch.float32, device=device),
            lo[gripper_idx], hi[gripper_idx],
        )
        for _ in range(steps):
            self.robot.control_dofs_position(arm_target, dofs_idx_local=self.arm_dofs)
            self.robot.control_dofs_position(gripper_close_target, dofs_idx_local=self.gripper_dofs)
            self.scene.step(1)

        # Lift via PD control
        lift_joints = self._solve_ik(object_pos[0], object_pos[1], object_pos[2] + 0.20)
        lift_target = torch.clamp(
            torch.tensor(lift_joints, dtype=torch.float32, device=device),
            lo[:self.n_arm], hi[:self.n_arm],
        )
        for _ in range(steps):
            self.robot.control_dofs_position(lift_target, dofs_idx_local=self.arm_dofs)
            self.robot.control_dofs_position(gripper_close_target, dofs_idx_local=self.gripper_dofs)
            self.scene.step(1)

    # ── Primitive 5: Place ───────────────────────────────────

    def place(self, target_pos: List[float], steps: int = 300):
        self.move_above_object(target_pos, height=0.10)
        self.move_to_object(target_pos, offset_z=self.FINGER_LENGTH)
        # Open gripper to release
        self.open_gripper(0.04, steps=steps)
        self.move_above_object(target_pos, height=0.10)

    # ── Pick & Place ─────────────────────────────────────────

    def pick_and_place(self, object_pos: List[float], target_pos: List[float]):
        self.pick(object_pos)
        self.place(target_pos)