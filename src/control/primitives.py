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
        self.ee_link = self._find_link("hand", "link7", "panda_hand")
        self.ee_link_idx = self.ee_link.idx if self.ee_link is not None else 10

        # Cache computed IK targets to avoid recomputation
        self._ik_cache = {}

    def _find_link(self, *candidates):
        names = {ln.name: ln for ln in self.robot.links}
        for c in candidates:
            if c in names:
                return names[c]
        return list(names.values())[-1]  # last link as fallback

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

    def control_arm_to(self, arm_qpos, steps: int = 60):
        """Drive arm joints toward target via PD controller."""
        target = torch.tensor(list(arm_qpos), dtype=torch.float32, device=self.robot.get_qpos().device)
        lo, hi = self._q_limit_tensors()
        target = torch.clamp(target, lo[:self.n_arm], hi[:self.n_arm])
        self.robot.control_dofs_position(target, dofs_idx_local=self.arm_dofs)
        self.scene.step(steps)

    def control_gripper_to(self, width: float, steps: int = 20):
        """Drive gripper DOFs toward target width."""
        target = torch.tensor([width] * len(self.gripper_dofs), dtype=torch.float32, device=self.robot.get_qpos().device)
        lo, hi = self._q_limit_tensors()
        gripper_idx = torch.tensor(self.gripper_dofs, dtype=torch.long, device=target.device)
        target = torch.clamp(target, lo[gripper_idx], hi[gripper_idx])
        self.robot.control_dofs_position(target, dofs_idx_local=self.gripper_dofs)
        self.scene.step(steps)

    # ── IK via Genesis built-in ──────────────────────────────

    # "Ready" pose init for IK — a moderately bent Franka arm pose that makes
    # Genesis IK converge reliably. The all-zeros home pose makes IK unstable.
    READY_INIT_QPOS = [0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0]

    def _solve_ik(self, x: float, y: float, z: float, quat: Optional[List[float]] = None) -> List[float]:
        """Use Genesis inverse_kinematics on the GPU.

        quat: optional target orientation (w,x,y,z). If None, position-only.
        Returns target qpos for the first n_arm DOFs.

        Uses a "ready" pose as IK init (instead of all-zeros) for reliable
        convergence — all-zeros init is at home pose far from cup workspace,
        causing IK to fail silently with 20+ cm errors.
        """
        device = self.robot.get_qpos().device
        target = torch.tensor([x, y, z], dtype=torch.float32, device=device)
        # Build init_qpos: ready arm pose + current gripper state
        current_qpos = self.robot.get_qpos()
        init_qpos = current_qpos.clone()
        for i, val in enumerate(self.READY_INIT_QPOS[:self.n_arm]):
            init_qpos[i] = val
        try:
            kwargs = dict(
                link=self.ee_link,
                pos=target,
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
            if hasattr(result, 'shape') and len(result.shape) > 0 and result.shape[-1] >= self.n_arm:
                return result[:self.n_arm].tolist()
            return result.tolist() if hasattr(result, 'tolist') else list(result)
        except Exception as e:
            print(f"    [IK] Genesis IK failed: {e}; using Jacobian fallback")
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

    def move_to(self, target_joints: List[float], steps: int = 60):
        self.control_arm_to(target_joints, steps)

    def move_above_object(self, object_pos: List[float], height: float = 0.10):
        """Move hand to be directly above the object by `height` (default 10 cm).
        Uses top-down approach orientation (gripper pointing down).
        """
        # Position-only IK: let the optimizer keep the orientation close to init
        # (avoids unreachable orientation constraints).
        joints = self._solve_ik(object_pos[0], object_pos[1], object_pos[2] + height)
        self.move_to(joints)

    def move_to_object(self, object_pos: List[float], offset_z: float = 0.04):
        """Move hand to be at object height + offset_z for gripper-finger alignment.

        offset_z is the height of the 'hand' link relative to the object CENTER,
        chosen so the fingers hang down past the object's midsection.
        Default 0.04 = finger length ~0.045 + slight clearance.
        """
        joints = self._solve_ik(object_pos[0], object_pos[1], object_pos[2] + offset_z)
        self.move_to(joints)

    # ── Primitive 2: Open Gripper ────────────────────────────

    def open_gripper(self, width: float = 0.04, steps: int = 20):
        self.control_gripper_to(width, steps)

    # ── Primitive 3: Close Gripper ───────────────────────────

    def close_gripper(self, width: float = 0.0, steps: int = 30, grasp_target_diameter: Optional[float] = None):
        """Gripper close. If grasp_target_diameter given, close fingers to that width
        so they press against the object's sides (good for position-control grips).
        Otherwise close fully (good for force-controlled grips).
        """
        if grasp_target_diameter is not None:
            width = max(width, grasp_target_diameter * 0.95)  # slight squeeze
        self.control_gripper_to(width, steps)

    # ── Primitive 4: Pick ────────────────────────────────────

    def pick(self, object_pos: List[float], steps: int = 60, grasp_diameter: float = 0.034):
        """Pick sequence: open → approach → descend → close(grip object) → lift.

        grasp_diameter sets the closed-target width so fingers press object's sides
        instead of fully closing on empty space. Default 0.034 ≈ our cylinder cup radius * 2 * 0.6.
        """
        self.open_gripper(0.04, steps)
        self.move_above_object(object_pos, height=0.10)
        self.move_to_object(object_pos, offset_z=0.04)
        # Hold arm at approach pose while closing gripper against object's diameter
        lo, hi = self._q_limit_tensors()
        gripper_idx = torch.tensor(self.gripper_dofs, dtype=torch.long, device=lo.device)
        gripper_close_target = torch.clamp(
            torch.tensor([grasp_diameter] * len(self.gripper_dofs), dtype=torch.float32, device=lo.device),
            lo[gripper_idx], hi[gripper_idx],
        )
        cur_arm = self._get_arm_qpos()
        arm_target = torch.clamp(
            torch.tensor(list(cur_arm), dtype=torch.float32, device=lo.device),
            lo[:self.n_arm], hi[:self.n_arm],
        )
        # Hold arm steady while gripping
        for _ in range(60):
            self.robot.control_dofs_position(arm_target, dofs_idx_local=self.arm_dofs)
            self.robot.control_dofs_position(gripper_close_target, dofs_idx_local=self.gripper_dofs)
            self.scene.step(1)
        # Lift with gripper still closed (keep applying close command during lift)
        lift_joints = self._solve_ik(object_pos[0], object_pos[1], object_pos[2] + 0.10)
        lift_target = torch.clamp(
            torch.tensor(lift_joints, dtype=torch.float32, device=lo.device),
            lo[:self.n_arm], hi[:self.n_arm],
        )
        for _ in range(80):
            self.robot.control_dofs_position(lift_target, dofs_idx_local=self.arm_dofs)
            self.robot.control_dofs_position(gripper_close_target, dofs_idx_local=self.gripper_dofs)
            self.scene.step(1)

    # ── Primitive 5: Place ───────────────────────────────────

    def place(self, target_pos: List[float], steps: int = 60):
        self.move_above_object(target_pos, height=0.10)
        self.move_to_object(target_pos, offset_z=0.02)
        # Open gripper to release
        self.open_gripper(0.04, steps=30)
        self.move_above_object(target_pos, height=0.10)

    # ── Pick & Place ─────────────────────────────────────────

    def pick_and_place(self, object_pos: List[float], target_pos: List[float]):
        self.pick(object_pos)
        self.place(target_pos)