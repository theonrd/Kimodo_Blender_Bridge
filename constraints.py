"""
Kimodo Blender Bridge — Constraints
====================================
Translates Blender scene objects (Empties, posed Armatures) into the
Kimodo constraints JSON format that gets sent alongside the text prompt.

Coordinate space conversion
----------------------------
Blender world space (Z-up)  →  Kimodo motion space (Y-up)
  X  (right)    →  X  (right)     same
  Y  (forward)  →  -Z (forward)   Y → -Z  (sign flip — see BVH import axis_forward='-Z')
  Z  (up)       →  Y  (up)        Z →  Y

The sign flip on Y is required because the BVH importer maps Kimodo +Z
back to Blender -Y (axis_forward='-Z' in operators.py).  Negating the
Blender-Y → Kimodo-Z conversion here keeps constraint positions consistent
with the imported animation.

All positions in meters (Blender scenes should use metric units).

Canonicalization
-----------------
Kimodo expects the root to start near (0,0) in XZ.  We auto-offset all
constraint positions by subtracting the XZ of the *earliest root2d* or
*earliest fullbody root* so the user can author constraints anywhere in
the scene.  If no frame-0 constraint exists the model still works — it
just starts from its learned distribution.

Frame mapping
-------------
  kimodo_frame = round((blender_frame - scene.frame_start) * kimodo_fps / blender_fps)
The Kimodo FPS defaults to 30; override it in the constraint settings.
"""

import bpy
import math
import json
import mathutils
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Bone names for SOMASkeleton30, exactly matching Kimodo's bone_order_names_with_parents.
# The INDEX in this list is the slot index that Kimodo reads from local_joints_rot.
# All names exist in somaskel77 (the BVH skeleton) so pose_bones.get() will find them.
SOMA_JOINT_ORDER = [
    "Hips",              # 0  — root
    "Spine1",            # 1
    "Spine2",            # 2
    "Chest",             # 3
    "Neck1",             # 4
    "Neck2",             # 5
    "Head",              # 6
    "Jaw",               # 7
    "LeftEye",           # 8
    "RightEye",          # 9
    "LeftShoulder",      # 10
    "LeftArm",           # 11
    "LeftForeArm",       # 12
    "LeftHand",          # 13
    "LeftHandThumbEnd",  # 14
    "LeftHandMiddleEnd", # 15
    "RightShoulder",     # 16
    "RightArm",          # 17
    "RightForeArm",      # 18
    "RightHand",         # 19
    "RightHandThumbEnd", # 20
    "RightHandMiddleEnd",# 21
    "LeftLeg",           # 22 — hip (not "LeftUpLeg")
    "LeftShin",          # 23 — knee (not "LeftLeg")
    "LeftFoot",          # 24
    "LeftToeBase",       # 25
    "RightLeg",          # 26 — hip
    "RightShin",         # 27 — knee
    "RightFoot",         # 28
    "RightToeBase",      # 29
]

# Parent index for each joint in SOMA_JOINT_ORDER (mirrors SOMASkeleton30.bone_order_names_with_parents).
# -1 = root (no parent).  Used to compute local rotations from world rotations.
SOMA_JOINT_PARENTS = [
    -1,  # 0  Hips        (root)
     0,  # 1  Spine1
     1,  # 2  Spine2
     2,  # 3  Chest
     3,  # 4  Neck1
     4,  # 5  Neck2
     5,  # 6  Head
     6,  # 7  Jaw
     6,  # 8  LeftEye
     6,  # 9  RightEye
     3,  # 10 LeftShoulder
    10,  # 11 LeftArm
    11,  # 12 LeftForeArm
    12,  # 13 LeftHand
    13,  # 14 LeftHandThumbEnd
    13,  # 15 LeftHandMiddleEnd
     3,  # 16 RightShoulder
    16,  # 17 RightArm
    17,  # 18 RightForeArm
    18,  # 19 RightHand
    19,  # 20 RightHandThumbEnd
    19,  # 21 RightHandMiddleEnd
     0,  # 22 LeftLeg
    22,  # 23 LeftShin
    23,  # 24 LeftFoot
    24,  # 25 LeftToeBase
     0,  # 26 RightLeg
    26,  # 27 RightShin
    27,  # 28 RightFoot
    28,  # 29 RightToeBase
]

# Coordinate-change matrix: v_kimodo = _M_BK @ v_blender
# Derived from blender_to_kimodo_pos: (Bx, By, Bz) → (Bx, Bz, -By)
# M_BK = [[1,0,0],[0,0,1],[0,-1,0]] — orthogonal, det=1
# For rotation matrices: R_kimodo = M_BK @ R_blender @ M_BK.T

# Bone names for each end-effector type (somaskel77 / SOMASkeleton30).
EFFECTOR_BONE = {
    'left_hand':  'LeftHand',
    'right_hand': 'RightHand',
    'left_foot':  'LeftFoot',
    'right_foot': 'RightFoot',
}
# Index of each end-effector in SOMA_JOINT_ORDER.
EFFECTOR_IDX = {
    'left_hand': 13, 'right_hand': 19, 'left_foot': 24, 'right_foot': 28,
}
# Fallback rest-pose offset from Hips to each effector in Kimodo Y-up meters
# (used when no source_armature is available to read true offsets from).
# Values approximate an adult SOMA T-pose: arms out sideways at shoulder height,
# feet straight below the hips at ground level (hips ~0.9 m up).
DEFAULT_TPOSE_OFFSETS = {
    'left_hand':  ( 0.75,  0.45, 0.0),
    'right_hand': (-0.75,  0.45, 0.0),
    'left_foot':  ( 0.10, -0.90, 0.0),
    'right_foot': (-0.10, -0.90, 0.0),
}


# ---------------------------------------------------------------------------
# Coordinate conversion helpers
# ---------------------------------------------------------------------------

def blender_to_kimodo_pos(v: mathutils.Vector) -> list[float]:
    """Convert Blender world position (Z-up) to Kimodo Y-up position [x, y, z]."""
    return [v.x, v.z, -v.y]   # X same; Z→Y (up); -Y→Z (sign flip for forward axis)


def blender_to_kimodo_2d(v: mathutils.Vector) -> list[float]:
    """Extract 2D ground-plane position [x, z_kimodo] from Blender world pos."""
    return [v.x, -v.y]   # Blender X→Kimodo X; Blender -Y→Kimodo Z (sign flip)


def quat_to_axis_angle_vec(q: mathutils.Quaternion) -> list[float]:
    """Convert Blender Quaternion to axis-angle 3-vector [ax, ay, az] * angle."""
    q = q.normalized()
    angle = 2.0 * math.acos(max(-1.0, min(1.0, q.w)))
    s = math.sqrt(max(0.0, 1.0 - q.w * q.w))
    if s < 1e-6:
        return [0.0, 0.0, 0.0]
    # Convert axis from Blender to Kimodo space too
    ax = q.x / s
    ay = q.z / s    # Blender Z → Kimodo Y
    az = -q.y / s   # Blender -Y → Kimodo Z (sign flip matches position conversion)
    return [ax * angle, ay * angle, az * angle]


def euler_to_axis_angle_vec(e: mathutils.Euler) -> list[float]:
    """Convert Blender Euler to axis-angle 3-vector."""
    return quat_to_axis_angle_vec(e.to_quaternion())


def heading_from_angle(angle_rad: float) -> list[float]:
    """Kimodo expects heading as [cos(θ), sin(θ)]."""
    return [math.cos(angle_rad), math.sin(angle_rad)]


def heading_angle_for_direction(dx: float, dy: float) -> float:
    """Heading angle (rad) that faces along the direction (dx, dy) in Blender.

    Empirical Kimodo convention (verified against generated motion; the UI
    tooltip claiming "0 = +Y forward" is wrong): θ = atan2(dx, −dy), i.e.
    0° faces Blender −Y, 90° faces +X, 180° faces +Y, 270° faces −X.
    """
    return math.atan2(dx, -dy)


def compute_canonical_rotation(constraint_items) -> "tuple[float, tuple[float, float]]":
    """Rotation (R, pivot) that maps the first root2d path segment onto the
    model's canonical start facing (Blender −Y).

    Kimodo always generates the first frames facing its canonical direction
    (−Y in Blender after BVH import), no matter what the prompt or heading
    constraints say — a different facing has to be steered in over ~1 s of
    "warm-up" frames, and a heading that fights the path direction makes the
    motion degenerate.  Rotating the waypoint positions by R about the first
    waypoint makes the canonical facing coincide with the direction of
    travel, so the character walks forward along the path from frame 1 with
    no warm-up and no heading constraints at all.
    bake_canonical_inverse_rotation() undoes R when the BVH is imported.

    Returns (0.0, (0, 0)) when no direction can be derived (fewer than two
    distinct root2d waypoints).
    """
    root2d = sorted(
        (ci for ci in constraint_items
         if ci.enabled and ci.marker_object and ci.constraint_type == 'root2d'),
        key=lambda ci: ci.frame,
    )
    if len(root2d) < 2:
        return 0.0, (0.0, 0.0)
    p0 = root2d[0].marker_object.location
    for ci in root2d[1:]:
        p1 = ci.marker_object.location
        dx, dy = p1.x - p0.x, p1.y - p0.y
        if math.hypot(dx, dy) > 1e-6:
            # φ = direction-of-travel angle; canonical facing −Y is φ = −90°.
            return math.radians(-90.0) - math.atan2(dy, dx), (p0.x, p0.y)
    return 0.0, (0.0, 0.0)


# ---------------------------------------------------------------------------
# Frame conversion
# ---------------------------------------------------------------------------

def blender_frame_to_kimodo(
    blender_frame: int,
    scene_start: int,
    blender_fps: float,
    kimodo_fps: float = 30.0,
) -> int:
    """Convert a Blender frame number to a 0-based Kimodo frame index."""
    elapsed_sec = (blender_frame - scene_start) / blender_fps
    return max(0, round(elapsed_sec * kimodo_fps))


# ---------------------------------------------------------------------------
# Pose reading helpers
# ---------------------------------------------------------------------------

def _rot3_to_axis_angle(m: mathutils.Matrix) -> list[float]:
    """3×3 rotation matrix (already in Kimodo Y-up space) → axis-angle 3-vector."""
    q = m.to_quaternion().normalized()
    angle = 2.0 * math.acos(max(-1.0, min(1.0, q.w)))
    s = math.sqrt(max(0.0, 1.0 - q.w * q.w))
    if s < 1e-6:
        return [0.0, 0.0, 0.0]
    return [q.x / s * angle, q.y / s * angle, q.z / s * angle]


def get_armature_joint_rots(
    armature_obj: bpy.types.Object,
    joint_order: list[str],
) -> list[list[float]]:
    """Extract SOMASkeleton30 local joint rotations from a posed armature.

    SMPL/Kimodo defines per-joint local rotations via:
        G[i] = G[parent[i]] @ R[i]
    where G[i] is the bone's cumulative world rotation (relative to the
    canonical T-pose where every G is identity) and R[i] is the local rotation
    stored in ``local_joints_rot``.

    The previous implementation read ``pb.rotation_euler`` directly and assumed
    the resulting matrix was already R[i] in Kimodo's Y-up basis.  That holds
    only when each Blender bone's rest orientation is identity — but Blender's
    BVH importer builds bones whose rest matrices encode the head→tail
    direction, so most bones have non-trivial rests (e.g. shoulders at ~−82°
    around Z, hips flipped 156° around X).  Treating the bone-local Euler as a
    Kimodo-frame rotation produced axis-swapped output on those bones.

    This implementation is rest-orientation-invariant:

      1. For each bone, compute its pose-vs-rest delta in armature (Blender
         Z-up) space:  delta = pose_arm @ rest_arm⁻¹.  This is the physical
         rotation the bone underwent in world space and is independent of how
         Blender chose to orient the bone at rest.
      2. Conjugate the delta into Kimodo's Y-up basis (M_BK · delta · M_BK⁻¹)
         to get G[i] in Kimodo world coordinates.
      3. Recover the local rotation as R[i] = G[parent]⁻¹ @ G[i].
    """
    pose_bones = armature_obj.pose.bones

    # Blender Z-up → Kimodo Y-up basis change (see module header).
    # Vectors:  (x, y, z)_blender → (x, z, -y)_kimodo
    M_BK = mathutils.Matrix(((1, 0, 0), (0, 0, 1), (0, -1, 0)))
    M_KB = M_BK.transposed()

    # G[i]: bone's pose-relative-to-rest rotation, in Kimodo Y-up basis.
    G_kimodo: dict[str, mathutils.Matrix] = {}
    for name in joint_order:
        pb = pose_bones.get(name)
        if pb is None:
            G_kimodo[name] = mathutils.Matrix.Identity(3)
            continue
        rest_arm = pb.bone.matrix_local.to_3x3()
        pose_arm = pb.matrix.to_3x3()
        # rest is a pure rotation, so .transposed() == .inverted()
        delta_arm = pose_arm @ rest_arm.transposed()
        G_kimodo[name] = M_BK @ delta_arm @ M_KB

    result: list[list[float]] = []
    for i, name in enumerate(joint_order):
        G_i = G_kimodo.get(name, mathutils.Matrix.Identity(3))
        parent_idx = SOMA_JOINT_PARENTS[i] if i < len(SOMA_JOINT_PARENTS) else -1
        if parent_idx < 0:
            R_local = G_i
        else:
            parent_name = joint_order[parent_idx]
            G_parent = G_kimodo.get(parent_name, mathutils.Matrix.Identity(3))
            R_local = G_parent.transposed() @ G_i
        result.append(_rot3_to_axis_angle(R_local))

    return result


def get_root_position(armature_obj: bpy.types.Object) -> list[float]:
    """Get hip/root world position from armature → Kimodo [x,y,z].

    Reads the Hips bone head in world space so the result is correct
    regardless of where the armature object's origin sits (BVH imports
    always place the armature origin at the scene origin; the Hips bone
    moves via keyframes/pose, not via the object location).
    """
    for bone_name in ("Hips", "hips", "Hip", "pelvis", "Pelvis"):
        pb = armature_obj.pose.bones.get(bone_name)
        if pb:
            world_pos = armature_obj.matrix_world @ pb.head
            return blender_to_kimodo_pos(world_pos)
    return blender_to_kimodo_pos(armature_obj.location)


def get_bone_world_position(
    armature_obj: bpy.types.Object,
    bone_name: str,
) -> list[float] | None:
    """World position of a specific pose bone → Kimodo [x,y,z]."""
    pb = armature_obj.pose.bones.get(bone_name)
    if not pb:
        return None
    # head world position
    world_pos = armature_obj.matrix_world @ pb.head
    return blender_to_kimodo_pos(world_pos)


def get_effector_tpose_offset(
    scene: bpy.types.Scene,
    effector_type: str,
) -> tuple[float, float, float]:
    """Rest-pose offset from Hips to the named end-effector, in Kimodo Y-up meters.

    Reads bone rest positions from scene.kimodo.source_armature when present so
    the offset matches the actual character proportions. Falls back to a
    hard-coded adult SOMA T-pose when no source armature is set.
    """
    arm = getattr(getattr(scene, "kimodo", None), "source_armature", None)
    if arm and arm.type == 'ARMATURE':
        bones = arm.data.bones
        hips = (bones.get("Hips") or bones.get("hips")
                or bones.get("Hip") or bones.get("pelvis") or bones.get("Pelvis"))
        eff = bones.get(EFFECTOR_BONE[effector_type])
        if hips and eff:
            d = eff.head_local - hips.head_local   # Blender Z-up local space
            return (d.x, d.z, -d.y)                # → Kimodo Y-up
    return DEFAULT_TPOSE_OFFSETS[effector_type]


def get_bone_world_rotation(
    armature_obj: bpy.types.Object,
    bone_name: str,
) -> list[float]:
    """World rotation of a specific pose bone → axis-angle 3-vector (Kimodo space)."""
    pb = armature_obj.pose.bones.get(bone_name)
    if not pb:
        return [0.0, 0.0, 0.0]
    world_mat = armature_obj.matrix_world @ pb.matrix
    q = world_mat.to_quaternion()
    return quat_to_axis_angle_vec(q)


# ---------------------------------------------------------------------------
# Constraint JSON builder
# ---------------------------------------------------------------------------

def build_constraints_json(
    constraint_items,           # iterable of KIMODO_ConstraintItem
    scene: bpy.types.Scene,
    kimodo_fps: float = 30.0,
    auto_canonicalize: bool = True,
    scene_start_override: "int | None" = None,
    canonical_rotation: bool = False,
) -> list[dict]:
    """
    Convert Blender constraint items to Kimodo constraints JSON list.

    Parameters
    ----------
    constraint_items : iterable of KIMODO_ConstraintItem PropertyGroup entries
    scene            : bpy.context.scene
    kimodo_fps       : Kimodo's motion FPS (default 30)
    auto_canonicalize: subtract XZ of earliest root waypoint so it lands at (0,0)
                       (ignored while canonical_rotation is active and applied)
    canonical_rotation: rotate root2d waypoints about the first one so the
                       direction of travel matches the model's canonical start
                       facing — the character walks forward from frame 1 with
                       no heading warm-up (see compute_canonical_rotation).
                       Only applies when every enabled constraint is root2d.

    Returns
    -------
    list of dicts ready to be json.dumps()-ed and sent to Kimodo
    """
    blender_fps = scene.render.fps / scene.render.fps_base
    scene_start = scene.frame_start if scene_start_override is None else scene_start_override

    # Collect enabled items sorted by frame
    items = sorted(
        [ci for ci in constraint_items if ci.enabled and ci.marker_object],
        key=lambda ci: ci.frame,
    )
    if not items:
        return []

    # Save scene frame — _evaluate_frame mutates it as a side effect.
    saved_frame = scene.frame_current

    try:
        # -----------------------------------------------------------------------
        # Canonical path rotation: rotate all root2d waypoints so the direction
        # of travel matches the model's canonical start facing (Blender −Y)
        # AND the first waypoint lands at the model's (0,0) origin — Kimodo is
        # trained on root-near-origin data, and a path offset by several
        # meters degrades the gait (swaying path, duck-footed steps).
        # The inverse (rotate + translate back) is baked into the imported
        # action (operators._apply_canonical_from_sidecar), so the motion
        # lands exactly on the authored world-space waypoints.
        # Only defined when every enabled constraint is a root2d waypoint —
        # fullbody/effector constraints encode pose orientations that would
        # have to rotate too, so those runs are left untouched.
        # -----------------------------------------------------------------------
        rot_rad = 0.0
        pivot = (0.0, 0.0)
        if (canonical_rotation
                and all(ci.constraint_type == 'root2d' for ci in items)):
            rot_rad, pivot = compute_canonical_rotation(items)
            if rot_rad != 0.0:
                # The transform below already centers the path on the model
                # origin, which is what auto-canonicalize would want; running
                # the offset on top would only move the path away from where
                # the inverse bake expects it — so canonicalize yields.
                auto_canonicalize = False

        def _rotated_marker_xy(obj):
            # Model-space ground coords: rotate the offset from the first
            # waypoint about the origin (direction fix) — the first waypoint
            # itself maps to (0, 0).
            dx, dy = obj.location.x - pivot[0], obj.location.y - pivot[1]
            if abs(rot_rad) > 1e-9:
                c, s = math.cos(rot_rad), math.sin(rot_rad)
                dx, dy = c * dx - s * dy, s * dx + c * dy
            return dx, dy

        # -----------------------------------------------------------------------
        # Auto-canonicalization: find the earliest XZ root position and use it
        # as the origin offset so the user can author constraints anywhere.
        # -----------------------------------------------------------------------
        origin_offset = [0.0, 0.0]  # [x_offset, z_offset] in Kimodo space
        if auto_canonicalize:
            for ci in items:
                ctype = ci.constraint_type
                if ctype == 'root2d':
                    pos2d = blender_to_kimodo_2d(ci.marker_object.location)
                    origin_offset = pos2d
                    break
                elif ctype == 'fullbody':
                    # Must evaluate the frame before reading the Hips bone position.
                    _evaluate_frame(scene, ci.frame)
                    pos3d = get_root_position(ci.marker_object)
                    origin_offset = [pos3d[0], pos3d[2]]
                    break

        def apply_offset_2d(pos2d):
            return [pos2d[0] - origin_offset[0], pos2d[1] - origin_offset[1]]

        def apply_offset_3d(pos3d):
            return [pos3d[0] - origin_offset[0], pos3d[1], pos3d[2] - origin_offset[1]]

        # -----------------------------------------------------------------------
        # Group ALL items of the same type into one block (not just consecutive).
        # Kimodo expects one block per constraint type with all frame_indices.
        # -----------------------------------------------------------------------
        type_order: list[str] = []
        type_to_items: dict[str, list] = {}
        for ci in items:
            ct = ci.constraint_type
            if ct not in type_to_items:
                type_order.append(ct)
                type_to_items[ct] = []
            type_to_items[ct].append(ci)

        out_constraints = []

        for ctype in type_order:
            group = type_to_items[ctype]
            frame_indices: list[int] = []
            smooth_root_2d: list = []
            root_positions: list = []
            local_joints_rot: list = []
            global_root_heading: list = []
            heading_slots: "list[float | None]" = []

            for ci in group:
                kframe = blender_frame_to_kimodo(ci.frame, scene_start, blender_fps, kimodo_fps)
                frame_indices.append(kframe)
                obj = ci.marker_object

                if ctype == 'root2d':
                    x, y = _rotated_marker_xy(obj)
                    pos2d = [x, -y]
                    smooth_root_2d.append(apply_offset_2d(pos2d))
                    heading_slots.append(
                        ci.heading_angle + rot_rad if ci.include_heading else None)

                elif ctype == 'fullbody':
                    if obj.type == 'ARMATURE':
                        # Evaluate the frame FIRST so Hips bone is at the right
                        # world position (BVH armatures animate via bone keyframes,
                        # not via the object location, so position must be sampled
                        # at the correct frame).
                        _evaluate_frame(scene, ci.frame)
                        pos3d = get_root_position(obj)
                        root_positions.append(apply_offset_3d(pos3d))
                        jrot = get_armature_joint_rots(obj, SOMA_JOINT_ORDER)
                        local_joints_rot.append(jrot)
                        pos2d = [pos3d[0], pos3d[2]]
                        smooth_root_2d.append(apply_offset_2d(pos2d))
                    else:
                        # Empty used for fullbody — just root position, identity pose.
                        pos3d = blender_to_kimodo_pos(obj.location)
                        root_positions.append(apply_offset_3d(pos3d))
                        pos2d = [pos3d[0], pos3d[2]]
                        smooth_root_2d.append(apply_offset_2d(pos2d))
                        local_joints_rot.append([[0.0, 0.0, 0.0]] * len(SOMA_JOINT_ORDER))

                elif ctype in ('left_hand', 'right_hand', 'left_foot', 'right_foot'):
                    # Kimodo derives the effector target from FK on (root_positions,
                    # local_joints_rot) — root_positions is the HIPS, NOT the hand.
                    eff_idx = EFFECTOR_IDX[ctype]

                    if obj.type == 'ARMATURE':
                        # Armature marker: read hips + full pose like fullbody.
                        # FK of this pose naturally places the effector wherever
                        # the user posed it.
                        _evaluate_frame(scene, ci.frame)
                        hips_pos = get_root_position(obj)
                        jrot = get_armature_joint_rots(obj, SOMA_JOINT_ORDER)
                    else:
                        # Empty marker: its location is the *target* end-effector
                        # position. Pick hips such that a T-pose places the
                        # effector exactly there, and use a T-pose (all zero
                        # rotations) for the body. The Empty's rotation is used
                        # as the effector's orientation.
                        target = blender_to_kimodo_pos(obj.location)
                        ox, oy, oz = get_effector_tpose_offset(scene, ctype)
                        hips_pos = [target[0] - ox, target[1] - oy, target[2] - oz]
                        jrot = [[0.0, 0.0, 0.0] for _ in SOMA_JOINT_ORDER]
                        # In T-pose all ancestors are identity, so the effector's
                        # local rotation equals its world rotation.
                        jrot[eff_idx] = quat_to_axis_angle_vec(
                            obj.matrix_world.to_quaternion()
                        )

                    pos3d = apply_offset_3d(hips_pos)
                    root_positions.append(pos3d)
                    smooth_root_2d.append([pos3d[0], pos3d[2]])
                    local_joints_rot.append(jrot)

            if ctype == 'root2d':
                if rot_rad != 0.0 and any(h is not None for h in heading_slots):
                    # With the canonical rotation active, headings must cover
                    # every root2d frame or Kimodo drops the whole list (the
                    # length guard below).  Fill missing ones from the local
                    # direction of the rotated path, so forgetting to tick
                    # "Include Heading" on one waypoint can't silently
                    # disable the headings on all of them.
                    n = len(heading_slots)
                    geom = [0.0] * n
                    last_geom = 0.0
                    for i in range(n):
                        if i + 1 < n:
                            a, b = smooth_root_2d[i], smooth_root_2d[i + 1]
                        elif i > 0:
                            a, b = smooth_root_2d[i - 1], smooth_root_2d[i]
                        else:
                            a = b = None
                        if a is not None:
                            ddx, ddz = b[0] - a[0], b[1] - a[1]
                            # θ = atan2(dx, dz) in Kimodo ground coords — the
                            # same convention as heading_angle_for_direction
                            # (Kimodo Z = −Blender Y).  Segments under 5 cm
                            # ("stand here" pairs) carry no direction — skip
                            # them so a degenerate delta can't poison the
                            # carried-over angle.
                            if math.hypot(ddx, ddz) > 0.05:
                                last_geom = math.atan2(ddx, ddz)
                        geom[i] = last_geom
                    global_root_heading = [
                        heading_from_angle(h if h is not None else geom[i])
                        for i, h in enumerate(heading_slots)
                    ]
                else:
                    global_root_heading = [
                        heading_from_angle(h) for h in heading_slots if h is not None
                    ]

            block: dict[str, Any] = {
                "type": ctype.replace("_", "-"),  # left_hand → left-hand
                "frame_indices": frame_indices,
            }

            if smooth_root_2d:
                block["smooth_root_2d"] = smooth_root_2d
            if root_positions:
                block["root_positions"] = root_positions
            if local_joints_rot:
                block["local_joints_rot"] = local_joints_rot
            if global_root_heading and len(global_root_heading) == len(frame_indices):
                block["global_root_heading"] = global_root_heading

            out_constraints.append(block)

        return out_constraints

    finally:
        # Restore the frame so constraint-building has no visible side effect.
        scene.frame_set(saved_frame)
        bpy.context.view_layer.update()


def _evaluate_frame(scene: bpy.types.Scene, frame: int):
    """Move timeline to a frame and update the depsgraph (for pose sampling)."""
    scene.frame_set(frame)
    bpy.context.view_layer.update()


def constraints_to_json_string(
    constraint_items,
    scene: bpy.types.Scene,
    kimodo_fps: float = 30.0,
    auto_canonicalize: bool = True,
    indent: int = 2,
    canonical_rotation: bool = False,
) -> str:
    """Return the constraints as a formatted JSON string."""
    data = build_constraints_json(
        constraint_items, scene, kimodo_fps, auto_canonicalize,
        canonical_rotation=canonical_rotation)
    return json.dumps(data, indent=indent)


# ---------------------------------------------------------------------------
# Canonical rotation bake (import side)
# ---------------------------------------------------------------------------

def bake_canonical_inverse_rotation(
    context,
    armature_obj: bpy.types.Object,
    rot_rad: float,
    pivot_xy: "tuple[float, float]",
    mode: str = "origin",
) -> bool:
    """Undo the canonical path rotation on an imported BVH action.

    The canonical rotation (compute_canonical_rotation) is applied to the
    constraint JSON only, so the BVH Kimodo generates is in rotated model
    coordinates.  This bakes the inverse rigid transform into the root
    bone's keyframes by rewriting each frame's armature-space pose matrix:
    the action lands back on the authored world-space waypoints and the
    armature object keeps an identity transform, so retargeting behaves
    exactly as with any other import.

    Modes (matching how the constraints were sent):
      * "origin" — p_model = Rot(R)·(p − pivot); inverse is
        p = Rot(−R)·p_model + pivot.  The model saw the first waypoint at
        its (0,0) origin; rotate about the action origin, then translate.
      * "pivot"  — legacy: p_model = pivot + Rot(R)·(p − pivot); inverse is
        a rotation about the pivot point inside model space.

    Only the root bone is touched — every other bone animates in its
    parent's local space, so rotating the root rotates the whole chain.

    Idempotent: the action is marked with a custom property so repeat
    imports of the same file don't rotate twice.
    """
    if abs(rot_rad) < 1e-9:
        return False
    ad = armature_obj.animation_data
    act = ad.action if ad else None
    if act is None or act.get("kimodo_canonical_unrotated"):
        return False

    root_pb = next((pb for pb in armature_obj.pose.bones if pb.parent is None), None)
    if root_pb is None:
        return False

    scene = context.scene
    saved_frame = scene.frame_current

    # BVH action coordinates ARE the model-space coordinates the constraints
    # were sent in (the importer builds the armature at the origin), so the
    # pivot is used directly in armature space.  Deliberately NOT mapped
    # through matrix_world: a moved or rotated source object changes where
    # the motion appears in the world, but the action's own coordinates stay
    # authored-space — and retarget constraints may be pointing at the
    # object, so we must not "compensate" for its transform here.
    pivot_arm = mathutils.Vector((pivot_xy[0], pivot_xy[1], 0.0))
    rot_back = mathutils.Matrix.Rotation(-rot_rad, 4, 'Z')
    if mode == "origin":
        M = mathutils.Matrix.Translation(pivot_arm) @ rot_back
    else:
        M = (
            mathutils.Matrix.Translation(pivot_arm)
            @ rot_back
            @ mathutils.Matrix.Translation(-pivot_arm)
        )

    rot_path = {
        'QUATERNION': 'rotation_quaternion',
        'AXIS_ANGLE': 'rotation_axis_angle',
    }.get(root_pb.rotation_mode, 'rotation_euler')

    f_start, f_end = act.frame_range
    try:
        for frame in range(int(math.floor(f_start)), int(math.ceil(f_end)) + 1):
            scene.frame_set(frame)
            context.view_layer.update()
            root_pb.matrix = M @ root_pb.matrix
            root_pb.keyframe_insert(data_path='location', frame=frame)
            root_pb.keyframe_insert(data_path=rot_path, frame=frame)
    finally:
        scene.frame_set(saved_frame)
        context.view_layer.update()

    act["kimodo_canonical_unrotated"] = True
    return True
