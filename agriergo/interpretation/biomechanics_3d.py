"""
3D Biomechanical Skeleton & Lumbar Spine (L5/S1) Stress Engine.

Provides:
1. 2.5D / 3D anthropometric kinematic lifting from 2D COCO keypoints.
2. Biomechanically validated L5/S1 vertebral disc compression & shear force calculation
   (based on Chaffin & Andersson's biomechanical model & NIOSH manual handling limits).
3. Regional spinal stress heatmap (Cervical, Thoracic, Lumbar, Sacral L5/S1).
4. Plotly 3D interactive rotatable skeleton visualization.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

from config.settings import (
    KP_NOSE, KP_LEFT_EYE, KP_RIGHT_EYE,
    KP_LEFT_EAR, KP_RIGHT_EAR,
    KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER,
    KP_LEFT_ELBOW, KP_RIGHT_ELBOW,
    KP_LEFT_WRIST, KP_RIGHT_WRIST,
    KP_LEFT_HIP, KP_RIGHT_HIP,
    KP_LEFT_KNEE, KP_RIGHT_KNEE,
    KP_LEFT_ANKLE, KP_RIGHT_ANKLE,
    NIOSH_COMPRESSION_ACTION_LIMIT,
    NIOSH_COMPRESSION_MAX_PERMISSIBLE,
    DEFAULT_WORKER_MASS_KG,
    L5S1_ERECTOR_SPINAE_MOMENT_ARM,
    TRUNK_CENTER_OF_MASS_RATIO,
    TRUNK_COM_LENGTH_METERS,
)


@dataclass
class Point3D:
    """A single anatomical joint position in 3D coordinate space."""
    x: float
    y: float
    z: float
    confidence: float = 1.0
    name: str = ""

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)


@dataclass
class SpineStress3D:
    """Biomechanical stress metrics along the vertebral spinal column."""
    l5s1_compression_n: float            # L5/S1 compressive force in Newtons
    l5s1_shear_n: float                  # L5/S1 anterior-posterior shear force in Newtons
    cervical_stress: float               # 0.0 – 1.0 normalized cervical spine stress
    thoracic_stress: float               # 0.0 – 1.0 normalized thoracic spine stress
    lumbar_stress: float                 # 0.0 – 1.0 normalized lumbar spine stress
    risk_level: str                      # "Safe", "Moderate", "Action Limit Exceeded", "Hazardous"
    action_required: str                 # Clinical/Ergonomic guidance
    color_hex: str                       # Hex color code representing risk tier
    action_limit_ratio: float            # Ratio of compression relative to NIOSH 3400 N limit


@dataclass
class Skeleton3D:
    """Complete 3D anatomical human skeleton with labeled joint coordinates."""
    joints: Dict[str, Point3D] = field(default_factory=dict)
    trunk_angle: float = 0.0
    neck_angle: float = 0.0
    knee_angle: float = 180.0
    center_of_mass: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Standard anatomical wireframe bone segments
    BONES: Tuple[Tuple[str, str], ...] = (
        ("head", "neck_c7"),
        ("neck_c7", "mid_spine_t12"),
        ("mid_spine_t12", "lumbar_l5s1"),
        ("lumbar_l5s1", "pelvis"),
        ("neck_c7", "left_shoulder"),
        ("neck_c7", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("pelvis", "left_hip"),
        ("pelvis", "right_hip"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
    )


def compute_l5s1_spinal_compression(
    trunk_flexion_deg: float,
    load_kg: float = 0.0,
    worker_mass_kg: float = DEFAULT_WORKER_MASS_KG,
    neck_flexion_deg: float = 0.0,
) -> SpineStress3D:
    """
    Compute L5/S1 spinal compressive and shear forces using Chaffin's biomechanical model.

    Mathematical formulation:
      W_upper = (TRUNK_CENTER_OF_MASS_RATIO * worker_mass_kg) * 9.81  [N]
      W_load  = (load_kg * 9.81)                                      [N]
      d_upper = TRUNK_COM_LENGTH_METERS * sin(rad(trunk_flexion))      [m]
      d_load  = 0.35 * sin(rad(trunk_flexion)) + 0.15                  [m]
      M_L5S1  = (W_upper * d_upper) + (W_load * d_load)               [N*m]
      F_muscle = M_L5S1 / L5S1_ERECTOR_SPINAE_MOMENT_ARM              [N]
      F_comp  = (W_upper + W_load) * cos(rad(trunk_flexion)) + F_muscle [N]
      F_shear = (W_upper + W_load) * sin(rad(trunk_flexion))          [N]

    Args:
        trunk_flexion_deg: Trunk flexion angle in degrees from vertical (0-90°).
        load_kg: Carried or handled external load in kilograms.
        worker_mass_kg: Worker body mass in kg (default 65 kg).
        neck_flexion_deg: Neck flexion angle in degrees (0-90°).

    Returns:
        SpineStress3D object with compression, shear, regional stress, and NIOSH status.
    """
    trunk_rad = np.radians(max(0.0, min(100.0, trunk_flexion_deg)))
    neck_rad = np.radians(max(0.0, min(90.0, neck_flexion_deg)))

    # Upper body gravitational force
    w_upper = (TRUNK_CENTER_OF_MASS_RATIO * worker_mass_kg) * 9.81
    w_load = load_kg * 9.81

    # Moment arms relative to L5/S1 disc axis
    d_upper = TRUNK_COM_LENGTH_METERS * np.sin(trunk_rad)
    # Load moment arm increases with forward reach
    d_load = 0.35 * np.sin(trunk_rad) + 0.15

    # Net moment around L5/S1 joint center (N*m)
    m_l5s1 = (w_upper * d_upper) + (w_load * d_load)

    # Erector spinae back extensor muscle counter-force
    f_muscle = m_l5s1 / L5S1_ERECTOR_SPINAE_MOMENT_ARM

    # Compressive force perpendicular to L5/S1 disc surface
    f_comp = ((w_upper + w_load) * np.cos(trunk_rad)) + f_muscle

    # Anterior-posterior shear force parallel to disc surface
    f_shear = (w_upper + w_load) * np.sin(trunk_rad)

    # Regional stress fractions (0.0 to 1.0)
    cervical_stress = float(np.clip(neck_flexion_deg / 45.0, 0.05, 1.0))
    thoracic_stress = float(np.clip(trunk_flexion_deg / 55.0, 0.08, 1.0))
    lumbar_stress = float(np.clip(f_comp / NIOSH_COMPRESSION_ACTION_LIMIT, 0.1, 1.5))

    action_ratio = f_comp / NIOSH_COMPRESSION_ACTION_LIMIT

    # Categorize against NIOSH standards
    if f_comp < 2000.0:
        risk = "Safe"
        action = "Acceptable spinal load; maintain dynamic mobility."
        color = "#10B981"  # Emerald Green
    elif f_comp < NIOSH_COMPRESSION_ACTION_LIMIT:
        risk = "Moderate"
        action = "Sub-critical spinal strain; limit prolonged static stoop."
        color = "#F59E0B"  # Amber
    elif f_comp < NIOSH_COMPRESSION_MAX_PERMISSIBLE:
        risk = "Action Limit Exceeded"
        action = "Exceeds NIOSH 3400 N limit! Immediate posture correction or tool aid required."
        color = "#EF4444"  # Red
    else:
        risk = "Hazardous"
        action = "Critical vertebral injury risk (>6400 N)! Cease manual task immediately."
        color = "#8B5CF6"  # Purple

    return SpineStress3D(
        l5s1_compression_n=round(float(f_comp), 1),
        l5s1_shear_n=round(float(f_shear), 1),
        cervical_stress=round(cervical_stress, 2),
        thoracic_stress=round(thoracic_stress, 2),
        lumbar_stress=round(lumbar_stress, 2),
        risk_level=risk,
        action_required=action,
        color_hex=color,
        action_limit_ratio=round(float(action_ratio), 2),
    )


def lift_2d_to_3d_skeleton(
    keypoints: np.ndarray,
    trunk_flexion_deg: float = 0.0,
    neck_flexion_deg: float = 0.0,
    knee_angle_deg: float = 180.0,
) -> Skeleton3D:
    """
    Lift 2D COCO keypoints into a standardized 3D anatomical skeletal frame.

    Applies anthropometric kinematics:
    - Normalizes coordinates so pelvis center is at (0, 0, 0).
    - Infers relative depth (Z-axis) from sagittal trunk flexion and knee flexion angles.
    - Synthesizes central vertebral nodes: C7 Neck, T12 Mid-Spine, L5/S1 Lumbar.

    Args:
        keypoints: Array of shape (17, 2) or (17, 3) containing 2D/3D joint coordinates.
        trunk_flexion_deg: Trunk flexion angle in degrees.
        neck_flexion_deg: Neck flexion angle in degrees.
        knee_angle_deg: Knee interior angle in degrees.

    Returns:
        Skeleton3D populated with labeled 3D anatomical joints.
    """
    kp = np.array(keypoints)
    if kp.shape[0] < 17:
        # Fallback default upright skeleton
        return _create_synthetic_skeleton(trunk_flexion_deg, neck_flexion_deg, knee_angle_deg)

    # Extract raw 2D coordinates (X, Y)
    # In standard image coords, Y is downward; we invert Y so up is +Y.
    def get_pt(idx: int) -> Tuple[float, float, float]:
        if idx < len(kp):
            x = float(kp[idx][0])
            y = -float(kp[idx][1])
            conf = float(kp[idx][2]) if len(kp[idx]) > 2 else 1.0
            return x, y, conf
        return 0.0, 0.0, 0.0

    # Calculate pelvis center (mid-hip)
    lh_x, lh_y, lh_c = get_pt(KP_LEFT_HIP)
    rh_x, rh_y, rh_c = get_pt(KP_RIGHT_HIP)
    pelvis_x = (lh_x + rh_x) / 2.0
    pelvis_y = (lh_y + rh_y) / 2.0

    # Calculate scale factor from torso length
    ls_x, ls_y, ls_c = get_pt(KP_LEFT_SHOULDER)
    rs_x, rs_y, rs_c = get_pt(KP_RIGHT_SHOULDER)
    mid_sh_x = (ls_x + rs_x) / 2.0
    mid_sh_y = (ls_y + rs_y) / 2.0

    torso_len_2d = max(20.0, np.sqrt((mid_sh_x - pelvis_x) ** 2 + (mid_sh_y - pelvis_y) ** 2))
    # Standard anatomical torso length = 0.50 meters
    scale = 0.50 / torso_len_2d

    trunk_rad = np.radians(trunk_flexion_deg)
    # Forward sagittal depth offset for upper torso based on flexion angle
    torso_z_offset = 0.50 * np.sin(trunk_rad)

    # Normalization helper
    def norm_joint(idx: int, default_z: float, name: str) -> Point3D:
        x, y, conf = get_pt(idx)
        nx = (x - pelvis_x) * scale
        ny = (y - pelvis_y) * scale
        return Point3D(x=round(nx, 3), y=round(ny, 3), z=round(default_z, 3), confidence=conf, name=name)

    joints: Dict[str, Point3D] = {}

    # Pelvis Center (Root origin)
    joints["pelvis"] = Point3D(x=0.0, y=0.0, z=0.0, confidence=min(lh_c, rh_c), name="pelvis")

    # Lower Limbs (Z ~ 0)
    joints["left_hip"] = norm_joint(KP_LEFT_HIP, 0.0, "left_hip")
    joints["right_hip"] = norm_joint(KP_RIGHT_HIP, 0.0, "right_hip")

    # Knees may project slightly forward during bending/squatting
    knee_z = -0.10 if knee_angle_deg < 140 else 0.0
    joints["left_knee"] = norm_joint(KP_LEFT_KNEE, knee_z, "left_knee")
    joints["right_knee"] = norm_joint(KP_RIGHT_KNEE, knee_z, "right_knee")

    joints["left_ankle"] = norm_joint(KP_LEFT_ANKLE, 0.0, "left_ankle")
    joints["right_ankle"] = norm_joint(KP_RIGHT_ANKLE, 0.0, "right_ankle")

    # Upper Limbs (Z projects forward proportionally to torso flexion)
    joints["left_shoulder"] = norm_joint(KP_LEFT_SHOULDER, torso_z_offset, "left_shoulder")
    joints["right_shoulder"] = norm_joint(KP_RIGHT_SHOULDER, torso_z_offset, "right_shoulder")

    joints["left_elbow"] = norm_joint(KP_LEFT_ELBOW, torso_z_offset + 0.05, "left_elbow")
    joints["right_elbow"] = norm_joint(KP_RIGHT_ELBOW, torso_z_offset + 0.05, "right_elbow")

    joints["left_wrist"] = norm_joint(KP_LEFT_WRIST, torso_z_offset + 0.10, "left_wrist")
    joints["right_wrist"] = norm_joint(KP_RIGHT_WRIST, torso_z_offset + 0.10, "right_wrist")

    # Synthesized Central Spine Nodes
    # 1. C7 Neck (Mid-shoulder)
    c7_x = (joints["left_shoulder"].x + joints["right_shoulder"].x) / 2.0
    c7_y = (joints["left_shoulder"].y + joints["right_shoulder"].y) / 2.0
    joints["neck_c7"] = Point3D(x=round(c7_x, 3), y=round(c7_y, 3), z=round(torso_z_offset, 3), name="neck_c7")

    # 2. Head (Nose or offset above neck)
    nose_x, nose_y, nose_c = get_pt(KP_NOSE)
    if nose_c > 0.2:
        head_pt = norm_joint(KP_NOSE, torso_z_offset + (0.15 * np.sin(np.radians(neck_flexion_deg))), "head")
        joints["head"] = head_pt
    else:
        joints["head"] = Point3D(x=c7_x, y=c7_y + 0.22, z=torso_z_offset + 0.05, name="head")

    # 3. L5/S1 Lumbosacral Disc Center (just above pelvis origin)
    joints["lumbar_l5s1"] = Point3D(
        x=0.0,
        y=0.08,
        z=round(torso_z_offset * 0.20, 3),
        name="lumbar_l5s1",
    )

    # 4. T12 Mid-Thoracic Vertebra (midpoint between C7 and L5/S1)
    t12_x = (c7_x + joints["lumbar_l5s1"].x) / 2.0
    t12_y = (c7_y + joints["lumbar_l5s1"].y) / 2.0
    t12_z = (torso_z_offset + joints["lumbar_l5s1"].z) / 2.0
    joints["mid_spine_t12"] = Point3D(x=round(t12_x, 3), y=round(t12_y, 3), z=round(t12_z, 3), name="mid_spine_t12")

    # Calculate Center of Mass
    all_x = [p.x for p in joints.values()]
    all_y = [p.y for p in joints.values()]
    all_z = [p.z for p in joints.values()]
    com = (float(np.mean(all_x)), float(np.mean(all_y)), float(np.mean(all_z)))

    return Skeleton3D(
        joints=joints,
        trunk_angle=trunk_flexion_deg,
        neck_angle=neck_flexion_deg,
        knee_angle=knee_angle_deg,
        center_of_mass=com,
    )


def _create_synthetic_skeleton(trunk_deg: float, neck_deg: float, knee_deg: float) -> Skeleton3D:
    """Generate a clean anthropometrically proportional 3D skeleton for given angles."""
    trunk_rad = np.radians(trunk_deg)
    torso_z = 0.50 * np.sin(trunk_rad)
    torso_y = 0.50 * np.cos(trunk_rad)

    joints = {
        "pelvis": Point3D(0.0, 0.0, 0.0, name="pelvis"),
        "lumbar_l5s1": Point3D(0.0, 0.08, torso_z * 0.2, name="lumbar_l5s1"),
        "mid_spine_t12": Point3D(0.0, torso_y * 0.5, torso_z * 0.6, name="mid_spine_t12"),
        "neck_c7": Point3D(0.0, torso_y, torso_z, name="neck_c7"),
        "head": Point3D(0.0, torso_y + 0.22, torso_z + 0.05, name="head"),
        "left_shoulder": Point3D(0.18, torso_y, torso_z, name="left_shoulder"),
        "right_shoulder": Point3D(-0.18, torso_y, torso_z, name="right_shoulder"),
        "left_elbow": Point3D(0.22, torso_y - 0.28, torso_z + 0.05, name="left_elbow"),
        "right_elbow": Point3D(-0.22, torso_y - 0.28, torso_z + 0.05, name="right_elbow"),
        "left_wrist": Point3D(0.20, torso_y - 0.52, torso_z + 0.10, name="left_wrist"),
        "right_wrist": Point3D(-0.20, torso_y - 0.52, torso_z + 0.10, name="right_wrist"),
        "left_hip": Point3D(0.12, -0.05, 0.0, name="left_hip"),
        "right_hip": Point3D(-0.12, -0.05, 0.0, name="right_hip"),
        "left_knee": Point3D(0.12, -0.45, 0.0, name="left_knee"),
        "right_knee": Point3D(-0.12, -0.45, 0.0, name="right_knee"),
        "left_ankle": Point3D(0.12, -0.85, 0.0, name="left_ankle"),
        "right_ankle": Point3D(-0.12, -0.85, 0.0, name="right_ankle"),
    }

    return Skeleton3D(
        joints=joints,
        trunk_angle=trunk_deg,
        neck_angle=neck_deg,
        knee_angle=knee_deg,
        center_of_mass=(0.0, float(torso_y * 0.3), float(torso_z * 0.4)),
    )


def generate_3d_skeleton_plotly_figure(
    skeleton: Skeleton3D,
    spine_stress: Optional[SpineStress3D] = None,
    title: str = "3D Biomechanical Skeleton & Spine Stress",
    height: int = 480,
) -> Any:
    """
    Render an interactive rotatable 3D wireframe skeleton with spinal stress heatmap in Plotly.

    Args:
        skeleton: Skeleton3D object with 3D joint coordinates.
        spine_stress: Optional SpineStress3D object with L5/S1 spinal metrics.
        title: Chart title.
        height: Plotly figure height in pixels.

    Returns:
        plotly.graph_objects.Figure ready for st.plotly_chart.
    """
    if go is None:
        return None

    fig = go.Figure()

    # 1. Plot Bones Wireframe
    bone_x, bone_y, bone_z = [], [], []
    for j1_name, j2_name in skeleton.BONES:
        if j1_name in skeleton.joints and j2_name in skeleton.joints:
            p1 = skeleton.joints[j1_name]
            p2 = skeleton.joints[j2_name]
            bone_x.extend([p1.x, p2.x, None])
            bone_y.extend([p1.z, p2.z, None])  # Swap Y and Z for standard Plotly ground view (Z is up)
            bone_z.extend([p1.y, p2.y, None])

    fig.add_trace(go.Scatter3d(
        x=bone_x,
        y=bone_y,
        z=bone_z,
        mode="lines",
        line=dict(color="#3B82F6", width=5),
        name="Limb Segments",
        hoverinfo="none",
    ))

    # 2. Highlight Spine Column with Stress Gradient
    spine_nodes = ["pelvis", "lumbar_l5s1", "mid_spine_t12", "neck_c7", "head"]
    sp_x, sp_y, sp_z, sp_text = [], [], [], []
    stress_color = spine_stress.color_hex if spine_stress else "#EF4444"

    for node_name in spine_nodes:
        if node_name in skeleton.joints:
            pt = skeleton.joints[node_name]
            sp_x.append(pt.x)
            sp_y.append(pt.z)
            sp_z.append(pt.y)
            sp_text.append(f"Spine: {node_name}")

    fig.add_trace(go.Scatter3d(
        x=sp_x,
        y=sp_y,
        z=sp_z,
        mode="lines+markers",
        line=dict(color=stress_color, width=9),
        marker=dict(size=7, color=stress_color),
        name="Vertebral Column",
        text=sp_text,
        hoverinfo="text",
    ))

    # 3. L5/S1 Disc Target Marker
    if "lumbar_l5s1" in skeleton.joints:
        l5 = skeleton.joints["lumbar_l5s1"]
        comp_text = f"<b>L5/S1 Disc</b><br>Compression: {spine_stress.l5s1_compression_n if spine_stress else 0} N<br>Status: {spine_stress.risk_level if spine_stress else 'Normal'}"
        fig.add_trace(go.Scatter3d(
            x=[l5.x],
            y=[l5.z],
            z=[l5.y],
            mode="markers+text",
            marker=dict(size=14, color=stress_color, symbol="diamond", line=dict(color="#FFFFFF", width=2)),
            name="L5/S1 Disc Joint",
            text=["L5/S1"],
            textposition="top center",
            hovertext=[comp_text],
            hoverinfo="text",
        ))

    # 4. Standard Joints Scatter
    joint_x, joint_y, joint_z, joint_names = [], [], [], []
    for name, pt in skeleton.joints.items():
        if name not in spine_nodes:
            joint_x.append(pt.x)
            joint_y.append(pt.z)
            joint_z.append(pt.y)
            joint_names.append(name.replace("_", " ").title())

    fig.add_trace(go.Scatter3d(
        x=joint_x,
        y=joint_y,
        z=joint_z,
        mode="markers",
        marker=dict(size=5, color="#60A5FA", opacity=0.9),
        name="Anatomical Joints",
        text=joint_names,
        hoverinfo="text",
    ))

    # 5. Center of Mass (COM) Ground Projection
    com_x, com_y, com_z = skeleton.center_of_mass
    fig.add_trace(go.Scatter3d(
        x=[com_x, com_x],
        y=[com_z, com_z],
        z=[com_y, -0.90],
        mode="lines+markers",
        line=dict(color="#10B981", width=3, dash="dash"),
        marker=dict(size=[6, 8], color="#10B981", symbol="circle"),
        name="Center of Gravity (COG)",
        hoverinfo="name",
    ))

    # Dark Aesthetic Layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#F3F4F6")),
        height=height,
        margin=dict(l=10, r=10, t=35, b=10),
        scene=dict(
            xaxis=dict(title="Lateral (X)", range=[-0.6, 0.6], backgroundcolor="#111827", gridcolor="#374151", showbackground=True),
            yaxis=dict(title="Sagittal Depth (Z)", range=[-0.6, 0.8], backgroundcolor="#111827", gridcolor="#374151", showbackground=True),
            zaxis=dict(title="Vertical (Y)", range=[-1.0, 0.8], backgroundcolor="#111827", gridcolor="#374151", showbackground=True),
            camera=dict(
                eye=dict(x=1.4, y=-1.5, z=0.8),
                up=dict(x=0, y=0, z=1),
            ),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1.3),
        ),
        paper_bgcolor="#111827",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.01,
            xanchor="center",
            x=0.5,
            font=dict(color="#D1D5DB", size=10),
        ),
    )

    return fig
