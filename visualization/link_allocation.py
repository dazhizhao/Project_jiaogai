from __future__ import annotations

from pathlib import Path
from typing import Sequence

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

from env.kinematics import forward_kinematics


def save_workspace_samples(
    output_path: str | Path,
    points: np.ndarray,
    lengths: Sequence[float],
    occupied_ratio: float,
    workspace_area_estimate: float,
    xy_bounds: Sequence[Sequence[float]],
    grid_shape: Sequence[int],
    joint_angle_samples: np.ndarray,
    representative_joint_angles: Sequence[float],
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        points=np.asarray(points, dtype=np.float32),
        lengths=np.asarray(lengths, dtype=np.float32),
        occupied_ratio=np.asarray(occupied_ratio, dtype=np.float32),
        workspace_area_estimate=np.asarray(workspace_area_estimate, dtype=np.float32),
        xy_bounds=np.asarray(xy_bounds, dtype=np.float32),
        grid_shape=np.asarray(grid_shape, dtype=np.int32),
        joint_angle_samples=np.asarray(joint_angle_samples, dtype=np.float32),
        representative_joint_angles=np.asarray(representative_joint_angles, dtype=np.float32),
    )
    return output


def render_workspace_preview(
    output_path: str | Path,
    points: np.ndarray,
    lengths: Sequence[float],
    representative_joint_angles: Sequence[float],
    xy_bounds: Sequence[Sequence[float]],
    occupied_ratio: float,
    workspace_area_estimate: float,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = _build_workspace_figure(
        points=points,
        lengths=lengths,
        representative_joint_angles=representative_joint_angles,
        xy_bounds=xy_bounds,
        occupied_ratio=occupied_ratio,
        workspace_area_estimate=workspace_area_estimate,
        point_count=len(points),
        point_size=10.0,
        alpha=0.3,
    )
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def export_workspace_video(
    output_path: str | Path,
    points: np.ndarray,
    lengths: Sequence[float],
    representative_joint_angles: Sequence[float],
    xy_bounds: Sequence[Sequence[float]],
    occupied_ratio: float,
    workspace_area_estimate: float,
    fps: int,
    frames: int,
    point_size: float,
    alpha: float,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    points_arr = np.asarray(points, dtype=np.float32)
    if len(points_arr) == 0:
        raise ValueError("points must contain at least one sample.")

    frame_counts = np.linspace(1, len(points_arr), num=frames, dtype=int)
    with imageio.get_writer(output, fps=fps, codec="libx264", format="FFMPEG", pixelformat="yuv420p") as writer:
        for point_count in frame_counts:
            fig, ax = _build_workspace_figure(
                points=points_arr,
                lengths=lengths,
                representative_joint_angles=representative_joint_angles,
                xy_bounds=xy_bounds,
                occupied_ratio=occupied_ratio,
                workspace_area_estimate=workspace_area_estimate,
                point_count=int(point_count),
                point_size=point_size,
                alpha=alpha,
            )
            fig.canvas.draw()
            frame_image = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            writer.append_data(frame_image)
            plt.close(fig)
    return output


def _build_workspace_figure(
    points: np.ndarray,
    lengths: Sequence[float],
    representative_joint_angles: Sequence[float],
    xy_bounds: Sequence[Sequence[float]],
    occupied_ratio: float,
    workspace_area_estimate: float,
    point_count: int,
    point_size: float,
    alpha: float,
):
    subset = np.asarray(points[:point_count], dtype=np.float32)
    bounds = np.asarray(xy_bounds, dtype=np.float32)
    kin = forward_kinematics(representative_joint_angles, lengths)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
    ax.scatter(
        subset[:, 0],
        subset[:, 1],
        s=point_size,
        alpha=alpha,
        color="#0a9396",
        edgecolors="none",
        label="workspace samples",
    )
    ax.plot(
        kin.joint_positions[:, 0],
        kin.joint_positions[:, 1],
        "-o",
        color="#005f73",
        linewidth=2.5,
        markersize=6,
        label="representative pose",
    )
    ax.scatter(0.0, 0.0, color="#9b2226", s=90, label="base", zorder=5)
    ax.set_xlim(bounds[0, 0], bounds[0, 1])
    ax.set_ylim(bounds[1, 0], bounds[1, 1])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("RL Workspace Coverage")
    ax.legend(loc="upper left")

    detail_text = "\n".join(
        [
            f"points: {point_count}/{len(points)}",
            f"occupied_ratio: {occupied_ratio:.4f}",
            f"workspace_area_estimate: {workspace_area_estimate:.4f}",
            "lengths: " + ", ".join(f"{value:.2f}" for value in np.asarray(lengths, dtype=float)),
        ]
    )
    ax.text(
        1.02,
        0.98,
        detail_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#94d2bd"},
    )
    fig.tight_layout()
    return fig, ax
