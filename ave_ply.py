#!/usr/bin/env python3
"""
Average multiple PLY meshes into one, preserving vertex colors.

Method:
- Use mesh 0 as the reference topology (faces).
- ICP-align all other meshes to reference.
- For each reference vertex:
    - find nearest sampled surface point on each aligned mesh
    - average positions
    - (if colors exist) average interpolated RGBA colors

Deps:
  pip install trimesh scipy numpy
"""

from __future__ import annotations

import argparse
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from trimesh.registration import icp


def load_mesh(path: str) -> trimesh.Trimesh:
    m = trimesh.load(path, force="mesh", process=False)
    if not isinstance(m, trimesh.Trimesh):
        raise TypeError(f"{path} did not load as a Trimesh (got {type(m)})")
    if m.vertices.size == 0 or m.faces.size == 0:
        raise ValueError(f"{path} appears empty (no vertices/faces)")
    return m


def get_vertex_colors(mesh: trimesh.Trimesh) -> np.ndarray | None:
    vc = getattr(mesh.visual, "vertex_colors", None)
    if vc is None or len(vc) == 0:
        return None
    vc = np.asarray(vc)
    if vc.ndim != 2:
        return None
    # Ensure RGBA
    if vc.shape[1] == 3:
        alpha = np.full((vc.shape[0], 1), 255, dtype=vc.dtype)
        vc = np.hstack([vc, alpha])
    return vc[:, :4]


def align_to_reference(
    moving: trimesh.Trimesh,
    reference_points: np.ndarray,
    icp_samples: int = 8000,
    max_iterations: int = 50,
    threshold: float = 1e-6,
) -> trimesh.Trimesh:
    moving_points = moving.sample(icp_samples)
    matrix, _, _ = icp(
        moving_points,
        reference_points,
        max_iterations=max_iterations,
        threshold=threshold,
    )
    aligned = moving.copy()
    aligned.apply_transform(matrix)
    return aligned


def sample_points_and_colors(mesh: trimesh.Trimesh, count: int):
    """
    Sample points uniformly on surface.
    If vertex colors exist, interpolate RGBA at sampled points.
    Returns (points, colors_or_None)
    """
    # trimesh.sample.sample_surface returns (points, face_index)
    pts, face_idx = trimesh.sample.sample_surface(mesh, count)
    pts = np.asarray(pts, dtype=np.float64)

    vc = get_vertex_colors(mesh)
    if vc is None:
        return pts, None

    face_idx = np.asarray(face_idx, dtype=np.int64)

    # For each sampled point, we know which face it came from
    faces = mesh.faces[face_idx]  # (count, 3)
    c0 = vc[faces[:, 0]].astype(np.float64)
    c1 = vc[faces[:, 1]].astype(np.float64)
    c2 = vc[faces[:, 2]].astype(np.float64)

    tris = mesh.triangles[face_idx].astype(np.float64)  # (count, 3, 3)
    bary = trimesh.triangles.points_to_barycentric(tris, pts)  # (count, 3)

    cols = (bary[:, [0]] * c0) + (bary[:, [1]] * c1) + (bary[:, [2]] * c2)
    cols = np.clip(cols, 0, 255)
    return pts, cols


def average_meshes_with_color(
    mesh_paths: list[str],
    out_path: str,
    ref_index: int = 0,
    icp_samples: int = 8000,
    nearest_samples: int = 200000,
) -> None:
    meshes = [load_mesh(p) for p in mesh_paths]
    ref = meshes[ref_index]

    ref_vertices = ref.vertices.astype(np.float64)
    ref_faces = ref.faces.copy()

    # ICP target points
    ref_points = ref.sample(icp_samples)

    # Align all meshes to the reference
    aligned = []
    for i, m in enumerate(meshes):
        if i == ref_index:
            aligned.append(m)
        else:
            aligned.append(
                align_to_reference(
                    m,
                    reference_points=ref_points,
                    icp_samples=icp_samples,
                )
            )

    # Geometry accumulation
    pos_acc = ref_vertices.copy()

    # Color accumulation (only if any mesh has vertex colors)
    ref_vc = get_vertex_colors(ref)
    if ref_vc is not None:
        col_acc = ref_vc.astype(np.float64).copy()
        col_count = 1.0
    else:
        col_acc = np.zeros((len(ref_vertices), 4), dtype=np.float64)
        col_count = 0.0

    for i, m in enumerate(aligned):
        if i == ref_index:
            continue

        samp_pts, samp_cols = sample_points_and_colors(m, nearest_samples)
        tree = cKDTree(samp_pts)
        _, idx = tree.query(ref_vertices, k=1, workers=-1)

        pos_acc += samp_pts[idx]

        if samp_cols is not None:
            col_acc += samp_cols[idx]
            col_count += 1.0

    avg_vertices = pos_acc / float(len(aligned))

    avg_mesh = trimesh.Trimesh(vertices=avg_vertices, faces=ref_faces, process=False)

    if col_count > 0:
        avg_cols = np.clip(np.rint(col_acc / col_count), 0, 255).astype(np.uint8)
        avg_mesh.visual = trimesh.visual.ColorVisuals(mesh=avg_mesh, vertex_colors=avg_cols)

    avg_mesh.export(out_path)
    print(f"Wrote averaged mesh: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("meshes", nargs="+", help="Input .ply mesh paths (2+)")
    ap.add_argument("-o", "--out", default="averaged_colored.ply", help="Output .ply path")
    ap.add_argument("--ref-index", type=int, default=0, help="Reference mesh index (topology source)")
    ap.add_argument("--icp-samples", type=int, default=8000)
    ap.add_argument("--nearest-samples", type=int, default=200000)
    args = ap.parse_args()

    if len(args.meshes) < 2:
        raise SystemExit("Need at least 2 meshes to average.")

    average_meshes_with_color(
        mesh_paths=args.meshes,
        out_path=args.out,
        ref_index=args.ref_index,
        icp_samples=args.icp_samples,
        nearest_samples=args.nearest_samples,
    )


if __name__ == "__main__":
    main()
