"""Static geometry and camera visibility shared by physics and teacher sensing."""

from __future__ import annotations

import math

import numpy as np
import torch

TILE_SIZE = 16.0
TILE_SPACING = 20.0
START_IN_TILE = (3.0, 8.0, 0.0)
COURSE_GOAL_OFFSET = 10.0
CAMERA_POSITION = (0.25, 0.0, 0.12)
CAMERA_PITCH = math.radians(20.0)
CAMERA_ROTATION = (math.cos(CAMERA_PITCH / 2), 0.0, math.sin(CAMERA_PITCH / 2), 0.0)


def quat_rotate(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by wxyz quaternions, with ordinary tensor broadcasting."""
    xyz = quaternion[..., 1:]
    cross = 2.0 * torch.cross(xyz.expand_as(vector), vector, dim=-1)
    return vector + quaternion[..., :1] * cross + torch.cross(xyz.expand_as(vector), cross, dim=-1)


def yaw_from_quat(quaternion: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quaternion.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))


def rotate_yaw(points: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cosine, sine = torch.cos(yaw), torch.sin(yaw)
    while cosine.ndim < points.ndim - 1:
        cosine, sine = cosine.unsqueeze(-1), sine.unsqueeze(-1)
    x, y, z = points.unbind(-1)
    return torch.stack((cosine * x - sine * y, sine * x + cosine * y, z), dim=-1)


def camera_coordinates(points: torch.Tensor, root_pos: torch.Tensor, root_quat: torch.Tensor):
    """Return optical depth, lateral/up coordinates, and the physical camera origin."""
    offset = points.new_tensor(CAMERA_POSITION).expand(root_pos.shape)
    camera_pos = root_pos + quat_rotate(root_quat, offset)
    delta = points - camera_pos.unsqueeze(-2)
    inverse = root_quat.clone()
    inverse[..., 1:] *= -1
    local = quat_rotate(inverse.unsqueeze(-2), delta)
    cosine, sine = math.cos(CAMERA_PITCH), math.sin(CAMERA_PITCH)
    depth = cosine * local[..., 0] - sine * local[..., 2]
    up = sine * local[..., 0] + cosine * local[..., 2]
    return depth, local[..., 1], up, camera_pos


def camera_frustum(points: torch.Tensor, root_pos: torch.Tensor, root_quat: torch.Tensor):
    depth, lateral, up, position = camera_coordinates(points, root_pos, root_quat)
    visible = (
        torch.isfinite(points).all(-1)
        & (depth >= 0.2)
        & (depth <= 6.0)
        & (lateral.abs() <= depth)
        & (up.abs() <= 0.75 * depth)
    )
    return visible, position


class VisibleSurfaceMemory:
    """Two-second world-cell memory; moving/turning does not rotate old visibility.

    The cells store observed surface height as well as time, so seeing ground in a
    cell does not grant knowledge of an unobserved wall above the same cell.
    """

    def __init__(self, origins: torch.Tensor, resolution: float = 0.1, ttl: float = 2.0):
        self.origins = origins
        self.resolution = resolution
        self.ttl = ttl
        self.side = math.ceil(TILE_SIZE / resolution)
        shape = (len(origins), self.side * self.side)
        self.seen_at = torch.full(shape, -torch.inf, device=origins.device)
        self.surface_z = torch.zeros(shape, device=origins.device)

    def _indices(self, points):
        offset = points.new_tensor(START_IN_TILE[:2])
        local = points[..., :2] - self.origins[:, None, :2] + offset
        finite = torch.isfinite(points).all(-1)
        cells = torch.floor(torch.nan_to_num(local) / self.resolution).long()
        inside = finite & (cells >= 0).all(-1) & (cells < self.side).all(-1)
        cells = cells.clamp(0, self.side - 1)
        return cells[..., 0] * self.side + cells[..., 1], inside

    def update(self, points: torch.Tensor, visible: torch.Tensor, timestamp: float):
        indices, inside = self._indices(points)
        selected = visible & inside
        rows = torch.arange(len(points), device=points.device)[:, None].expand_as(indices)
        self.seen_at[rows[selected], indices[selected]] = timestamp
        self.surface_z[rows[selected], indices[selected]] = points[..., 2][selected]

    def query(self, points: torch.Tensor, timestamp: float, height_tolerance: float = 0.08):
        indices, inside = self._indices(points)
        last = self.seen_at.gather(1, indices)
        height = self.surface_z.gather(1, indices)
        return inside & (timestamp - last <= self.ttl) & ((height - points[..., 2]).abs() <= height_tolerance)

    def reset(self, env_ids):
        self.seen_at[env_ids] = -torch.inf
        self.surface_z[env_ids] = 0


def _height_patch(x, y, z):
    import trimesh

    grid_x, grid_y = np.meshgrid(x, y, indexing="ij")
    vertices = np.column_stack((grid_x.ravel(), grid_y.ravel(), z.ravel()))
    ids = np.arange(grid_x.size).reshape(grid_x.shape)
    a = ids[:-1, :-1].ravel()
    b = ids[1:, :-1].ravel()
    c = ids[1:, 1:].ravel()
    d = ids[:-1, 1:].ravel()
    faces = np.concatenate((np.stack((a, b, c), -1), np.stack((a, c, d), -1)))
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def build_tile(kind: str, difficulty: float, rng: np.random.Generator, obstacles: bool):
    """Build traversable geometry with a clear spawn and goal platform."""
    import trimesh

    if kind not in {"flat", "slopes", "rough", "stairs", "course"}:
        raise ValueError(f"Unknown terrain kind: {kind}")
    if not 0 <= difficulty <= 1:
        raise ValueError("Terrain difficulty must be in [0, 1].")

    def box(extents, position):
        mesh = trimesh.creation.box(extents=extents)
        mesh.apply_translation(position)
        return mesh

    pieces = [box((TILE_SIZE, TILE_SIZE, 0.2), (8, 8, -0.1))]
    if kind == "slopes":
        x = np.array([4.5, 6.5, 8.5, 10.5])
        y = np.array([2.5, 13.5])
        h = 0.15 + 0.45 * difficulty
        pieces.append(_height_patch(x, y, np.array([0, h, h, 0])[:, None] * np.ones((1, 2))))
    elif kind == "rough":
        x, y = np.linspace(4.5, 10.5, 41), np.linspace(3, 13, 51)
        gx, gy = np.meshgrid(x, y, indexing="ij")
        taper = np.sin(np.pi * (gx - x[0]) / (x[-1] - x[0])) * np.sin(np.pi * (gy - y[0]) / (y[-1] - y[0]))
        z = (0.015 + 0.05 * difficulty) * (1 + np.sin(5 * gx) * np.cos(4 * gy))
        z += rng.uniform(0, 0.01 + 0.02 * difficulty, size=z.shape)
        pieces.append(_height_patch(x, y, z * taper))
    elif kind == "stairs":
        step_height = 0.04 + 0.10 * difficulty
        for step in range(5):
            left = 5.0 + step * 0.3
            right = 9.0 - step * 0.3
            h = (step + 1) * step_height
            pieces.append(box((right - left, 9.0, h), ((left + right) / 2, 8.0, h / 2)))
    elif kind == "course":
        # One x-aligned task route: flat launch, uphill, plateau, downhill,
        # flat recovery, stairs up, plateau, stairs down, and flat finish.
        x = np.array([3.5, 5.0, 6.0, 7.5])
        y = np.array([4.0, 12.0])
        slope_height = 0.20 + 0.35 * difficulty
        z = np.array([0.0, slope_height, slope_height, 0.0])[:, None] * np.ones((1, 2))
        pieces.append(_height_patch(x, y, z))
        step_height = 0.05 + 0.08 * difficulty
        for step in range(5):
            left = 8.5 + step * 0.30
            right = 12.1 - step * 0.30
            h = (step + 1) * step_height
            pieces.append(box((right - left, 8.0, h), ((left + right) / 2, 8.0, h / 2)))
    traversable_face_count = sum(len(piece.faces) for piece in pieces)
    if obstacles:
        # The shared mesh includes these solids for rendering, collision and every ray query.
        offset = float(rng.uniform(-0.5, 0.5))
        pieces.extend(
            (
                box((0.8, 1.1, 1.0), (6.0, 7.6 + offset, 0.5)),
                box((0.6, 1.4, 1.0), (8.4, 9.3, 0.5)),
                box((0.25, 2.5, 1.2), (10.7, 5.4, 0.6)),
                box((1.5, 0.25, 1.2), (10.0, 4.3, 0.6)),
            )
        )
    result = trimesh.util.concatenate(pieces)
    result.metadata["obstacle_faces"] = np.arange(len(result.faces)) >= traversable_face_count
    return result


def build_scene(num_envs: int, mode: str, obstacles: bool, seed: int, difficulty: float):
    import trimesh

    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    kinds = ("flat", "slopes", "rough", "stairs")
    if mode != "mixed" and mode not in (*kinds, "course"):
        raise ValueError(f"Unknown terrain mode: {mode}")
    width = math.ceil(math.sqrt(num_envs))
    origins, meshes = [], []
    rng = np.random.default_rng(seed)
    for index in range(num_envs):
        translation = np.array([(index // width) * TILE_SPACING, (index % width) * TILE_SPACING, 0.0])
        mesh = build_tile(kinds[index % len(kinds)] if mode == "mixed" else mode, difficulty, rng, obstacles)
        mesh.apply_translation(translation)
        meshes.append(mesh)
        origins.append(translation + np.array(START_IN_TILE))
    result = trimesh.util.concatenate(meshes)
    result.metadata["obstacle_faces"] = np.concatenate([mesh.metadata["obstacle_faces"] for mesh in meshes])
    return result, np.asarray(origins, dtype=np.float32)
