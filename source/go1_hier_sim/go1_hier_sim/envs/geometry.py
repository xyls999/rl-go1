"""Static geometry and camera visibility shared by physics and teacher sensing."""

from __future__ import annotations

import math

import numpy as np
import torch

TILE_SIZE = 40.0
TILE_SPACING = 44.0
START_IN_TILE = (3.0, 8.0, 0.0)
COURSE_GOAL_OFFSET = 34.0
# Course layout. Treads must be long enough for the ~0.6 m trunk to bridge one
# without its leading edge jamming on the next riser, so they stay >= 0.35 m.
# The route is a dense mix of slopes, dips, stairways, corrugated patches and
# bumps: every block starts and ends at the course datum, so blocks compose in
# any order and the level finish strip joins the last block without a lip. The
# feature span stops short of the finish line, leaving only a short level
# approach, and the route is long enough that the low end of the speed band still
# needs the full episode budget to reach the line.
COURSE_TREAD_DEPTH = 0.40
COURSE_LEAD_IN_END = 5.30
COURSE_FEATURE_END = 35.00
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

    # The base plate covers one whole tile so a straight course is never
    # interrupted by an inter-tile void. Its top stays level with z = 0, the
    # plane the course surface patches use, so the plate keeps acting as a
    # continuous floor directly beneath them.
    pieces = [box((TILE_SIZE, TILE_SIZE, 0.2), (TILE_SIZE / 2.0, TILE_SIZE / 2.0, -0.1))]
    course_metadata = None
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
        # One x-aligned route inside the certified physical envelope: a short
        # level lead-in, then a dense mix of slopes, dips, stairways whose risers
        # vary in height, corrugated patches and isolated bumps, with almost no
        # level ground between them and only a short level approach to the finish
        # line. Every block starts and ends at the course datum, so blocks compose
        # in any order and the level finish strip joins the last one without a
        # lip. NOTE: this route is authored in tile coordinates (the spawn sits at
        # START_IN_TILE) while the finish offsets below are spawn-relative. Both
        # frames are load-bearing; mixing them is what once left the goal beyond
        # the end of the mesh.
        #
        # Magnitudes come from the robot rather than from taste: a 0.25 + 0.25 m
        # leg standing at 0.365 m can raise the body by roughly 0.16 m, and the
        # worst sampled static friction (0.6) caps a slope at atan(0.6) ~ 31
        # degrees, so risers stay <= 0.16 m and slopes <= 30 degrees.
        #
        # Triangle budget is a hard constraint, not a preference: the 256
        # environment terrain has to be cookable by PhysX, and an over-dense mesh
        # silently fails to cook and drops every robot. Patches are therefore
        # sampled at 0.10 m along the route, which still matches the 0.1 m height
        # scan, and the level approach is sampled at 0.20 m because it is flat.
        roughness_amplitude = 0.004 + 0.012 * difficulty
        slope_angles = []
        ramp_peaks = []
        risers = []
        treads = []
        blocks = []

        def x_envelope(mesh_x, x0, x1):
            return np.sin(np.pi * (mesh_x - x0) / (x1 - x0)).clip(0.0, 1.0)

        def course_surface(x0, x1, field, resolution=0.10):
            grid_x = np.linspace(x0, x1, max(6, int(round((x1 - x0) / resolution)) + 1))
            grid_y = np.linspace(4.0, 12.0, 41)
            mesh_x, mesh_y = np.meshgrid(grid_x, grid_y, indexing="ij")
            y_taper = np.sin(
                np.pi * (mesh_y - grid_y[0]) / (grid_y[-1] - grid_y[0])
            ).clip(0.0, 1.0)
            pieces.append(_height_patch(grid_x, grid_y, field(mesh_x, mesh_y) * y_taper))

        def corrugation(mesh_x, mesh_y):
            waves = (
                0.55 * np.sin(13.0 * mesh_x + rng.uniform(-math.pi, math.pi))
                + 0.30 * np.sin(8.0 * mesh_y + rng.uniform(-math.pi, math.pi))
                + 0.15 * np.sin(27.0 * mesh_x + rng.uniform(-math.pi, math.pi))
            )
            noise = rng.uniform(-1.0, 1.0, size=mesh_x.shape)
            noise = (
                noise
                + np.roll(noise, 1, axis=0)
                + np.roll(noise, -1, axis=0)
                + np.roll(noise, 1, axis=1)
                + np.roll(noise, -1, axis=1)
            ) / 5.0
            return waves + 0.55 * noise

        def slope_block(x0):
            # Half the grade blocks run up and half dip below the datum first, so
            # the route mixes climbs and descents instead of climbing only.
            angle_deg = float(rng.uniform(5.0, 5.0 + 25.0 * difficulty))
            run_up = float(rng.uniform(0.90, 1.60))
            run_down = float(rng.uniform(0.90, 1.60))
            peak = float(run_up * math.tan(math.radians(angle_deg)))
            # A cross-slope (camber) makes the ground uneven across the walking
            # band as well; it is tapered to zero at the block ends.
            camber = float(rng.uniform(-1.0, 1.0)) * math.radians(3.0 + 3.0 * difficulty)
            x1 = x0 + run_up + run_down
            knees = np.array([x0, x0 + run_up, x1])
            profile = np.array([0.0, peak, 0.0])

            def field(mesh_x, mesh_y):
                base = np.interp(mesh_x, knees, profile)
                lateral = (mesh_y - 8.0) * math.tan(camber)
                rough = roughness_amplitude * corrugation(mesh_x, mesh_y)
                return (base + lateral + rough) * x_envelope(mesh_x, x0, x1)

            course_surface(x0, x1, field)
            slope_angles.append(angle_deg)
            ramp_peaks.append(abs(peak))
            blocks.append(
                {
                    "kind": "slope",
                    "start": float(x0),
                    "end": float(x1),
                    "peak": peak,
                    "camber_deg": float(math.degrees(camber)),
                }
            )
            return x1

        def ledge_block(x0):
            # A blind step-down, which is the hardest uneven feature inside the
            # envelope: ramp up, cross a short plateau, drop one legal riser, then
            # ramp back to the datum. The drop is capped by the same 0.16 m
            # kinematic limit as a stairway riser.
            angle_deg = float(rng.uniform(8.0, 8.0 + 22.0 * difficulty))
            run_up = float(rng.uniform(0.80, 1.20))
            run_down = float(rng.uniform(0.80, 1.20))
            plateau = float(rng.uniform(0.50, 0.90))
            top = float(run_up * math.tan(math.radians(angle_deg)))
            drop = float(min(0.16, max(0.06, top)))
            camber = float(rng.uniform(-1.0, 1.0)) * math.radians(2.5 + 2.5 * difficulty)
            x1 = x0 + run_up
            x2 = x1 + plateau
            x3 = x2 + run_down
            plateau_height = top - drop

            def field(mesh_x, mesh_y):
                stage = np.select(
                    [mesh_x < x1, mesh_x < x2],
                    [top * (mesh_x - x0) / (x1 - x0), np.full_like(mesh_x, plateau_height)],
                    default=plateau_height * (x3 - mesh_x).clip(min=0.0) / (x3 - x2),
                )
                lateral = (mesh_y - 8.0) * math.tan(camber)
                rough = roughness_amplitude * corrugation(mesh_x, mesh_y)
                return (stage + lateral + rough) * x_envelope(mesh_x, x0, x3)

            course_surface(x0, x3, field)
            slope_angles.append(angle_deg)
            ramp_peaks.append(top)
            blocks.append(
                {
                    "kind": "ledge",
                    "start": float(x0),
                    "end": float(x3),
                    "peak": top,
                    "drop": drop,
                }
            )
            return x3

        def stairs_block(x0, steps=None, kind="stairs", max_end=None, tread_range=None):
            # Short stairways mix 3-5 risers. The long variant below reuses this
            # construction with 7-9 treads so the policy also sees sustained climbs
            # and descents instead of isolated steps.
            count = int(rng.integers(3, 6)) if steps is None else int(steps)
            low_tread, high_tread = (
                (COURSE_TREAD_DEPTH - 0.03, COURSE_TREAD_DEPTH + 0.03)
                if tread_range is None
                else (float(tread_range[0]), float(tread_range[1]))
            )
            if max_end is not None:
                # Truncate the stairway to the space actually left, so a long
                # staircase can never push the route past the feature end. Three
                # risers per direction is the smallest useful staircase.
                headroom = max(0.0, float(max_end) - x0)
                fits = int((headroom - 0.30) // (2.0 * high_tread))
                count = max(3, min(count, fits))
            rise = rng.uniform(0.04, 0.04 + 0.12 * difficulty, size=count)
            tread_up = rng.uniform(low_tread, high_tread, size=count)
            tread_down = rng.uniform(low_tread, high_tread, size=count)
            # The descent reorders the same riser heights, so the stairway keeps
            # varied step heights while returning exactly to the course datum.
            descent = rng.permutation(rise)
            landing = float(rng.uniform(0.30, 0.60))
            cursor = x0
            height = 0.0
            for index in range(count):
                height += float(rise[index])
                tip = float(tread_up[index])
                pieces.append(
                    box((tip, 8.0, height), (cursor + tip / 2.0, 8.0, height / 2.0))
                )
                cursor += tip
            pieces.append(box((landing, 8.0, height), (cursor + landing / 2.0, 8.0, height / 2.0)))
            cursor += landing
            for index in range(count):
                height -= float(descent[index])
                tip = float(tread_down[index])
                if height > 1e-6:
                    pieces.append(
                        box((tip, 8.0, height), (cursor + tip / 2.0, 8.0, height / 2.0))
                    )
                cursor += tip
            if abs(height) > 1e-9:
                raise ValueError("course stairway does not return to the course datum")
            risers.extend(float(value) for value in rise)
            treads.extend(float(value) for value in tread_up)
            treads.extend(float(value) for value in tread_down)
            blocks.append(
                {
                    "kind": str(kind),
                    "start": float(x0),
                    "end": float(cursor),
                    "peak": float(np.sum(rise)),
                    "steps": count,
                }
            )
            return cursor

        def long_stairs_block(x0):
            # 7-9 risers per direction: ~6-7 m of continuous staircase inside one
            # tile, still capped by the same kinematic envelope (riser <= 0.16 m,
            # tread >= 0.35 m) and still returning exactly to the course datum. The
            # riser count shrinks automatically when the short blocks ahead of it
            # consumed most of the feature span.
            return stairs_block(
                x0, steps=int(rng.integers(7, 10)), kind="long_stairs",
                max_end=COURSE_FEATURE_END - 8.0, tread_range=(0.35, 0.39),
            )

        def rough_block(x0, x1=None):
            length = float(rng.uniform(1.00, 2.20)) if x1 is None else float(x1) - x0
            x1 = x0 + max(length, 0.10)
            amplitude = roughness_amplitude * 1.8

            def field(mesh_x, mesh_y):
                return amplitude * corrugation(mesh_x, mesh_y) * x_envelope(mesh_x, x0, x1)

            course_surface(x0, x1, field)
            blocks.append({"kind": "rough", "start": float(x0), "end": float(x1), "peak": 0.0})
            return x1

        def bumps_block(x0):
            x1 = x0 + float(rng.uniform(1.20, 2.00))
            count = int(rng.integers(2, 6))
            centres = np.sort(rng.uniform(x0 + 0.22, x1 - 0.22, size=count))
            peaks = rng.uniform(0.04, 0.05 + 0.06 * difficulty, size=count)
            widths = rng.uniform(0.16, 0.30, size=count)

            def field(mesh_x, mesh_y):
                total = np.zeros_like(mesh_x)
                for centre, peak, width in zip(centres, peaks, widths):
                    total = total + peak * np.exp(-((mesh_x - centre) / width) ** 2)
                return total * x_envelope(mesh_x, x0, x1)

            course_surface(x0, x1, field)
            blocks.append(
                {"kind": "bumps", "start": float(x0), "end": float(x1), "peak": float(peaks.max())}
            )
            return x1

        # Every tile opens with one guarantee block of each feature kind, so the
        # route always mixes grades, uneven stairways, rough ground and bumps
        # instead of leaving the mix to chance. The rest of the span is filled from
        # a reshuffled short-block pool. Block length bounds keep the mix out of
        # the level approach: a long stairway is at most 9 treads * 0.43 m per
        # direction plus a landing (8.4 m), a short stairway 5 treads (4.9 m), a
        # slope at most 4.0 m, a rough patch 2.2 m and a bump group 2.0 m, and the
        # opening plus fillers never exceeds COURSE_FEATURE_END, which the check
        # below enforces. The opening guarantees one of every block kind and three
        # separate stairways (two short, one long) so every tile contains
        # sustained climbing and descending.
        opening = (slope_block, ledge_block, stairs_block, stairs_block, rough_block, bumps_block)
        graded_builders = (rough_block, bumps_block, slope_block, ledge_block, stairs_block)
        flat_builders = (rough_block, bumps_block)
        cursor = COURSE_LEAD_IN_END
        for index in rng.permutation(len(opening)):
            cursor = opening[int(index)](cursor)
        # The long stairway closes the guaranteed opening and truncates itself to
        # whatever span is left, so the mix cannot overrun the feature end.
        cursor = long_stairs_block(cursor)
        # Fill thresholds are set so the longest remaining block still fits before
        # COURSE_FEATURE_END: 5.20 m for the graded pool, 2.20 m for the flat pool.
        while cursor < COURSE_FEATURE_END - 5.30:
            for index in rng.permutation(len(graded_builders)):
                if cursor >= COURSE_FEATURE_END - 5.30:
                    break
                cursor = graded_builders[int(index)](cursor)
        while cursor < COURSE_FEATURE_END - 2.30:
            for index in rng.permutation(len(flat_builders)):
                if cursor >= COURSE_FEATURE_END - 2.30:
                    break
                cursor = flat_builders[int(index)](cursor)
        if COURSE_FEATURE_END - cursor > 0.10:
            cursor = rough_block(cursor, COURSE_FEATURE_END)
        if cursor > COURSE_FEATURE_END + 1e-6:
            raise ValueError("course features overrun the level approach")

        # Level ground after the last feature. It must extend past both the
        # navigation goal and the declared finish region: ending the mesh where
        # the finish region begins leaves the last metre without ground, so every
        # episode fell off the edge and the goal was unreachable by construction.
        finish_x = np.linspace(
            COURSE_FEATURE_END, TILE_SIZE - 0.90,
            int(round((TILE_SIZE - 0.90 - COURSE_FEATURE_END) / 0.20)) + 1,
        )
        finish_y = np.linspace(4.0, 12.0, 41)
        pieces.append(
            _height_patch(
                finish_x, finish_y, np.zeros((finish_x.size, finish_y.size), dtype=np.float64)
            )
        )
        course_metadata = {
            "roughness_amplitude": float(roughness_amplitude),
            "slope_angle_deg": float(max(slope_angles) if slope_angles else 0.0),
            "slope_height": float(max(ramp_peaks) if ramp_peaks else 0.0),
            "step_rises": list(risers),
            "max_step_rise": float(max(risers) if risers else 0.0),
            "tread_depths": list(treads),
            "tread_depth_min": float(min(treads) if treads else COURSE_TREAD_DEPTH),
            "blocks": blocks,
            "feature_kinds": sorted({block["kind"] for block in blocks}),
            "long_stairways": sorted(
                block["steps"] for block in blocks if block["kind"] == "long_stairs"
            ),
            "stairway_count": sum(1 for block in blocks if block["kind"] in ("stairs", "long_stairs")),
            "feature_end": float(COURSE_FEATURE_END),
            "flat_approach_m": float(START_IN_TILE[0] + 33.10 - COURSE_FEATURE_END),
            "finish_start_offset": 33.10,
            "finish_end_offset": 34.10,
        }

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
    if course_metadata is not None:
        result.metadata["course_profile"] = course_metadata
    return result


def build_scene(num_envs: int, mode: str, obstacles: bool, seed: int, difficulty: float,
                course_difficulty_min: float | None = None,
                course_difficulty_max: float | None = None):
    import trimesh

    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    kinds = ("flat", "slopes", "rough", "stairs")
    if mode != "mixed" and mode not in (*kinds, "course"):
        raise ValueError(f"Unknown terrain mode: {mode}")
    if (course_difficulty_min is None) != (course_difficulty_max is None):
        raise ValueError("course difficulty bounds must be provided together")
    if course_difficulty_min is not None:
        if not 0 <= course_difficulty_min <= course_difficulty_max <= 1:
            raise ValueError("course difficulty bounds must satisfy 0 <= min <= max <= 1")
    width = math.ceil(math.sqrt(num_envs))
    origins, meshes = [], []
    rng = np.random.default_rng(seed)
    for index in range(num_envs):
        translation = np.array([(index // width) * TILE_SPACING, (index % width) * TILE_SPACING, 0.0])
        tile_kind = kinds[index % len(kinds)] if mode == "mixed" else mode
        tile_difficulty = difficulty
        if tile_kind == "course":
            if course_difficulty_min is None:
                tile_difficulty = float(np.clip(difficulty + rng.uniform(-0.20, 0.20), 0.45, 1.0))
            else:
                tile_difficulty = float(rng.uniform(course_difficulty_min, course_difficulty_max))
        mesh = build_tile(tile_kind, tile_difficulty, rng, obstacles)
        mesh.apply_translation(translation)
        meshes.append(mesh)
        origins.append(translation + np.array(START_IN_TILE))
    result = trimesh.util.concatenate(meshes)
    result.metadata["obstacle_faces"] = np.concatenate([mesh.metadata["obstacle_faces"] for mesh in meshes])
    return result, np.asarray(origins, dtype=np.float32)
