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
# The gap floor sits this far below the datum so a robot that steps into the
# trench cannot step back out (deeper than the 0.16 m riser envelope), which
# makes "fell into the gap" an unambiguous failure instead of a recoverable dip.
GAP_FLOOR_DEPTH = 0.55
GAP_DEPTH_SPAN = 0.15
GAP_WIDTH_MIN = 0.08
GAP_WIDTH_MAX = 0.70
GAP_JUMP_WIDTH_M = 0.26
# Single-line geometry discriminator. <= threshold is physically short enough
# to walk; > threshold needs a leap. The mixed family has BOTH regimes, so
# terrain-family identity alone cannot select the runtime jump window.
GAP_JUMP_THRESHOLD_M = GAP_JUMP_WIDTH_M
GAP_SHORT_WIDTH_MAX = 0.24
GAP_LONG_WIDTH_MIN = 0.29
GAP_LONG_WIDTH_MAX = 0.42
# Reference tyre proportions (xyls999/27rc-dog-race modelling notes): outer
# 0.55, hole 0.33, axial thickness 0.165 and centre pitch 0.55. This training
# course retains the diameters but temporarily simplifies height/pitch below.
TIRE_OUTER_R = 0.275
TIRE_INNER_R = 0.165
# Official axial thickness is 0.165 m, which sits exactly at the Go1 climb
# limit (~0.16 m): the first row then behaves as a wall the policy can only
# grind against. Train at a mountable height and raise it once the family passes.
TIRE_THICKNESS = 0.05
TIRE_PITCH = 0.62
TIRE_LANES = 4
TIRE_SECTORS = 8
# Features must stop before the finish strip (33.10-34.10), otherwise a dense
# gap family would put trenches inside the goal region and the goal would be
# unreachable by construction.
FEATURE_STOP = 33.00  # at or above this a stride cannot cross it
# Solid ground between two trenches. The policy's leap covers roughly
# 0.6-1.0 m in the 0.45-0.60 m/s command band, so a shorter pad would leave the
# leap nowhere to land and the robot would touch down inside the next void.
GAP_LANDING_PAD_M = 1.10
GAP_RHYTHM_COUNT = 6   # consecutive voids at full difficulty
GAP_RHYTHM_MIN = 3     # ...and the run present at every difficulty
GAP_RHYTHM_PAD_M = 0.70
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

    soft_entrance = kind == "combined_soft"
    entrance_width_scale = {"combined_soft2": 0.25, "combined_soft3": 0.6}.get(kind, 1.0)
    entrance_depth_floor = 0.10 if kind in ("combined_soft4", "combined_soft15") else None
    all_void_depth_floor = 0.10 if kind == "combined_soft15" else None
    entrance_pad = 1.10 if kind == "combined_soft9" else None
    entrance_band = {"combined_soft6": (0.29, 0.42), "combined_soft7": (0.18, 0.26)}.get(kind)
    orig_kind = kind
    if kind not in {
        "flat", "slopes", "rough", "stairs", "course", "gap_course", "course_gaps",
        "stairs_up_long", "stairs_down_long", "stairs_mixed",
        "stairs_narrow_8_mixed", "slope_up", "slope_down", "slope_mixed",
        "gaps_only", "combined_flat",
        "obstacle_wall", "obstacle_clearance", "obstacle_gap_assembly",
        "obstacle_brick_pit", "obstacle_tire_array", "obstacle_beam",
        "obstacle_mixed",
        "tire_field", "gaps_short", "gaps_long", "combined",
        "combined_soft", "combined_soft2", "combined_soft3", "combined_soft4",
        "combined_soft6", "combined_soft7", "combined_soft8",
        "combined_soft9", "gaps_short_dense", "combined_soft12",
        "combined_soft13", "combined_soft14", "combined_soft15",
    }:
        raise ValueError(f"Unknown terrain kind: {kind}")
    if not 0 <= difficulty <= 1:
        raise ValueError("Terrain difficulty must be in [0, 1].")
    # The LP-ACRL terrain families are named per terrain type; two of them are
    # exactly an existing route with a clearer name, so they alias instead of
    # duplicating the assembly code:
    #   gaps_only     == the dedicated trench route ("gap_course")
    #   combined_flat == the mixed course, which already keeps a level approach
    if kind == "gaps_only":
        kind = "gap_course"
    elif kind == "combined_flat":
        kind = "course"
    elif kind in ("combined", "combined_soft", "combined_soft2", "combined_soft3",
                  "combined_soft4", "combined_soft6", "combined_soft7",
                  "combined_soft8", "combined_soft9", "combined_soft12",
                  "combined_soft13", "combined_soft14", "combined_soft15"):
        # The generic mixed route already packs grades, uneven stairways,
        # rough ground, bumps and leaps across the whole span, so the
        # combination family reuses it instead of a second implementation.
        # combined_soft (v32 P10 soft-curriculum) is the same route with the
        # three opening recovery pads widened by 0.15 m; it is training-only.
        kind = "course_gaps"

    def box(extents, position):
        mesh = trimesh.creation.box(extents=extents)
        mesh.apply_translation(position)
        return mesh

    # The base plate covers one whole tile so a straight course is never
    # interrupted by an inter-tile void. Its top stays level with z = 0, the
    # plane the course surface patches use, so the plate keeps acting as a
    # continuous floor directly beneath them.
    # Ground is one solid slab by default. Courses with trenches replace it with
    # x-segmented slabs plus a low floor under each void: a negative height patch
    # inside the slab is buried geometry, not a hole, so the robot would walk on
    # the slab top and never fall in.
    split_ground = kind in ("gap_course", "course_gaps", "gaps_short", "gaps_long",
                            "slope_down", "slope_mixed",
                            "obstacle_gap_assembly", "obstacle_beam",
                            "obstacle_brick_pit", "obstacle_mixed")
    pieces = [] if split_ground else [
        box((TILE_SIZE, TILE_SIZE, 0.2), (TILE_SIZE / 2.0, TILE_SIZE / 2.0, -0.1))
    ]
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
    elif kind in ("obstacle_wall", "obstacle_clearance", "obstacle_gap_assembly",
                  "obstacle_brick_pit", "obstacle_tire_array", "obstacle_beam",
                  "obstacle_mixed"):
        # 2027 ROBOCON obstacle course rebuilt as trimesh. Each segment starts
        # and ends on the datum so segments compose in any order, and each
        # returns (next_x, void) where void is (x0, x1, floor_depth) or None.
        # obstacle_mixed shuffles the segment order per environment, which is
        # how "mixed" and "every permutation" are covered by one family: 7! =
        # 5040 families is not a usable registry, and per-environment ordering
        # teaches order-invariance directly.
        route_x0 = COURSE_LEAD_IN_END + 0.10
        route_y0, route_y1 = 4.0, 12.0
        route_yc = 0.5 * (route_y0 + route_y1)
        route_w = route_y1 - route_y0

        def slab(x_len, y_width, height, x_centre, y_centre, z_base=0.0):
            pieces.append(box((x_len, y_width, height),
                              (x_centre, y_centre, z_base + height / 2.0)))

        def positioned(mesh, x, y, z, angle=0.0):
            if angle:
                import trimesh
                mesh.apply_transform(
                    trimesh.transformations.rotation_matrix(angle, [0.0, 0.0, 1.0]))
            mesh.apply_translation((x, y, z))
            pieces.append(mesh)

        def seg_wall(x, d):
            """Vertical wall across the corridor; it must be climbed. The
            reference value is 0.30 m for a wheeled robot; a Go1 clears about
            0.16 m, so this stops at 0.22 m and starts at 0.10 m."""
            # A 0.10 m thin lip gives the feet nowhere to land: the robot
            # rams the edge and topples. A tread-depth barrier behaves like a
            # stair rise, which the stair families already climb. The rise is
            # also held near what a Go1 actually clears (~0.10 m at level 3).
            height = 0.08 + 0.10 * d
            thickness = COURSE_TREAD_DEPTH
            slab(thickness, route_w, height, x + thickness / 2.0, route_yc)
            return x + thickness, None

        def seg_clearance(x, d):
            """Height-limit gate: posts outside the corridor, a beam spanning it,
            so the only way through is under the beam. Reference clearance is
            0.29 m and a standing Go1 is ~0.32-0.35 m, so it starts at 0.50 m
            and tightens toward the reference value."""
            import trimesh

            clearance = 0.50 - 0.21 * d
            depth, beam, post = 0.30, 0.05, 0.10
            for y_post in (route_y0 - 0.35, route_y1 + 0.35):
                slab(depth, post, clearance, x + depth / 2.0, y_post)
            slab(depth, route_w + 0.8, beam, x + depth / 2.0, route_yc,
                 z_base=clearance)
            return x + depth, None

        def seg_gap_assembly(x, d):
            """Step up, cross a gap, descend a 20 degree ramp. Reference: 0.36 m
            platforms with a 0.36 m gap, 0.12 m steps, 20 degree ramp."""
            platform = 0.08 + 0.10 * d
            step = 0.06 + 0.06 * d
            length = GAP_LANDING_PAD_M
            slab(0.32, route_w, step, x + 0.16, route_yc)
            x += 0.32
            slab(0.32, route_w, platform, x + 0.16, route_yc)
            x += 0.32
            # One continuous pad instead of two short platforms with a trench
            # between them. With the trench cut to 3 cm the trunk was still
            # pinned at its far lip, the body dropped to ground stance height
            # and slid sideways off the pad, so the discontinuity was the trap.
            # A single pad has none; the entry step and the 20 degree descent
            # below are the parts the other nine families prove are learnable.
            pad = 2.0 * length
            slab(pad, route_w, platform, x + pad / 2.0, route_yc)
            x += pad
            void = None
            ramp_len = float(max(0.60, platform / math.tan(math.radians(20.0))))
            steps = max(3, int(round(ramp_len / 0.12)))
            for i in range(steps):
                frac = (steps - i) / steps
                slab(ramp_len / steps, route_w, platform * frac,
                     x + (i + 0.5) * ramp_len / steps, route_yc)
            return x + ramp_len, void

        def seg_brick_pit(x, d):
            """Rough brick-and-bark pit with a low guardrail. Reference: 2x2 m
            L-shaped outline, 0.15 m rail, fixed irregular fill. The pit floor is
            lowered by splitting the ground; loose material is not simulated."""
            length = 2.0
            amplitude = 0.015 + 0.035 * d
            depth = 0.04 + 0.06 * d
            # Coarse on purpose: this patch was 25 x 100 faces per environment,
            # and at 512 envs the combined mesh became too heavy to build a
            # collider for every tile (only the first ~181 env ids were ever
            # simulated). 0.25 m still resolves the corrugation.
            gx = np.linspace(x, x + length, max(6, int(length / 0.25)))
            gy = np.linspace(route_y0, route_y1, max(6, int(route_w / 0.25)))
            mesh_x, mesh_y = np.meshgrid(gx, gy, indexing="ij")
            taper = np.sin(np.pi * (mesh_x - x) / length).clip(0.0, 1.0)
            corrugation = np.sin(9.0 * mesh_x) * np.cos(7.0 * mesh_y)
            pieces.append(_height_patch(
                gx, gy, -depth + amplitude * taper * corrugation))
            slab(length, 0.06, 0.15, x + length / 2.0, route_y0 + 0.03)
            return x + length, (x, x + length, depth)

        def seg_tire_array(x, d):
            """2x4 tire array. Reference: outer 0.55 m, inner 0.33 m, 0.165 m
            thick. Each tire is a ring of boxes, the same discretisation the
            reference uses for its own collision mesh. At ~0.165 m tall this is
            the easiest obstacle."""
            import trimesh

            # A wheel rolls over a hollow tyre; a foot needs a rim wide enough
            # to stand on, or a hole shallow enough to step into. Rim 0.20 m,
            # hole 0.20 m across and only 0.08-0.16 m deep.
            # Low and spaced out: 0.05-0.10 m tall, one tyre gap between
            # centres, so this is a first lesson rather than a trap.
            outer_r, inner_r, thickness = 0.30, 0.10, 0.05 + 0.05 * d
            rows = 1 + int(round(1 * d))
            cols = 4
            pitch = outer_r * 2.0 + 0.20
            segments = 6
            ring_r = 0.5 * (outer_r + inner_r)
            width = outer_r - inner_r
            block_len = 2.0 * math.pi * ring_r / segments * 1.05
            for row in range(rows):
                for col in range(cols):
                    cx = x + outer_r + row * pitch
                    cy = route_yc + (col - (cols - 1) / 2.0) * pitch
                    for k in range(segments):
                        angle = 2.0 * math.pi * k / segments
                        mesh = trimesh.creation.box((width, block_len, thickness))
                        positioned(mesh, cx + ring_r * math.cos(angle),
                                   cy + ring_r * math.sin(angle), thickness / 2.0,
                                   angle - math.pi / 2.0)
            return x + rows * pitch, None

        def seg_beam(x, d):
            """Single-log bridge over a void. Narrow by design, so this segment
            reports a trench (0.55 m down) and the branch floors it: leaving the
            beam is a trench fall, not a free detour."""
            # Start generous and tighten: 0.77 m at level 3, 0.45 m at full
            # difficulty, still a genuine balance requirement.
            # Widened and the drop made shallow: leaving the plank is a step
            # down, not a 0.55 m pit. A Go1 trained on flat ground and stairs
            # cannot cross a deep void on sight (measured 0.000 with fallen=1.0
            # at both 16 and 256 envs).
            width = 1.30 - 0.50 * d
            span = 1.20 + 1.00 * d
            drop = 0.10
            slab(span + 0.40, width, 0.10, x + span / 2.0, route_yc)
            return x + span, (x, x + span, drop)

        segments = {
            "wall": seg_wall,
            "clearance": seg_clearance,
            "gap_assembly": seg_gap_assembly,
            "brick_pit": seg_brick_pit,
            "tire_array": seg_tire_array,
            "beam": seg_beam,
        }
        if kind == "obstacle_mixed":
            # Internal curriculum: three obstacles at difficulty 0, all six at
            # full difficulty, always a fresh random order per environment.
            # Placing all six at every difficulty left this family the only
            # holdout for six straight level-1 stages (15-18 of 51 envs
            # reporting, against a full 51 for every other family), because the
            # policy rarely met any given ordering.
            count = min(len(segments), 3 + int(round(3.0 * float(difficulty))))
            names = [str(name) for name in rng.permutation(sorted(segments))][:count]
        else:
            names = [kind[len("obstacle_"):]]
        cursor = route_x0
        voids = []
        for name in names:
            cursor, void = segments[name](cursor, difficulty)
            if void is not None:
                voids.append(void)
            cursor += 0.70
        if split_ground:
            # Emit the corridor floor by hand: solid slab on the datum, with the
            # reported voids dropped to their own floor height.
            x_cut = 0.0
            for v0, v1, depth in sorted(voids):
                slab(v0 - x_cut, TILE_SIZE, 0.2, (x_cut + v0) / 2.0,
                     TILE_SIZE / 2.0, z_base=-0.2)
                slab(v1 - v0, TILE_SIZE, 0.2, (v0 + v1) / 2.0,
                     TILE_SIZE / 2.0, z_base=-depth - 0.2)
                x_cut = v1
            slab(TILE_SIZE - x_cut, TILE_SIZE, 0.2, (x_cut + TILE_SIZE) / 2.0,
                 TILE_SIZE / 2.0, z_base=-0.2)
    elif kind in ("course", "gap_course", "course_gaps",
                  "stairs_up_long", "stairs_down_long", "stairs_mixed",
                  "stairs_narrow_8_mixed", "slope_up", "slope_down", "slope_mixed",
                  "gaps_short", "gaps_long", "tire_field"):
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
        gap_widths = []
        gap_drops = []
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

        def slope_block(x0, sign=1, angle_max=None, run_max=1.60):
            # Half the grade blocks run up and half dip below the datum first, so
            # the route mixes climbs and descents instead of climbing only.
            # The single-direction terrain families pass sign=-1 for downhill and
            # an explicit angle_max so a pure-slope route follows the LP-ACRL
            # parameterisation (gradient 0.00 -> 0.40, i.e. 0 -> 21.8 degrees)
            # instead of the mixed route's 5 -> 30 degree envelope.
            upper = 5.0 + 25.0 * difficulty if angle_max is None else float(angle_max)
            angle_deg = float(rng.uniform(min(5.0, upper), max(5.0, upper)))
            run_up = float(rng.uniform(0.90, max(0.90, min(1.60, run_max))))
            run_down = float(rng.uniform(0.90, max(0.90, min(1.60, run_max))))
            peak = float(sign * run_up * math.tan(math.radians(angle_deg)))
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

        def stairs_block(x0, steps=None, kind="stairs", max_end=None,
                         tread_range=None, riser_height=None, landing_range=(0.30, 0.60),
                         high_step_rise=None):
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
            rise = (rng.uniform(0.04, 0.04 + 0.12 * difficulty, size=count)
                    if riser_height is None else np.full(count, float(riser_height)))
            tread_up = rng.uniform(low_tread, high_tread, size=count)
            tread_down = rng.uniform(low_tread, high_tread, size=count)
            # Reserve a single true high step on the finishing staircase, with
            # a long tread on both its ascent and matching descent. Other stair
            # families and the three-riser approach remain their own lessons.
            if high_step_rise is not None:
                if count < 4 or high_step_rise <= 0:
                    raise ValueError("high step requires at least four stairs and positive rise")
                high_index = 2
                rise[high_index] = max(float(rise[high_index]), float(high_step_rise))
                tread_up[high_index] = max(float(tread_up[high_index]), 0.47)
                descent = rise[::-1].copy()
                tread_down[count - 1 - high_index] = max(
                    float(tread_down[count - 1 - high_index]), 0.47
                )
            else:
                descent = rng.permutation(rise)
            landing = float(rng.uniform(*landing_range))
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
                    "riser_heights": [float(value) for value in rise],
                    "descent_heights": [float(value) for value in descent],
                    "tread_depths": [float(value) for value in (*tread_up, *tread_down)],
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

        def gap_block(x0, pad=GAP_LANDING_PAD_M, rhythm=False,
                      width_lo=None, width_hi=None, depth_floor=None):
            # A rectangular trench: short ones are walked and long ones leaped.
            # The void spans
            # [x0, x2] across the whole tile width, so it cannot be walked
            # around; its floor sits a fixed depth below the datum (deeper than
            # the 0.16 m step envelope) so stepping in is unrecoverable. The far
            # lip may sit up to one legal riser lower than the near edge. Width
            # and drop come from the difficulty, so the same builder serves the
            # dedicated-gap curriculum and the mixed route. The solid ground and
            # the low floor are emitted by the caller from this registration.
            # Within-tile curriculum: the route opens with gaps the warmed
            # walking gait can clear and widens toward the finish, so each
            # episode is itself a lesson from easy to hard (the user-approved
            # standard-to-aggressive progression).
            # Width is the curriculum variable. It starts narrow enough that the
            # walking gait already carries it at difficulty 0 and widens towards
            # the published generalist range (~0.4 m) at difficulty 1. Each
            # episode still runs easy-to-hard because the width also grows along
            # the route, but the whole range stays inside what one policy can
            # learn instead of opening at a jump specialist's exam width.
            if width_lo is None or width_hi is None:
                # Default band stops at the physically crossable limit: the old
                # 0.70 m maximum could not be cleared at the 0.8 m/s speed cap,
                # so every un-banded route (gap_course, combined) now builds
                # gaps a Go1 can actually clear.
                base_width = GAP_WIDTH_MIN + (GAP_LONG_WIDTH_MAX - GAP_WIDTH_MIN) * difficulty
            else:
                # Explicit band: short and long gap widths remain separated
                # even when both kinds are placed on a mixed tile.
                base_width = float(width_lo) + (float(width_hi) - float(width_lo)) * difficulty
            progress = float(np.clip((x0 - COURSE_LEAD_IN_END)
                                     / (COURSE_FEATURE_END - COURSE_LEAD_IN_END), 0.0, 1.0))
            band_lo = GAP_WIDTH_MIN if width_lo is None else float(width_lo)
            band_hi = GAP_LONG_WIDTH_MAX if width_hi is None else float(width_hi)
            width = float(np.clip(base_width * (0.85 + 0.30 * progress), band_lo, band_hi))
            # The floor sits deep enough that standing in the trench is
            # impossible and a foot below the lip trips the fell-into-gap
            # termination, so the route cannot be cheated by walking through
            # the void and climbing out.
            depth = ((GAP_FLOOR_DEPTH if depth_floor is None else float(depth_floor))
                   + GAP_DEPTH_SPAN * difficulty + 0.10 * progress)
            drop = 0.0 if depth_floor is not None else depth - GAP_FLOOR_DEPTH
            lip = 0.05
            x1 = x0 + width
            x2 = x1 + lip
            x3 = x2 + pad  # landing pad before the next feature
            gap_widths.append(width)
            gap_drops.append(drop)
            blocks.append(
                {
                    "kind": "gap",
                    "start": float(x0),
                    "end": float(x3),
                    "void_end": float(x2),
                    "width": width,
                    "drop": drop,
                    "depth": depth,
                    "rhythm": bool(rhythm),
                    "jump_required": bool(width > GAP_JUMP_WIDTH_M),
                }
            )
            return x3

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
        # Route length is part of the curriculum: the easy end packs roughly
        # half the span with features and leaves the remainder as flat finish
        # ground, so the policy is not asked to hold a full-length obstacle
        # course before it can hold a short one. Full length at difficulty 1.
        # A block can overshoot fill_stop, so the realised route is longer than
        # the factor suggests. At 0.45 the mixed route still ran ~26 m and
        # compounded per-block failures kept combined at ~2.5%; 0.20 leaves a
        # short route the policy can actually finish, growing to the full
        # course at difficulty 1.
        fill_stop = COURSE_LEAD_IN_END + (COURSE_FEATURE_END - COURSE_LEAD_IN_END) * (
            0.20 + 0.80 * min(1.0, max(0.0, difficulty)))

        flat_builders = (rough_block, bumps_block)
        cursor = COURSE_LEAD_IN_END
        if kind == "gap_course":
            # Dedicated jump curriculum: spawn 1.3 m before the first trench so
            # every rollout is one full leap attempt, not a long walk to it.
            # Shorter than the plain course's 2.3 m lead-in; the V18 lesson
            # (>=1.3 m of gait roll-in before the first feature) still holds.
            cursor = 4.30
        if kind == "course_gaps":
            # A true mixed course must contain EVERY requested obstacle, even at
            # difficulty 0. Mix the order of the middle sections; keep clustered
            # short and long trenches separate so the single-width discriminator
            # has unambiguous walking and jumping lessons on every tile.
            def mixed_tires(x0, rows):
                ring_r = 0.5 * (TIRE_OUTER_R + TIRE_INNER_R)
                ring_w = TIRE_OUTER_R - TIRE_INNER_R
                sector_len = 2.0 * math.pi * ring_r / TIRE_SECTORS * 1.05
                for row in range(rows):
                    centre_x = x0 + row * TIRE_PITCH + TIRE_OUTER_R
                    for lane in range(TIRE_LANES):
                        centre_y = 8.0 + (lane - (TIRE_LANES - 1) / 2.0) * TIRE_PITCH
                        for sector in range(TIRE_SECTORS):
                            angle = 2.0 * math.pi * sector / TIRE_SECTORS
                            ring_piece = trimesh.creation.box((ring_w, sector_len, TIRE_THICKNESS))
                            ring_piece.apply_transform(trimesh.transformations.rotation_matrix(
                                angle - math.pi / 2.0, [0.0, 0.0, 1.0]))
                            ring_piece.apply_translation((
                                centre_x + ring_r * math.cos(angle),
                                centre_y + ring_r * math.sin(angle),
                                TIRE_THICKNESS / 2.0,
                            ))
                            pieces.append(ring_piece)
                x1 = x0 + rows * TIRE_PITCH
                blocks.append({"kind": "tire_field", "start": float(x0),
                               "end": float(x1), "peak": float(TIRE_THICKNESS),
                               "tyres": rows * TIRE_LANES})
                return x1

            def mixed_grades(x0):
                return slope_block(slope_block(x0, sign=1), sign=-1)

            # Pick one riser height per tile: the first and last stair sections
            # must match step-for-step, while the narrower tread belongs only
            # to the 8-up / 8-down finish section.
            shared_riser = float(rng.uniform(0.04, 0.04 + 0.12 * difficulty))

            def mixed_stairs(x0):
                # The late eight-up/eight-down feature carries the sustained
                # climbing lesson; keep the middle block at three steps so
                # neither tire field nor the final landing overruns the tile.
                return stairs_block(x0, steps=3, riser_height=shared_riser)

            entrance_count = (1 if orig_kind == "combined_soft13"
                              else 2 if orig_kind == "combined_soft14" else 3)
            middle_tire_rows = 4 if difficulty <= 0.30 else 3
            if orig_kind == "combined_soft12":
                # diagnostic: no tyre band right after the entrance voids
                middle_tire_rows = 0
            cursor = 4.30
            for _ in range(entrance_count):
                # These voids are walked, so a shorter pad is safe; long-gap
                # takeoff/landing platforms below remain at 1.10 m.
                cursor = gap_block(cursor,
                                   pad=entrance_pad if entrance_pad is not None
                                   else 0.65 + (0.15 if soft_entrance else 0.0),
                                   width_lo=(entrance_band[0] if entrance_band
                                             else GAP_WIDTH_MIN * entrance_width_scale),
                                   width_hi=(entrance_band[1] if entrance_band
                                             else GAP_SHORT_WIDTH_MAX * entrance_width_scale),
                                   depth_floor=entrance_depth_floor)
            # Follow the approved overview: tire -> up/down slope -> ordinary
            # stairs. Widths, heights and contact patterns still vary by seed.
            middle = (lambda x: mixed_tires(x, middle_tire_rows), mixed_grades, mixed_stairs)
            for section in middle:
                cursor = section(cursor)
            for _ in range(3):
                cursor = gap_block(cursor, width_lo=GAP_LONG_WIDTH_MIN,
                                   width_hi=GAP_LONG_WIDTH_MAX, rhythm=True,
                                   depth_floor=all_void_depth_floor)
            # Preserve tyres on both sides of the late, narrower staircase.
            # The shorter first band reserves 5.9 m for 16 narrow treads and a
            # landing without moving any obstacle into the finish strip.
            cursor = mixed_tires(cursor, middle_tire_rows)
            cursor = stairs_block(cursor, steps=8, kind="final_stairs",
                                  tread_range=(0.35, 0.35),
                                  landing_range=(0.30, 0.30),
                                  riser_height=shared_riser, high_step_rise=0.17)
            # The goal starts at x=33.10.  Budget the final rows to x=33.05:
            # the last ring ends another 0.07 m earlier than the row envelope.
            # The older 32.90 row budget excluded a valid third row for some
            # four-rank terrain seeds, despite every ring remaining before goal.
            finishing_tire_end = FEATURE_STOP + 0.05
            rows = int((finishing_tire_end - cursor + 1e-9) // TIRE_PITCH)
            if rows < 3:
                raise ValueError(f"combined route has no room for finishing tyres: {cursor:.3f}")
            cursor = mixed_tires(cursor, rows)
        elif kind in ("stairs_up_long", "stairs_down_long", "stairs_mixed",
                    "stairs_narrow_8_mixed", "slope_up", "slope_down", "slope_mixed",
                    "gaps_short", "gaps_long", "tire_field"):
            # LP-ACRL terrain families. Each training environment carries one
            # terrain type, so the task space becomes the paper's categorical
            # terrain dimension crossed with its continuous difficulty and
            # velocity levels. Every family reuses the datum-preserving block
            # builders, which is what keeps blocks composable and leaves the
            # level approach lip-free.
            def ramp(x0, h_from, h_to):
                # Grade between two heights at a fixed ~20 deg so the return to
                # the datum stays inside the trained gradient envelope: the
                # LP-ACRL slope families top out at 0.40 (21.8 deg) and clear
                # their 90% gate there, while the previous 24 deg return ramp
                # left both long-stair families stuck at the ramp foot
                # (measured completion 0.016 / 0.035 with failure at course x ~ 0).
                run = float(max(0.90, abs(h_to - h_from) / math.tan(math.radians(20.0))))
                grid_x = np.linspace(x0, x0 + run, 40)
                grid_y = np.linspace(4.0, 12.0, 41)
                mesh_x, mesh_y = np.meshgrid(grid_x, grid_y, indexing="ij")
                taper = np.sin(
                    np.pi * (mesh_y - grid_y[0]) / (grid_y[-1] - grid_y[0])
                ).clip(0.0, 1.0)
                field = h_from + (h_to - h_from) * (mesh_x - x0) / run
                pieces.append(_height_patch(grid_x, grid_y, field * taper))
                blocks.append({
                    "kind": "ramp", "start": float(x0), "end": float(x0 + run),
                    "peak": float(max(abs(h_from), abs(h_to))),
                })
                return x0 + run

            def stairway(descending):
                # Continuous single-direction staircase at the paper's step
                # heights (0.05 -> 0.20 m). The trained direction is the stair
                # run itself: an ascending family climbs the risers then ramps
                # back to the datum, a descending family ramps up first and then
                # walks the risers down. Either way the block ends at the datum.
                cursor_inner = 4.30
                while True:
                    # Cost the block before building it: one stairway plus its
                    # ramp is up to ~12 m, far more than the 5.3 m headroom the
                    # mixed-route loop assumes, so an unguarded loop overran
                    # COURSE_FEATURE_END and tripped the route assertion.
                    rise = float(min(0.20, (0.05 + 0.15 * difficulty) * rng.uniform(0.85, 1.15)))
                    tread = float(rng.uniform(0.36, 0.40))
                    # Scale the riser count with difficulty. A long staircase
                    # at the easy end forces a 0.8-0.96 m summit that the policy
                    # could not hold heading on (stairs_up_long 0.298); the
                    # shorter 5-7 riser staircase at d1 was worth +0.40 on that
                    # family in s18, and the route length is unchanged because
                    # the fill loop always runs to COURSE_FEATURE_END.
                    span = min(1.0, max(0.0, difficulty / 0.6))
                    count_lo = int(round(3 + 5 * span))
                    count_hi = int(round(5 + 7 * span))
                    count = int(rng.integers(count_lo, count_hi + 1))
                    plateau = float(rng.uniform(0.55, 0.85))
                    # The ascending run starts at the datum and climbs to
                    # count*rise; the descending run starts at the summit and
                    # steps back down to it. Seeding `height` with the summit on
                    # both paths double-counted the climb (the ramp then
                    # descended from twice the real height and overran the
                    # route), so only the descending path preloads it.
                    height = float(count * rise) if descending else 0.0
                    ramp_run = float(max(0.90, (count * rise) / math.tan(math.radians(20.0))))
                    cost = count * tread + plateau + ramp_run
                    if cursor_inner + cost > FEATURE_STOP - 0.10:
                        # Fit a shorter final stairway rather than replacing
                        # the rest of a pure stair course with rough/flat filler.
                        for candidate in range(count - 1, 2, -1):
                            candidate_cost = (candidate * tread + plateau + max(
                                0.90, candidate * rise / math.tan(math.radians(20.0))))
                            if cursor_inner + candidate_cost <= FEATURE_STOP - 0.10:
                                count = candidate
                                break
                        else:
                            break
                        height = float(count * rise) if descending else 0.0
                    block_start = cursor_inner
                    if descending:
                        cursor_inner = ramp(cursor_inner, 0.0, height)
                    for _ in range(count):
                        if descending:
                            height -= rise
                        else:
                            height += rise
                        if height > 1e-6:
                            pieces.append(box(
                                (tread, 8.0, height),
                                (cursor_inner + tread / 2.0, 8.0, height / 2.0),
                            ))
                        cursor_inner += tread
                    # A descending staircase finishes on the datum, so its
                    # plateau would be a zero-height box: PhysX refuses to
                    # cook that, but dropping it removes the route's floor
                    # (stairs_down_long went to 0.000). Clamp to a thin slab.
                    plateau_height = max(height, 0.02)
                    pieces.append(box(
                        (plateau, 8.0, plateau_height),
                        (cursor_inner + plateau / 2.0, 8.0, plateau_height / 2.0),
                    ))
                    cursor_inner += plateau
                    if not descending:
                        cursor_inner = ramp(cursor_inner, height, 0.0)
                    risers.extend([rise] * count)
                    treads.extend([tread] * count)
                    blocks.append({
                        "kind": "stair_descent" if descending else "stair_ascent",
                        "start": float(block_start),
                        "end": float(cursor_inner),
                        "peak": float(count * rise),
                        "steps": count,
                    })
                    if FEATURE_STOP - cursor_inner < 2.50:
                        break
                return cursor_inner

            def tyre_band(x0):
                """Continuous tyre band across the whole remaining span.

                Official proportions, two lanes, centres one outer diameter
                apart so the tyres touch: the robot is on tyres for the entire
                segment instead of meeting a single 2x4 array. Two lanes only,
                because the reference discretises each tyre into 48 convex
                sectors and a wider field cannot cook a collider at 512 envs.
                """
                import trimesh
                y_centre = 8.0
                ring_r = 0.5 * (TIRE_OUTER_R + TIRE_INNER_R)
                ring_w = TIRE_OUTER_R - TIRE_INNER_R
                block_len = 2.0 * math.pi * ring_r / TIRE_SECTORS * 1.05
                stop = FEATURE_STOP
                x = float(x0)
                rows = 0
                while x + TIRE_PITCH <= stop - 0.10:
                    centre_x = x + TIRE_OUTER_R
                    for lane in range(TIRE_LANES):
                        centre_y = y_centre + (lane - (TIRE_LANES - 1) / 2.0) * TIRE_PITCH
                        for k in range(TIRE_SECTORS):
                            angle = 2.0 * math.pi * k / TIRE_SECTORS
                            mesh = trimesh.creation.box((ring_w, block_len, TIRE_THICKNESS))
                            mesh.apply_transform(trimesh.transformations.rotation_matrix(
                                angle - math.pi / 2.0, [0.0, 0.0, 1.0]))
                            mesh.apply_translation((
                                centre_x + ring_r * math.cos(angle),
                                centre_y + ring_r * math.sin(angle),
                                TIRE_THICKNESS / 2.0,
                            ))
                            pieces.append(mesh)
                    x += TIRE_PITCH
                    rows += 1
                blocks.append({"kind": "tire_field", "start": float(x0), "end": float(x),
                               "peak": float(TIRE_THICKNESS), "tyres": int(rows * TIRE_LANES)})
                return x

            if kind in ("gaps_short", "gaps_short_dense"):
                cursor = 4.30
                dense = kind == "gaps_short_dense"
                while cursor < FEATURE_STOP - 2.30:
                    cursor = gap_block(cursor,
                                       pad=0.65 if dense else GAP_LANDING_PAD_M,
                                       width_lo=GAP_WIDTH_MIN,
                                       width_hi=GAP_SHORT_WIDTH_MAX)
            elif kind == "gaps_long":
                cursor = 4.30
                while cursor < FEATURE_STOP - 2.30:
                    cursor = gap_block(cursor, width_lo=GAP_LONG_WIDTH_MIN,
                                       width_hi=GAP_LONG_WIDTH_MAX)
            elif kind == "tire_field":
                cursor = tyre_band(4.30)
            elif kind in ("stairs_mixed", "stairs_narrow_8_mixed"):
                # Repeated up-eight/down-eight sections for the full route.
                # A final shorter stairway, never rough filler, closes the span.
                narrow = kind == "stairs_narrow_8_mixed"
                tread_span = (0.35, 0.35) if narrow else (0.37, 0.43)
                stair_kind = "final_stairs" if narrow else "long_stairs"
                common_riser = (float(rng.uniform(0.04, 0.04 + 0.12 * difficulty))
                                if narrow else None)
                cursor = 4.30
                while cursor + 16 * tread_span[1] + 0.30 <= FEATURE_STOP - 0.10:
                    cursor = stairs_block(cursor, steps=8, kind=stair_kind,
                                          tread_range=tread_span,
                                          landing_range=(0.30, 0.30),
                                          riser_height=common_riser)
                remaining = FEATURE_STOP - 0.10 - cursor
                partial_steps = min(7, int((remaining - 0.30) // (2 * tread_span[1])))
                if partial_steps >= 3:
                    cursor = stairs_block(cursor, steps=partial_steps, kind=stair_kind,
                                          tread_range=tread_span,
                                          landing_range=(0.30, 0.30),
                                          riser_height=common_riser)
            elif kind == "stairs_up_long":
                cursor = stairway(descending=False)
            elif kind == "stairs_down_long":
                cursor = stairway(descending=True)
            else:
                # Pure grade route at the LP-ACRL slope parameterisation
                # (gradient 0.00 -> 0.40, i.e. 0 -> 21.8 degrees).
                paper_angle = float(min(30.0, max(4.0, math.degrees(math.atan(0.40 * difficulty)))))
                cursor = 4.30
                flip = -1 if kind == "slope_down" else 1
                while cursor < FEATURE_STOP - 1.90:
                    available_run = min(1.60, (FEATURE_STOP - 0.10 - cursor) / 2.0)
                    cursor = slope_block(cursor, sign=flip, angle_max=paper_angle,
                                         run_max=available_run)
                    if kind == "slope_mixed":
                        flip = -flip
        elif kind == "gap_course":
            # Dedicated jump-curriculum route: dense gaps separated by short
            # landing strips, nothing else. Difficulty controls width and the
            # drop of the far lip.
            while cursor < FEATURE_STOP - 2.30:
                cursor = gap_block(cursor)
            if COURSE_FEATURE_END - cursor > 0.10:
                cursor = rough_block(cursor, fill_stop)
        else:
            for index in rng.permutation(len(opening)):
                cursor = opening[int(index)](cursor)
            # The long stairway closes the guaranteed opening and truncates itself to
            # whatever span is left, so the mix cannot overrun the feature end.
            cursor = long_stairs_block(cursor)
            # Fill thresholds are set so the longest remaining block still fits before
            # COURSE_FEATURE_END: 5.20 m for the graded pool, 2.20 m for the flat pool.
            while cursor < fill_stop - 5.30:
                for index in rng.permutation(len(graded_builders)):
                    if cursor >= fill_stop - 5.30:
                        break
                    cursor = graded_builders[int(index)](cursor)
            while cursor < fill_stop - 2.30:
                for index in rng.permutation(len(flat_builders)):
                    if cursor >= fill_stop - 2.30:
                        break
                    cursor = flat_builders[int(index)](cursor)
            if COURSE_FEATURE_END - cursor > 0.10:
                cursor = rough_block(cursor, fill_stop)
        if cursor > COURSE_FEATURE_END + 1e-6:
            raise ValueError(
                "course features overrun the level approach: "
                f"cursor={cursor:.3f} end={COURSE_FEATURE_END:.3f} kind={kind} "
                f"last={blocks[-1] if blocks else None}"
            )

        # Level ground after the last feature. It must extend past both the
        # navigation goal and the declared finish region: ending the mesh where
        # the finish region begins leaves the last metre without ground, so every
        # episode fell off the edge and the goal was unreachable by construction.
        # Start the flat strip where the features actually stopped, not at
        # COURSE_FEATURE_END: a family that stops early would otherwise leave
        # an unpaved hole between its last feature and the strip.
        strip_start = float(min(max(cursor, COURSE_LEAD_IN_END), fill_stop))
        finish_x = np.linspace(
            strip_start, TILE_SIZE - 0.90,
            max(2, int(round((TILE_SIZE - 0.90 - strip_start) / 0.20)) + 1),
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
            "gap_count": len(gap_widths),
            "gap_widths": list(gap_widths),
            "gap_max_width": float(max(gap_widths) if gap_widths else 0.0),
            "gap_drops": list(gap_drops),
            "wide_gap_count": int(sum(1 for w in gap_widths if w > GAP_JUMP_WIDTH_M)),
            "feature_end": float(COURSE_FEATURE_END),
            "flat_approach_m": float(START_IN_TILE[0] + 33.10 - COURSE_FEATURE_END),
            "finish_start_offset": 33.10,
            "finish_end_offset": 34.10,
        }

    if split_ground and course_metadata is not None:
        # Beneath a downward grade, a z=0 slab would cover the negative
        # height patch: both contact and ray cast would see flat ground. Cut
        # only its walking corridor, keeping solid shoulders outside y=4..12.
        # Trenches still retain their deep floors across the FULL tile width.
        ground_cursor = 0.0
        cutouts = sorted(
            (b for b in course_metadata["blocks"] if b["kind"] == "gap"
             or (b["kind"] == "slope" and b["peak"] < 0)),
            key=lambda b: b["start"],
        )
        for block in cutouts:
            start = block["start"]
            if start > ground_cursor + 1e-6:
                pieces.append(box(
                    (start - ground_cursor, TILE_SIZE, 0.2),
                    (0.5 * (ground_cursor + start), TILE_SIZE / 2.0, -0.1),
                ))
            if block["kind"] == "gap":
                void_end, depth = block["void_end"], block["depth"]
                pieces.append(box(
                    (void_end - start, TILE_SIZE, 0.10),
                    (0.5 * (start + void_end), TILE_SIZE / 2.0, -depth - 0.05),
                ))
                ground_cursor = void_end
            else:
                length = block["end"] - start
                for y0, y1 in ((0.0, 4.0), (12.0, TILE_SIZE)):
                    pieces.append(box(
                        (length, y1-y0, 0.2),
                        (start + length / 2.0, (y0+y1) / 2.0, -0.1),
                    ))
                ground_cursor = block["end"]
        if ground_cursor < TILE_SIZE - 1e-6:
            pieces.append(box(
                (TILE_SIZE - ground_cursor, TILE_SIZE, 0.2),
                (0.5 * (ground_cursor + TILE_SIZE), TILE_SIZE / 2.0, -0.1),
            ))
        course_metadata["ground_segments"] = len(pieces)

    # PhysX refuses to cook a mesh that contains a degenerate face, and a
    # failed cook leaves the whole terrain without a collider: every
    # environment then falls through the floor and terminates at the same
    # step (stage s19: all 512 environments ended after exactly 11 steps).
    # Drop zero-volume pieces before the traversable count is taken so the
    # obstacle face mask stays aligned with the final mesh.
    # Trimesh.volume is only meaningful for a watertight mesh; the ramps and
    # height patches here are open surfaces, whose volume reads as zero, so
    # filtering on volume silently deleted the ramps and left
    # stairs_down_long with no way up to its summit (completion 0.000).
    # A piece is degenerate only if one of its bounding-box extents is zero.
    pieces = [piece for piece in pieces if float(min(piece.extents)) > 1e-9]
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
                course_difficulty_max: float | None = None,
                tasks: list[tuple[str, float]] | None = None):
    import trimesh

    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    kinds = ("flat", "slopes", "rough", "stairs")
    # Keep in step with task_curriculum.TERRAIN_FAMILIES; this copy validates an
    # explicit LP-ACRL task table and is imported by the sampler path as well.
    terrain_families = (
        "stairs_up_long", "stairs_down_long", "stairs_mixed",
        "stairs_narrow_8_mixed", "slope_up", "slope_down", "slope_mixed",
        "tire_field", "gaps_short", "gaps_long", "combined",
        "combined_soft", "combined_soft2", "combined_soft3", "combined_soft4",
        "combined_soft6", "combined_soft7", "combined_soft8", "combined_soft9",
        "gap_course", "gaps_short_dense", "combined_flat", "course",
        "combined_soft12", "combined_soft13", "combined_soft14", "combined_soft15",
    )
    if mode != "mixed" and mode not in (
        *kinds, "course", "gap_course", "course_gaps", "lp_acrl", *terrain_families
    ):
        raise ValueError(f"Unknown terrain mode: {mode}")
    if (course_difficulty_min is None) != (course_difficulty_max is None):
        raise ValueError("course difficulty bounds must be provided together")
    if course_difficulty_min is not None:
        if not 0 <= course_difficulty_min <= course_difficulty_max <= 1:
            raise ValueError("course difficulty bounds must satisfy 0 <= min <= max <= 1")
    if tasks is not None:
        if len(tasks) != num_envs:
            raise ValueError(f"task table has {len(tasks)} entries for {num_envs} environments")
        unknown = {str(kind) for kind, _ in tasks} - set(terrain_families)
        if unknown:
            raise ValueError(f"task table contains unknown terrain families: {sorted(unknown)}")
    width = math.ceil(math.sqrt(num_envs))
    origins, meshes = [], []
    mixed_reward_segments = []  # per-tile bounds for optional reward only; no mesh changes
    long_gap_segments = []  # reward-only [start, far_lip, end]; never exposed to actor
    all_gap_segments = []  # termination-only bounds; no actor observations
    rng = np.random.default_rng(seed)
    for index in range(num_envs):
        translation = np.array([(index // width) * TILE_SPACING, (index % width) * TILE_SPACING, 0.0])
        if tasks is not None:
            # Explicit LP-ACRL stage plan: this environment's family and
            # difficulty are decided by the sampler, not resampled here.
            tile_kind, tile_difficulty = str(tasks[index][0]), float(tasks[index][1])
        else:
            if mode == "lp_acrl":
                # LP-ACRL task space under uniform sampling: the categorical terrain
                # dimension cycles so every terrain type appears in each batch, while
                # the continuous difficulty is drawn per environment. This is the
                # paper's Uniform baseline and the substrate that the
                # learning-progress sampler replaces once its estimates are warm.
                tile_kind = terrain_families[index % len(terrain_families)]
            else:
                tile_kind = kinds[index % len(kinds)] if mode == "mixed" else mode
            tile_difficulty = difficulty
            if tile_kind in (
                "course", "gap_course", "course_gaps",
                "stairs_up_long", "stairs_down_long", "stairs_mixed",
                "slope_up", "slope_down", "slope_mixed", "gaps_only", "combined_flat",
            ):
                if course_difficulty_min is None:
                    tile_difficulty = float(np.clip(difficulty + rng.uniform(-0.20, 0.20), 0.45, 1.0))
                else:
                    tile_difficulty = float(rng.uniform(course_difficulty_min, course_difficulty_max))
        for attempt in range(6):
            try:
                stream = rng if attempt == 0 else np.random.default_rng(int(rng.integers(1 << 62)))
                mesh = build_tile(tile_kind, tile_difficulty, stream, obstacles)
                break
            except ValueError as error:
                if tile_kind not in (
                        "combined", "combined_soft", "combined_soft2",
                        "combined_soft3", "combined_soft4", "combined_soft6",
                        "combined_soft7", "combined_soft8", "combined_soft9",
                        "combined_soft12", "combined_soft13", "combined_soft14", "combined_soft15"
                ) or "finishing tyres" not in str(error) or attempt == 5:
                    raise
        bounds = np.full((24, 3), np.nan, dtype=np.float32)
        profile = mesh.metadata.get("course_profile", {})
        import os as _os
        _threshold = float(_os.environ.get("GO1_V8_LONG_GAP_THRESHOLD",
                                           GAP_JUMP_WIDTH_M))
        long_gaps = [block for block in profile.get("blocks", [])
                     if block["kind"] == "gap" and block["width"] > _threshold]
        if len(long_gaps) > len(bounds):
            raise ValueError("too many long gaps for the reward metadata")
        for slot, block in enumerate(long_gaps):
            bounds[slot] = [block[key] - START_IN_TILE[0]
                            for key in ("start", "void_end", "end")]
        long_gap_segments.append(bounds)
        all_gaps = [block for block in profile.get("blocks", []) if block["kind"] == "gap"]
        all_bounds = np.full((24, 3), np.nan, dtype=np.float32)
        if len(all_gaps) > len(all_bounds):
            raise ValueError("too many gaps for the termination metadata")
        for slot, block in enumerate(all_gaps):
            all_bounds[slot] = [block[key] - START_IN_TILE[0]
                                for key in ("start", "void_end", "end")]
        all_gap_segments.append(all_bounds)
        if tile_kind in ("combined", "combined_soft", "combined_soft2",
                         "combined_soft3", "combined_soft4", "combined_soft6",
                         "combined_soft7", "combined_soft8", "combined_soft9",
                         "combined_soft12", "combined_soft13", "combined_soft14", "combined_soft15"):
            blocks = mesh.metadata["course_profile"]["blocks"]
            gaps = blocks[:3]
            tires = [block for block in blocks[3:] if block["kind"] == "tire_field"]
            if len(gaps) != 3 or any(block["kind"] != "gap" for block in gaps) or not tires:
                if tile_kind == "combined":
                    raise ValueError("combined reward geometry requires three opening gaps and a tire band")
                mixed_reward_segments.append(np.full((4, 3), np.nan, dtype=np.float32))
                bounds = None
            if bounds is not None:
                bounds = [[b["start"] - START_IN_TILE[0], b["void_end"] - START_IN_TILE[0],
                           b["end"] - START_IN_TILE[0]] for b in gaps]
                tire = tires[0]
                bounds.append([tire["start"] - START_IN_TILE[0],
                               tire["end"] - START_IN_TILE[0], tire["end"] - START_IN_TILE[0]])
                mixed_reward_segments.append(np.asarray(bounds, dtype=np.float32))
        else:
            mixed_reward_segments.append(np.full((4, 3), np.nan, dtype=np.float32))
        mesh.apply_translation(translation)
        meshes.append(mesh)
        origins.append(translation + np.array(START_IN_TILE))
    result = trimesh.util.concatenate(meshes)
    result.metadata["obstacle_faces"] = np.concatenate([mesh.metadata["obstacle_faces"] for mesh in meshes])
    result.metadata["mixed_reward_segments"] = np.stack(mixed_reward_segments)
    result.metadata["long_gap_segments"] = np.stack(long_gap_segments)
    result.metadata["all_gap_segments"] = np.stack(all_gap_segments)
    return result, np.asarray(origins, dtype=np.float32)
