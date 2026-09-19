# Architecture contract

| Layer | Raw policy input | Encoder and recurrent core | Output | Rate |
|---|---:|---|---:|---:|
| Locomotion | 1073 | terrain+confidence `1024-128-32`, concat 81, GRU 128, MLP `256-128` | 12 joint offsets | 50 Hz |
| Navigation | 193 | obstacles `144-128-32`, concat 81, GRU 128, MLP `128-64` | 3 velocity actions | 10 Hz |
| Terrain vision | image `2x60x80` + state 45 | CNN `16-32-64`, feature 128, concat 173, GRU 256 | 256 heights + 256 confidence | 20 Hz |
| Obstacle vision | image `2x60x80` + state 45 | independent copy of the vision network | 72 ranges + 72 confidence | 20 Hz |

The physics rate is 200 Hz. All recurrent states are per environment. Resetting one environment
must clear only its policy, visual, command filter, last-action, sensor-age, and visibility memory.

Actor proprioception is angular velocity 3, projected gravity 3, relative joint position 12,
joint velocity 12, and previous executed joint action 12. Locomotion additionally receives command 3
and a continuous sine/cosine gait phase 2 plus the immutable reset heading expressed
as a body-frame XY unit vector 2. It excludes simulated linear velocity,
absolute pose, and contact truth. Privileged critics append linear velocity 3 and four foot-contact
flags to clean actor observations.

The teacher geometry values and their observability are separate. A confidence of zero removes a
value before policy encoding. Camera visibility, occlusion, and the two-second world-cell history
determine confidence; unknown space is not free space.

The initial simulated camera is 80 by 60, front-mounted, pitched down 20 degrees, with 0.2–6 metre
Z-depth. These are experiment configuration values, not a hardware calibration.
