#!/usr/bin/env bash
cd /config/Desktop/sim
ls source/go1_hier_sim/go1_hier_sim/training/
echo === on_policy rollout/update skeleton ===
grep -nE 'def (rollouts|update|act|_rollout)|actions_buf|obs_buf|hidden_state|reset_idx|self.policy\(' source/go1_hier_sim/go1_hier_sim/training/on_policy.py | head -40
echo === v23 runtime brake integration ===
grep -n 'v23' source/go1_hier_sim/go1_hier_sim/envs/runtime.py | head
echo === checkpoint load API ===
grep -nE 'def load_actor|def load_checkpoint|class .*Actor|def save_checkpoint' source/go1_hier_sim/go1_hier_sim/checkpoints.py source/go1_hier_sim/go1_hier_sim/models.py | head