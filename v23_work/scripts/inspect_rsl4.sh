#!/usr/bin/env bash
cd /config/Desktop/sim
/root/env_isaaclab/bin/python -c "import inspect; from rsl_rl.algorithms import PPO; src=inspect.getsource(PPO.update); import re; print([w for w in re.findall(r.\x27distill\w*\x27, src)]); print(len(src))" 2>&1 | head -3
grep -rn "distillation" /root/env_isaaclab/lib/python3.11/site-packages/rsl_rl/runners/on_policy_runner.py | head -8
grep -rn "training_type" /root/env_isaaclab/lib/python3.11/site-packages/rsl_rl/runners/on_policy_runner.py | head -6
grep -rn "act_distributions\|def act\b" /root/env_isaaclab/lib/python3.11/site-packages/rsl_rl/modules/actor_critic_recurrent.py /root/env_isaaclab/lib/python3.11/site-packages/rsl_rl/modules/actor_critic.py 2>/dev/null | head -8