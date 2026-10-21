import glob
import json

ps = sorted(glob.glob('/config/Desktop/sim/reports/evaluate_stage1_*gate*v53*.json'))
print(ps)
if ps:
    d = json.load(open(ps[-1]))
    for k in ("episodes", "completion_rate", "fallen_rate", "collision_rate",
              "fell_into_gap_rate", "timeout_rate", "command_speed_mean", "achieved_speed_mean"):
        v = d.get(k)
        print(k, round(v, 4) if isinstance(v, float) else v)
