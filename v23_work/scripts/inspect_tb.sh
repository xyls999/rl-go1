#!/usr/bin/env bash
cd /config/Desktop/sim
d=$(ls -d checkpoints/stage1/*go1-v29-smoke4x16 2>/dev/null | head -1)
echo DIR=$d
/root/env_isaaclab/bin/python - <<PY
import glob, os
from tensorboard.backend.event_processing import event_accumulator
d = "$d"
files = glob.glob(os.path.join(d, "**", "*events*"), recursive=True)
print("events files:", files[:3])
if files:
    ea = event_accumulator.EventAccumulator(files[0], size_guidance={"*": [1]})
    ea.Reload()
    tags = [t for t in ea.Tags()["scalars"]]
    print("TAGS sample:", sorted(tags)[:40])
    hits = [t for t in tags if "distill" in t.lower() or "teacher" in t.lower()]
    print("DISTILL TAGS:", hits)
    for t in hits[:2]:
        print(t, ea.Scalars(t)[-1])
PY