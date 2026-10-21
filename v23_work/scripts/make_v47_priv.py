from pathlib import Path
import ast, hashlib

p = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/config.py')
s = p.read_text(encoding='utf-8')

old = '''def actor_observation(env):
    env.hier.refresh()
    obs = env.hier.locomotion_obs if env.cfg.hier_stage == "locomotion" else env.hier.navigation_obs'''
assert s.count(old) == 1
s = s.replace(old, '''def actor_observation(env):
    env.hier.refresh()
    # P10 privileged specialist (v47): same observation width, but the terrain
    # feature vector comes from exact simulator geometry instead of the camera
    # render. Training-only; the deployed student always uses the camera path.
    import os as _os
    if _os.environ.get("GO1_PRIVILEGED_ACTOR") == "1":
        obs = (env.hier.teacher_locomotion_obs if env.cfg.hier_stage == "locomotion"
               else env.hier.teacher_navigation_obs)
        return obs
    obs = env.hier.locomotion_obs if env.cfg.hier_stage == "locomotion" else env.hier.navigation_obs''')

p.write_text(s, encoding='utf-8', newline='\n')
ast.parse(s)
print('config', hashlib.sha256(p.read_bytes()).hexdigest())
