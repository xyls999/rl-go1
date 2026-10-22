from pathlib import Path
import hashlib

def sh(f):
    return hashlib.sha256(Path('v23_work/' + f).read_bytes()).hexdigest()

s = Path('v23_work/scripts/start_v59_respawn.sh').read_text(encoding='utf-8')
# config swap + pins
s = s.replace('curriculum_phases_respawn_v59.json', 'curriculum_phases_pen_v60.json')
s = s.replace('9aa2746175a06ea2daf3b4e48896e437a1ef5b67e5bf39402fafe45e0799053c',
              sh('configs/curriculum_phases_pen_v60.json'))
s = s.replace('v59', 'v60').replace('respawn', 'penetration')
s = s.replace('export GO1_SOFT_RESPAWN=1\n',
              'export GO1_V8_LONG_GAP_THRESHOLD=0.05 GO1_V19_MIN_WIDTH=0.05\n')
# pins that differ from v59's
s = s.replace('319277289699f60731f32f95ea11d82aca59aa50b67f549afa26fc17f33f98c2',
              sh('source/go1_hier_sim/go1_hier_sim/envs/geometry.py'))
s = s.replace('595db49f915f0c16367cf3ab43475f8f4bc4ff03fdeae9a2133da63fe2c6c830',
              sh('source/go1_hier_sim/go1_hier_sim/envs/runtime.py'))
q = Path('v23_work/scripts/start_v60_penetration.sh')
q.write_text(s, encoding='utf-8', newline='\n')
q.write_bytes(q.read_bytes().replace(b'\r\n', b'\n'))
assert "''" not in s
print('config', sh('configs/curriculum_phases_pen_v60.json'))
print('launcher', hashlib.sha256(q.read_bytes()).hexdigest())
