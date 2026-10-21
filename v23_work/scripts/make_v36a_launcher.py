from pathlib import Path
import hashlib

def sh(f):
    return hashlib.sha256(Path('v23_work/' + f).read_bytes()).hexdigest()

s = Path('v23_work/scripts/start_v33a_combined_soft6.sh').read_text(encoding='utf-8')
s = s.replace('curriculum_phases_combined_soft_v33a.json', 'curriculum_phases_combined_soft_v36a.json')
s = s.replace('combined-soft6', 'combined-soft8').replace('combined_soft6', 'combined_soft8').replace('v33a', 'v36a')
s = s.replace('86934e2af9849d7c074dd4aa411e0f9ce55fd54ae6dc2b2bcfb0dcfc52fa61fd', sh('configs/curriculum_phases_combined_soft_v36a.json'))
s = s.replace('d71d0be131861cb4160075413faa56fc5f8ef74e7319c07ca8527cc23555098b', sh('scripts/run_student_v32.py'))
s = s.replace('3f9cb342f53159cb5bdb1960afb0a7413bc3c19baebee7b71be9d00ffe376a15', sh('source/go1_hier_sim/go1_hier_sim/envs/geometry.py'))
s = s.replace('89d54edf74f677a5a560841990ab9438657a7e759c5f11c0c6d4dca698dd5c9c', sh('source/go1_hier_sim/go1_hier_sim/envs/runtime.py'))
s = s.replace('export GO1_POLICY_VARIANT=map_attn_v3', 'export GO1_V36_SOFT_TERMINATION=1\nexport GO1_POLICY_VARIANT=map_attn_v3')
csha = sh('source/go1_hier_sim/go1_hier_sim/envs/config.py')
anchor = "'source/go1_hier_sim/go1_hier_sim/cli.py':"
assert anchor in s
s = s.replace(anchor, "'source/go1_hier_sim/go1_hier_sim/envs/config.py': '%s',\n " % csha + anchor, 1)
q = Path('v23_work/scripts/start_v36a_combined_soft8.sh')
q.write_text(s, encoding='utf-8', newline='\n')
q.write_bytes(q.read_bytes().replace(b'\r\n', b'\n'))
assert 'combined_soft8' in s and 'GO1_V36_SOFT_TERMINATION=1' in s and csha in s
print('launcher', hashlib.sha256(q.read_bytes()).hexdigest())
