import json
from pathlib import Path
import ast
import hashlib

# 1) runner: gate speed override
p = Path('v23_work/scripts/run_student_v32.py')
s = p.read_text(encoding='utf-8')
old = '''        "--speed-min", str(config["speed_min"]), "--speed-max", str(config["speed_max"]),
        "--seed", "2026", "--checkpoint", str(checkpoint),'''
assert s.count(old) == 1
s = s.replace(old, '''        "--speed-min", str(config.get("gate_speed_min", config["speed_min"])),
        "--speed-max", str(config.get("gate_speed_max", config["speed_max"])),
        "--seed", "2026", "--checkpoint", str(checkpoint),''')
p.write_text(s, encoding='utf-8', newline='\n')
ast.parse(s)
print('runner', hashlib.sha256(p.read_bytes()).hexdigest())

# 2) v36d config: fast command band, acceptance-band gate
d = json.loads(Path('v23_work/configs/curriculum_phases_hard_v36c.json').read_text(encoding='utf-8'))
d['comment'] = ('v36d pace fix: train at the top command band 0.6-0.8 m/s (P10 leap uses 1.5 m/s); '
                'gate still on the acceptance band 0.35-0.8 m/s.')
d['speed_min'] = 0.6
d['speed_max'] = 0.8
d['gate_speed_min'] = 0.35
d['gate_speed_max'] = 0.8
d['state'] = 'lp_combined_v36d_state.json'
d['sampler_state'] = 'lp_combined_v36d_acrl_state.json'
d['results_file'] = 'lp_combined_v36d_results.txt'
d['init_checkpoint'] = 'checkpoints/stage1/2026-10-19_go1-v36c-hard-finetune2-s01-p1-d1/model_799.pt'
for phase in d['phases']:
    phase['iterations'] = 600
cfg = Path('v23_work/configs/curriculum_phases_fast_v36d.json')
cfg.write_text(json.dumps(d, indent=2) + "\n", encoding='utf-8', newline='\n')
print('config', hashlib.sha256(cfg.read_bytes()).hexdigest())

# 3) launcher
s2 = Path('v23_work/scripts/start_v36c_hard_finetune2.sh').read_text(encoding='utf-8')
s2 = s2.replace('curriculum_phases_hard_v36c.json', 'curriculum_phases_fast_v36d.json')
s2 = s2.replace('v36c', 'v36d')
s2 = s2.replace('hard-finetune2', 'fast-command')
s2 = s2.replace('8495a46615886c9e062c134ddb6107b743a6b28621bccfb2b687e3ebd5f63226',
                hashlib.sha256(cfg.read_bytes()).hexdigest())
s2 = s2.replace('checkpoints/stage1/2026-10-19_go1-v36b-hard-finetune-s01-p1-d1/model_399.pt',
                'checkpoints/stage1/2026-10-19_go1-v36c-hard-finetune2-s01-p1-d1/model_799.pt')
s2 = s2.replace('472ae1c319d302fd94dc8a607234fa11cb64b1a7b96cce8ecf5a38035a1457b7',
                'bc8501a445dbfd7fb1d95351d3372a498f5604753427b9ad542e705bde497989')
s2 = s2.replace('472ae1c3', 'bc8501a4')
s2 = s2.replace('INIT=v36b', 'INIT=v36c')
q = Path('v23_work/scripts/start_v36d_fast_command.sh')
q.write_text(s2, encoding='utf-8', newline='\n')
q.write_bytes(q.read_bytes().replace(b'\r\n', b'\n'))
assert 'bc8501a4' in s2 and 'v36d' in s2 and 'fast_v36d' in s2
print('launcher', hashlib.sha256(q.read_bytes()).hexdigest())
