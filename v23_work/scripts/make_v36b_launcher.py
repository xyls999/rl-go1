from pathlib import Path
import hashlib

def sh(f):
    return hashlib.sha256(Path('v23_work/' + f).read_bytes()).hexdigest()

s = Path('v23_work/scripts/start_v36a_combined_soft8.sh').read_text(encoding='utf-8')
s = s.replace('combined-soft8', 'hard-finetune').replace('combined_soft8', 'combined')
s = s.replace('v36a', 'v36b')
s = s.replace('curriculum_phases_combined_soft_v36b.json', 'curriculum_phases_hard_v36b.json')
s = s.replace('curriculum_phases_combined_soft_v36a.json', 'curriculum_phases_hard_v36b.json')
s = s.replace('ec4bef9487f64f5b5a882f28d24808a81199c65d413f0213c1b562cce7563937', sh('configs/curriculum_phases_hard_v36b.json'))
s = s.replace('export GO1_V36_SOFT_TERMINATION=1', 'unset GO1_V36_SOFT_TERMINATION')
q = Path('v23_work/scripts/start_v36b_hard_finetune.sh')
q.write_text(s, encoding='utf-8', newline='\n')
q.write_bytes(q.read_bytes().replace(b'\r\n', b'\n'))
# sanity
assert 'single:combined' in s.replace('ROSTER=single:combined', 'single:combined')
assert 'GO1_V36_SOFT_TERMINATION' in s and 'unset GO1_V36_SOFT_TERMINATION' in s
assert 'v36b' in s
assert '1c4a3852' in s
print('launcher', hashlib.sha256(q.read_bytes()).hexdigest())
