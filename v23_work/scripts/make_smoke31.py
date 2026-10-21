from pathlib import Path
s = Path('v23_work/scripts/v30_smoke4x16.sh').read_text()
s = s.replace('v30_smoke4x16', 'v31_smoke4x16').replace('go1-v30-smoke4x16', 'go1-v31-smoke4x16')
s = s.replace('GO1_V19_PRELAUNCH_REWARD=1 GO1_V19_PRELAUNCH_SCALE=4', 'GO1_V19_PRELAUNCH_REWARD=1 GO1_V19_PRELAUNCH_SCALE=4 GO1_V19_COMBINED=1')
s = s.replace('checkpoints/stage1/2026-10-18_go1-v26-triple-refine-s01-p1-d1/model_899.pt', 'checkpoints/stage1/2026-10-19_go1-v30b-combined-teacher-s01-p1-d1/model_899.pt')
q = Path('v23_work/scripts/v31_smoke4x16.sh')
q.write_text(s); q.write_bytes(q.read_bytes().replace(b'\r\n', b'\n'))
assert 'GO1_V19_COMBINED=1' in s and 'go1-v30b-combined-teacher' in s
print('smoke31 ok')