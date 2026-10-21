from pathlib import Path
import hashlib

q = Path('v23_work/scripts/start_v36b_hard_finetune.sh')
s = q.read_text(encoding='utf-8')
assert s.count('6607e26d81afeb60d5cfa45b453efe833c61d134d0e3a3ae52c2254b19e45674') == 1
s = s.replace('checkpoints/stage1/2026-10-18_go1-v26-triple-refine-s01-p1-d1/model_899.pt',
              'checkpoints/stage1/2026-10-19_go1-v36a-combined-soft8-s01-p1-d1/model_799.pt')
s = s.replace('6607e26d81afeb60d5cfa45b453efe833c61d134d0e3a3ae52c2254b19e45674',
              '2baa9497ca618e7db45a580302e1d2b85a766fc30718701b41210ec6911afa85')
s = s.replace('INIT=v26', 'INIT=v36a')
q.write_text(s, encoding='utf-8', newline='\n')
q.write_bytes(q.read_bytes().replace(b'\r\n', b'\n'))
assert '2baa9497' in s and 'model_799.pt' in s and 'v36b' in s
print('launcher final', hashlib.sha256(q.read_bytes()).hexdigest())
