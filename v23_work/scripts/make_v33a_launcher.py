from pathlib import Path
import hashlib

p = Path('v23_work/scripts/start_v32a_combined_soft.sh')
s = p.read_text(encoding='utf-8')
s = s.replace('curriculum_phases_combined_soft_v32a.json', 'curriculum_phases_combined_soft_v33a.json')
s = s.replace('6934dcc8b6af9fc246dfd458e790edac02b95762ddcdb611e2a79d872289e779',
              'd71d0be131861cb4160075413faa56fc5f8ef74e7319c07ca8527cc23555098b')
s = s.replace('81dd2d6c046b32c5eeb77b13d3d90366a4aaa9b990f389114e73917ebdfbc50c',
              'fd63e677448f09cdf929661d0b782a89b05ab929bdf7dacd45e7c834b78eda34')
s = s.replace('0eeb50462a5ff86d02b8a4ae3e1ae3c2d4855596450834161e9fdcd680812d1f',
              '3f9cb342f53159cb5bdb1960afb0a7413bc3c19baebee7b71be9d00ffe376a15')
s = s.replace('069245693c5e1d17f25d05f9496d2d975c240eb647dfbf128c54e4e46ae0e7b7',
              '89d54edf74f677a5a560841990ab9438657a7e759c5f11c0c6d4dca698dd5c9c')
s = s.replace('v32a', 'v33a').replace('combined-soft', 'combined-soft6').replace('combined_soft', 'combined_soft6')
q = Path('v23_work/scripts/start_v33a_combined_soft6.sh')
q.write_text(s, encoding='utf-8', newline='\n')
q.write_bytes(q.read_bytes().replace(b'\r\n', b'\n'))
print('launcher', hashlib.sha256(q.read_bytes()).hexdigest())
print('config', hashlib.sha256(Path('v23_work/configs/curriculum_phases_combined_soft_v33a.json').read_bytes()).hexdigest())
