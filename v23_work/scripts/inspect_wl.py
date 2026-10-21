from pathlib import Path

s = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/geometry.py').read_text(encoding='utf-8')
i = s.find('"combined_soft9",')
print('WHITELIST >>>', repr(s[i - 60:i + 120]))
j = s.find('"gap_course"', s.find('terrain_families = ('))
print('SCENE >>>', repr(s[j - 120:j + 120]))
