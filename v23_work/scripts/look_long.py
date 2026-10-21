from pathlib import Path
s = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/geometry.py').read_text(encoding='utf-8')
i = s.find('GAP_LONG_WIDTH_MIN,', s.find('rhythm=True') - 400)
print(repr(s[i - 200:i + 260]))
