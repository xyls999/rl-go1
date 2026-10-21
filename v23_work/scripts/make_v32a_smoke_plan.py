import json
from pathlib import Path

d = json.load(open('/config/Desktop/sim/reports/v30_smoke_plan16.json'))
d['families'] = ['combined_soft'] * len(d['families'])
d['instances'] = [['combined_soft'] + list(rest[1:]) for rest in d['instances']]
Path('/config/Desktop/sim/reports/v32a_smoke_plan16.json').write_text(json.dumps(d, indent=2) + "\n")
print('plan ok', len(d['families']))
