import json

d = json.load(open('/config/Desktop/sim/reports/v30_smoke_plan16.json'))
n = 32
e = dict(d)
fam = 'combined_soft9'
e['families'] = [fam] * n
e['difficulty'] = [0.1] * n
e['difficulty_level'] = [1] * n
inst = [[fam, 1, int(x[2])] for x in d['instances']] * 2
e['instances'] = inst[:n]
e['speed_level'] = [x[2] for x in e['instances']]
json.dump(e, open('/config/Desktop/sim/reports/v37_plan_soft9_32.json', 'w'), indent=2)
print('32-slot soft9 plan ok')
