import json

d = json.load(open('/config/Desktop/sim/reports/v30_smoke_plan16.json'))
fam = 'combined_soft8'
n = 16
e = dict(d)
e['families'] = [fam] * n
e['difficulty'] = [0.1] * n
e['difficulty_level'] = [1] * n
e['instances'] = [[fam, 1, int(x[2])] for x in d['instances']]
e['speed_level'] = [x[2] for x in e['instances']]
json.dump(e, open('/config/Desktop/sim/reports/v59_plan_soft8_16.json', 'w'), indent=2)
print('soft8 16 plan ok')
