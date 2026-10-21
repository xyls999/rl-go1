import json

d = json.load(open('/config/Desktop/sim/reports/v30_smoke_plan16.json'))
fam = 'combined'
n = 8
e = dict(d)
e['families'] = [fam] * n
e['difficulty'] = [0.1] * n
e['difficulty_level'] = [1] * n
e['instances'] = [[fam, 1, int(x[2])] for x in d['instances']][:n]
e['speed_level'] = [x[2] for x in e['instances']]
json.dump(e, open('/config/Desktop/sim/reports/v50_plan_combined8.json', 'w'), indent=2)
print('combined8 plan ok')
