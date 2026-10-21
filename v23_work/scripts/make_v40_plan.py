import json

d = json.load(open('/config/Desktop/sim/reports/v30_smoke_plan16.json'))
fam = 'gaps_short_dense'
n = 128
e = dict(d)
e['families'] = [fam] * n
e['difficulty'] = [0.1] * n
e['difficulty_level'] = [1] * n
e['instances'] = [[fam, 1, int(x[2])] for x in d['instances']] * 8
e['instances'] = e['instances'][:n]
e['speed_level'] = [x[2] for x in e['instances']]
json.dump(e, open('/config/Desktop/sim/reports/v40_plan_gaps_short_dense.json', 'w'), indent=2)
print('dense plan ok')
