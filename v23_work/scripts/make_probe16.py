import json
d = json.load(open('/config/Desktop/sim/reports/v30_smoke_plan16.json'))
e = dict(d)
e['families'] = ['combined_soft6'] * 16
e['difficulty'] = [0.1] * 16
e['difficulty_level'] = [1] * 16
e['speed_level'] = [x[2] for x in d['instances']]
e['instances'] = [['combined_soft6', 1, int(x[2])] for x in d['instances']]
json.dump(e, open('/config/Desktop/sim/reports/v33_probe_plan_soft6_16.json', 'w'), indent=2)
print('16-slot soft6 plan ok')