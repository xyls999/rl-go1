import json

d = json.load(open('/config/Desktop/sim/reports/v30_smoke_plan16.json'))
for fam in ('combined_soft13', 'combined_soft14'):
    n = 128
    e = dict(d)
    e['families'] = [fam] * n
    e['difficulty'] = [0.1] * n
    e['difficulty_level'] = [1] * n
    e['instances'] = [[fam, 1, int(x[2])] for x in d['instances']] * 8
    e['instances'] = e['instances'][:n]
    e['speed_level'] = [x[2] for x in e['instances']]
    json.dump(e, open(f'/config/Desktop/sim/reports/v45_plan_{fam}.json', 'w'), indent=2)
    print(fam, 'ok')
