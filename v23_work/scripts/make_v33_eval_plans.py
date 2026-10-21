import json

d = json.load(open('/config/Desktop/sim/reports/v30_smoke_plan16.json'))
for fam in ('combined_soft2', 'combined_soft3'):
    e = dict(d)
    e['families'] = [fam] * 128
    e['difficulty'] = [0.1] * 128
    e['difficulty_level'] = [1] * 128
    e['instances'] = [[fam, 1, int(x[2])] for x in d['instances']] * 8
    e['instances'] = e['instances'][:128]
    e['speed_level'] = [x[2] for x in e['instances']]
    json.dump(e, open(f'/config/Desktop/sim/reports/v33_eval_plan_{fam}.json', 'w'), indent=2)
    print(fam, 'ok')
