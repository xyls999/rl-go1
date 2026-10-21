import json
d = json.load(open('/config/Desktop/sim/reports/v30_smoke_plan16.json'))
for fam in ('combined_soft6',):
    e = dict(d)
    e['families'] = [fam] * 128
    e['difficulty'] = [0.1] * 128
    e['difficulty_level'] = [1] * 128
    inst = [[fam, 1, int(x[2])] for x in d['instances']] * 8
    e['instances'] = inst[:128]
    e['speed_level'] = [x[2] for x in e['instances']]
    json.dump(e, open('/config/Desktop/sim/reports/v33_eval_plan_' + fam + '.json', 'w'), indent=2)
    print(fam, 'ok')