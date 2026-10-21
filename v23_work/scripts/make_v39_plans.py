import json
d = json.load(open('/config/Desktop/sim/reports/v30_smoke_plan16.json'))
for fam in ('gap_course','gaps_short'):
    e=dict(d); n=128
    e['families']=[fam]*n; e['difficulty']=[0.1]*n; e['difficulty_level']=[1]*n
    e['instances']=[[fam,1,int(x[2])] for x in d['instances']]*8
    e['instances']=e['instances'][:n]; e['speed_level']=[x[2] for x in e['instances']]
    json.dump(e, open('/config/Desktop/sim/reports/v39_plan_'+fam+'.json','w'), indent=2)
    print(fam,'ok')