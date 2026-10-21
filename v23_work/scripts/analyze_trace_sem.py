import json
import numpy as np

rows = [json.loads(l) for l in open('/config/Desktop/sim/reports/v30b_combined_trace512.jsonl')]
succ = [r for r in rows if r.get('success')]
fail = [r for r in rows if not r.get('success')]
print('n', len(rows), 'succ', len(succ))
print('success course_x:', [round(r['course_x'], 2) for r in succ[:6]] or 'none')
fx = np.array([r['course_x'] for r in fail])
fall = np.array([r['course_x'] for r in fail if r.get('fallen')])
tof = np.array([r['course_x'] for r in fail if not r.get('fallen')])
print('fail course_x quantiles:', [round(float(q), 2) for q in np.quantile(fx, [0, .1, .25, .5, .75, .9, 1])])
print('fallen n=', len(fall), [round(float(q), 2) for q in np.quantile(fall, [0, .1, .5, .9, 1])])
print('timeout n=', len(tof), [round(float(q), 2) for q in np.quantile(tof, [0, .5, 1])])
fg = sum(1 for r in fail if r.get('fell_into_gap'))
print('fallen&fell_into_gap:', fg, 'of', len(fall))
# first reward x events near death
sx = [r['first_reward_x'].get('supported_short') for r in fall_rows] if False else [
    (r['course_x'], r['first_reward_x'].get('supported_short'), r['steps'],
     round(r.get('terminal_gravity_z', 0), 2), r.get('net_foot_force_max'))
    for r in rows if r.get('fallen') and r['course_x'] < 3.5][:12]
for t in sx:
    print('early-fall row:', t)
