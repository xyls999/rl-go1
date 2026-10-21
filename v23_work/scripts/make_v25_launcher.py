from pathlib import Path

p = Path('v23_work/scripts/start_v24_triple_refine.sh')
s = p.read_text()
reps = [
    ('87a3c53061b38d238ffc7c22ebd9d1fa7d452d2d51b5733cb8940f2bf848b54b',
     '294ee5815e74fc8e548224acbaec0659ce92f768a26c4541d3362d65d7f6df7b'),
    ('curriculum_phases_triple_refine_v24.json', 'curriculum_phases_triple_refine_v25.json'),
    ('8f3138d88e8e587621ee750d98a12ba1e01d30ba82a27dfb8f1cce96e9142c61',
     'c93ab53b0f8ac6c49358e80657c9a21f00c82b8ab397fe333e89a68d79efac7e'),
    ('45e98ab7c7e47f9629ad0a901d9bc3dbc19738dc821e04893c84c09e327aa878',
     'd41e627f9d8d0417fe02d989fb5c4ffc80be91a5a6bfdbf80cdbbf4c7d8d60d1'),
    ('checkpoints/stage1/2026-10-17_go1-v23-triple-refine-s01-p1-d1/model_899.pt',
     'checkpoints/stage1/2026-10-18_go1-v24-triple-refine-s01-p1-d1/model_899.pt'),
    ('cb324e8235eed24425334d815dc5ffb8f80d0fb1bd7ac95dc6c38772d9be97a7',
     '0d801c3733979cd66ae87bf8a34b2fe431939f45b29cf3fc13617622e7c99956'),
    ('reports/v23_ten_gate_trace512.jsonl', 'reports/v24_ten_gate_trace512.jsonl'),
    ('v23_triple_refine.exit', 'v24_triple_refine.exit'),
    ('v24_smoke4x16', 'v25_smoke4x16'),
    ('V24_TRIPLE_PREFLIGHT_OK', 'V25_TRIPLE_PREFLIGHT_OK'),
    ('V24_START_UTC', 'V25_START_UTC'),
    ('INIT=v23 BRAKE=on', 'INIT=v24 BRAKE=cap.65/pen.35'),
    ('GO1_V23_TIRE_BRAKE=1 GO1_DDP_BACKEND',
     'GO1_V23_TIRE_BRAKE=1 GO1_V23_TIRE_CAP=.65 GO1_V23_TIRE_PEN=.35 GO1_DDP_BACKEND'),
    ('2026-10-18_go1-v24-triple-refine', '2026-10-18_go1-v25-triple-refine'),
    ('v24_triple_refine', 'v25_triple_refine'),
    ('lp_triple_refine_v24', 'lp_triple_refine_v25'),
]
for old, new in reps:
    if old not in s:
        raise SystemExit(f'missing: {old}')
    s = s.replace(old, new)
q = Path('v23_work/scripts/start_v25_triple_refine.sh')
q.write_text(s, encoding='utf-8')
q.write_bytes(q.read_bytes().replace(b'\r\n', b'\n'))
print('launcher v25 written')
