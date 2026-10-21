from pathlib import Path

src = Path('v23_work/scripts/run_student_v30.py').read_text(encoding='utf-8')

old = 'from go1_hier_sim.envs.combined_roster_v30 import select_training_roster  # noqa: E402'
assert src.count(old) == 1
src = src.replace(old, 'from go1_hier_sim.envs.combined_roster_v32 import select_training_roster  # noqa: E402')

old = '    families = select_training_roster(config, TERRAIN_FAMILIES)\n'
assert src.count(old) == 1
new = ('    families = select_training_roster(config, TERRAIN_FAMILIES + ("combined_soft",))\n'
       '    gate_families = (str(config["gate_family"]),) if config.get("gate_family") else families\n')
src = src.replace(old, new)

old = '''        gate_plan = make_balanced_gate_plan(plan, families, float(config["difficulty_max"]))
        if (len(gate_plan["instances"]) != envs or set(gate_plan["families"]) != set(families)
                or min(gate_plan["families"].count(family) for family in families) < 40):
            raise SystemExit("Stage-C teacher gate must cover combined with at least 40 slots")'''
assert src.count(old) == 1
new = '''        gate_plan = make_balanced_gate_plan(plan, gate_families, float(config["difficulty_max"]))
        if (len(gate_plan["instances"]) != envs or set(gate_plan["families"]) != set(gate_families)
                or min(gate_plan["families"].count(family) for family in gate_families) < 40):
            raise SystemExit("Stage-C teacher gate must cover the real family with 40+ slots")'''
src = src.replace(old, new)

old = '        record(f"[{run_id}] GATE_PLAN {gate_plan_path.name} balanced_{len(families)}_families")'
assert src.count(old) == 1
src = src.replace(old, '        record(f"[{run_id}] GATE_PLAN {gate_plan_path.name} '
                       'balanced_{len(gate_families)}_families")')

dst = Path('v23_work/scripts/run_student_v32.py')
dst.write_text(src, encoding='utf-8', newline='\n')
import ast
ast.parse(src)
print('run_student_v32 ok')
