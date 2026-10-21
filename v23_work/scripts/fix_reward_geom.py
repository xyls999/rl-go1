from pathlib import Path
import ast, hashlib

p = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/geometry.py')
s = p.read_text(encoding='utf-8')

old = '''            if len(gaps) != 3 or any(block["kind"] != "gap" for block in gaps) or not tires:
                raise ValueError("combined reward geometry requires three opening gaps and a tire band")'''
assert s.count(old) == 1
s = s.replace(old, '''            if len(gaps) != 3 or any(block["kind"] != "gap" for block in gaps) or not tires:
                if tile_kind == "combined":
                    raise ValueError("combined reward geometry requires three opening gaps and a tire band")
                mixed_reward_segments.append(np.full((4, 3), np.nan, dtype=np.float32))
                bounds = None''')

old = '''            bounds = [[b["start"] - START_IN_TILE[0], b["void_end"] - START_IN_TILE[0],
                       b["end"] - START_IN_TILE[0]] for b in gaps]
            tire = tires[0]
            bounds.append([tire["start"] - START_IN_TILE[0],
                           tire["end"] - START_IN_TILE[0], tire["end"] - START_IN_TILE[0]])
            mixed_reward_segments.append(np.asarray(bounds, dtype=np.float32))'''
assert s.count(old) == 1
s = s.replace(old, '''            if bounds is not None:
                bounds = [[b["start"] - START_IN_TILE[0], b["void_end"] - START_IN_TILE[0],
                           b["end"] - START_IN_TILE[0]] for b in gaps]
                tire = tires[0]
                bounds.append([tire["start"] - START_IN_TILE[0],
                               tire["end"] - START_IN_TILE[0], tire["end"] - START_IN_TILE[0]])
                mixed_reward_segments.append(np.asarray(bounds, dtype=np.float32))''')

p.write_text(s, encoding='utf-8', newline='\n')
ast.parse(s)
print('geometry', hashlib.sha256(p.read_bytes()).hexdigest())
