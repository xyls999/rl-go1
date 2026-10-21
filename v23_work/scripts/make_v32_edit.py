from pathlib import Path
p = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/geometry.py')
s = p.read_text(encoding='utf-8')
old = """                cursor = gap_block(cursor, pad=0.65, width_lo=GAP_WIDTH_MIN,
                                   width_hi=GAP_SHORT_WIDTH_MAX)"""
new = """                cursor = gap_block(cursor, pad=0.65 + _entrance_pad_extra(),
                                   width_lo=GAP_WIDTH_MIN,
                                   width_hi=GAP_SHORT_WIDTH_MAX)"""
assert old in s, 'combined entrance block not found'
s = s.replace(old, new)
helper = """def _entrance_pad_extra() -> float:
    # P10-style soft-curriculum knob (v32): widening the recovery pads of the
    # three opening voids teaches the dense triple-gap rhythm on training-only
    # tiles; the default 0 keeps every published terrain byte-identical.
    import os
    return float(os.environ.get("GO1_V32_ENTRANCE_PAD", "0"))


"""
anchor = "def build_tile(kind: str, difficulty: float, rng: np.random.Generator, obstacles: bool):"
assert anchor in s
s = s.replace(anchor, helper + anchor, 1)
p.write_text(s, encoding='utf-8', newline='\n')
import ast
ast.parse(s)
print('patched+parse-ok')
