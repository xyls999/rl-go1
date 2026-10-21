from pathlib import Path
import ast, hashlib

p = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/teacher_distill_v29.py')
s = p.read_text(encoding='utf-8')
old = '''def family_teacher_keys(families: list[str], course_x: torch.Tensor,
                        segments: torch.Tensor) -> tuple[list[str], torch.Tensor]:'''
assert s.count(old) == 1, ('sig', s.count(old))
s = s.replace(old, '''def family_teacher_keys(families: list[str], course_x: torch.Tensor,
                        segments: torch.Tensor,
                        combined_leap: bool = False) -> tuple[list[str], torch.Tensor]:''')

old = """                name = 'tire' if int(codes[index]) == 1 else 'general'
"""
n = s.count(old)
print('assign count', n)
assert n == 1
s = s.replace(old, """                name = 'tire' if int(codes[index]) == 1 else 'general'
                if combined_leap and int(codes[index]) == 0:
                    # Route the three opening voids of the mixed route to the
                    # long-gap expert (leap-shaped part of the course).
                    row = segments[index]
                    if bool((course_x[index] >= row[0, 0] - 0.30)
                            and (course_x[index] <= row[2, 2] + 0.20)):
                        name = 'long'
""")
p.write_text(s, encoding='utf-8', newline='\n')
ast.parse(s)
print('tmap', hashlib.sha256(p.read_bytes()).hexdigest())

p2 = Path('v23_work/source/go1_hier_sim/go1_hier_sim/envs/runtime.py')
s2 = p2.read_text(encoding='utf-8')
old = '''            names, valid = family_teacher_keys(self._v29_families, local_x,
                                               self._v29_segments_cpu)'''
assert s2.count(old) == 1, ('rt', s2.count(old))
s2 = s2.replace(old, '''            names, valid = family_teacher_keys(
                self._v29_families, local_x, self._v29_segments_cpu,
                combined_leap=(os.environ.get("GO1_V29_COMBINED_LEAP") == "1"))''')
p2.write_text(s2, encoding='utf-8', newline='\n')
ast.parse(s2)
print('runtime', hashlib.sha256(p2.read_bytes()).hexdigest())
