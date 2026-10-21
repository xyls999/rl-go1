#!/usr/bin/env bash
cd /config/Desktop/sim
grep -anE 'TEACHERS|teachers' scripts/start_v29b_student_distill.sh | head -8
ls reports/ | grep -ai 'teacher\|v29' | head -10