#!/usr/bin/env bash
cd /config/Desktop/sim
tail -6 reports/v39_gap_course.log | cut -c1-150
echo '--- errors'
grep -aE 'Error|error|raise|ValueError' reports/v39_gap_course.log | tail -4 | cut -c1-150