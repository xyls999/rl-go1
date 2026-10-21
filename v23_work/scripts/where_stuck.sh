#!/usr/bin/env bash
pid=$(ps -eo pid,args | grep -E 'env_isaaclab/bin/python.*sim\.py' | grep -v grep | awk '{print $1}' | head -1)
echo "PID=$pid"
[ -z "$pid" ] && exit 0
echo '--- state'
grep -E 'State|Threads' /proc/$pid/status
echo '--- wchan (main)'
cat /proc/$pid/wchan 2>/dev/null; echo
echo '--- top threads by state'
for t in /proc/$pid/task/*; do
  st=$(awk '{print $3}' $t/stat 2>/dev/null)
  wn=$(cat $t/wchan 2>/dev/null)
  echo "$(basename $t) $st $wn"
done | awk '{print $2, $3}' | sort | uniq -c | sort -rn | head -10
echo '--- open files (tail)'
sudo -n ls -l /proc/$pid/fd 2>/dev/null | tail -8
