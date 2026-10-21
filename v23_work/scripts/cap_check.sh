#!/usr/bin/env bash
echo "PID1: $(cat /proc/1/comm 2>/dev/null)"
ls -la /.dockerenv /run/.containerenv 2>/dev/null
echo "--- caps"
which capsh >/dev/null 2>&1 && capsh --print 2>/dev/null | grep -aiE 'sys_boot|Current:' | head -3
grep -a CapEff /proc/self/status
python3 - <<'PY'
eff = 0
for line in open('/proc/self/status'):
    if line.startswith('CapEff'):
        eff = int(line.split()[1], 16)
print('CAP_SYS_BOOT(21):', bool(eff & (1 << 21)))
print('CAP_SYS_ADMIN(21? no, 21 boot; admin=21?) cap bits ok')
PY
echo "--- sysrq"
sudo -n sh -c 'cat /proc/sys/kernel/sysrq 2>/dev/null; test -w /proc/sysrq-trigger && echo sysrq_writable || echo sysrq_ro'
