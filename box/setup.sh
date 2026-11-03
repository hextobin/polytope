#!/bin/bash
# First boot (run by cloud-init as root): arm the dead-man timer, schedule
# result syncs, start the experiment.
set -euo pipefail
source /root/.exp.env
source /root/exp/box/lib.sh
mkdir -p "$LOGS"
date +%s > "$S/boot_time"
echo $(( $(date +%s) + DEADLINE_HOURS * 3600 )) > "$DEADLINE"
chmod +x "$EXP"/box/*.sh

unit() { cat > "/etc/systemd/system/$1"; }

unit exp-deadman.service <<UNIT
[Service]
Type=oneshot
TimeoutStartSec=infinity
ExecStart=$EXP/box/deadman.sh
UNIT
unit exp-deadman.timer <<'UNIT'
[Timer]
OnBootSec=5min
OnUnitActiveSec=5min
[Install]
WantedBy=timers.target
UNIT

unit exp-sync.service <<UNIT
[Service]
Type=oneshot
TimeoutStartSec=10min
ExecStart=$EXP/box/sync.sh
UNIT
unit exp-sync.timer <<'UNIT'
[Timer]
OnBootSec=15min
OnUnitActiveSec=15min
[Install]
WantedBy=timers.target
UNIT

unit exp-run.service <<UNIT
[Service]
Type=simple
ExecStart=$EXP/box/run.sh
UNIT

systemctl daemon-reload
systemctl enable --now exp-deadman.timer exp-sync.timer
systemctl start --no-block exp-run.service
