#!/usr/bin/env bash
# One-shot NTP correction: set each device's clock to the server's
# current time via global.cgi?action=setCurrentTime.
#
# Why setCurrentTime and not full configManager NTP configuration:
#   - Devices are behind customer's firewall and can't reach internet NTP
#   - Our own server has NTP via systemd-timesyncd, so server clock is good
#   - One-shot setCurrentTime is non-invasive (no daemon restart)
#   - We can do this safely DURING business hours — no service interruption
#
# Run periodically via cron to keep clocks aligned:
#   0 3 * * *  /opt/renessans/deploy/ntp_sync_devices.sh
#
# Verify drift before/after via getCurrentTime.

set -uo pipefail

USER="admin:Q1234567q"
IPS=(10.10.7.201 10.10.7.202 10.10.7.203 10.10.7.204 10.10.7.205 \
     10.10.7.206 10.10.7.207 10.10.7.208 10.10.7.209)

# Dahua wants `YYYY-MM-DD%20HH:MM:SS` (space url-encoded).
NOW=$(date -u +"%Y-%m-%d%%20%H:%M:%S")
NOW_DISPLAY=$(date -u +"%Y-%m-%d %H:%M:%S UTC")
echo "server time: $NOW_DISPLAY"
echo ""

for ip in "${IPS[@]}"; do
  short=$(echo "$ip" | awk -F. '{print $4}')

  # Read device clock first (drift measurement)
  BEFORE=$(timeout 5 curl -sS --max-time 5 --user "$USER" --digest \
    "http://$ip/cgi-bin/global.cgi?action=getCurrentTime" 2>/dev/null \
    | tr -d '\r' | head -1)

  # Set new time (uses UTC; device may be configured to a different TZ
  # but the underlying clock is what matters for our timestamps).
  RESP=$(timeout 5 curl -sS --max-time 5 --user "$USER" --digest \
    "http://$ip/cgi-bin/global.cgi?action=setCurrentTime&time=$NOW" 2>/dev/null \
    | head -1)

  # Read after — confirms it took.
  AFTER=$(timeout 5 curl -sS --max-time 5 --user "$USER" --digest \
    "http://$ip/cgi-bin/global.cgi?action=getCurrentTime" 2>/dev/null \
    | tr -d '\r' | head -1)

  printf "%-12s %s  | before: %-30s after: %s\n" \
    "Turniket-$((short - 200))" "$ip" "$BEFORE" "$AFTER"
done
