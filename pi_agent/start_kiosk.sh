#!/usr/bin/env bash
# 두뇌 PC 서버가 뜰 때까지 기다렸다가 Chromium을 키오스크 모드로 띄운다 (labwc autostart에서 호출).
#
# 키오스크 페이지가 두뇌 PC에 실제로 붙었는지 /healthz의 pi_kiosk로 확인하고, 제한 시간 안에 안 붙으면
# Chromium을 다시 띄운다. 11/27 첫 부팅 때 Chromium 내부 네트워크 프로세스가 시작 직후 죽으면서
# 첫 페이지를 못 열고 흰 화면으로 멈췄다(부팅 직후 업데이트 확인으로 SD 카드가 바쁠 때).
# 페이지가 한 번 붙으면 이후 연결은 app.js가 스스로 다시 붙이고, Chromium이 꺼지면 다시 띄운다.
# 기록: $XDG_RUNTIME_DIR/reachy-kiosk.log (재부팅하면 지워짐)
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
set -a
# shellcheck disable=SC1091
source "$DIR/kiosk.env"
set +a
BASE="http://${BRAIN_HOST}:${BRAIN_PORT:-8000}"
# ?device=pi: 커서를 숨기고(Pi는 터치를 마우스로 보내서 누른 자리에 커서가 남음), 서버가 이 페이지를 pi_kiosk로 센다
URL="$BASE/kiosk?device=pi"
BROWSER="$(command -v chromium || command -v chromium-browser)"
LOG="${XDG_RUNTIME_DIR:-/tmp}/reachy-kiosk.log"

log() { echo "$(date '+%F %T') $*" >>"$LOG"; }
brain_up() { curl -fsS -o /dev/null --max-time 2 "$BASE/healthz"; }
page_up() {
  local health
  health="$(curl -fsS --max-time 2 "$BASE/healthz" 2>/dev/null)" || return 1
  case "$health" in
    *'"pi_kiosk"'*) echo "$health" | grep -q '"pi_kiosk": *[1-9]' ;;
    *) return 0 ;;  # pi_kiosk를 모르는 옛 서버: 확인할 수 없으니 다시 띄우지 않는다
  esac
}

first="${KIOSK_PAGE_TIMEOUT:-60}"  # 첫 실행은 프로필을 만드느라 느리니 넉넉히. 안 붙을 때마다 두 배로, 최대 두 배까지
limit="$first"
while true; do
  until brain_up; do sleep 2; done
  pkill -x chromium 2>/dev/null && sleep 1
  log "Chromium 실행: $URL (${limit}초 안에 페이지 연결 확인)"
  "$BROWSER" --kiosk --noerrdialogs --disable-infobars --no-first-run \
    --disable-session-crashed-bubble --disable-translate --overscroll-history-navigation=0 \
    --password-store=basic --check-for-update-interval=31536000 "$URL" &
  pid=$!
  ok=0
  for _ in $(seq "$limit"); do
    sleep 1
    if page_up; then ok=1; break; fi
    kill -0 "$pid" 2>/dev/null || break
  done
  if [ "$ok" = 1 ]; then
    log "키오스크 페이지 연결됨"
    limit="$first"
    wait "$pid"
    log "Chromium이 꺼짐 → 다시 띄움"
  else
    log "페이지가 연결되지 않음 (흰 화면이거나 Chromium이 꺼짐) → 다시 띄움"
    kill "$pid" 2>/dev/null
    limit=$(( limit * 2 > first * 2 ? first * 2 : limit * 2 ))
  fi
  sleep 3
done
