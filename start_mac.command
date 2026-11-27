#!/bin/bash
# Reachy Café - macOS 시작 스크립트. Finder에서 더블클릭하면 메뉴가 뜬다 (터미널: ./start_mac.command [명령] [인자...]).
#   setup preflight spike calibrate kiosk mock sim test  (start_linux.sh와 같은 명령)
# 필요: Apple Silicon(M1 이상) + macOS 14 이상. mediapipe 1.0.1·onnxruntime 1.27.0에 Intel 맥·macOS 13 이하용
# 패키지가 없다 (uv로 확인, 2026-11-24). GStreamer는 pip 패키지(gstreamer-bundle)에 들어 있어 따로 설치하지 않는다.
# 맥에는 CUDA가 없어 음성 인식(Whisper)은 CPU에서 돈다 (자동 전환). 사전 점검이 인식 시간을 재 준다.
cd "$(dirname "$0")" || exit 1
PY=.venv/bin/python

say() { echo "[reachy-cafe] $*"; }

pause_end() {
  if [ -t 0 ] && [ -z "${NO_PAUSE:-}" ]; then echo; read -r -p "Enter를 누르면 창을 닫아도 됩니다..." _; fi
}

check_mac() {
  if [ "$(uname -m)" != "arm64" ]; then
    if [ "$(sysctl -n hw.optional.arm64 2>/dev/null)" = "1" ]; then
      say "이 터미널이 Rosetta(x86_64)로 실행 중입니다. 터미널 앱 '정보 가져오기'에서 'Rosetta를 사용하여 열기'를 끄고 다시 실행하세요."
    else
      say "이 맥은 $(uname -m)입니다. Apple Silicon(M1 이상)만 지원합니다 (mediapipe·onnxruntime 패키지 없음)."
    fi
    return 1
  fi
  local major
  major=$(sw_vers -productVersion | cut -d. -f1)
  if [ "${major:-0}" -lt 14 ]; then
    say "macOS $(sw_vers -productVersion): macOS 14(Sonoma) 이상이 필요합니다."
    return 1
  fi
}

find_uv() {
  for d in "$HOME/.local/bin" "$HOME/.cargo/bin" /opt/homebrew/bin; do
    if [ -x "$d/uv" ]; then PATH="$d:$PATH"; fi
  done
  if ! command -v uv >/dev/null 2>&1; then
    say "uv가 없습니다. 터미널에서 아래 한 줄로 설치하고 다시 실행하세요:"
    echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"
    return 1
  fi
}

need_venv() {
  if [ ! -x "$PY" ]; then say ".venv가 없습니다. 먼저 메뉴 1) 설치 (또는 ./start_mac.command setup)"; return 1; fi
}

run_py() {
  need_venv || return 1
  "$PY" -X utf8 "$@"
}

do_setup() {
  check_mac || return 1
  find_uv || return 1
  if [ ! -x "$PY" ]; then uv venv --python 3.12 .venv || { say "실패: .venv를 만들지 못했습니다"; return 1; }; fi
  uv pip install --python "$PY" -e ".[dev,robot,speech,sim]" || { say "실패: 위 메시지를 보세요"; return 1; }
  chmod +x start_mac.command 2>/dev/null
  say "설치 끝. 다음: 메뉴 2) 사전 점검"
  say "처음 로봇에 연결할 때 macOS가 '로컬 네트워크' 접근을 물으면 허용하세요 (거부하면 로봇에 연결되지 않음)."
}

dispatch() {
  local cmd="$1"
  shift
  case "$cmd" in
    setup) do_setup ;;
    preflight) run_py tools/preflight.py --load-models "$@" ;;
    spike) run_py tools/spike_connection.py "$@" ;;
    calibrate) run_py tools/spike_connection.py --skip audio --skip asr --skip motion --aim --calibrate "$@" ;;
    kiosk)
      if [ -f config/calibration.toml ]; then run_py -m reachy_cafe --config config/calibration.toml "$@"
      else run_py -m reachy_cafe "$@"; fi ;;
    mock) run_py -m reachy_cafe --config config/dev_laptop.toml "$@" ;;
    sim) run_py tools/sim_check.py "$@" ;;
    test) run_py -m pytest -q "$@" ;;
    bundle) run_py tools/make_bundle.py "$@" ;;
    *) say "모르는 명령: $cmd (setup preflight spike calibrate kiosk mock sim test)"; return 2 ;;
  esac
}

menu() {
  echo "Reachy Café (macOS)"
  echo "  1) 설치 (처음 한 번)"
  echo "  2) 사전 점검 (로봇 안 움직임)"
  echo "  3) 연결 점검 (로봇 움직임)"
  echo "  4) 배치 보정 (화면·슬롯 위치, 고개 방향)"
  echo "  5) 키오스크 실행 (실제 로봇)"
  echo "  6) 키오스크 실행 (로봇 없이 mock)"
  echo "  7) MuJoCo 시뮬 점검"
  echo "  8) 테스트"
  read -r -p "번호: " n
  case "$n" in
    1) dispatch setup ;; 2) dispatch preflight ;; 3) dispatch spike ;; 4) dispatch calibrate ;;
    5) dispatch kiosk ;; 6) dispatch mock ;; 7) dispatch sim ;; 8) dispatch test ;;
    *) say "끝" ;;
  esac
}

if [ $# -eq 0 ]; then
  menu
  status=$?
  pause_end
  exit $status
fi
dispatch "$@"
