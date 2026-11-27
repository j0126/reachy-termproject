#!/usr/bin/env bash
# Reachy Café - Linux 시작 스크립트 (Ubuntu 24.04 권장, x86_64). 터미널에서:  ./start_linux.sh [명령] [인자...]
# 명령 없이 실행하면 메뉴가 뜬다.
#   setup      처음 한 번: .venv 만들고 설치 (NVIDIA GPU가 있으면 GPU 음성 인식 포함)
#   preflight  사전 점검 (로봇은 읽기만, 모델 실제 로딩 포함)
#   spike      연결 점검 (로봇이 움직임)
#   calibrate  배치 보정: 화면·슬롯 위치(--aim) + 고개 방향(--calibrate) → config/calibration.toml
#   kiosk      키오스크 실행 (실제 로봇, config/calibration.toml이 있으면 함께 적용)
#   mock       키오스크 실행 (로봇 없이)
#   sim        MuJoCo 시뮬 점검 (로봇 없이)
#   test       테스트
# Linux는 GStreamer를 직접 설치해야 한다 (Windows·맥은 pip 패키지에 들어 있음): docs/desktop_check.md 'Linux 추가 준비',
# 로봇 카메라·마이크(WebRTC)에는 Rust webrtc 플러그인이 필요하다: scripts/linux_gstreamer_webrtc.sh
set -u
cd "$(dirname "$(readlink -f "$0")")" || exit 1
PY=.venv/bin/python
APT_PACKAGES="build-essential pkg-config git curl libffi-dev libssl-dev libglib2.0-dev libcairo2-dev \
libgirepository1.0-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev \
gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-alsa gstreamer1.0-nice libnice10 libportaudio2 \
python3-gi python3-gi-cairo libgl1"

say() { echo "[reachy-cafe] $*"; }

pause_end() {
  if [ -t 0 ] && [ -z "${NO_PAUSE:-}" ]; then echo; read -r -p "Enter를 누르면 끝납니다..." _; fi
}

find_uv() {
  for d in "$HOME/.local/bin" "$HOME/.cargo/bin"; do
    if [ -x "$d/uv" ]; then PATH="$d:$PATH"; fi
  done
  if ! command -v uv >/dev/null 2>&1; then
    say "uv가 없습니다. 아래 한 줄로 설치하고, 새 터미널에서 다시 실행하세요:"
    echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"
    return 1
  fi
}

gst_env() {
  # Rust webrtc 플러그인을 /opt/gst-plugins-rs에 설치했으면 (~/.bashrc에 없어도) 여기서 GST_PLUGIN_PATH를 잡는다
  local dir="/opt/gst-plugins-rs/lib/$(uname -m)-linux-gnu"
  if [ -d "$dir" ]; then
    case ":${GST_PLUGIN_PATH:-}:" in
      *":$dir:"*) ;;
      *) export GST_PLUGIN_PATH="$dir${GST_PLUGIN_PATH:+:$GST_PLUGIN_PATH}" ;;
    esac
  fi
}

cuda_env() {
  # pip로 설치한 CUDA 라이브러리(nvidia-cublas-cu12, nvidia-cudnn-cu12)를 CTranslate2(Whisper)가 찾게 한다
  [ -x "$PY" ] || return 0
  local libs
  libs=$("$PY" -c 'import importlib.util as u, os
dirs = []
for pkg in ("nvidia.cublas", "nvidia.cudnn"):
    s = u.find_spec(pkg)
    if s and s.submodule_search_locations:
        d = os.path.join(list(s.submodule_search_locations)[0], "lib")
        if os.path.isdir(d):
            dirs.append(d)
print(":".join(dirs))' 2>/dev/null)
  if [ -n "$libs" ]; then export LD_LIBRARY_PATH="$libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; fi
  return 0
}

check_system() {
  local missing=0
  command -v cc >/dev/null 2>&1 || missing=1
  command -v pkg-config >/dev/null 2>&1 || missing=1
  if [ "$missing" -eq 0 ]; then
    pkg-config --exists cairo gobject-introspection-1.0 libffi gstreamer-1.0 || missing=1
  fi
  if [ "$missing" -eq 1 ]; then
    say "시스템 패키지가 부족합니다 (C 컴파일러, pkg-config, cairo·gobject-introspection·libffi·GStreamer 헤더)."
    say "reachy-mini가 리눅스에서는 PyGObject를 소스에서 빌드하고 시스템 GStreamer를 씁니다. Ubuntu/Debian:"
    echo "    sudo apt-get update && sudo apt-get install -y $APT_PACKAGES"
    return 1
  fi
  local gv
  gv=$(pkg-config --modversion gstreamer-1.0 2>/dev/null || echo 0)
  if ! printf '%s\n%s\n' "1.22" "$gv" | sort -V -C; then
    say "GStreamer $gv: 1.22 이상이 필요합니다 (Ubuntu 24.04 권장, 22.04는 docs/desktop_check.md의 PPA 안내)."
    return 1
  fi
  return 0
}

check_webrtc() {
  gst_env
  if command -v gst-inspect-1.0 >/dev/null 2>&1 && gst-inspect-1.0 webrtcsrc >/dev/null 2>&1; then
    say "GStreamer webrtcsrc 있음 → 로봇 카메라·마이크(WebRTC) 사용 가능"
    return 0
  fi
  say "주의: GStreamer Rust webrtc 플러그인(webrtcsrc)이 없습니다. 로봇 카메라·마이크를 쓸 수 없습니다 (mock·시뮬은 됨)."
  say "설치: ./scripts/linux_gstreamer_webrtc.sh  (Rust로 gst-plugins-rs 0.14.1 빌드, 10분 안팎)"
  return 1
}

need_venv() {
  if [ ! -x "$PY" ]; then say ".venv가 없습니다. 먼저: ./start_linux.sh setup"; return 1; fi
}

run_py() {
  need_venv || return 1
  gst_env
  cuda_env
  "$PY" -X utf8 "$@"
}

do_setup() {
  find_uv || return 1
  check_system || return 1
  local extras="dev,robot,speech,sim"
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    extras="$extras,cuda"
    say "NVIDIA GPU 발견 → GPU 음성 인식(cuda) 포함"
  else
    say "NVIDIA GPU 없음 → 음성 인식은 CPU에서 (자동 전환)"
  fi
  if [ ! -x "$PY" ]; then uv venv --python 3.12 .venv || { say "실패: .venv를 만들지 못했습니다"; return 1; }; fi
  uv pip install --python "$PY" -e ".[$extras]" \
    || { say "실패: 위 메시지를 보세요 (PyGObject·pycairo 빌드 오류면 apt 패키지가 부족한 것)"; return 1; }
  chmod +x start_linux.sh scripts/*.sh 2>/dev/null
  check_webrtc
  say "설치 끝. 다음: ./start_linux.sh preflight"
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
    webrtc) check_webrtc ;;
    *) say "모르는 명령: $cmd (setup preflight spike calibrate kiosk mock sim test)"; return 2 ;;
  esac
}

menu() {
  echo "Reachy Café (Linux)"
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
