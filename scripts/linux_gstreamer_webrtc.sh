#!/usr/bin/env bash
# 리눅스에서 Reachy Mini SDK의 WebRTC 카메라·마이크를 쓰기 위한 GStreamer 준비 (한 번만).
# Windows·맥은 pip 패키지(gstreamer-bundle)에 들어 있어 필요 없다. 순서는 Seeed Studio의 Reachy Mini 안내와 같다:
#   https://wiki.seeedstudio.com/reachymini_sdk_gstreamer-installation/
#   1) apt: GStreamer 1.22+ 개발 패키지와 플러그인   2) Rust (rustup)
#   3) gst-plugins-rs 0.14.1의 webrtc 플러그인을 cargo-c로 빌드해 /opt/gst-plugins-rs에 설치
#   4) GST_PLUGIN_PATH (start_linux.sh는 자동으로 잡고, 원하면 ~/.bashrc에도 넣는다)
# 단계마다 묻고 진행한다. sudo 비밀번호는 apt와 /opt 폴더 만들 때만 필요하다.
set -eu
APT_PACKAGES="build-essential pkg-config git curl libffi-dev libssl-dev libglib2.0-dev libcairo2-dev \
libgirepository1.0-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev \
gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-alsa gstreamer1.0-nice libnice10 libportaudio2 \
python3-gi python3-gi-cairo libgl1"
PREFIX=/opt/gst-plugins-rs
TAG=0.14.1
ARCH_DIR="$PREFIX/lib/$(uname -m)-linux-gnu"
WORK="${XDG_CACHE_HOME:-$HOME/.cache}/reachy-cafe/gst-plugins-rs"

ask() { read -r -p "$1 [Y/n] " a; case "$a" in [Nn]*) return 1 ;; *) return 0 ;; esac; }

echo "== 1/4 apt 패키지 (GStreamer 1.22 이상 필요: Ubuntu 24.04 기본, 22.04는 PPA ppa:savoury1/multimedia)"
if ask "sudo apt-get install로 설치할까요?"; then
  sudo apt-get update
  # shellcheck disable=SC2086
  sudo apt-get install -y $APT_PACKAGES
fi
gv=$(pkg-config --modversion gstreamer-1.0 2>/dev/null || echo 0)
echo "GStreamer $gv"
if ! printf '%s\n%s\n' "1.22" "$gv" | sort -V -C; then
  echo "GStreamer 1.22 이상이 필요합니다. Ubuntu 22.04라면:"
  echo "    sudo add-apt-repository ppa:savoury1/multimedia && sudo apt update"
  echo "    sudo apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-good1.0-dev libgstreamer-plugins-bad1.0-dev"
  exit 1
fi

echo "== 2/4 Rust"
if ! command -v cargo >/dev/null 2>&1 && [ -f "$HOME/.cargo/env" ]; then . "$HOME/.cargo/env"; fi
if ! command -v cargo >/dev/null 2>&1; then
  if ask "rustup(공식 설치 스크립트)으로 Rust를 설치할까요?"; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    . "$HOME/.cargo/env"
  else
    echo "Rust가 있어야 플러그인을 빌드할 수 있습니다."; exit 1
  fi
fi
cargo --version

echo "== 3/4 gst-plugins-rs $TAG webrtc 플러그인 빌드 → $PREFIX (몇 분 걸림)"
if ask "빌드할까요?"; then
  mkdir -p "$(dirname "$WORK")"
  if [ ! -d "$WORK/.git" ]; then
    git clone https://gitlab.freedesktop.org/gstreamer/gst-plugins-rs.git "$WORK"
  fi
  cd "$WORK"
  git fetch --tags
  git checkout "$TAG"
  cargo install cargo-c
  sudo mkdir -p "$PREFIX"
  sudo chown "$USER" "$PREFIX"
  cargo cinstall -p gst-plugin-webrtc --prefix="$PREFIX" --release
fi

echo "== 4/4 확인"
export GST_PLUGIN_PATH="$ARCH_DIR${GST_PLUGIN_PATH:+:$GST_PLUGIN_PATH}"
if gst-inspect-1.0 webrtcsrc >/dev/null 2>&1; then
  echo "webrtcsrc OK ($ARCH_DIR)"
else
  echo "webrtcsrc를 찾지 못했습니다: 빌드 메시지를 확인하세요"; exit 1
fi
if ! grep -q "gst-plugins-rs" "$HOME/.bashrc" 2>/dev/null && ask "~/.bashrc에 GST_PLUGIN_PATH를 넣을까요? (start_linux.sh는 없어도 자동으로 잡음)"; then
  echo "export GST_PLUGIN_PATH=$ARCH_DIR:\$GST_PLUGIN_PATH" >> "$HOME/.bashrc"
fi
echo "끝. 다음: ./start_linux.sh setup (또는 이미 설치했다면 ./start_linux.sh preflight)"
