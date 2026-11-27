#!/usr/bin/env bash
# Raspberry Pi 4 키오스크 설치 (Raspberry Pi OS 64-bit 데스크톱, labwc).
# 사용법: bash ~/reachy-cafe/pi_agent/install_pi.sh
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

echo "[1/5] 패키지 설치 (websockets, gpiozero, 한글·이모지 폰트)"
sudo apt update
sudo apt install -y python3-websockets python3-gpiozero fonts-noto-cjk fonts-noto-color-emoji curl
if ! command -v chromium >/dev/null && ! command -v chromium-browser >/dev/null; then
  sudo apt install -y chromium || sudo apt install -y chromium-browser
fi

echo "[2/5] 설정 파일"
[ -f "$DIR/kiosk.env" ] || cp "$DIR/kiosk.env.example" "$DIR/kiosk.env"
chmod +x "$DIR/start_kiosk.sh" "$DIR/card_agent.py"

echo "[3/5] 카드 슬롯 에이전트를 systemd 서비스로 등록"
sed -e "s|__USER__|$USER|g" -e "s|__DIR__|$DIR|g" "$DIR/reachy-card-agent.service" \
  | sudo tee /etc/systemd/system/reachy-card-agent.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable --now reachy-card-agent.service

echo "[4/5] 로그인하면 키오스크 화면 자동 실행 (labwc autostart)"
mkdir -p "$HOME/.config/labwc"
AUTOSTART="$HOME/.config/labwc/autostart"
touch "$AUTOSTART"
grep -q "start_kiosk.sh" "$AUTOSTART" || echo "\"$DIR/start_kiosk.sh\" &" >> "$AUTOSTART"

echo "[5/5] 화면 보호기 끄기"
sudo raspi-config nonint do_blanking 1 || echo "  raspi-config → Display Options → Screen Blanking → No 로 직접 꺼 주세요"

echo
echo "완료. $DIR/kiosk.env 의 BRAIN_HOST를 두뇌 PC IP로 고친 뒤 재부팅하세요: sudo reboot"
