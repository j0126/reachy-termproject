# Raspberry Pi 4 + Touch Display 2 키오스크 설치

Pi 4는 화면 표시(Chromium 키오스크)와 카드 슬롯(GPIO)만 맡습니다. 인식·대화·FSM은 전부 두뇌 PC(데스크톱)에서 돕니다.

## 1. 준비물

| 항목 | 비고 |
|---|---|
| Raspberry Pi 4 | RAM 2 GB 이상 권장 |
| 공식 15.3 W(5.1 V 3 A) USB-C 전원 | 디스플레이도 Pi의 5 V 핀에서 전원을 받으므로 정품 권장 |
| microSD 16 GB 이상 | Raspberry Pi OS (64-bit) with desktop |
| 공식 Touch Display 2 (7") | 동봉된 15핀↔15핀 FFC 케이블, GPIO 전원 케이블 사용 |
| 레버형 마이크로스위치, LED, 220 Ω 저항, 점퍼선 | 모의 카드 슬롯 |

## 2. OS 설치

1. Raspberry Pi Imager → **Raspberry Pi OS (64-bit)** (데스크톱 포함) 선택.
2. 고급 설정에서 사용자 이름·비밀번호, 공유기 Wi-Fi, SSH 사용을 미리 넣는다.
3. 발표 직전에 핫스팟을 쓸 수도 있으면, 첫 부팅 후 그 핫스팟 Wi-Fi도 미리 한 번 연결해 저장해 둔다.

## 3. 디스플레이 연결 (전원을 끈 상태에서)

1. **영상·터치**: FFC 케이블을 디스플레이와 Pi 4의 `DISPLAY` 커넥터에 연결. 금속 접점은 디스플레이 쪽에서는 위(디스플레이 반대쪽), Pi 쪽에서는 이더넷·USB 포트 쪽을 향하게. 터치 신호도 이 케이블로 간다.
2. **전원**: GPIO 전원 케이블을 디스플레이 `J1`에 꽂고, 반대쪽은 Pi **핀 2(5 V, 빨강)**, **핀 6(GND, 검정)**에 꽂는다. 잘못 꽂으면 디스플레이가 손상될 수 있으니 두 번 확인한다.
3. 7"는 Pi 4에서 자동 인식된다. 화면이 안 나오면 `/boot/firmware/config.txt`에 `dtoverlay=vc4-kms-dsi-ili9881-7inch`를 추가한다.
4. 기본 방향이 세로(720×1280)라 그대로 쓴다. 돌려야 하면 Control Centre → Screens → `DSI-1` → Orientation.
5. **터치는 마우스 클릭으로 들어온다** (11/27 팀 Pi에서 확인, Raspberry Pi OS 기본값 labwc `mouseEmulation="yes"`). 탭은 클릭이 되지만 손가락으로 밀어도 스크롤되지 않는다. 그래서 키오스크 화면은 720×1280에서 스크롤 없이 들어가게 만들었고, 누른 자리에 남는 커서는 키오스크 주소(`/kiosk?device=pi`)에서 숨긴다.

## 4. 카드 슬롯 배선

| 부품 | Pi 핀 (BCM) |
|---|---|
| 마이크로스위치 NO 단자 | 핀 11 (GPIO17) |
| 마이크로스위치 COM 단자 | 핀 9 (GND) |
| LED 양극(긴 다리) → 220 Ω 저항 | 핀 13 (GPIO27) |
| LED 음극(짧은 다리) | 핀 14 (GND) |

디스플레이 전원 핀(2, 6)과 겹치지 않습니다. 카드를 넣으면 레버가 눌려 스위치가 닫히도록 슬롯에 고정하세요.

## 5. 코드 복사와 설치

프로젝트의 `pi_agent` 폴더를 Pi의 `~/reachy-cafe/pi_agent`로 복사한다 (git clone 또는 scp). Windows에서는 프로젝트 폴더에서:

```powershell
ssh <사용자>@<Pi IP> "mkdir -p ~/reachy-cafe"
scp -r pi_agent <사용자>@<Pi IP>:~/reachy-cafe/
```

그다음 Pi에서 (SSH로 접속해도 됨):

```bash
bash ~/reachy-cafe/pi_agent/install_pi.sh
nano ~/reachy-cafe/pi_agent/kiosk.env      # BRAIN_HOST를 두뇌 PC IP로 (예: 192.168.0.10)
sudo reboot
```

11/27 팀 Pi(`irl-rpi`)에 이 순서로 설치했다. 약 7분 걸렸고, 대부분은 패키지 설치와 글꼴 캐시 생성이다. SSH 비밀번호는 접속 후 2분 안에 넣어야 한다(넘기면 `Connection reset`).

설치 스크립트가 하는 일:
- `python3-websockets`, `python3-gpiozero`, 한글 폰트(`fonts-noto-cjk`), 이모지 폰트(`fonts-noto-color-emoji`) 설치
- 카드 슬롯 에이전트를 systemd 서비스(`reachy-card-agent`)로 등록 → 부팅하면 자동 실행
- 데스크톱 로그인 시 `start_kiosk.sh` 자동 실행 → 두뇌 PC 서버가 뜰 때까지 기다렸다가 Chromium 키오스크로 `/kiosk?device=pi`를 연다. 페이지가 60초 안에 두뇌 PC에 붙지 않으면(두뇌 PC `/healthz`의 `pi_kiosk`가 0, 흰 화면) Chromium을 다시 띄우고, Chromium이 꺼져도 다시 띄운다. 기록은 `/run/user/$(id -u)/reachy-kiosk.log`
- 화면 보호기 끄기

추가로 Control Centre → Display에서 **화상 키보드(On-screen keyboard)를 Disabled**로 바꾼다 (키오스크에는 입력창이 없음).

## 6. 동작 확인

```bash
curl http://<BRAIN_HOST>:8000/healthz          # {"ok": true, ..., "pi_kiosk": 1} ← 1이면 Pi 화면이 붙은 것
journalctl -u reachy-card-agent -f              # "두뇌 PC에 연결됨", 카드를 넣으면 "카드 in"
```

두뇌 PC의 오퍼레이터 패널(`http://localhost:8000/operator`) 상단에 **화면 1 · 카드 슬롯 1**이 보이면 연결 완료입니다. 카드를 넣었다 빼면 이벤트 로그에 `card` 이벤트가 찍힙니다.

GPIO 없이 에이전트만 시험하려면 (PC에서도 가능):

```bash
python3 card_agent.py --keyboard --host <BRAIN_HOST>   # i + Enter = 넣기, o + Enter = 빼기
```

## 7. 문제 해결

| 증상 | 확인할 것 |
|---|---|
| 화면이 안 켜짐 | FFC 방향, 전원 핀 2·6, `config.txt`의 dtoverlay |
| 터치가 반응 없음 | `grep -i goodix /proc/bus/input/devices`에 `10-005d Goodix Capacitive TouchScreen`이 나오는지. 나오면 `sudo apt install evtest` 후 `sudo evtest`로 누를 때 이벤트가 찍히는지 본다. 안 나오면 `sudo apt update && sudo apt full-upgrade` 후 재부팅, FFC 양쪽 재장착 |
| 두뇌 PC에서 화면(app.js·style.css)을 고쳤는데 Pi는 그대로 | 화면은 페이지를 다시 열어야 바뀐다 → Pi 재부팅(`sudo reboot`). 서버가 `no-cache`를 보내므로 다시 열면 새 파일을 받는다 |
| 화면 아래쪽이 잘림 | 화면은 720×1280 기준으로 맞췄다. 노트북에서 볼 때는 창을 9:16 비율로 맞춰서 확인 |
| 화면이 하얗게 멈춤 (11/27 설치 직후 첫 부팅에서 겪음) | Chromium이 첫 페이지를 못 연 상태. `start_kiosk.sh`가 60–120초 안에 다시 띄운다. 계속되면 `cat /run/user/$(id -u)/reachy-kiosk.log`와 `tail ~/.xsession-errors`를 보고, 두뇌 PC `/healthz`의 `pi_kiosk`가 1이 되는지 확인 |
| 화면에 "연결 중…"만 보임 | `kiosk.env`의 BRAIN_HOST, 두뇌 PC가 켜져 있는지, 두뇌 PC 방화벽에서 TCP 8000 허용 |
| 한글이 네모로 보임 | `sudo apt install fonts-noto-cjk` 후 재부팅 |
| 메뉴 그림(이모지)이 안 보임 | `sudo apt install fonts-noto-color-emoji` |
| 번개 모양 저전압 경고, 화면 깜빡임 | 정품 15.3 W 전원 사용 |
| 카드 이벤트가 두 번씩 찍힘 | 스위치 고정 상태, `bounce_time`(기본 0.05초)을 늘려 보기 |

## 8. 발표 직전 핫스팟으로 바꿀 때

1. Pi와 로봇이 그 핫스팟에 미리 연결돼 있는지 확인한다 (Wi-Fi 설정은 IP와 별개).
2. 핫스팟에서 받은 두뇌 PC IP로 `kiosk.env`의 `BRAIN_HOST`를 고친다.
3. `sudo systemctl restart reachy-card-agent` 후 재부팅한다 (키오스크 화면도 새 주소로 다시 열림).
4. 두뇌 PC 쪽은 `config/default.toml`의 `[robot] host`(또는 `REACHY_HOST`)만 로봇의 새 IP로 바꾼다.
