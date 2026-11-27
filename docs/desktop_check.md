# 데스크톱 점검 안내 (실제 로봇 첫 연결) — Windows · macOS · Linux

목표: 데모를 돌릴 PC와 실제 Reachy Mini에서 2단계 코드가 도는지 확인하고, 키오스크 배치에 맞는 보정 값을 잡는다.
기준 PC는 HW3를 돌렸던 Windows 데스크톱(RTX 4060 Ti)이고, 같은 절차를 맥과 리눅스에서도 쓸 수 있다.
예상 시간: 처음 설치 포함 60–90분 (리눅스는 GStreamer 준비로 +15분). 각 단계는 실패해도 다음 단계로 넘어가며 결과를 `logs/`에 남긴다.

## OS별 시작 스크립트

모든 단계는 시작 스크립트 하나로 한다. 인자 없이 실행(더블클릭)하면 번호 메뉴가 뜨고, 명령을 직접 줄 수도 있다.

| | Windows | macOS | Linux |
|---|---|---|---|
| 스크립트 | `start_windows.bat` | `start_mac.command` | `start_linux.sh` |
| 메뉴 | 더블클릭 | Finder에서 더블클릭 | 터미널에서 `./start_linux.sh` |
| 명령 실행 | `.\start_windows.bat <명령>` (PowerShell) | `./start_mac.command <명령>` | `./start_linux.sh <명령>` |

| 단계 | 메뉴 | 명령 | 로봇이 움직이나 | 시간 |
|---|---|---|---|---|
| 1 설치 | 1 | `setup` | 아니요 | 10–20분 |
| 2 사전 점검 | 2 | `preflight` | 아니요 (읽기만) | 2분 |
| 3 연결 점검 | 3 | `spike` | 예 | 5분 |
| 4 배치 보정 | 4 | `calibrate` | 예 | 10분 |
| 5 키오스크 실행 | 5 | `kiosk` | 예 | 자유 |
| (로봇 없이) | 6 / 7 / 8 | `mock` / `sim` / `test` | 아니요 | |

## 0. 준비물과 OS 조건

| | Windows | macOS | Linux |
|---|---|---|---|
| 지원 | Windows 10/11 64비트 | **Apple Silicon(M1 이상) + macOS 14 이상** | **Ubuntu 24.04** x86_64 권장 (22.04는 GStreamer PPA 필요) |
| 음성 인식(Whisper) | NVIDIA GPU면 GPU, 없으면 CPU | CPU (CUDA 없음) | NVIDIA GPU면 GPU, 없으면 CPU |
| GStreamer (로봇 카메라·마이크) | pip에 포함 | pip에 포함 | **직접 설치** (아래 'Linux 추가 준비') |
| 코드 받기 | `git clone` (권장) 또는 `reachy_cafe_windows_*.zip` | `git clone` 또는 `reachy_cafe_mac_*.zip` | `git clone` 또는 `reachy_cafe_linux_*.tar.gz` |

- Intel 맥과 macOS 13 이하는 안 된다: mediapipe 1.0.1·onnxruntime 1.27.0에 해당 패키지가 없다 (uv로 확인).
- 음성 인식은 GPU가 없으면 자동으로 CPU(int8)로 돈다. 사전 점검이 인식 시간을 재 주니, 느리면(2초 이상) `config/default.toml`의 `[speech] asr_model = "small"`.
- 공통: Reachy Mini Wireless(충전), 공유기(로봇 192.168.0.15, PC도 같은 공유기), [uv](https://docs.astral.sh/uv/) (없으면 시작 스크립트가 설치 명령을 알려 준다)
- 배치: 손님 기준 **로봇이 화면 왼쪽** (로봇 기준 화면이 왼쪽). 화면 가운데가 로봇 정면에서 **40° 안쪽**이면 시선이 끝까지 간다
- (선택) Pi 4 + Touch Display 2, 카드 슬롯: 없으면 오퍼레이터 패널의 화면·카드 버튼으로 대신한다

코드는 git으로 받는다. HW3·HW4 코드와 HW2 얼굴 모델이 저장소 안(`vendor/`, `models/`)에 들어 있어 이것만 받으면 된다:

```powershell
git clone https://github.com/j0126/reachy-termproject.git termproject
```

git을 쓸 수 없으면 노트북에서 묶음을 만든다: `.venv\Scripts\python -X utf8 tools\make_bundle.py` (`dist/`에 세 OS 묶음이 생김)

## 1. 설치 (처음 한 번)

받은 `termproject` 폴더에서 설치한다. 기존 HW3 폴더 안에 받거나 덮어쓰지 않는다. 아래 예시는 `C:\ReachyCafe\termproject`에 받은 경우다.

**Windows** (PowerShell):

```powershell
cd C:\ReachyCafe\termproject
```

```powershell
.\start_windows.bat setup
```

**macOS**: `termproject/start_mac.command` 더블클릭 → 1. "확인되지 않은 개발자"로 막히면 우클릭 > 열기, 또는:

```bash
xattr -dr com.apple.quarantine ~/ReachyCafe && chmod +x ~/ReachyCafe/termproject/start_mac.command
```

**Linux**: 먼저 아래 'Linux 추가 준비'를 한 뒤

```bash
cd ~/ReachyCafe/termproject && ./start_linux.sh setup
```

HW3를 돌렸던 PC라면 Whisper(large-v3-turbo)·Supertonic 모델이 캐시에 있어 다시 받지 않는다. 처음이면 첫 실행 때 약 1.6 GB를 받는다.

### Linux 추가 준비 (한 번)

reachy-mini는 리눅스에서 시스템 GStreamer와 PyGObject(소스 빌드)를 쓰고, 로봇 카메라·마이크(WebRTC)에는 GStreamer **Rust webrtc 플러그인**(`webrtcsrc`)이 필요하다. Windows·맥은 pip 패키지에 들어 있다. 순서는 [Seeed Studio의 Reachy Mini GStreamer 안내](https://wiki.seeedstudio.com/reachymini_sdk_gstreamer-installation/)와 같고, 도우미 스크립트가 단계마다 물어보며 진행한다 (apt → Rust → gst-plugins-rs 0.14.1 빌드 → `/opt/gst-plugins-rs`):

```bash
cd ~/ReachyCafe/termproject && ./scripts/linux_gstreamer_webrtc.sh
```

- GStreamer 1.22 이상 필요: Ubuntu 24.04는 기본 패키지로 된다. 22.04는 `sudo add-apt-repository ppa:savoury1/multimedia` 후 설치 (스크립트가 안내)
- `start_linux.sh`는 `GST_PLUGIN_PATH`(플러그인)와 `LD_LIBRARY_PATH`(pip로 설치한 CUDA 라이브러리)를 스스로 잡는다
- 플러그인 없이도 mock·시뮬·테스트는 된다. `./start_linux.sh webrtc`로 플러그인 상태만 확인할 수 있다

## 2. 사전 점검 (로봇은 읽기만, 모터 명령 없음)

메뉴 2, 또는 `preflight` (Whisper·Supertonic·MediaPipe를 실제로 한 번 돌려 본다).

확인할 것:
- **FAIL 0개**. OS 조건, 패키지, GStreamer WebRTC 요소(`webrtcsrc` 등), HW 파일, 얼굴 모델
- 음성 인식: GPU면 `cuda`, 맥·GPU 없는 PC는 `cpu` — "2초 소리 인식" 시간이 1–2초 안인지
- 로봇: 응답 시간, daemon `1.11.0`, 카메라·오디오 **사용 가능**, IK `AnalyticalKinematics`
- 이 PC와 로봇이 **같은 네트워크**인지 (이 PC IP가 Pi 4의 `BRAIN_HOST`)

| 결과 | 할 일 |
|---|---|
| 로봇 daemon에 연결할 수 없음 | 로봇 전원, 같은 공유기인지, 로봇 IP (다르면 `preflight --robot-host <IP>`, 계속 쓸 거면 `config/default.toml`의 `[robot] host`) |
| 카메라·오디오 해제됨 | `preflight --fix-media` |
| GStreamer WebRTC 요소 없음 | Windows·맥: `setup` 다시. 리눅스: 'Linux 추가 준비' |
| Whisper GPU 로딩 실패 | 자동으로 CPU로 다시 시도한다. GPU를 쓰려면 `setup` 다시 (NVIDIA 드라이버, cuda 포함) |
| 키오스크 포트 8000 사용 중 | 다른 키오스크·HW3 대시보드 끄기 |

## 3. 연결 점검 (로봇이 움직임: 주변을 비운다)

메뉴 3, 또는 `spike`. 단계마다 Enter로 진행:
- **faces**: 로봇 앞 약 1 m에 서서 로봇을 본다
- **audio**: 로봇 스피커에서 문장이 들렸는지 `y`/`n`
- **asr**: "아메리카노 주세요"라고 말한다
- **motion**: 로봇이 화면 쪽을 봤다가 Aha! 모션 + 효과음 + 문장을 동시에 한다. 효과음이 "팝" 하는 순간과 맞는지, 끊김이 없는지 본다

여기서 잰 명령→움직임 지연이 라벨 지연(`label_delay_s`)으로 `config/calibration.toml`에 저장된다.

## 4. 배치 보정 (키오스크 배치를 고정한 뒤)

메뉴 4, 또는 `calibrate`:
- **화면·슬롯 위치**: 로봇이 보는 곳이 화면 가운데, 다음엔 카드 슬롯이 될 때까지 옮긴다
  - `l`/`r` = 로봇 기준 왼쪽/오른쪽, `u`/`d` = 위/아래, `f`/`b` = 앞/뒤 (2 cm, 대문자면 5 cm), Enter = 확정
  - "(범위·관절 한계로 덜 돌아감)"이 뜨면 값을 더 옮기지 말고 로봇이나 화면을 조금 옮긴다
- **고개 방향**: 안내에 따라 로봇 보기 → 화면 보기 → 딴 곳 보기 (각 4초)

결과는 `config/calibration.toml`에 합쳐진다 (다시 하면 같은 키만 바뀐다. 손으로 고쳐도 된다).

## 5. 키오스크 실행 (실제 로봇)

메뉴 5, 또는 `kiosk` (`config/calibration.toml`이 있으면 자동으로 함께 적용).

- 오퍼레이터 패널 `http://localhost:8000/operator`: 상단 "로봇 real", **상태 → 채널**에 모션 ✓ · 음성 ✓ · 카메라 ✓
- S1을 실제로 해 본다: 다가가면 인사 → 말로 주문 → 화면 터치로 옵션 → 결제 (카드 슬롯이 없으면 오퍼레이터의 "카드 넣기/빼기")
- 방화벽 (Pi 4가 이 PC의 8000 포트로 접속): Windows는 처음 실행 때 **개인 네트워크 허용**, 맥은 시스템 설정 > 네트워크 > 방화벽에서 Python 허용, 리눅스는 방화벽을 쓰면 `sudo ufw allow 8000/tcp`
- (선택) Pi 4: `pi_agent/kiosk.env`의 `BRAIN_HOST`를 2단계에서 나온 IP로 (pi_agent/SETUP_PI4.md)
- 끝낼 때는 실행 창에서 **Ctrl+C**: 로봇이 기본 자세(REST)로 돌아간 뒤 연결을 닫는다

## 6. 노트북으로 보내 줄 것

- `logs/preflight_*.json`, `logs/spike_*.json`, `config/calibration.toml`
- 키오스크를 돌렸으면 `logs/<시각>/` 폴더 (`events.jsonl`, `speech_events.jsonl`)
- 눈으로 본 것 메모: 효과음·화면이 모션과 맞았는지, 시선이 화면·슬롯·손님을 제대로 봤는지, 음성 인식이 틀린 말, 끊김·지연

## 문제가 생기면

| 증상 | 확인·해결 |
|---|---|
| (Windows) PowerShell에서 `start_windows.bat`를 못 찾음 | 앞에 `.\`를 붙인다: `.\start_windows.bat setup` |
| (맥) 로봇에 연결이 안 됨 (No route to host) | 시스템 설정 > 개인정보 보호 및 보안 > 로컬 네트워크에서 터미널 허용 (macOS 15+) |
| (맥) 스크립트가 열리지 않음 | 1단계의 `xattr`·`chmod` 명령 |
| (리눅스) setup이 PyGObject·pycairo 빌드에서 실패 | `start_linux.sh`가 알려 주는 apt 패키지 설치 |
| (리눅스) 카메라 ✗, "webrtcsrc" 오류 | 'Linux 추가 준비'의 플러그인 빌드, `./start_linux.sh webrtc`로 확인 |
| spike camera fps가 10 미만 | Wi-Fi 혼잡: 공유기를 가까이, 다른 기기 줄이기 |
| 소리가 안 들림 | 로봇 볼륨(HW3 `TROUBLESHOOTING.md`), 사전 점검의 카메라·오디오 상태 |
| 모션 중 멈칫함 (late_ticks) | Wi-Fi 상태, PC의 무거운 프로그램 종료 |
| 음성 인식이 느림 (CPU) | `[speech] asr_model = "small"`, 또는 NVIDIA GPU가 있는 PC에서 실행 |
| 키오스크가 "mock (connect failed)" | 로봇 연결 실패 → IP·전원 확인 후 다시 실행 (화면·오퍼레이터는 계속 동작) |
| 채널에 음성 ✗ / 카메라 ✗ | 그 채널만 실패: 음성은 텍스트 입력, 카메라는 얼굴 프리셋으로 진행 가능. 사전 점검으로 원인 확인 |
| 로봇 대신 미리 해 보기 | 메뉴 7 / `sim` (MuJoCo 시뮬, 설치에 포함) |
