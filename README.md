# Reachy Café: 표현적인 멀티모달 로봇 키오스크

HRI 텀프로젝트 3조. Reachy Mini가 손님을 감지하고 음성·시선·모션·터치스크린(Raspberry Pi 4 + Touch Display 2)·모의 카드 슬롯으로 카페 주문을 인사부터 주문번호까지 안내합니다.

- 설계와 일정: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)
- Pi 4 키오스크 설치: [pi_agent/SETUP_PI4.md](pi_agent/SETUP_PI4.md)

## 구성

```
 Reachy Mini Wireless (192.168.0.15)          두뇌 PC (데스크톱)                       Raspberry Pi 4
 카메라·마이크·스피커·모션  ◀── SDK ──▶  인식 → 사람 상태 → 주문 FSM → 행동  ◀── WebSocket ──▶  Touch Display 2 (Chromium)
                                       FastAPI 서버 :8000 (/kiosk, /operator)                 카드 슬롯 GPIO 에이전트
```

로봇 없이도 **mock 로봇**과 **오퍼레이터 패널(시뮬레이터)**로 전체 주문 흐름이 끝까지 돕니다 (1단계). 실제 로봇 백엔드(2단계)는 HW2(얼굴 인식)·HW3(음성)·HW4(모션) 코드를 연결 하나로 묶어 `reachy_cafe/robot/`에 있고, 같은 인터페이스(`reachy_cafe/behavior/robot_api.py`)로 갈아 끼웁니다.

## 빠른 시작 (로봇 없이)

Python 3.12와 [uv](https://docs.astral.sh/uv/)가 필요합니다. OS별 시작 스크립트가 설치와 실행을 맡습니다. 인자 없이 실행(더블클릭)하면 번호 메뉴가 뜹니다 (1 설치 → 6 mock).

| OS | 설치 | 로봇 없이 실행 |
|---|---|---|
| Windows 10/11 | `.\start_windows.bat setup` | `.\start_windows.bat mock` |
| macOS (Apple Silicon, 14 이상) | `./start_mac.command setup` | `./start_mac.command mock` |
| Linux (Ubuntu 24.04) | `./start_linux.sh setup` | `./start_linux.sh mock` |

`setup`은 실제 로봇용 패키지(음성·카메라·시뮬)까지 설치합니다. 화면 흐름만 볼 거라면 가볍게 설치할 수도 있습니다 (리눅스도 GStreamer 없이 됨):

```powershell
uv venv --python 3.12 .venv
```

```powershell
uv pip install -e ".[dev]"
```

```powershell
.venv\Scripts\python -m reachy_cafe --config config/dev_laptop.toml
```

(맥·리눅스는 `.venv/bin/python -m reachy_cafe --config config/dev_laptop.toml`)

- 오퍼레이터 패널: http://localhost:8000/operator (오른쪽에 키오스크 미리보기가 같이 뜸)
- 키오스크 화면: http://localhost:8000/kiosk (Pi 4가 여는 주소는 `http://<두뇌 PC IP>:8000/kiosk`)
- 청중 화면(프로젝터용 "Reachy의 생각"): http://localhost:8000/stage
- Pi 4가 8000 포트로 접속하므로 방화벽에서 허용합니다 (Windows는 처음 실행 때 **개인 네트워크 허용**).

오퍼레이터 패널에서 데모 시나리오를 그대로 흉내 낼 수 있습니다.
1. 얼굴 시뮬레이션 → **1명 가까이** (로봇이 인사)
2. 발화 주입의 **S1/S2/S3** 버튼을 차례로 누르거나 직접 입력
3. **카드 넣기 / 빼기**로 결제
4. 자리 비움(**없음**), 두 명(**2명**), 프레임 끊김, 오인식("바닐") 같은 실패 상황도 버튼으로 재현
5. **조건 A/B**로 with/without 비교 (A = 모션·시선·LED·상태 스트립 끔)
6. **보완 기능**(호객·접근 의도, 주의 방향, 백채널, 날씨 인사)을 켜고 끄며 유지할지 판단. 얼굴 프리셋 "다가옴·지나감·가까이·화면 봄"으로 시험
7. **평가 기록**에 참가자·조건·과제를 넣고 시행 → `tools/analyze_logs.py`로 집계

## 실제 로봇과 데스크톱 (2단계)

기준은 HW3를 돌렸던 Windows 데스크톱(RTX 4060 Ti)이고, 같은 절차를 macOS·Linux에서도 쓸 수 있습니다. **처음 로봇에 연결하는 날은 [docs/desktop_check.md](docs/desktop_check.md)의 순서를 따릅니다** (OS별 조건·명령·문제 해결 포함).

| 단계 | 시작 스크립트 명령 (메뉴 번호) | 하는 일 |
|---|---|---|
| 설치 | `setup` (1) | `.venv` + 로봇·음성·시뮬 패키지. NVIDIA GPU가 있으면 GPU 음성 인식(cuda) 포함 |
| 사전 점검 | `preflight` (2) | 로봇은 읽기만: OS 조건, 패키지, GStreamer WebRTC, HW 파일, 모델 실제 실행, daemon·미디어·네트워크 |
| 연결 점검 | `spike` (3) | 카메라 fps, 얼굴, 마이크·TTS, "아메리카노 주세요" 인식, 시선 + Aha! + 효과음 + 동시 TTS, 라벨 지연 측정 |
| 배치 보정 | `calibrate` (4) | 화면·카드 슬롯 위치를 로봇 시선으로 맞추고 고개 방향 부호·임계값 측정 → `config/calibration.toml` |
| 실행 | `kiosk` (5) | 실제 로봇으로 키오스크 (`config/calibration.toml`이 있으면 함께 적용) |

OS별로 다른 점:
- **Windows**: GStreamer가 pip 패키지(gstreamer-bundle)에 들어 있어 바로 됩니다. PowerShell에서는 `.\start_windows.bat <명령>`
- **macOS**: Apple Silicon(M1 이상) + macOS 14 이상만 됩니다 (mediapipe 1.0.1·onnxruntime 1.27.0 패키지). CUDA가 없어 음성 인식은 CPU(int8)
- **Linux**: GStreamer 1.22+와 Rust webrtc 플러그인을 직접 설치합니다: `./scripts/linux_gstreamer_webrtc.sh` ([Seeed Studio 안내](https://wiki.seeedstudio.com/reachymini_sdk_gstreamer-installation/)와 같은 순서)
- GPU가 없거나 GPU 음성 인식 로딩이 실패하면 자동으로 CPU로 돌립니다 (느리면 `[speech] asr_model = "small"`)

**코드는 git으로 받습니다.** 키오스크가 쓰는 HW3·HW4 코드와 HW2 얼굴 모델이 저장소 안(`vendor/`, `models/`)에 들어 있어 HW 폴더를 따로 둘 필요가 없습니다.

```powershell
git clone https://github.com/j0126/reachy-termproject.git termproject
```

- `vendor/hw3`(HW3 `reachy_s2s` + 설정), `vendor/hw4`(HW4 `reachy_motion` + 모션 YAML·효과음), `models/face_landmarker.task`(HW2)는 원본 HW 폴더에서 `tools/vendor_hw.py`로 복사한 것입니다. 여기서 고치지 말고, 원본을 고친 뒤 다시 복사합니다. 앱은 시작할 때 `config/default.toml`의 `[hw]` 폴더(기본 `vendor/`)를 먼저 불러오므로 PC에 따로 설치된 HW 패키지와 섞이지 않습니다.
- git을 쓸 수 없으면 노트북에서 OS별 묶음을 만듭니다: `.venv\Scripts\python -X utf8 tools\make_bundle.py` (`dist/`에 zip·tar.gz)
- **NVIDIA GPU가 없는 PC에서 실행할 때**: 음성 인식만 데스크톱 GPU에 맡길 수 있습니다. 데스크톱에서 `.venv\Scripts\python -m reachy_cafe.asr_server`(포트 8001)를 띄우고, 현장 PC의 `config`에 `[speech] asr = "remote"`, `asr_url = "http://<데스크톱 IP>:8001"`을 둡니다. 데스크톱이 응답하지 않으면 현장 PC CPU의 `small`(약 2초)로 넘어갑니다(`asr_fallback`). 11/27 실측으로 GPU 없는 노트북 CPU는 large-v3-turbo가 말 한 번에 6–14초라 그대로는 못 씁니다.

- 로봇 IP는 `config/default.toml`의 `[robot] host` (기본 `192.168.0.15`). 핫스팟으로 바꿀 때는 이 값만 바꾸거나 `--robot-host`, `REACHY_HOST`를 씁니다.
- 처음 실행할 때 Whisper(large-v3-turbo)와 Supertonic 모델을 내려받습니다 (HW3를 돌린 PC면 캐시에 이미 있음). 고정 문장 TTS는 `assets/tts_cache`에 미리 합성해 둡니다.
- 로봇 연결이 안 되면 mock 로봇으로 바뀌고 오류를 로그에 남깁니다 (화면·오퍼레이터 패널은 계속 동작). 카메라·음성·모션은 따로 실패합니다: 카메라가 안 되면 오퍼레이터의 얼굴 프리셋, 음성이 안 되면 화면과 텍스트 입력, 모션이 안 되면 음성·화면만으로 진행합니다. 채널 상태(모션·음성·카메라 ✓/✗, ASR·TTS 로딩 중)는 오퍼레이터 패널의 "상태 → 채널"에 보입니다.
- 마이크 없이 말하기만 쓰려면 `[speech] asr = "text"` (발화는 오퍼레이터 패널에서 입력).

### 로봇 없이 MuJoCo 시뮬로 모션 경로 확인

MuJoCo 시뮬 daemon을 127.0.0.1에 띄워 실제 SDK 경로(명령 전송·상태 수신·시선·감정 모션·선점·종료 자세)를 확인합니다. MuJoCo는 `setup`이 함께 설치합니다 (`sim` 옵션, 없으면 `../reachy_hw4/.venv`의 것을 씀). HW4 안전 인터록으로 시뮬레이터에만 명령이 갑니다. 시뮬에는 카메라·마이크가 없어 모션만 봅니다. 시작 스크립트의 `sim` (메뉴 7), 또는:

```powershell
.venv\Scripts\python -X utf8 tools\sim_check.py
```

키오스크 앱 전체를 시뮬 로봇에 붙이려면 daemon만 띄워 두고(첫 번째 창) 앱을 시뮬 설정으로 실행합니다(두 번째 창). 얼굴은 오퍼레이터 프리셋, 발화는 텍스트 입력, 음성 출력은 시간만 흉내 냅니다 (카메라·음성 폴백 경로 확인도 겸함).

```powershell
.venv\Scripts\python -X utf8 tools\sim_check.py --daemon-only
```

```powershell
.venv\Scripts\python -X utf8 -m reachy_cafe --config config/sim.toml
```

## 폴더

| 경로 | 내용 |
|---|---|
| `reachy_cafe/dialogue/` | 메뉴·장바구니(`menu.py`), 규칙 기반 NLU(`nlu.py`), 추천(`recommender.py`), 문장 템플릿(`nlg.py`), 주문 FSM(`fsm.py`) |
| `reachy_cafe/perception/` | 사람 상태 추정(`human_state.py`), 시뮬레이션 얼굴(`sim.py`) |
| `reachy_cafe/behavior/` | Act 정의, 채널별 타이밍(`manager.py`), 모션 시간표(`motion_catalog.py`), mock 로봇, 로봇 인터페이스 |
| `reachy_cafe/robot/` | 실제 로봇: 공유 연결(`connection.py`), 모션·시선(`motion.py`, HW4 MotionService 확장), 모션 라이브러리(`motion_library.py`, HW4 모션 + 카페 모션), 음성(`speech.py`, HW3 VAD·Whisper·Supertonic), 카메라(`vision.py`, HW2 FaceLandmarker 확장), 묶음(`real.py`) |
| `motions/` | 카페에서 더한 모션 (HW4 YAML 형식): `greeting` 손님 인사 (expressive·neutral, 초안 `draft: true`) |
| `vendor/` | HW 코드 복사본: `hw3/`(HW3 `reachy_s2s` + 설정), `hw4/`(HW4 `reachy_motion` + 모션 YAML·효과음). `tools/vendor_hw.py`로 원본에서 복사 |
| `models/` | HW2 얼굴 모델 `face_landmarker.task` (MediaPipe) |
| `reachy_cafe/asr_server.py` | 음성 인식 서버: GPU 없는 현장 PC가 `asr = "remote"`로 데스크톱 GPU에 인식을 맡길 때 데스크톱에서 실행 |
| `tools/` | 데스크톱 사전 점검(`preflight.py`), 실제 로봇 연결 점검·배치 보정(`spike_connection.py`), MuJoCo 시뮬 점검(`sim_check.py`), 인사 모션 시뮬 영상(`greeting_video.py`, HW4 가상환경으로 실행), 데스크톱 묶음(`make_bundle.py`), 평가 로그 집계(`analyze_logs.py`) |
| `results/` | 만든 결과물: 인사 모션 시뮬 영상(`videos/`), 화면 캡처(`gui/`). 데스크톱 묶음에는 넣지 않음 |
| `reachy_cafe/kiosk_server/` | FastAPI + WebSocket (`/ws/ui`, `/ws/device`, `/ws/operator`) |
| `kiosk_ui/` | 키오스크 화면(`index.html`, `app.js`, `style.css`), 오퍼레이터 패널(`operator.html`, `operator.js`) |
| `pi_agent/` | Pi 4 카드 슬롯 에이전트, 키오스크 자동 실행, 설치 스크립트 |
| `config/` | `default.toml`(데스크톱·로봇 기준), `dev_laptop.toml`(mock), `sim.toml`(MuJoCo 시뮬 로봇), `menu.json`, `calibration.toml`(데스크톱에서 잰 보정 값, 생기면) |
| `start_windows.bat`, `start_mac.command`, `start_linux.sh` | OS별 시작 스크립트 (메뉴 / 명령: setup preflight spike calibrate kiosk mock sim test) |
| `scripts/` | 리눅스 GStreamer webrtc 플러그인 준비(`linux_gstreamer_webrtc.sh`) |
| `tests/` | NLU·추천·문장·사람 상태·FSM 시나리오(S1–S3, 실패 F1–F7)·자연스러움과 직원 안내(`test_naturalness.py`)·행동 타이밍·인사 모션(`test_greeting.py`)·앱 라우팅(`test_app.py`)·화면 파일 캐시(`test_server.py`)·원격 음성 인식과 CPU 대비책(`test_remote_asr.py`)·로봇 연결 순서(`test_connection.py`)·평가 집계 테스트 |
| `logs/` | 실행마다 `events.jsonl` (보고서 수치와 실패 분석용) |

## 평가 (참가자 10명)

절차·설문 문항·분석 방법은 [docs/evaluation_protocol.md](docs/evaluation_protocol.md)에 있습니다. 시행이 끝나면 로그를 세션별 지표로 집계합니다.

```powershell
.venv\Scripts\python tools/analyze_logs.py --only-trials
```

결과는 `logs/summary_sessions.csv`(엑셀에서 바로 열림)에 저장되고, 조건별 요약이 화면에 출력됩니다.

## 테스트

```powershell
.venv\Scripts\python -m pytest -q
```

실제 로봇 모듈 테스트(`test_motion_player`, `test_motion_safety`, `test_speech`, `test_vision`, `test_real_robot`)는 로봇 없이 HW4 MockBackend·HW3 MockAudioBackend·가짜 ASR/TTS/VAD로 실제 시간에 돌고, `.[robot]` 의존성이나 HW 폴더가 없으면 건너뜁니다. `test_motion_safety`는 시선 레이어가 보낼 수 있는 자세와 빨라진 전이를 HW4 안전 검사기(IK·관절 여유·관절 속도·자동 몸통 회전)로 확인합니다.
