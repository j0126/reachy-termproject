# TROUBLESHOOTING — 자주 생기는 문제와 해결

먼저 `check_setup.bat`을 실행하면 대부분의 원인을 [FAIL]/[WARN]과 해결 방법으로 보여 준다(모터 명령 없음, 키 값은 출력하지 않음).

## 1. Reachy 연결 실패

| 증상 | 원인 / 해결 |
|---|---|
| `check_setup`: "로봇 데몬에 연결할 수 없음" / 대시보드 "connect failed" | IP가 틀렸거나 다른 네트워크다. `.env`의 `REACHY_HOST`, PC와 로봇이 같은 Wi-Fi인지, 로봇 전원(데몬은 켜진 뒤 1분쯤 걸림)을 확인한다 |
| 연결은 되는데 `TimeoutError: no microphone audio …` | WebRTC(UDP)가 막혔다. Windows 방화벽에서 Python(`.venv\Scripts\python.exe`)의 네트워크 허용, 같은 네트워크인지 확인한다 |
| SDK 버전 경고 / `check_setup`: 데몬 버전 다름 | PC의 `reachy-mini`(1.11.0)와 로봇 데몬 버전을 맞춘다(로봇 업데이트 또는 `pyproject.toml` 버전) |
| 로그에 `No Reachy Mini Audio USB device found!` | Wireless에서는 정상이다(무시) |
| `rtpsession … empty RTCP`, `Signalling error … receiver is gone` | GStreamer 내부 메시지다. 기능에는 영향이 없다 |

## 2. 마이크·스피커

| 증상 | 원인 / 해결 |
|---|---|
| 대시보드 Mic level이 움직이지 않음 / 로봇 소리가 안 남 | 로봇의 카메라·오디오(미디어)가 해제된 채일 수 있다. 다른 프로그램(`motion_preview --pose` 등 no-media 도구)이 끝나지 않았던 경우다. `check_setup.bat --fix-media` → 대시보드 다시 실행 |
| 로봇 목소리가 작음 / 안 들림 | 로봇 볼륨 확인(Audio 탭 `Test tone 440 Hz`). 녹음 재생은 `normalize playback`을 켠다 |
| 로봇이 말하는 중 내 말의 첫 단어가 빠짐 | 로봇 칩(XVF3800)의 에코 억제 때문이다(측정된 한계). 끼어들 때는 조금 길게 말한다 |
| 소음 속에서 짧은 말을 잘못 알아들음 | 측정된 한계다(PHASE5_REPORT). 문장을 또렷하게, 조금 길게 말한다 |

## 3. 로컬 LLM 서버 (A0 / A1)

| 증상 | 원인 / 해결 |
|---|---|
| `Load engines` 뒤 "LLM server not reachable" / "warm-up FAILED" | `start_llm_server.bat`을 먼저 실행하고 창을 켜 둔다 |
| `llama-server not found: …` | llama.cpp b11680 CUDA 12.4를 `%LOCALAPPDATA%\Programs\llama.cpp\b11680-cuda12.4\`에 풀었는지 확인한다([QUICK_START](QUICK_START.md) 1-5). 다른 위치면 `config/default.toml`의 `[llm_server] exe`를 고친다 |
| 포트 8080 사용 중 | 이미 서버가 떠 있거나 다른 프로그램이 쓰는 중이다. 기존 서버 창을 쓰거나 그 프로그램을 끈다 |
| GPU 메모리 부족(out of memory) | 다른 GPU 앱(게임, 동영상, 움직이는 배경화면)을 끈다. 8 GB에서 실측 7.3 GB를 쓴다. 작은 모델: `--config config/llm_local_gemma4_e2b.toml`(결과는 A0/A1과 섞지 않는다) |
| 첫 실행이 오래 걸림 | 모델 내려받기(LLM 4.6 GB, ASR 1.6 GB, TTS 0.4 GB)다. 한 번만 한다 |

## 4. CUDA / GPU

| 증상 | 원인 / 해결 |
|---|---|
| `check_setup`: "NVIDIA GPU 0개" | GPU가 없거나 드라이버가 없다. A0/A1은 불가, B를 쓴다 |
| `Load engines`에서 `cublas64_12.dll` / `cudnn` 관련 오류 | CUDA 라이브러리가 빠졌다. NVIDIA 드라이버를 설치한 뒤 `setup.bat`을 다시 실행한다(`nvidia-smi`가 보이면 `[cuda]` 패키지를 설치함) |
| ASR이 매우 느림 | GPU가 아니라 CPU로 돌고 있다. 위와 같이 확인한다 |

## 5. Gemini API (B)

| 증상 | 원인 / 해결 |
|---|---|
| "GEMINI_API_KEY is not set (.env)" | `.env`에 `GEMINI_API_KEY=...`를 넣고 대시보드를 다시 실행한다 |
| 429 / quota / 한도 오류, 연결이 반복해서 끊김 | 무료 티어 한도다. 기다렸다가 다시 한다. **임의로 다른 모델로 바꾸지 않는다**(결과 비교가 깨진다). 언제 무엇이 실패했는지 기록한다 |
| 회사·학교 네트워크에서 연결 안 됨 | 방화벽이 WebSocket(HTTPS)을 막을 수 있다. 다른 네트워크로 확인한다 |
| B 자막이 실제 말과 다름 | 서버 자막의 특성이다. 판정은 녹음(음성) 기준으로 한다([TEST_GUIDE](TEST_GUIDE.md) §4.2) |
| B가 "지금은 움직임 기능이 꺼져 있다"고 말함 | 공통 시스템 프롬프트(Phase 2–4, 모션 없음 전제)의 문장 때문이다. 비교 조건을 같게 하려고 그대로 두었다 |

## 6. 모션

| 증상 | 원인 / 해결 |
|---|---|
| 연결했는데 모터가 자동으로 안 켜짐 | Motion 탭 빨간 글씨에 이유가 나온다. "허용 범위 밖"이면 머리를 잠자기 자세 가까이 두고 `Motor ON`을 누른다. "모터 상태 'enabled'"면 다른 앱(데몬 대시보드 등)을 끈다 |
| 창을 닫았는데 바로 안 닫힘 | 정상이다. 로봇을 잠자기 자세로 내리고 모터를 끈 뒤(약 8초) 닫힌다 |
| "Normal shutdown failed" / 모드 FAULT | 토크가 켜진 채 멈췄다. 주변을 확인하고 `Normal shutdown`을 다시 누르거나, 머리를 받친 채 F12를 누른다 |
| 중립에서 머리가 약간 숙여져 보임 | 정상이다. 모터가 머리 무게를 다 못 버텨 약 4° 처진다(R1 측정) |
| 모션이 잠깐 멈춤 | 엔진 로드·Cloud 연결 순간에는 생길 수 있다(1초 이내). 대화 중에는 측정되지 않았다 |
| 프로그램이 응답 없음, 모터는 켜져 있음 | PowerShell: `.venv\Scripts\python.exe -m reachy_s2s.apps.motion_preview --estop`(REST로 모터 끔, 머리를 받친다) |
| 몸통이 비스듬함(약 8°) | 토크가 꺼진 상태의 몸통 위치다. 모션은 몸통 기준으로 계산하고 몸통은 움직이지 않는다 |

## 7. 대시보드·실행 파일

| 증상 | 원인 / 해결 |
|---|---|
| `.bat`을 실행하면 "[FAIL] .venv not found" | `setup.bat`을 먼저 실행한다 |
| `setup.bat`: "Python 3.12 was not found" | Python 3.12를 설치한다(py launcher 포함). 3.13은 지원하지 않는다 |
| PySide6 `DLL load failed` | PowerShell에서 `.venv\Scripts\python.exe -m pip install --force-reinstall pyside6-essentials shiboken6` |
| 기록이 여러 폴더로 나뉨 | `New session`을 누른 경우다. 과제 사이에는 `Reset conversation`을 쓴다. 나뉜 폴더는 manifest에 같은 회차로 여러 줄 적으면 합쳐진다 |
| CSV의 한글이 깨짐 | 결과 정리 CSV는 UTF-8(BOM)이다. Excel "데이터 → 텍스트/CSV"에서 UTF-8로 연다 |
| 결과 폴더가 어디인지 모름 | Conversation 탭 "결과 폴더"와 `Open results folder` 버튼, 또는 `results\sessions\` (이름 = 시작 시각) |

해결되지 않으면 세션 폴더의 `app.log`와 `events.jsonl`을 함께 공유한다(API 키는 기록되지 않는다).
