# Reachy Mini Speech-to-Speech — HRI Homework 3

Reachy Mini(Wireless) 로봇으로 **세 가지 음성 대화 시스템**을 직접 구현하고, 같은 대본·같은 조건에서 응답 지연·정확도·턴 교대(멈춤, 정정, 끼어들기)를 비교하는 프로젝트다.
로봇은 대화 상태에 따라 머리·안테나를 움직인다(상태 기반 모션).

| 시스템 | 한 줄 설명 | 구현 |
|---|---|---|
| **A0 — Basic Cascaded** | 로컬 VAD → ASR(faster-whisper) → LLM(Gemma 4 E4B) → TTS(Supertonic 3). **Phase 2에서 처음 만든 기본 시스템 그대로**(비교용으로 일부러 낮춘 것이 아님) | Phase 2 |
| **A1 — Enhanced Cascaded** | A0와 같은 모듈 + 대화 제어 개선(끼어들기, 끊긴 응답 이력, 멈춤·정정 시 이어 붙이기) | Phase 4 |
| **B — Cloud S2S** | Gemini Live(`gemini-3.8-live`)가 음성을 받아 바로 음성으로 답하는 클라우드 통합형 | Phase 3 |

- **A0 vs A1**: 같은 ASR·LLM·TTS에서 대화 제어 개선의 효과를 본다.
- **A1 vs B**: 로컬 모듈형과 클라우드 통합형의 비교다. 모델·실행 환경도 다르므로 아키텍처만의 차이로 해석하지 않는다. 자세한 설계는 [docs/EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md)에 있다.

**현재 결과 요약**(Phase 5, 조건별 1회, 사람 판정 전 — [docs/PHASE5_REPORT.md](docs/PHASE5_REPORT.md))
- 응답 지연 중앙값(발화 끝 → 첫 오디오 전송): B 1.08 s, A0 2.62 s, A1 2.63 s
- 끼어들기·멈춤 후 이어 붙이기: A1만 성공(A0는 설계상 불가). B는 끼어들기 성공, 멈춤 뒤 뒷말은 놓침
- 소음(핑크 노이즈): 지연·용어 인식 영향은 작고, 짧은 발화 오인식이 늘었다.

## 처음 읽는 순서 (팀원)

1. **이 README** — 개요와 문서 지도
2. [**QUICK_START.md**](QUICK_START.md) — 설치, 로봇 연결, A0/A1/B 대화, 모션 데모, 안전한 종료 (.bat 파일만 사용)
3. [**docs/EVALUATION_PROTOCOL.md**](docs/EVALUATION_PROTOCOL.md) — 무엇을 왜 비교하나: 시스템 정의, 기준선 선정 이유, 설정 파일, 조건, 지표 정의
4. [**TEST_GUIDE.md**](TEST_GUIDE.md) — 실험 방법: 대본, 반복·순서, 성공·실패 판정, 결과 저장·표·그래프 만들기
5. [**docs/PHASE5_REPORT.md**](docs/PHASE5_REPORT.md) — 결과, 분석, 한계, 미측정 항목 (보고서·발표 자료의 근거)
6. [**TROUBLESHOOTING.md**](TROUBLESHOOTING.md) — 연결·오디오·LLM 서버·GPU·API·모션 문제 해결

## 실행 환경

| 환경 | 필요한 것 | 할 수 있는 것 |
|---|---|---|
| **Full Evaluation** | Windows 11, Python 3.12, NVIDIA GPU(8 GB 권장), llama.cpp + 로컬 모델, 로봇, 인터넷·API 키(B) | A0 / A1 / B 전부, Phase 5 실험 |
| **Cloud B Demo** | Windows 11, Python 3.12, 인터넷, Gemini API 키, 로봇 | B 대화, 모션 데모 |

개발 PC(Windows 11, RTX 4060 Ti 8 GB)에서만 검증했다. **다른 PC는 아직 검증하지 않았다**(검증 상태: [docs/HANDOVER_PLAN.md](docs/HANDOVER_PLAN.md)).

## 실행 파일 (프로젝트 폴더에서 더블클릭)

| 파일 | 하는 일 |
|---|---|
| `setup.bat` | 처음 한 번: 가상환경·패키지 설치(GPU가 있으면 CUDA 라이브러리 포함), `.env` 만들기, 점검 |
| `check_setup.bat` | PC·모델·LLM 서버·API 키·로봇 상태 점검. 무엇이 가능한지와 해결 방법을 출력한다(모터 명령 없음, 키는 출력 안 함) |
| `start_llm_server.bat` | A0/A1용 로컬 LLM 서버(Full Evaluation PC). 켜 둔 채로 쓴다 |
| `run_dashboard.bat` | 대화·데모용 대시보드. Motion ON, 연결하면 안전할 때 모터 자동 시작, 창을 닫으면 잠자기 자세로 정상 종료 |
| `run_phase5.bat [회차]` | Phase 5 측정 한 회차: A0 → A1 → B 대시보드가 차례로 열림(모터 OFF, 회차마다 순서 회전) |
| `make_phase5_report.bat` | `results/phase5/manifest.csv`의 세션으로 표·그래프(`report.html`) 만들기 |

## 문서 지도

| 문서 | 내용 |
|---|---|
| [QUICK_START.md](QUICK_START.md) · [TEST_GUIDE.md](TEST_GUIDE.md) · [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | 팀원용: 실행, 실험, 문제 해결 |
| [docs/EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md) | 비교 설계·지표 정의 |
| [docs/PHASE5_REPORT.md](docs/PHASE5_REPORT.md) | Phase 5 결과·분석·한계 |
| [docs/PHASE2_REPORT.md](docs/PHASE2_REPORT.md), [docs/PHASE2_MODEL_SELECTION.md](docs/PHASE2_MODEL_SELECTION.md) | A0(Basic Cascaded) 구현, 로컬 모델로 바꾼 이유 |
| [docs/PHASE3_PLAN.md](docs/PHASE3_PLAN.md), [docs/PHASE3_REPORT.md](docs/PHASE3_REPORT.md) | B(Gemini Live) 설계·결과 |
| [docs/PHASE4_PLAN.md](docs/PHASE4_PLAN.md), [docs/PHASE4_REPORT.md](docs/PHASE4_REPORT.md) | A1 턴 교대(멈춤·정정·끼어들기) 설계·결과, 대본 S1–S6 |
| [docs/PHASE4_5_DESIGN.md](docs/PHASE4_5_DESIGN.md), [docs/PHASE4_5_PLAN.md](docs/PHASE4_5_PLAN.md), [docs/PHASE4_5_REPORT.md](docs/PHASE4_5_REPORT.md) | 상태 기반 로봇 모션 설계·안전 장치·실기기 검증(R0–R3), TTS 스레드 개선 |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 코드 구조, 스레드, 시간 측정 방법 |
| [docs/IMPLEMENTATION_NOTES.md](docs/IMPLEMENTATION_NOTES.md), [docs/HF_S2S_NOTES.md](docs/HF_S2S_NOTES.md) | SDK·실기기 조사 기록, 참고 구현 분석 |
| [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) | 명령줄 실행, 측정 도구, 결과 파일 형식, 테스트, 폴더 구조 |
| [docs/HANDOVER_PLAN.md](docs/HANDOVER_PLAN.md) | 인수인계 완료 기준과 검증 상태 |

## 진행 상태 (2026-10-17)

| Phase | 내용 | 상태 |
|---|---|---|
| 0–1 | 환경·SDK 조사, 로봇 오디오 입출력, Mel, 이벤트 로그, GUI | ✅ 실기기 검증 |
| 2 | A0 Basic Cascaded | ✅ 실기기 — [PHASE2_REPORT](docs/PHASE2_REPORT.md) |
| 3 | B Cloud S2S (Gemini Live) | ✅ 실기기 — [PHASE3_REPORT](docs/PHASE3_REPORT.md) |
| 4 | A1 턴 교대(멈춤·정정·끼어들기) | ✅ 실기기: 끼어들기 A0 0/3 → A1 3/3(B 3/3), 정정 1/3 → 3/3(B 3/3) — [PHASE4_REPORT](docs/PHASE4_REPORT.md) |
| 4.5 | 상태 기반 로봇 모션 | ✅ 실기기 R1–R3 통과, 데모 기본 Motion ON — [PHASE4_5_REPORT](docs/PHASE4_5_REPORT.md) |
| 5 | A0 / A1 / B 비교 실험 | 🟡 조용함·소음 각 1회 측정·자동 채점 완료, **사람 판정·반복 측정 미완료** — [PHASE5_REPORT](docs/PHASE5_REPORT.md) |
| 인수인계 | 팀원 PC 실행·실험·분석·발표 지원 | 🟡 문서·실행 파일·점검 도구 완료, 다른 PC 검증 전 — [HANDOVER_PLAN](docs/HANDOVER_PLAN.md) |

## 폴더 구조

```
reachy_hw3/
  *.bat                  실행 파일 (위 표)
  README.md QUICK_START.md TEST_GUIDE.md TROUBLESHOOTING.md
  config/                설정: default.toml(기본), local_e4b_supertonic.toml(A0), a1.toml(A1 추가분),
                         tts_threads2.toml(개선형 TTS), demo.toml(데모 Motion ON) ...
  reachy_s2s/            코드 (audio, robot, conversation, engines, motion, gui, evaluation, apps)
  docs/                  설계·보고서
  results/sessions/      세션마다 turns.csv, events.jsonl, session.json, audio/*.wav
  results/phase5/        Phase 5 세션 목록(manifest.csv), 표·그래프·판정표
  scripts/ tests/        보조 스크립트, 하드웨어 없는 테스트
```

## 알려진 한계 (요약)

- Phase 5는 조건별 1회·화자 1명이다. 사람 판정(녹음 청취)이 아직 없다.
- 지연은 로봇으로 오디오를 보낸 시점까지다. 실제 스피커 출력 지연(약 0.4 s)은 추정치다.
- 로봇이 말하는 중 사용자 말의 앞부분이 로봇 칩의 에코 억제로 눌려 첫 단어를 잘못 알아들을 수 있다.
- 공통 시스템 프롬프트에 "움직임 기능이 꺼져 있다"는 문장이 남아 있다(Phase 2–4 전제, 비교 조건 유지).
- 자세한 한계는 [docs/PHASE5_REPORT.md](docs/PHASE5_REPORT.md) §7, [docs/PHASE4_5_REPORT.md](docs/PHASE4_5_REPORT.md)에 있다.

## 개발자

- 명령줄 사용법과 측정 도구: [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md)
- 테스트(로봇·GPU·API 없이): `.venv\Scripts\python.exe -m unittest discover -s tests -t .`
- API 키는 `.env`에만 둔다(git 제외). 설정은 `config/*.toml`을 겹쳐 쓴다(뒤에 오는 파일이 우선).
