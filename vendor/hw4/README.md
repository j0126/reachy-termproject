# Reachy Mini 표현적 모션 — HRI Homework 4 소스 코드

DGIST HRI Homework 4 제출용 소스 코드입니다. Reachy Mini(Wireless)가 **의도·상태·반응**을 전달하는 표현적 모션 4종을
Expressive / Neutral 두 버전으로 설계했고, 이를 실행·안전 검사·추종 측정·분석하는 도구 **Motion Lab GUI**를 함께 만들었습니다.

| 범주 | 모션 | 전달하는 것 | 파일 (설정값이 그대로 들어 있음) |
|---|---|---|---|
| (A) 의도 | `aha_recommend` | "좋은 생각이 났어요, 이걸 추천해요" | `motions/aha_recommend.expressive.yaml`, `.neutral.yaml` |
| (B) 상태·감정 | `boredom` | "지루하다, 아무도 안 온다" | `motions/boredom.*.yaml` |
| (C) 반응 | `success` / `failure` | "됐다!" / "아, 안 됐다…" | `motions/success.*.yaml`, `motions/failure.*.yaml` |

이 묶음에는 소스 코드와 이 안내서(설치·실행·사용 방법)만 넣었습니다. 설계 근거, 실험 결과와 분석은 보고서와 시연 영상에 있습니다.

## 1. 필요한 것

| 운영체제 | 조건 | 확인한 정도 |
|---|---|---|
| **macOS** | **Apple Silicon(M1 이후) + macOS 14 이상** | 설치에 필요한 패키지 100개가 모두 이 환경용 설치 파일로 배포되어 있음을 확인했습니다(2026-11-16). 실제 Mac에서 실행해 보지는 못했습니다 |
| Windows | 10/11, 64-bit | 개발·시험한 환경입니다 |
| Linux | Ubuntu 22.04 이상, x86_64 | 스크립트만 준비했고 시험하지 않았습니다 |

- 인터넷: 처음 설치할 때 한 번 필요합니다(설치 뒤 약 1 GB).
- Python: 따로 설치하지 않아도 됩니다. 설치 도구 uv가 Python 3.12를 받습니다.
- 로봇: 없어도 됩니다. 기본 실행 대상은 이 컴퓨터 안의 **MuJoCo 시뮬레이션**입니다. 실제 로봇은 Reachy Mini Wireless(daemon 1.11.0).
- Intel Mac은 설치되지 않습니다. 로봇 SDK(`reachy-mini` 1.11.0)가 요구하는 onnxruntime 1.27.0이 Intel Mac용으로 배포되지 않습니다.

## 2. 설치와 실행 — macOS

1. **터미널**을 엽니다(응용 프로그램 → 유틸리티 → 터미널).
2. **uv 설치** (한 번만): `curl -LsSf https://astral.sh/uv/install.sh | sh` → 터미널 창을 닫고 새로 엽니다.
3. 압축을 푼 폴더로 이동합니다: `cd ` (끝에 공백)를 입력하고 폴더를 터미널 창에 끌어다 놓은 뒤 Enter.
4. **설치** (한 번만, 1–3분): `bash setup_mac.command` → 마지막 줄에 `pins OK`가 나오면 끝입니다.
5. **실행**: `bash run_gui.command` → Motion Lab 창이 뜹니다.

- Finder에서 `setup_mac.command`, `run_gui.command`를 더블클릭해도 됩니다. "확인되지 않은 개발자"로 열리지 않으면 위의 터미널
  방법을 쓰거나, 그 폴더에서 `xattr -dr com.apple.quarantine .`을 한 번 실행한 뒤 더블클릭합니다.
- MuJoCo 시뮬레이션 창은 macOS에서 `mjpython`으로 열립니다(설치에 포함, 프로그램이 알아서 처리).

## 3. 설치와 실행 — Windows

1. **uv 설치** (한 번만, PowerShell): `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"` → 새 창.
2. **설치**: `setup_windows.bat` 더블클릭 → 마지막 줄 `pins OK`.
3. **실행**: `run_gui.bat` 더블클릭.

## 4. 설치와 실행 — Linux (Ubuntu 22.04 이상)

1. 필요한 시스템 패키지(로봇 SDK가 쓰는 PyGObject를 설치 때 컴파일함):
   `sudo apt install -y build-essential pkg-config libffi-dev libcairo2-dev libgirepository1.0-dev libxcb-cursor0 fonts-nanum pulseaudio-utils`
2. **uv 설치**: `curl -LsSf https://astral.sh/uv/install.sh | sh` → 새 터미널.
3. **설치**: 압축을 푼 폴더에서 `bash setup_linux.sh` → `pins OK` (끝에 앱 메뉴 바로가기를 만들지 묻습니다).
4. **실행**: `bash run_gui.sh` 또는 앱 메뉴의 `Reachy HW4 Motion Lab`.

### 스크립트 없이 직접 설치하려면 (모든 OS, 압축을 푼 폴더에서)

```
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -c constraints.txt -e ".[sim,analysis,gui]" av            (macOS·Linux)
uv pip install --python .venv\Scripts\python.exe -c constraints.txt -e ".[sim,analysis,gui]" av    (Windows)
.venv/bin/python -m reachy_motion env        → pins OK: reachy-mini==1.11.0, ...   (Windows: .venv\Scripts\python)
.venv/bin/python -m reachy_motion gui        → Motion Lab
```

- `constraints.txt`가 모든 패키지 버전을 고정합니다. `reachy-mini`는 로봇 daemon과 같은 **1.11.0**이어야 합니다.
- 아래에서 `python`은 macOS·Linux의 `.venv/bin/python`, Windows의 `.venv\Scripts\python`을 뜻합니다.
- 반드시 이 폴더에서 실행합니다. 실행 기록이 이 폴더의 `results/`에 저장됩니다.

## 5. 사용 방법 (Motion Lab)

창에는 세 탭(① 모션 · ② 실험 · ③ 결과)과, 탭 위쪽에 **'실제 로봇' 상자**(IP · 연결 · Wake-up · 토크 OFF · 연결 끊기)가 있습니다.
버튼이 회색이면 창 위쪽 상태 줄에 이유가 나오고, 버튼·표 머리글에 마우스를 올리면 설명이 나옵니다.
오른쪽 위 `? 사용법`은 이 안내서를 창 안에서 엽니다.

### 5.1 로봇 없이 모션 보기 (① 모션 탭)

1. 모션과 변형(Expressive / Neutral)을 고릅니다. 설명(의도·애니메이션 원리·valence/arousal 목표), 머리·안테나 위치·속도 그래프,
   Expressive vs Neutral 비교, 설정값 표(키프레임별 이동 시간·정지·변화량·최고 속도·보간)가 나옵니다.
2. `▶ MuJoCo 재생`: 시뮬레이션 창에서 재생합니다(처음에는 약 10초 뒤 창이 뜹니다).
3. `▶ Expressive → Neutral`: 같은 모션의 두 버전을 이어서 재생합니다.
4. `▶ E/N 나란히 영상`: 두 버전을 한 화면에 나란히 렌더링해 창 안에서 재생합니다(처음 10–15초).
5. `효과음`: Expressive의 효과음을 낼 곳(로봇 스피커 · 노트북 스피커 · 끄기). 노트북 스피커로 두면 MuJoCo 재생에서도 들립니다.

### 5.2 실험 (② 실험 탭)

백엔드를 **MuJoCo**(기본) 또는 **실제 로봇**으로 고르고 실험을 고른 뒤 `▶ 실행`합니다. 결과는 자동으로
`results/lab/<시각>_<sim|robot>_<R1|R2|R3>/`에 저장됩니다(명령 `commands.csv`, 측정 `present_*.csv`, 조건 `manifest.json`).

| 실험 | 하는 일 |
|---|---|
| R1 동작 확인 | 한 축(pitch·yaw·roll·z·안테나)을 작게 움직였다 돌아옵니다. 진폭은 목록에서 고르거나 직접 입력(회전 0.1–15°, z 0.1–8 mm, 안테나 0.1–45°) |
| R2 표현 모션 | 최종 모션을 Expressive / Neutral로 실행하고 명령과 측정을 기록합니다 |
| R3 한계 시험 | pitch 20°(minjerk)에서 이동 시간 D를 1.0 → 0.6 → 0.4 → 0.3 → 0.25 → 0.2 s로 한 단계씩 줄입니다. 앞 단계가 기준(진폭비 < 0.8, 지연을 뺀 RMSE > 진폭의 15 %, 오버슈트 > 15 %, 지연이 D 1.0보다 100 ms 넘게 늘어남, 완료 못 함·정지·오류)을 넘으면 다음 단계는 거부되고, 그 단계가 추종이 무너진 지점입니다. 선택: D 1.0에서 진폭 A 10 → 20 → 30° |

### 5.3 결과 (③ 결과 탭)

1. 저장된 세션을 불러오고, 볼 시행을 체크합니다.
2. 추종 지표: RMSE·Peak 오차(지연 포함), 지연(Lag), 진폭비(측정 크기 ÷ 명령 크기), R3의 지연을 뺀 RMSE, 정지 중 떨림 RMS.
3. 그림: 명령(점선) vs 측정(실선), R1 진폭비, R3 단계 비교, 한계 지도(요구 속도·가속 계산값 위에 실행 결과 표시), Expressive vs Neutral.
4. `보고서 자료 내보내기` → `results/lab/exports/<시각>_report/` (CSV · PNG · 요약).
- 모든 숫자에는 출처(`data_kind`)가 붙습니다: `measured` 실제 로봇, `simulated` MuJoCo, `computed` 계산. 다른 출처가 섞이면 빨간 경고가 뜹니다.

### 5.4 실제 로봇에서

1. 로봇과 컴퓨터를 같은 Wi-Fi에 둡니다. 대시보드 앱·다른 컴퓨터 등으로 **동시에** 조작하지 않습니다(SDK에 잠금이 없습니다).
2. '실제 로봇' 상자: IP(기본값 `192.168.0.15`는 개발에 쓴 로봇 주소이므로 로봇에 맞게 고칩니다), 포트 `8000` → `연결`.
   - 잠든 로봇(토크 꺼짐: 전원을 켠 직후, 앱으로 재운 뒤): 지금 자세 그대로 토크가 켜집니다. 머리가 관절 한계를 조금(10° 이내)
     넘어 쉬고 있어도 됩니다 - 그때는 토크가 켜질 때 그 관절이 한계 쪽으로 조금 움직일 수 있으니 머리를 받친 채 누릅니다.
     이어서 `Wake-up`을 누르면 기본 자세(INIT)로 갑니다.
   - Reachy Mini Control 앱으로 이미 깨운 로봇(토크 켜짐): 토크를 그대로 둔 채 연결되고 바로 준비됩니다(앱은 닫습니다).
   - 연결이 거부되면 로봇 상자에 이유와 할 일이 나옵니다.
3. 상태가 `INIT 대기 · 시행 가능 · 토크 ON · 연결 열림`이면 준비가 끝난 것입니다.
4. ① 모션 탭에서 모션·변형을 고르고 `▶ 실제 로봇 재생`(클릭 1번 = 1번 실행: INIT → 모션 → INIT). ② 실험 탭의 R1·R2·R3도 같은 방식입니다.
   Expressive는 기본으로 로봇 스피커에서 효과음이 납니다.
5. 끝낼 때: 머리를 손으로 받치고 `토크 OFF`(머리가 내려갑니다), 또는 `연결 끊기`(토크를 켠 채 둡니다).

**안전**: 떨림·열·이상한 소리가 보이면 `■ 정지` → 머리를 받치고 `토크 OFF` → 그래도 안 되면 전원. 새 조건은 MuJoCo에서 먼저 해 봅니다.
실행 전에 계산 안전 검사(관절 여유 ≥ 10°, 모터 속도 비 ≤ 0.5, 자세 범위)를 통과한 시행만 실행되고, 로봇에서는 한 번에 하나씩만,
R3는 한 단계씩만 되며 D < 0.2 s는 막힙니다. 이 검사는 초기 정책이고 기구적 안전을 보장하지는 않습니다.

## 6. 명령줄 (선택)

```
python -m reachy_motion check success --variant both            계산 안전 검사와 Expressive/Neutral 비교 조건
python -m reachy_motion play success                            Mock(가상 모델)로 재생: 로봇·시뮬레이션 없이 실행 기록만
python -m reachy_motion sim-play success --variant expressive   MuJoCo 창에서 재생 (Enter 다시, v 다른 변형, q 끝)
python -m reachy_motion video --dir motions                     E|N 나란히 영상 → results/phase3_sim/videos/
python -m reachy_motion phase1                                  보간 4종 그림·표 (과제 1) → results/phase1/
```

## 7. 모션 파일 읽는 법

- `motions/<모션>.<expressive|neutral>.yaml` (도·mm 단위). 키프레임마다 `duration_s`(이동 시간), `hold_s`(정지), `easing`
  (`linear`·`minjerk`·`ease_in_out`·`cartoon`), `pose`(머리 roll·pitch·yaw, x·y·z), 안테나 `[오른쪽, 왼쪽]` 각도.
- `expected`는 valence·arousal **설계 목표**(측정값 아님), `principles`는 적용한 애니메이션 원리, `sound`는 효과음 파일과 시각입니다.
- 부호: +pitch = 고개 숙임, +yaw = 로봇 왼쪽.
- Neutral: 같은 시작·끝 자세, 일정한 속도(linear), 안테나 움직임·소리 없음(`check <모션> --variant both`로 확인).
- 효과음(`motions/sounds/`)은 직접 합성했습니다. 규칙: [motions/sounds/README.md](motions/sounds/README.md).
- `motions/candidates/`는 설계 과정의 이전 버전과 boredom 후보입니다(최종본 아님).

## 8. 폴더 구조

```
reachy_hw4/
├── README.md                            이 안내서
├── setup_mac.command    run_gui.command       macOS 설치 · 실행
├── setup_windows.bat    run_gui.bat           Windows
├── setup_linux.sh       run_gui.sh            Linux
├── pyproject.toml  constraints.txt            패키지 정의 · 전체 버전 고정
├── motions/          최종 모션 8개, sounds/ 효과음, candidates/ 이전 버전·후보
├── reachy_motion/    소스 코드
│   ├── gui/              Motion Lab (PySide6): 창, 실험·결과 탭, 실제 로봇 상자, E/N 영상
│   ├── library/          모션 스펙, YAML 읽기·쓰기, Expressive/Neutral 비교 조건
│   ├── planner.py  timeline.py   SDK 보간 함수로 50 Hz 명령 타임라인 만들기
│   ├── safety/           계산 안전 검사(관절 여유·속도 비·자세 범위), 기동 절차, 실행 승인
│   ├── executor/         실행기: 명령·측정 기록, 정지(감속 정지, 측정 끊김, 오류)
│   ├── backends/         Mock, 로컬 MuJoCo daemon, 실제 로봇(SDK 1.11.0) 연결
│   ├── analysis/         보간 4종·IK·모터 요구량 계산, 추종 지표, 그림
│   ├── sound.py          효과음 합성·재생
│   └── cli.py …          명령줄
└── tests/            자동 테스트 (네트워크·로봇에 접속하지 않음)
```

## 9. 테스트

```
python -m unittest discover -s tests -t .
```

737개, 약 2분, 네트워크·로봇에 접속하지 않습니다. 마지막 줄이 `OK (skipped=5)`면 정상입니다(5개는 이 묶음에 없는 개발 도구용).

## 10. 참고

- 코드는 설치된 `reachy-mini` 1.11.0 소스에서 확인한 동작에 맞췄습니다(주석에 `파일:줄` 근거).
- MuJoCo 시뮬레이션은 모터의 속도·토크 포화를 모델링하지 않아 실제 로봇보다 훨씬 잘 따라갑니다. 시뮬레이션 값은 실제 로봇 예측이 아닙니다.
