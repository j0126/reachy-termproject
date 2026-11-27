# 효과음 (Expressive 모션)

확정 모션 파일에 적힌 효과음 큐의 음원이다(2026-11-14 추가, 사용자 결정 "로봇 스피커로"; failure·boredom은 모션 수정 2차 "둘 다 넣기").

| 파일 | 쓰는 모션 | 큐 시각 (모션 시작부터) | 소리 |
|---|---|---|---|
| `aha.wav` | aha_recommend expressive | 0.85 s (번뜩 고개를 드는 순간) | 빠르게 올라가는 종소리 두 번 (E6 → B6), 0.5 s |
| `success.wav` | success expressive | 0.3 s (튀어 오르는 순간) | 밝게 올라가는 장조 아르페지오 (C6–E6–G6–C7), 0.6 s |
| `failure.wav` | failure expressive | 0.2 s (고개를 떨구기 시작할 때) | 내려가는 단조 두 음 "어-어" (E5 → C#5가 C5로 처짐), 0.75 s, 조금 작게 |
| `boredom.wav` | boredom expressive | 4.9 s (처지기 시작할 때) | 한숨처럼 천천히 내려가는 낮은 소리 (D5 → F#4) + 약한 숨소리, 1.1 s, 아주 작게 |

- Neutral 4개에는 효과음이 없다(과제: Neutral은 "no antenna movement or sound").
- 소리 규칙(V/A 설계 목표에서): valence → 음의 방향·조(긍정 = 올라감·장조, 부정 = 내려감·단조), arousal → 빠르기·높이·크기(aha·success 빠르고 높게 -6 dBFS, failure 느리게 한 옥타브 아래 -8 dBFS, boredom 아주 느리고 낮게 -10 dBFS). 밝기(2 kHz 위 에너지, computed): aha −9 dB, success −2 dB, failure −54 dB, boredom −45 dB.
- 네 파일은 `reachy_motion/sound.py`(`synth_aha`, `synth_success`, `synth_failure`, `synth_boredom`)로 합성한 원본 소리다. 다른 사람의 음원을 쓰지 않았다.
  다시 만들기: `.venv\Scripts\python -c "from reachy_motion.sound import write_default_sounds; write_default_sounds('motions/sounds')"`
- 다른 소리로 바꾸려면 **같은 이름**으로 덮어쓴다(WAV, 16-bit 권장, 25 MB 이하). 로봇은 연결마다 처음 재생할 때 파일을 다시 올린다.
- 재생 위치는 GUI ① 모션 탭 `효과음`에서 고른다: 로봇 스피커 / 노트북 스피커 / 끄기, 시각 보정(ms).
