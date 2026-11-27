"""Korean labels of the Motion Lab GUI and its Korean fonts (user-facing text only).

Technical terms stay as in the HW4 documents and the assignment: RMSE, Lag, Static Bias, Peak Error, Overshoot,
roll / pitch / yaw, Stewart, Expressive / Neutral, and the data_kind values (computed / mock / simulated / measured).
Messages that come from the motion engine itself (safety-checker violations, SDK errors) are shown as they are.
"""

from __future__ import annotations

import sys
from typing import Any

#: Korean font families, first found wins (Windows: Malgun Gothic, macOS: Apple SD Gothic Neo, Linux: fonts-nanum /
#: fonts-noto-cjk). DejaVu Sans stays as the glyph fallback.
FONT_CANDIDATES = ("Malgun Gothic", "Apple SD Gothic Neo", "NanumGothic", "Noto Sans KR", "AppleGothic",
                   "Noto Sans CJK KR")
#: The GUI font size: 9 pt at Windows / Linux 96 dpi is 12 px; Qt on macOS counts 1 pt = 1 px, so 12 pt there.
UI_POINT_SIZE = 12 if sys.platform == "darwin" else 9

POLICY_METRICS = {
    "min_stewart_margin_deg": "최소 Stewart 관절 여유 (deg)", "min_stewart_margin_joint": "최소 여유 관절",
    "min_stewart_margin_t": "최소 여유 시각 (s)", "peak_stewart_speed_dps": "Stewart 최고 속도 (deg/s)",
    "peak_stewart_speed_ratio": "Stewart 최고 속도 / 무부하 속도", "peak_body_speed_dps": "몸통 최고 속도 (deg/s)",
    "peak_body_speed_ratio": "몸통 최고 속도 / 무부하 속도", "peak_antenna_speed_dps": "안테나 최고 속도 (deg/s)",
    "peak_antenna_speed_ratio": "안테나 최고 속도 / 무부하 속도", "max_abs_roll_deg": "최대 |roll| (deg)",
    "max_abs_pitch_deg": "최대 |pitch| (deg)", "max_abs_rel_yaw_deg": "최대 머리–몸통 상대 yaw (deg)",
    "body_yaw_range_deg": "몸통 yaw 범위 (deg)", "x_range_mm": "x 범위 (mm)", "y_range_mm": "y 범위 (mm)",
    "z_range_mm": "z 범위 (mm)", "antenna_range_deg": "안테나 범위 (deg)",
    "max_auto_body_shift_deg": "자동 body yaw 최대 이동 (deg)", "max_psi_step_deg": "표본 간 최대 ψ 변화 (deg)",
    "max_body_step_deg": "표본 간 최대 몸통 관절 변화 (deg)", "max_stream_step_deg": "tick 간 최대 관절 변화 (deg)",
    "peak_joint_accel_dps2": "최고 관절 가속도 (deg/s²)", "duration_s": "길이 (s)", "n_dense": "조밀 표본 수",
    "n_stream": "50 Hz 표본 수", "ik_nan_samples": "IK 실패 표본 수",
}
PARAMS = {
    "track": "트랙", "index": "#", "label": "이름", "t_depart_s": "출발 s", "t_arrive_s": "도착 s",
    "t_release_s": "해제 s", "duration_s": "이동 s", "hold_s": "정지 s", "easing": "보간",
    "peak_ratio": "최고 속도 계수", "right_deg": "오른쪽 deg", "left_deg": "왼쪽 deg",
    "delta_right_deg": "Δ오른쪽 deg", "delta_left_deg": "Δ왼쪽 deg", "rot_deg": "회전량 deg", "trans_mm": "이동량 mm",
    "peak_rot_dps": "최고 회전 속도 deg/s", "peak_rot_path_dps": "최고 회전 속도(경로) deg/s",
    "peak_trans_mms": "최고 이동 속도 mm/s", "peak_antenna_dps": "최고 안테나 속도 deg/s",
    "peak_body_dps": "최고 몸통 속도 deg/s",
}
COMPARE = {
    "data_kind": "데이터 종류", "basis": "기준", "tolerance": "허용 오차", "same_start": "같은 시작",
    "start_difference": "시작 차이", "same_end": "같은 끝", "end_difference": "끝 차이", "motion": "모션",
    "spec_digest": "spec 다이제스트", "draft": "초안", "scale": "배율", "start_state": "시작 상태",
    "end_state": "끝 상태", "duration_s": "길이 (s)", "track_durations_s": "트랙 길이 (s)", "key_counts": "키 개수",
    "antennas_used": "안테나 사용", "easings": "보간", "max_head_rotation_deg": "최대 머리 회전 (deg)",
    "max_head_translation_mm": "최대 머리 이동 (mm)", "max_antenna_change_deg": "최대 안테나 변화 (deg)",
    "max_body_yaw_change_deg": "최대 몸통 yaw 변화 (deg)", "speeds": "속도", "sound": "소리",
    "principles": "애니메이션 원리",
}
COLUMNS = {
    "run": "시행", "stage": "단계", "condition": "조건", "channel": "채널", "role": "역할", "unit": "단위",
    "vary": "바꾼 파라미터", "amplitude_deg": "A (deg)", "duration_s": "D (s)", "rep": "반복", "motion": "모션",
    "variant": "변형", "unsettled_steps": "미정착 단계 수", "lag_at_bound": "지연 탐색 경계", "legacy_time_base": "이전 시간 기준",
    "kind": "종류", "speed_ratio": "Stewart 최고 속도 / 무부하 속도", "accel_dps2": "최고 관절 가속도 (deg/s²)",
    "robot_scope": "실제 로봇 범위", "runnable": "초기 정책 통과", "data_kind": "data_kind",
}
MOTIONS = {"aha_recommend": "A. Aha! 추천 (aha_recommend)", "boredom": "B. Boredom 지루함 (boredom)",
           "success": "C. Success 성공 (success)", "failure": "C. Failure 실패 (failure)"}
AXES = {"yaw": "yaw (좌우 회전)", "pitch": "pitch (끄덕임)", "roll": "roll (갸웃)", "z": "z (위아래, mm)",
        "antenna_right": "오른쪽 안테나", "antenna_left": "왼쪽 안테나"}
PLAYBACK = {"queued": "대기열", "started": "재생 중", "finished": "완료", "cancelled": "취소", "preempted": "선점됨",
            "rejected": "거부", "failed": "실패"}
STATUS = {"pending": "대기", "queued": "대기열", "running": "실행 중", "done": "완료", "anomaly": "이상",
          "stopped": "정지", "refused": "거부", "error": "오류", "cancelled": "실행 취소"}
POLICY_DISCLAIMER = ("초기 정책(Phase 2)은 기구적 안전을 보장하지 않는다. URDF 한계는 추정한 원시 각도 대응으로 계산했고, "
                     "충돌 모델이 없으며, 속도는 5 V 무부하 데이터시트 값(상한)이고, 가속도·토크 모델이 없다. "
                     "통과 = '정책 안'이지 '실제 로봇에서 안전'이 아니다.")


#: Glossary for first-time users: term -> (Korean name with the assignment's English term, explanation).
GLOSSARY: dict[str, tuple[str, str]] = {
    "RMSE": ("RMSE (Root Mean Square Error, 평균 제곱근 오차, 지연 포함)",
             "움직이는 구간에서 명령값과 측정값의 차이를 제곱해 평균낸 뒤 제곱근을 취한 값. 작을수록 명령을 잘 따라간 것이다. "
             "정적 편향(Static Bias)은 빼지만 지연은 그대로 들어 있다(실제 로봇 약 190 ms라 빠른 동작일수록 커진다). 동작 모양의 "
             "오차는 '지연 보정 RMSE'로 본다. 단위는 deg 또는 mm."),
    "Lag-compensated RMSE": ("지연 보정 RMSE (Lag-compensated RMSE)",
                             "명령을 Lag만큼 늦춘 뒤 계산한 RMSE: 지연을 뺀 '동작 모양'의 오차. R3 그림·표에서는 명령 진폭 대비 %로 "
                             "보여 주며, 15 %를 넘으면 실제 로봇 R3 사다리를 멈춘다."),
    "Hold Ripple": ("정지 중 떨림 RMS (Hold Ripple)",
                    "명령이 멈춰 있는 구간(움직임이 끝나고 0.5 s 뒤부터)에서 측정값을 직선 추세에서 뺀 나머지의 RMS. 머리가 떨리면 "
                    "커지고, 천천히 흘러내리는 것은 세지 않는다. 엔코더 해상도·센서 잡음 수준이면 떨림이 없는 것이다. 멈춤 기준은 "
                    "아니고 표시만 한다."),
    "Peak Error": ("Peak Error (최대 오차, 지연 포함)", "기록 전체에서 명령값과 측정값의 차이가 가장 컸던 순간의 크기. 지연과 "
                   "편향이 모두 포함된다."),
    "Lag": ("Lag (지연)", "명령을 몇 ms 늦춰야 실제 움직임과 가장 잘 맞는지 찾은 값. PC에서 명령을 보낸 시각부터 측정이 도착한 "
            "시각까지의 전체 지연이다. r2(설명력)가 0.9보다 작으면 순수 지연으로 설명되지 않으므로 참고만 한다."),
    "Amplitude Ratio": ("Amplitude Ratio (진폭비)", "실제 움직임의 크기(최댓값-최솟값) ÷ 명령의 크기. 1이면 명령만큼 움직였고, "
                        "1보다 작으면 덜 움직였다."),
    "Overshoot": ("Overshoot (오버슈트)", "목표 자세에 도착한 뒤 목표를 지나쳐 넘어간 최대량. 정지 구간이 있는 단계에서만 계산한다."),
    "Settling Time": ("Settling Time (정착 시간)", "명령 이동이 끝난 뒤 실제 값이 최종값 근처(변화량의 5% 또는 0.2°/0.2 mm)에 "
                      "들어와 계속 머물기 시작할 때까지 걸린 시간."),
    "Static Bias": ("Static Bias (정적 편향)", "멈춰 있는 구간에서 실제 값이 명령값보다 평균적으로 얼마나 어긋나 있는지(예: "
                    "중력 때문에 머리가 약간 처짐). 움직임이 끝난 직후 0.5 s는 빼고 계산한다."),
    "Valence": ("Valence (정서가)", "모션이 주려는 감정이 긍정적인지(+1) 부정적인지(-1)를 나타내는 설계 목표값. 측정값이 아니다."),
    "Arousal": ("Arousal (각성도)", "모션이 주려는 감정의 에너지 수준. 차분함(-1)부터 흥분(+1)까지의 설계 목표값. 측정값이 아니다."),
    "IK": ("IK (Inverse Kinematics, 역기구학)", "원하는 머리 자세(위치·각도)를 만들기 위해 Stewart 관절 6개와 몸통을 각각 "
           "얼마나 돌려야 하는지 계산하는 과정. 해가 없으면 그 자세는 만들 수 없다."),
    "R0": ("상태 점검 (R0)", "실제 로봇 상태만 읽는 단계(GET). 모터를 움직이지 않는다. 로봇 상자의 '연결'과 '새로고침'이 "
           "자동으로 한다. 깨우기는 로봇 상자의 'Wake-up'."),
    "R1": ("동작 확인 (R1)", "깨운 로봇에서 한 축만 작게(회전 2·5·10°, z 1·3·5 mm, 안테나 5·15·30°, 범위 안에서 직접 "
           "입력 가능) 움직여 방향·부호·편향·지연과 작은 명령을 얼마나 따라가는지(진폭비)를 확인한다. 실제 로봇 첫 시행."),
    "R2": ("표현 모션 (R2)", "확정된 모션 8개(4개 모션 × Expressive/Neutral)를 그대로 재생하고 명령값과 측정값을 기록한다. "
           "같은 모션의 Expressive와 Neutral을 짝으로 실행해 비교한다."),
    "R3": ("한계 시험 (R3)", "pitch 끄덕임의 시간 D를 줄이거나 진폭 A를 키우며(한 번에 하나만) 추종 성능이 어떻게 바뀌는지 본다. "
           "한 단계씩 실행하고 첫 이상에서 멈춘다. 실제 로봇 D 사다리는 D ≥ 0.2 s, 진폭 사다리는 D ≥ 1 s에서 작은 A부터."),
    "Mock": ("Mock (모의 백엔드)", "daemon의 명령 경로를 흉내 낸 합성 모델. 로봇도 시뮬레이터도 없이 프로그램 동작을 확인할 때 "
             "쓴다. 결과는 측정값이 아니다 (data_kind: mock)."),
    "MuJoCo": ("MuJoCo (물리 시뮬레이터)", "이 PC에서 도는 Reachy Mini 시뮬레이션(127.0.0.1). viewer 창으로 움직임을 볼 수 있다. "
               "마찰이 크고 실시간보다 느려 로봇 성능 예측은 아니다 (data_kind: simulated)."),
    "data_kind": ("data_kind (데이터 출처)", "computed = 계산한 계획값, mock = Mock 모델 출력, simulated = MuJoCo, measured = "
                  "실제 로봇 측정. 서로 다른 출처의 값을 섞어 비교하지 않는다."),
    "Expressive": ("Expressive (표현형)", "감정을 전달하도록 애니메이션 원리(예비동작·과장·후속동작 등)를 넣은 모션."),
    "Neutral": ("Neutral (중립형)", "같은 기능을 일정한 속도로만 하는 비교용 모션. 감정 표현 요소를 뺐다."),
    "Initial Policy": ("초기 정책 (Initial Policy)", "실행 전에 계산으로 확인하는 안전 기준: 관절이 한계에서 10° 이상 떨어져 "
                       "있고, 요구 속도가 모터 무부하 속도의 50% 이하 등. 통과해도 기구적 안전을 보장하지는 않는다."),
    "Speed Ratio": ("요구 속도 비 (Speed Ratio)", "계획한 관절 속도 ÷ 모터 무부하 속도(XL330 618 deg/s). 정책 한계는 0.5."),
    "Joint Margin": ("관절 여유 (Joint Margin)", "Stewart 관절이 기구 한계(URDF)까지 남은 각도. 정책은 10° 이상."),
    "Stewart": ("Stewart 관절", "머리를 받치는 6개 모터(stewart_1~6). 이 관절들의 조합으로 머리의 기울기와 위치가 정해진다."),
    "Trial axis": ("시행 축 / 대표 관절", "시행 축 = 그 시행에서 주로 움직이는 머리 축(예: R3는 pitch). 대표 관절 = 명령 진폭이 "
                   "가장 큰 Stewart 관절."),
    "D / A": ("D (편도 이동 시간) / A (진폭)", "R3에서 D는 rest에서 목표까지 가는 시간(s), A는 pitch 끄덕임 크기(deg). 한 번에 "
              "하나만 바꾼다."),
    "minjerk": ("minjerk (최소 저크 보간)", "시작과 끝이 부드러운 기본 보간 방식. linear(일정 속도), ease_in_out, cartoon(예비동작·"
                "오버슈트 포함)도 있다."),
    "Tick": ("tick / 실행기 주기 (pc_measured)", "실행기가 20 ms(50 Hz)마다 명령을 보낸 실제 간격을 이 PC에서 잰 값. 로봇 측정이 아니다."),
    "S1/S2": ("정지 S1 / S2", "S1 = '정지' 버튼: 경로를 따라 0.3 s 동안 부드럽게 멈춘다. S2 = 전송 실패나 측정 끊김: 즉시 멈추고 "
              "마지막 목표를 유지한다."),
}


#: Short everyday description of each signed-off motion (Basic-mode card), written from the YAML description of
#: the 2026-11-11 sign-off. Shown only while the motion is exactly that file (same digest); otherwise the YAML intent.
MOTION_SUMMARY: dict[tuple[str, str], str] = {
    ("aha_recommend", "expressive"): "생각하듯 고개를 숙이고 안테나를 내렸다가, 갑자기 고개를 들며 두 안테나를 번쩍 "
                                     "세웁니다(아하!). 그다음 왼쪽의 추천 물건을 바라보고 잠시 멈춘 뒤 돌아옵니다.",
    ("aha_recommend", "neutral"): "왼쪽의 추천 물건을 일정한 속도로 바라보고 잠시 멈췄다가 돌아옵니다. 안테나와 소리는 쓰지 "
                                  "않습니다.",
    ("boredom", "expressive"): "무언가를 기다리듯 천천히 왼쪽과 오른쪽을 둘러봅니다. 아무 일도 없자 고개와 안테나가 축 "
                               "처지고, 잠시 뒤 천천히 돌아옵니다.",
    ("boredom", "neutral"): "같은 곳(왼쪽, 오른쪽)을 일정한 속도로 둘러보고 돌아옵니다. 처지는 동작, 안테나, 소리는 "
                            "없습니다.",
    ("success", "expressive"): "살짝 웅크렸다가 통통 튀듯 올라와 고개를 두 번 끄덕입니다. 안테나가 서서 흔들립니다.",
    ("success", "neutral"): "일정한 속도로 고개를 두 번 끄덕입니다(예). 안테나와 소리는 없습니다.",
    ("failure", "expressive"): "움찔한 뒤 고개를 떨구며 기울이고, 안테나가 처진 채 고개를 젓습니다. 그다음 천천히 원래대로 "
                               "돌아옵니다.",
    ("failure", "neutral"): "일정한 속도로 고개를 좌우로 젓습니다(아니요). 안테나와 소리는 없습니다.",
}


#: The lecture's principle names of the signed-off motions (docs/phase3/MOTION_REPORT_TABLE.md "원리 (강의 용어)");
#: the motion files record the Disney names (library.spec.Principle). The Motion tab card shows these for the
#: signed-off files only (same digest check as MOTION_SUMMARY); the Neutral versions are the constant-speed baseline.
MOTION_PRINCIPLES: dict[tuple[str, str], str] = {
    ("aha_recommend", "expressive"): "forethought, anticipation, timing, exaggeration, slow in·slow out",
    ("boredom", "expressive"): "timing, slow in·slow out, exaggeration, Laban Effort",
    ("success", "expressive"): "reaction, anticipation, exaggeration, slow in·slow out",
    ("failure", "expressive"): "reaction, anticipation, timing, exaggeration, slow in·slow out",
    ("aha_recommend", "neutral"): "없음 (등속 비교 기준)",
    ("boredom", "neutral"): "없음 (등속 비교 기준)",
    ("success", "neutral"): "없음 (등속 비교 기준)",
    ("failure", "neutral"): "없음 (등속 비교 기준)",
}


def va_words(valence: float, arousal: float) -> tuple[str, str]:
    """Plain words for the design-target valence / arousal (a reading of the sign and size, not a measurement)."""
    v = "긍정적" if valence >= 0.3 else ("부정적" if valence <= -0.3 else "중립에 가까움")
    a = "활발함" if arousal >= 0.3 else ("차분하고 가라앉음" if arousal <= -0.3 else "보통")
    return v, a


def glossary_text(*terms: str) -> str:
    """Tooltip text for one or more glossary terms."""
    out = []
    for t in terms:
        name, text = GLOSSARY.get(t, (t, ""))
        out.append(f"{name}: {text}" if text else name)
    return "\n\n".join(out)


def glossary_html() -> str:
    rows = "".join(f"<tr><td style='padding:4px 10px 4px 0; vertical-align:top'><b>{n}</b></td><td style='padding:4px 0'>{t}</td></tr>"
                   for n, t in GLOSSARY.values())
    return f"<h3>용어 설명 (Glossary)</h3><table>{rows}</table>"


#: Violation code -> Korean sentence (value, limit) for refused trials.
def violation_text(v: Any) -> str:
    code, val, lim = getattr(v, "code", "?"), getattr(v, "value", None), getattr(v, "limit", None)

    def n(x: Any, fmt: str = ".3g") -> str:
        return format(x, fmt) if isinstance(x, (int, float)) else "-"

    texts = {
        "stewart_speed": f"Stewart 관절 요구 속도가 무부하 속도의 {n(val)}배로 정책 한계 {n(lim)}배를 넘습니다",
        "body_speed": f"몸통 요구 속도 비 {n(val)}가 정책 한계 {n(lim)}를 넘습니다",
        "antenna_speed": f"안테나 요구 속도 비 {n(val)}가 정책 한계 {n(lim)}를 넘습니다",
        "stewart_margin": f"Stewart 관절 여유가 {n(val, '.1f')}°로 정책 최소 {n(lim)}°보다 작습니다",
        "stream_step": f"실행기 tick(20 ms)당 관절 변화가 {n(val, '.2f')}°로 한계 {n(lim, '.2f')}°를 넘습니다 (너무 빠름)",
        "ik_nan": "IK 해가 없는 자세가 있습니다 (로봇이 만들 수 없는 자세)",
        "envelope_roll": f"roll이 {n(val, '.1f')}°로 범위 {n(lim)}°를 넘습니다",
        "envelope_pitch": f"pitch가 {n(val, '.1f')}°로 범위 {n(lim)}°를 넘습니다",
        "envelope_rel_yaw": f"머리–몸통 상대 yaw가 {n(val, '.1f')}°로 범위 {n(lim)}°를 넘습니다",
        "envelope_body_yaw": f"몸통 yaw가 {n(val, '.1f')}°로 범위 {n(lim)}°를 넘습니다",
        "envelope_x": f"머리 x 위치가 범위를 벗어납니다 ({n(val, '.1f')} mm)",
        "envelope_y": f"머리 y 위치가 범위를 벗어납니다 ({n(val, '.1f')} mm)",
        "envelope_z": f"머리 z 위치가 범위를 벗어납니다 ({n(val, '.1f')} mm)",
        "antenna_range": "안테나 각도가 허용 범위를 벗어납니다",
        "auto_body_shift": f"자동 body yaw가 개입합니다 ({n(val, '.1f')}°, 한계 {n(lim)}°)",
        "body_step": f"표본 사이 몸통 관절 변화가 {n(val, '.1f')}°로 한계 {n(lim)}°를 넘습니다",
        "psi_wrap": "머리 yaw가 ±180° 경계를 넘나들어 몸통 목표가 뒤집힐 수 있습니다",
    }
    return texts.get(code, f"{code}: {n(val)} (한계 {n(lim)})")


def label(table: dict[str, str], key: str) -> str:
    return table.get(key, key)


def korean_font_family() -> str | None:
    """The first Korean font matplotlib knows (None if there is none)."""
    from matplotlib import font_manager

    names = {f.name for f in font_manager.fontManager.ttflist}
    return next((n for n in FONT_CANDIDATES if n in names), None)


def use_korean_matplotlib() -> str | None:
    """Korean text in every figure drawn after this (family list = per-glyph fallback, matplotlib >= 3.6)."""
    from matplotlib import rcParams

    fam = korean_font_family()
    if fam is not None:
        rcParams["font.family"] = [fam, "DejaVu Sans"]
        rcParams["axes.unicode_minus"] = False  # Korean fonts may lack U+2212
    return fam


def use_korean_qt(app: Any) -> str | None:
    """The Qt application font: a Korean UI font when the system has one."""
    from PySide6 import QtGui

    fams = set(QtGui.QFontDatabase.families())
    fam = next((n for n in FONT_CANDIDATES + ("맑은 고딕",) if n in fams), None)
    if fam is not None and app is not None:
        f = QtGui.QFont(fam)
        f.setPointSize(UI_POINT_SIZE)
        app.setFont(f)
    return fam


__all__ = ["POLICY_METRICS", "PARAMS", "COMPARE", "COLUMNS", "PLAYBACK", "STATUS", "POLICY_DISCLAIMER", "label",
           "GLOSSARY", "glossary_text", "glossary_html", "violation_text", "MOTIONS", "AXES", "MOTION_SUMMARY", "MOTION_PRINCIPLES",
           "va_words",
           "use_korean_matplotlib", "use_korean_qt", "korean_font_family"]
