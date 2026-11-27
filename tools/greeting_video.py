"""인사 모션(greeting)을 MuJoCo 시뮬로 재생해 영상으로 만든다 (모션 YAML을 고친 뒤 다시 찍을 때).

HW4 `reachy_motion.sim_video`의 오프라인 재생(SDK 1.11.0 MuJoCo 모델, 50 Hz 명령, HW4 안전 검사 통과한 모션만)을
그대로 쓰고, 화면 아래에 한국어 자막(지금 키프레임, 인사말이 시작되는 순간 = bow + label_delay_s)을 붙인다.
시뮬레이션이라 실제 로봇과 다를 수 있다 (MuJoCo는 명령을 최대 약 8° 늦게 따라감).

결과 (results/videos/, 데스크톱 묶음에는 안 들어감):
  greeting_views.mp4 / .webp       expressive를 정면(손님 시점) + 비스듬히 45°로, 실제 속도 다음 0.5× 느리게
  greeting_vs_neutral.mp4 / .webp  expressive vs neutral (정면, 같은 시간축)

실행: MuJoCo와 PyAV(H.264)가 있는 HW4 가상환경으로
  ..\\reachy_hw4\\.venv\\Scripts\\python.exe tools\\greeting_video.py        (Windows)
  ../reachy_hw4/.venv/bin/python tools/greeting_video.py                   (macOS·Linux)
PyAV가 없으면 WebP만 만든다.
"""

from __future__ import annotations

import argparse
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reachy_cafe.config import load_config  # noqa: E402
from reachy_cafe.robot.hwpaths import ensure_hw_paths  # noqa: E402

HEAD_KO = {"notice": "알아챔", "bow": "숙여 인사", "rise": "올려다보며 갸웃", "back": "복귀", "start": "대기", "end": "끝"}
ANT_KO = {"perk": "세움", "fold": "내림", "wave-a": "손 흔들기", "wave-b": "손 흔들기", "wave-c": "손 흔들기",
          "settle": "제자리", "(none)": "움직임 없음", "start": "-", "end": "-", "-": "-"}
FONTS = {  # 한글이 되는 글꼴 (OS별), 없으면 PIL 기본 글꼴 (한글이 네모로 보임)
    False: [r"C:\Windows\Fonts\malgun.ttf", "/System/Library/Fonts/AppleSDGothicNeo.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"],
    True: [r"C:\Windows\Fonts\malgunbd.ttf", "/System/Library/Fonts/AppleSDGothicNeo.ttc",
           "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"],
}
SIZE = (560, 420)
BG, PANEL_FOOT, FG, DIM = (22, 27, 38), (30, 36, 50), (236, 240, 246), (150, 160, 178)
ACCENT, WARM, TRACK = (110, 168, 254), (250, 190, 90), (56, 64, 82)


@lru_cache(maxsize=None)
def font(size: int, bold: bool = False) -> Any:
    from PIL import ImageFont

    for path in FONTS[bold]:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


class Captions:
    def __init__(self, line: str, sync_label: str, label_delay_s: float):
        self.line, self.sync_label, self.delay = line, sync_label, label_delay_s

    @staticmethod
    def head_keys(spec: Any) -> list[tuple[str, float]]:
        eff = spec.effective()
        return [(k.label, t0) for (t0, _a, _r), k in zip(eff.key_times("head"), eff.track("head"))]

    def speech_start(self, spec: Any) -> float:
        return dict(self.head_keys(spec)).get(self.sync_label, 0.0) + self.delay

    def footer(self, draw: Any, x: int, y: int, w: int, info: dict[str, Any], spec: Any) -> None:
        """패널 아래: 시각·지금 키프레임, 인사말(bow + 지연부터), 키프레임 시간 막대."""
        dur, t, phase = float(spec.duration), float(info["motion_t"]), info["phase"]
        hk = info["head_key"] if phase == "move" else phase
        ak = info["antenna_key"] if phase == "move" else phase
        draw.rectangle((x, y, x + w - 1, y + 118), fill=PANEL_FOOT)
        draw.text((x + 12, y + 8), f"t {t:4.2f} / {dur:.2f} s", fill=FG, font=font(15, True))
        draw.text((x + 130, y + 8), f"머리: {HEAD_KO.get(hk, hk)} ({hk})   안테나: {ANT_KO.get(ak, ak)}", fill=DIM,
                  font=font(15))
        ts = self.speech_start(spec)
        if phase != "start" and t >= ts:
            draw.rounded_rectangle((x + 10, y + 36, x + w - 10, y + 70), radius=8, outline=ACCENT, width=2,
                                   fill=(26, 40, 66))
            draw.text((x + 22, y + 41), f"말: “{self.line}”", fill=FG, font=font(17, True))
        else:
            draw.text((x + 22, y + 43), f"말: 고개를 숙이기 시작하면({self.sync_label} + {self.delay:g} s) 인사말 시작",
                      fill=DIM, font=font(15))
        x0, x1, yb = x + 16, x + w - 16, y + 92
        draw.rectangle((x0, yb, x1, yb + 6), fill=TRACK)
        frac = min(max(t / dur, 0.0), 1.0) if phase != "start" else 0.0
        draw.rectangle((x0, yb, x0 + int((x1 - x0) * frac), yb + 6), fill=ACCENT)
        for label, t0 in self.head_keys(spec):
            kx = x0 + int((x1 - x0) * t0 / dur)
            draw.rectangle((kx - 1, yb - 4, kx + 1, yb + 10), fill=FG)
            draw.text((kx + 3, yb + 8), label, fill=DIM, font=font(11))
        sx = x0 + int((x1 - x0) * ts / dur)
        draw.polygon([(sx - 6, yb - 12), (sx + 6, yb - 12), (sx, yb - 3)], fill=WARM)

    def compose(self, panels: list[tuple[Any, dict[str, Any], str, Any]], title: str, subtitle: str,
                speed: str) -> Any:
        """panels = [(rgb, frame_info, 이름표, spec), ...] 나란히, 위에 제목, 각 패널 아래 자막."""
        from PIL import Image, ImageDraw

        head_h, foot_h, gap = 64, 120, 4
        w = sum(p[0].shape[1] for p in panels) + gap * (len(panels) - 1)
        img = Image.new("RGB", (w, head_h + max(p[0].shape[0] for p in panels) + foot_h), BG)
        d = ImageDraw.Draw(img)
        d.text((14, 6), title, fill=FG, font=font(21, True))
        d.text((14, 37), subtitle, fill=DIM, font=font(13))
        tw = int(d.textlength(speed, font=font(17, True)))
        d.rounded_rectangle((w - tw - 34, 14, w - 12, 46), radius=8, fill=(40, 48, 66))
        d.text((w - tw - 23, 17), speed, fill=WARM if "느리게" in speed else FG, font=font(17, True))
        x = 0
        for rgb, info, tag, spec in panels:
            img.paste(Image.fromarray(rgb), (x, head_h))
            tw = int(d.textlength(tag, font=font(16, True)))
            d.rounded_rectangle((x + 8, head_h + 8, x + 24 + tw, head_h + 36), radius=6, fill=(16, 20, 28))
            d.text((x + 16, head_h + 9), tag, fill=ACCENT, font=font(16, True))
            self.footer(d, x, head_h + rgb.shape[0], rgb.shape[1], info, spec)
            x += rgb.shape[1] + gap
        return img


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="인사 모션 MuJoCo 시뮬 영상 (HW4 sim_video 재생 + 한국어 자막)")
    parser.add_argument("--out", default=str(ROOT / "results" / "videos"), help="결과 폴더")
    parser.add_argument("--line", default="안녕하세요! 주문 도와드릴까요?", help="자막으로 보일 인사말")
    parser.add_argument("--config", action="append", default=[], help="덮어쓸 TOML (label_delay_s·sync_labels)")
    args = parser.parse_args(argv)
    cfg = load_config(*args.config, env={})
    ensure_hw_paths(cfg)

    from PIL import Image
    from reachy_motion.sim_video import CameraView, mp4_available, replay, save_mp4, save_webp

    from reachy_cafe.robot.motion_library import cafe_library

    caps = Captions(args.line, cfg["motion"]["sync_labels"].get("greeting", "bow"),
                    float(cfg["motion"].get("label_delay_s", 0.1)))
    front = CameraView(0.68, 170.0, -8.0, (0.0, 0.0, 0.21))
    three_q = CameraView(0.66, 135.0, -10.0, (0.0, 0.0, 0.20))
    lib = cafe_library()
    expressive, neutral = lib.get("greeting", "expressive"), lib.get("greeting", "neutral")
    draft = " · 초안" if expressive.draft else ""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    small = lambda frames: [im.resize((im.width * 2 // 3, im.height * 2 // 3), Image.LANCZOS) for im in frames]  # noqa: E731

    # 1) expressive: 정면 + 45°, 실제 속도(25 fps 재생) 다음 0.5× (50 fps로 찍어 25 fps로 재생)
    title = f"인사 모션 greeting (expressive{draft}) · MuJoCo 시뮬"
    subtitle = ("SDK 1.11.0 MuJoCo 모델로 오프라인 재생 (실제 로봇 아님) · 키오스크에서는 앞에 지금 시선 → 시작 자세 "
                "전이(0.3 s 이상)가 붙음")
    passes = []
    for fps, speed in ((25, "실제 속도"), (50, "0.5× 느리게")):
        runs = [replay(expressive, lead_s=0.5, tail_s=1.0, fps=fps, render_size=SIZE, camera=cam)
                for cam in (front, three_q)]
        n = min(len(r.frames) for r in runs)
        passes.append([caps.compose([(runs[0].frames[k], runs[0].frame_info[k], "정면 (손님 시점)", expressive),
                                     (runs[1].frames[k], runs[1].frame_info[k], "비스듬히 45°", expressive)],
                                    title, subtitle, speed) for k in range(n)])
        print(f"expressive {speed}: {n} frames, sim이 명령을 따라가는 최대 차이 {runs[0].max_command_gap_deg:.1f}°")
    real, slow = passes
    save_webp(small(real), out / "greeting_views.webp", fps=25, quality=72)
    if mp4_available():
        save_mp4(real + slow, out / "greeting_views.mp4", fps=25)

    # 2) expressive vs neutral, 정면, 같은 시간축
    title = f"인사 모션 greeting · expressive vs neutral (HW4 변형 비교){draft} · MuJoCo 시뮬"
    subtitle = "정면(손님 시점) · 같은 시간축 · 인사말은 각 변형의 bow 키프레임 + 지연에 시작"
    total = 0.5 + max(expressive.duration, neutral.duration) + 1.0
    runs = [replay(s, lead_s=0.5, total_s=total, fps=25, render_size=SIZE, camera=front) for s in (expressive, neutral)]
    n = min(len(r.frames) for r in runs)
    pair = [caps.compose([(runs[0].frames[k], runs[0].frame_info[k], f"EXPRESSIVE {expressive.duration:.2f} s",
                           expressive),
                          (runs[1].frames[k], runs[1].frame_info[k], f"NEUTRAL {neutral.duration:.2f} s", neutral)],
                         title, subtitle, "실제 속도") for k in range(n)]
    save_webp(small(pair), out / "greeting_vs_neutral.webp", fps=25, quality=72)
    if mp4_available():
        save_mp4(pair, out / "greeting_vs_neutral.mp4", fps=25)
    else:
        print("PyAV(H.264)가 없어 MP4는 건너뜀 (WebP만). HW4 가상환경으로 실행하면 MP4도 만든다.")
    for p in sorted(out.glob("greeting_*")):
        print(f"  {p.name}: {p.stat().st_size / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
