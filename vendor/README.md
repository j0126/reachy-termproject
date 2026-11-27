# vendor: 키오스크가 쓰는 HW 코드의 복사본

`git clone` 하나로 실행되도록 HW3·HW4 코드에서 키오스크가 쓰는 부분을 여기 복사했다 (2026-11-27, `tools/vendor_hw.py`). `config/default.toml`의 `[hw]`가 이 폴더를 가리키고, `reachy_cafe/robot/hwpaths.py`가 앱 시작 때 sys.path 앞에 넣는다.

- `vendor/hw4/` ← `reachy_hw4`: reachy_motion, motions, pyproject.toml, README.md (97개 파일)
- `vendor/hw3/` ← `hw3/reachy-hw3-main`: reachy_s2s, config, pyproject.toml, README.md, TROUBLESHOOTING.md (98개 파일)
- `models/face_landmarker.task` ← HW2(reachy-chamchamcham) 얼굴 모델 (MediaPipe)

여기서 고치지 않는다. 원본 HW 폴더에서 고친 뒤 `tools/vendor_hw.py`로 다시 복사한다. 의존 패키지는 이 저장소의 `pyproject.toml`(robot·speech·cuda·sim)에 HW3·HW4와 같은 버전으로 들어 있다.
