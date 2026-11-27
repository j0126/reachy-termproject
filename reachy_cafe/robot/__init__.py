"""실제 로봇 백엔드 (2단계).

연결 하나(`RobotLink`)를 세 과제 코드가 공유한다:
- 모션·시선 (HW4 `reachy_motion`): `motion.MotionPlayer` (MotionService 상속)
- 음성 (HW3 `reachy_s2s`): `speech.KioskSpeech` (VAD → ASR → 버스, TTS 캐시 → 로봇 스피커)
- 카메라 (HW2 방식 MediaPipe FaceLandmarker): `vision.FacePerception`
`real.RealRobot`이 이들을 묶어 `behavior.robot_api.RobotOutputs`를 구현한다.

이 패키지는 mock 모드에서는 import되지 않는다 (SDK·MediaPipe·HW 코드가 없어도 1단계는 돈다).
"""
