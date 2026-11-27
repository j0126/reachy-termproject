"""키오스크 서버: Pi의 Chromium이 화면 파일을 캐시에 붙잡아 두지 않게 한다.

캐시 헤더가 없으면 Chromium이 몇 시간씩 옛 app.js·style.css를 써서, 두뇌 PC에서 화면을 고쳐도
키오스크에 바로 반영되지 않았다 (11/27 720×1280 메뉴 수정 때 발견).
"""

import time
from pathlib import Path

from fastapi.testclient import TestClient

from reachy_cafe.bus import EventBus
from reachy_cafe.kiosk_server.hub import Hub
from reachy_cafe.kiosk_server.server import create_app

UI_DIR = Path(__file__).resolve().parents[1] / "kiosk_ui"


def make_client() -> TestClient:
    return TestClient(create_app(Hub(), EventBus(), UI_DIR))


def test_ui_files_are_revalidated_every_time() -> None:
    client = make_client()
    for path in ("/kiosk", "/kiosk?device=pi", "/operator", "/stage", "/ui/app.js", "/ui/style.css"):
        response = client.get(path)
        assert response.status_code == 200, path
        assert response.headers["cache-control"] == "no-cache", path


def test_healthz_counts_only_pi_kiosk_pages() -> None:
    # start_kiosk.sh는 pi_kiosk가 0이면 흰 화면으로 보고 Chromium을 다시 띄운다 (11/27 첫 부팅 때 흰 화면)
    client = make_client()
    assert client.get("/healthz").json()["pi_kiosk"] == 0
    with client.websocket_connect("/ws/ui") as laptop_preview:
        laptop_preview.receive_json()
        assert client.get("/healthz").json()["pi_kiosk"] == 0
        with client.websocket_connect("/ws/ui?device=pi") as pi_page:
            pi_page.receive_json()
            health = client.get("/healthz").json()
            assert health["pi_kiosk"] == 1 and health["ui"] == 2
    deadline = time.monotonic() + 2
    while client.get("/healthz").json()["pi_kiosk"] and time.monotonic() < deadline:
        time.sleep(0.05)
    assert client.get("/healthz").json()["pi_kiosk"] == 0


def test_unchanged_file_answers_not_modified() -> None:
    client = make_client()
    first = client.get("/ui/app.js")
    again = client.get("/ui/app.js", headers={"If-None-Match": first.headers["etag"]})
    assert again.status_code == 304
