"""공공데이터포털 한국마사회 API 클라이언트."""
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import unquote

import requests

ROOT = Path(__file__).resolve().parent.parent
BASE = "https://apis.data.go.kr/B551015"
RESULT_PATH = "API214_1/RaceDetailResult_1"  # 경주성적정보 (배당률·구간기록 포함)
ENTRY_PATH = "API26_2/entrySheet_2"  # 출전표 상세정보 (data.go.kr 15058677)
MEETS = {1: "서울", 2: "제주", 3: "부경"}


class ApiError(RuntimeError):
    pass


def today():
    """한국 시간 기준 오늘 (서버가 해외에 있어도 같게)."""
    return datetime.now(timezone(timedelta(hours=9))).date()


def load_key():
    key = os.environ.get("KRA_SERVICE_KEY", "").strip()
    env = ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text(encoding="utf-8-sig").splitlines():
            if line.strip().startswith("KRA_SERVICE_KEY="):
                key = line.split("=", 1)[1].strip().strip("\"'")
    if not key:
        raise SystemExit(
            "인증키가 없습니다. .env.example 을 .env 로 복사하고 KRA_SERVICE_KEY 를 채워 주세요."
        )
    # Encoding 키(%2B 등 포함)를 넣어도 requests 가 다시 인코딩하므로 원래 값으로 되돌린다.
    return unquote(key)


def _get(path, params):
    last = None
    for attempt in range(4):
        try:
            r = requests.get(f"{BASE}/{path}", params=params, timeout=30)
            r.raise_for_status()
        except requests.RequestException as e:
            last = e
            time.sleep(2 ** attempt)
            continue
        try:
            return r.json()
        except ValueError:
            # 인증키 오류 등은 _type=json 이어도 XML 로 내려온다.
            raise ApiError(f"JSON 이 아닌 응답입니다 (인증키/활용신청 확인):\n{r.text[:500]}")
    raise ApiError(f"요청 실패: {last}")


def fetch_all(path, **params):
    """페이지를 끝까지 넘기며 item 목록 전체를 돌려준다."""
    key = load_key()
    items, page = [], 1
    while True:
        data = _get(path, {"ServiceKey": key, "_type": "json", "numOfRows": 1000,
                           "pageNo": page, **params})
        resp = data.get("response", {})
        code = str(resp.get("header", {}).get("resultCode", ""))
        if code not in ("00", "0"):
            raise ApiError(f"API 오류 {code}: {resp.get('header', {}).get('resultMsg')}")
        body = resp.get("body") or {}
        got = (body.get("items") or {}).get("item") or []
        if isinstance(got, dict):
            got = [got]
        items.extend(got)
        if not got or len(items) >= int(body.get("totalCount") or 0):
            return items
        page += 1
