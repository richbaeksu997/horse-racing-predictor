"""날짜·경주를 골라 출전 정보와 예측 확률을 보는 웹 화면.

    python app.py          # http://127.0.0.1:5000 (이 PC 에서만)
    python app.py --lan    # 같은 와이파이의 다른 기기에서도 접속 가능

화면(web/index.html)은 data/dates.json, data/<날짜>.json 을 읽는다.
여기서는 요청 때마다 계산해 주고, build_site.py 는 같은 내용을 파일로 미리 만들어 둔다.
"""
import socket
import sys
import time

from flask import Flask, jsonify

from kra.api import ApiError
from predict import NoEntries
from views import day_payload, group_dates, race_dates

app = Flask(__name__, static_folder="web", static_url_path="")
_cache = {}


def cached(key, ttl, fn):
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    value = fn()
    _cache[key] = (time.time(), value)
    return value


@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.get("/data/dates.json")
def dates():
    return jsonify(group_dates(cached("dates", 600, race_dates)))


@app.get("/data/<day>.json")
def predict(day):
    if not (day.isdigit() and len(day) == 8):
        return jsonify(error="날짜 형식이 올바르지 않습니다."), 400
    try:
        return jsonify(cached(("day", day), 600, lambda: day_payload(day)))
    except NoEntries as e:
        return jsonify(error=str(e)), 404
    except (ApiError, FileNotFoundError) as e:
        return jsonify(error=str(e)), 502


def _lan_ip():
    # 실제로 패킷을 보내지는 않고, 공유기 쪽으로 나가는 주소만 알아낸다.
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]


if __name__ == "__main__":
    # --lan: 같은 와이파이에 연결된 휴대폰 등에서도 접속할 수 있게 연다.
    lan = "--lan" in sys.argv
    if lan:
        print(f"\n휴대폰 브라우저에서 열 주소:  http://{_lan_ip()}:5000\n")
    app.run(host="0.0.0.0" if lan else "127.0.0.1", port=5000)
