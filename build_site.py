"""서버 없이 볼 수 있는 웹페이지를 docs/ 에 만든다 (GitHub Pages 게시용).

    python build_site.py

아직 열리지 않은 날의 예측은 실행할 때마다 새로 계산한다(출전 취소·기수 변경 반영).
경주 당일부터는 이미 만들어 둔 예측을 그대로 두고 실제 결과만 채워 넣는다.
"""
import json
import shutil

from kra.api import ROOT, today
from predict import NoEntries
from views import apply_results, day_payload, group_dates, race_dates

OUT = ROOT / "docs"


def main():
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    published = []
    for day in race_dates():
        path = OUT / "data" / f"{day}.json"
        if path.exists() and day <= f"{today():%Y%m%d}":
            payload = apply_results(json.loads(path.read_text(encoding="utf-8")))
            print(f"{day}: 기존 예측 유지, 결과 갱신")
        else:
            try:
                payload = day_payload(day)
            except NoEntries:
                continue
            print(f"{day}: 예측 계산 ({len(payload['races'])}경주)")
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        published.append(day)

    (OUT / "data" / "dates.json").write_text(json.dumps(group_dates(published), ensure_ascii=False), encoding="utf-8")
    shutil.copy(ROOT / "web" / "index.html", OUT / "index.html")
    (OUT / ".nojekyll").touch()
    print(f"완료: {OUT}")


if __name__ == "__main__":
    main()
