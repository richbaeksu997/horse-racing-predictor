"""화면에 보여 줄 데이터(날짜 목록, 하루치 출전표·예측)를 만든다. app.py 와 build_site.py 가 함께 쓴다."""
from datetime import datetime, timedelta, timezone

import pandas as pd

from kra.api import ENTRY_PATH, MEETS, ApiError, fetch_all, today
from predict import history, predict_day


def _val(v):
    """NaN 은 null 로, numpy 값은 파이썬 값으로."""
    if pd.isna(v):
        return None
    return v.item() if hasattr(v, "item") else v


def weeks():
    """지난주·이번 주·다음 주의 (이름, 월요일, 일요일)."""
    monday = today() - timedelta(days=today().weekday())
    return [(name, monday + timedelta(days=7 * i), monday + timedelta(days=7 * i + 6))
            for i, name in [(1, "다음 주"), (0, "이번 주"), (-1, "지난주")]]


def race_dates():
    """세 주 범위 안에서 성적이 있거나 출전표가 공개된 날짜 (YYYYMMDD)."""
    spans = weeks()
    first, last = spans[-1][1], spans[0][2]
    found = set(history()["date"].dt.strftime("%Y%m%d").unique())
    try:
        for month in sorted({first.strftime("%Y%m"), last.strftime("%Y%m")}):
            for meet in MEETS:
                found |= {str(it["rcDate"]) for it in fetch_all(ENTRY_PATH, meet=meet, rc_month=month)}
    except ApiError:
        pass
    return sorted(d for d in found if f"{first:%Y%m%d}" <= d <= f"{last:%Y%m%d}")


def group_dates(dates):
    return [{"label": name, "dates": sorted((d for d in dates if f"{start:%Y%m%d}" <= d <= f"{end:%Y%m%d}"),
                                            reverse=True)} for name, start, end in weeks()]


def apply_results(payload):
    """이미 치러진 경주에 실제 착순과 단승 배당을 채워 넣는다."""
    hist = history()
    day = hist[hist["date"] == pd.to_datetime(payload["date"], format="%Y%m%d")]
    actual = {(int(r.meet), int(r.rcNo), int(r.chulNo)): (_val(r.ord), _val(r.winOdds)) for r in day.itertuples()}
    for race in payload["races"]:
        for h in race["horses"]:
            h["ord"], h["odds"] = actual.get((race["meet"], race["rcNo"], h["chulNo"]), (None, None))
        race["finished"] = any(h["ord"] is not None for h in race["horses"])
    return payload


def day_payload(day):
    df = predict_day(day)
    races = []
    for (meet, rc_no), race in df.groupby(["meet", "rcNo"], sort=True):
        head = race.iloc[0]
        horses = [{
            "chulNo": int(r.chulNo), "hrName": r.hrName, "sex": _val(r.sexName), "age": _val(r.age),
            "prd": _val(r.prd), "wgBudam": _val(r.wgBudam), "rating": _val(r.rating),
            "jkName": r.jkName, "trName": r.trName, "owName": _val(r.owName),
            "record": [_val(r.rcCntT), _val(r.ord1CntT), _val(r.ord2CntT), _val(r.ord3CntT)],
            "daysSince": _val(r.h_days_since),
            "pWin": round(float(r.p_win), 4), "pTop3": round(float(r.p_top3), 4),
        } for r in race.itertuples()]
        races.append({
            "meet": int(meet), "meetName": MEETS[int(meet)], "rcNo": int(rc_no),
            "dist": _val(head["rcDist"]), "rank": _val(head["rank"]), "rcName": _val(head["rcName"]),
            "stTime": str(_val(head["stTime"]) or "").replace("출발", "").strip(" :"),
            "budam": _val(head["budam"]), "ageCond": _val(head["ageCond"]), "horses": horses,
        })
    now = datetime.now(timezone(timedelta(hours=9)))
    payload = apply_results({"date": day, "predictedAt": f"{now:%Y-%m-%d %H:%M}", "races": races})
    # 결과가 나온 뒤에 계산한 예측이면, 그 결과가 모델 학습에 들어갔을 수 있다.
    payload["afterResult"] = any(r["finished"] for r in races)
    return payload
