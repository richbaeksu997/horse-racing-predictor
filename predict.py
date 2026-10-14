"""출전표를 받아 말별 1착·3착 이내 확률을 예측한다.

    python predict.py 20261017            # 해당 날짜의 전 경마장
    python predict.py 20261017 --meet 1   # 서울만 (1 서울, 2 제주, 3 부경)

출전표는 보통 경주 며칠 전에 공개된다. 결과는 predictions/ 에 CSV 로도 저장된다.
"""
import argparse
from functools import lru_cache

import lightgbm as lgb
import pandas as pd

from kra.api import ENTRY_PATH, MEETS, ROOT, fetch_all
from kra.data import HISTORY, load_results, normalize
from kra.features import FEATURES, build_features
from train import MODELS

KEYS = ["meet", "rcNo", "chulNo"]
# 화면 표시용으로 출전표 원본에서 그대로 가져오는 항목
ENTRY_EXTRA = ["prd", "owName", "rank", "rcName", "stTime", "budam", "ageCond", "prizeCond",
               "rcCntT", "ord1CntT", "ord2CntT", "ord3CntT"]


class NoEntries(Exception):
    pass


@lru_cache(maxsize=1)
def _history(stamp):
    return load_results()


def history():
    # 수집 파일이 바뀌면 다시 읽는다.
    return _history(max((f.stat().st_mtime for f in HISTORY.glob("*.csv.gz")), default=0))


@lru_cache(maxsize=4)
def _model(target, stamp):
    return lgb.Booster(model_str=(MODELS / f"{target}.txt").read_text(encoding="utf-8"))


def model(target):
    path = MODELS / f"{target}.txt"
    if not path.exists():
        raise FileNotFoundError("학습된 모델이 없습니다. 먼저 train.py 를 실행하세요.")
    return _model(target, path.stat().st_mtime)


def predict_day(date, meets=None):
    """date(YYYYMMDD) 의 출전마별 특성·예측 확률 표."""
    frames = []
    for meet in meets or MEETS:
        items = fetch_all(ENTRY_PATH, meet=meet, rc_date=date)
        if items:
            frames.append(pd.DataFrame(items).assign(meet=meet))
    if not frames:
        raise NoEntries(f"{date} 출전표가 없습니다 (아직 공개 전이거나 경주가 없는 날).")
    raw = pd.concat(frames, ignore_index=True)
    entries = normalize(raw)
    # 출전표에는 결과가 없어야 한다.
    entries[["ord", "rcTime", "wgHr", "winOdds"]] = float("nan")
    entries["is_entry"] = True

    day = pd.to_datetime(date, format="%Y%m%d")
    hist = history()
    past = hist[hist["date"] < day].assign(is_entry=False)
    df = build_features(pd.concat([past, entries], ignore_index=True))
    df = df[df["is_entry"]].copy()

    for target in ["win", "top3"]:
        df[f"p_{target}"] = model(target).predict(df[FEATURES])
    # 한 경주 안에서 1착 확률 합은 1, 3착내 확률 합은 3 이 되도록 맞춘다.
    by_race = df.groupby("race_id")
    df["p_win"] = df["p_win"] / by_race["p_win"].transform("sum")
    places = by_race["chulNo"].transform("size").clip(upper=3)
    df["p_top3"] = (df["p_top3"] * places / by_race["p_top3"].transform("sum")).clip(upper=1)

    extra = raw.reindex(columns=["meet", "rcNo", "chulNo", "sex"] + ENTRY_EXTRA).rename(columns={"sex": "sexName"})
    df = df.merge(extra.drop_duplicates(KEYS), on=KEYS, how="left")
    return df.sort_values(["meet", "rcNo", "p_win"], ascending=[True, True, False]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date", help="경주일 YYYYMMDD")
    ap.add_argument("--meet", type=int, choices=list(MEETS))
    args = ap.parse_args()

    try:
        df = predict_day(args.date, [args.meet] if args.meet else None)
    except (NoEntries, FileNotFoundError) as e:
        raise SystemExit(str(e))
    for (meet, rc_no), race in df.groupby(["meet", "rcNo"], sort=True):
        print(f"\n=== {MEETS[meet]} {int(rc_no)}경주 ({int(race['rcDist'].iloc[0])}m, {len(race)}두) ===")
        for r in race.itertuples():
            starts = 0 if pd.isna(r.h_starts) else int(r.h_starts)
            print(f"  {int(r.chulNo):>2}번 {r.hrName:<10} 기수 {r.jkName:<6} "
                  f"1착 {r.p_win:5.1%} | 3착내 {r.p_top3:5.1%} | 통산 {starts}전")

    out = ROOT / "predictions" / f"{args.date}.csv"
    out.parent.mkdir(exist_ok=True)
    cols = ["meet", "rcNo", "chulNo", "hrName", "jkName", "trName", "p_win", "p_top3", "h_starts"]
    df[cols].to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
