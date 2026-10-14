"""원본 JSON 을 분석용 표로 정리한다."""
import numpy as np
import pandas as pd

from .api import ROOT

RAW_RESULTS = ROOT / "data" / "raw" / "results"  # API 응답 원본 (PC 에만 보관)
HISTORY = ROOT / "data" / "history"  # 정리된 월별 성적
ID_TEXT = ["hrNo", "hrName", "jkNo", "jkName", "trNo", "trName"]

NUMERIC = ["rcNo", "chulNo", "age", "wgBudam", "rating", "rcDist", "ord", "wgHr", "winOdds"]
TEXT = ["hrNo", "hrName", "jkNo", "jkName", "trNo", "trName", "sex"]
SEX = {"수": 0, "암": 1, "거": 2}
CLASS_TYPE = {"국": 0, "혼": 1, "제": 2, "한": 3}  # 국산/혼합/제주마/한라마
BUDAM_TYPE = {"핸디": 0, "별정": 1, "마령": 2}


def _num(s):
    # "470(-2)", "4세" 같은 값에서 첫 숫자만 뽑는다.
    return pd.to_numeric(s.astype(str).str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce")


def _seconds(s):
    # 주파기록: 75.3 또는 1:15.3 두 형식 모두 처리
    s = s.astype(str).str.strip()
    parts = s.str.extract(r"^(?:(\d+):)?(\d+\.?\d*)$")
    return pd.to_numeric(parts[0], errors="coerce").fillna(0) * 60 + pd.to_numeric(parts[1], errors="coerce")


def _first(raw, cols):
    """경마장마다 필드명이 달라서, 후보 컬럼 중 값이 있는(0 보다 큰) 첫 값을 고른다."""
    out = pd.Series(np.nan, index=raw.index)
    for c in cols:
        if c in raw:
            v = pd.to_numeric(raw[c], errors="coerce")
            out = out.fillna(v.where(v > 0))
    return out


def _text(raw, cols):
    for c in cols:
        if c in raw:
            return raw[c].astype(str).str.strip()
    return pd.Series("", index=raw.index)


def normalize(raw):
    """API item 표(경주성적/출전표 공통)를 표준 컬럼으로 맞춘다. meet 은 호출 시 넣은 코드(1/2/3)."""
    df = pd.DataFrame(index=raw.index)
    df["meet"] = raw["meet"].astype(int)
    df["date"] = pd.to_datetime(_num(raw["rcDate"]).astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    for c in NUMERIC:
        df[c] = _num(raw[c]) if c in raw else float("nan")
    for c in TEXT:
        df[c] = raw[c].astype(str).str.strip() if c in raw else None
    df["rcTime"] = _seconds(raw["rcTime"]) if "rcTime" in raw else float("nan")
    # 번호는 API 에 따라 "0037452" / 37452 로 다르게 올 수 있어 형태를 통일한다.
    for c in ["hrNo", "jkNo", "trNo"]:
        if c in raw:
            df[c] = df[c].str.replace(r"\.0$", "", regex=True).str.lstrip("0")
    df["sex"] = pd.to_numeric(df["sex"].map(SEX))
    # 0 은 기록 없음(출전취소·실격·주행중지 등)
    for c in ["ord", "rcTime", "winOdds", "wgHr", "rating"]:
        df[c] = df[c].where(df[c] > 0)
    # 91~99 는 착순이 아니라 실격·주행중지·경주제외·출전취소·경주취소 등을 뜻하는 코드
    df["ord"] = df["ord"].where(df["ord"] < 90)

    # 경주 조건 (성적·출전표 양쪽에 모두 있음)
    rank = _text(raw, ["rank"])
    df["class_level"] = _num(rank).where(~rank.str.contains("OPEN|오픈"), 0)  # 6등급 → 6, 오픈 → 0
    df["class_type"] = pd.to_numeric(rank.str[:1].map(CLASS_TYPE))
    df["prize"] = np.log1p(_first(raw, ["chaksun1"]))
    df["budam_type"] = pd.to_numeric(_text(raw, ["budam"]).str[:2].map(BUDAM_TYPE))
    prd = _text(raw, ["prd", "name"])  # 산지: 출전표는 prd, 성적은 name
    df["foreign"] = (~prd.str.startswith("한")).astype(float).where(prd != "")

    # 구간 기록 (성적에만 있음)
    df["s1f_time"] = _first(raw, ["seS1fAccTime", "buS1fAccTime", "jeS1fTime"])  # 출발 후 200m
    df["s1f_ord"] = _first(raw, ["sjS1fOrd", "buS1fOrd"])
    to_g3f = _first(raw, ["seG3fAccTime", "buG3fAccTime"])
    l3f = _first(raw, ["jeG3fTime"]).fillna(df["rcTime"] - to_g3f)  # 마지막 600m
    df["l3f_time"] = l3f.where(l3f > 0)
    wg_chg = raw["wgHr"].astype(str).str.extract(r"\(\s*([+-]?\d+)\s*\)")[0] if "wgHr" in raw else np.nan
    df["wg_chg"] = pd.to_numeric(wg_chg, errors="coerce")

    df = df.dropna(subset=["date", "rcNo", "chulNo"])
    return df.drop_duplicates(["meet", "date", "rcNo", "chulNo"]).reset_index(drop=True)


def save_month(meet, month, items):
    """한 달치 원본을 정리해 data/history 에 저장한다. 분석은 이 파일들만 읽는다."""
    HISTORY.mkdir(parents=True, exist_ok=True)
    if items:
        df = normalize(pd.DataFrame(items).assign(meet=meet))
        # 착순이 없는 말은 성적으로 쓰지 않는다.
        df = df[df["ord"].notna()]
    else:
        df = pd.DataFrame(columns=["date"])  # 경주가 없던 달도 받았다는 표시로 빈 파일을 남긴다.
    df.to_csv(HISTORY / f"{meet}_{month}.csv.gz", index=False)


def load_results():
    frames = [pd.read_csv(f, dtype={c: str for c in ID_TEXT}, parse_dates=["date"])
              for f in sorted(HISTORY.glob("*.csv.gz"))]
    frames = [f for f in frames if len(f)]
    if not frames:
        raise SystemExit("수집된 성적 데이터가 없습니다. 먼저 collect.py 를 실행하세요.")
    return pd.concat(frames, ignore_index=True)
