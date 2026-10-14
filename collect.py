"""경주 성적을 월 단위로 내려받아 data/history 에 정리해 저장한다.

    python collect.py                  # 2020-01 부터 이번 달까지 (이미 받은 달은 건너뜀)
    python collect.py --start 2023-01
    python collect.py --recent         # 지난달과 이번 달만 (자동 갱신용)
    python collect.py --probe          # 성적·출전표 원본 1건씩 출력 (필드명 확인용)
"""
import argparse
import json

from kra.api import ENTRY_PATH, MEETS, RESULT_PATH, ApiError, fetch_all, today
from kra.data import HISTORY, RAW_RESULTS, save_month


def months(start, end):
    y, m = start
    while (y, m) <= end:
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def prev_month(y, m):
    return (y, m - 1) if m > 1 else (y - 1, 12)


def probe():
    y, m = prev_month(today().year, today().month)
    for name, path in [("경주성적정보", RESULT_PATH), ("출전표", ENTRY_PATH)]:
        try:
            items = fetch_all(path, meet=1, rc_month=f"{y}{m:02d}")
        except ApiError as e:
            print(f"--- {name}: 실패 ---\n{e}")
            continue
        print(f"--- {name}: {len(items)}건 ---")
        if items:
            print(json.dumps(items[0], ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-01", help="시작 월 YYYY-MM")
    ap.add_argument("--recent", action="store_true")
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()
    if args.probe:
        return probe()

    RAW_RESULTS.mkdir(parents=True, exist_ok=True)
    now = (today().year, today().month)
    start = prev_month(*now) if args.recent else tuple(map(int, args.start.split("-")))
    # 이번 달은 경주가 계속 추가되므로 매번 다시 받는다. 월초에는 지난달 마지막 경주도 다시 확인한다.
    refresh = {now} | ({prev_month(*now)} if today().day <= 7 else set())
    for y, m in months(start, now):
        for meet, name in MEETS.items():
            month = f"{y}{m:02d}"
            raw = RAW_RESULTS / f"{meet}_{month}.json"
            hist = HISTORY / f"{meet}_{month}.csv.gz"
            if (y, m) in refresh or not (raw.exists() or hist.exists()):
                items = fetch_all(RESULT_PATH, meet=meet, rc_month=month)
                raw.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
                print(f"{y}-{m:02d} {name}: {len(items)}건")
            elif hist.exists():
                continue
            save_month(meet, month, json.loads(raw.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
