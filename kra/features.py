"""예측에 쓰는 특성. 모든 특성은 해당 경주 '이전' 기록만으로 계산한다.

당일 마체중·주로상태·장구·배당률은 출전표 시점에 알 수 없으므로 특성에서 제외했다.
"""
import numpy as np
import pandas as pd

BASE_FEATURES = ["meet", "rcDist", "chulNo", "field_size", "age", "sex", "wgBudam", "rating"]
HORSE_FEATURES = [
    "h_starts", "h_win_rate", "h_top3_rate",
    "h_fin_pct_last", "h_fin_pct_avg3", "h_fin_pct_avg",
    "h_speed_z_last", "h_speed_z_avg3", "h_speed_z_avg", "h_speed_z_best",
    "h_days_since", "h_prev_wg", "budam_diff", "dist_diff",
    "h_dist_starts", "h_dist_fin_pct_avg",
]
PEOPLE_FEATURES = ["jk_starts", "jk_win_rate", "jk_top3_rate", "tr_starts", "tr_win_rate", "tr_top3_rate"]
# 경주 조건과 등급 변화 (class_diff < 0 이면 승급, prize_diff > 0 이면 더 큰 상금의 경주)
CLASS_FEATURES = ["class_level", "class_type", "prize", "budam_type", "foreign", "class_diff", "prize_diff", "h_prize_avg_diff"]
# 구간 기록으로 본 주행 습성: 초반 위치·초반 200m·막판 600m
PACE_FEATURES = ["h_early_pos_avg3", "h_s1f_z_avg3", "h_l3f_z_avg3", "h_l3f_z_best", "h_gain_avg3", "h_prev_wg_chg"]
# 기수-말 조합, 최근 1년 기수·조교사 성적
COMBO_FEATURES = ["same_jockey", "hj_starts", "hj_fin_pct_avg", "jk1y_starts", "jk1y_win_rate", "jk1y_top3_rate",
                  "tr1y_starts", "tr1y_win_rate", "tr1y_top3_rate"]
# 같은 경주 출전마들과 비교한 상대값(_rel)과 경주 내 순위(_rk, 0 = 가장 높음)
RELATIVE = ["h_speed_z_avg3", "h_fin_pct_avg3", "h_win_rate", "jk_win_rate", "wgBudam", "rating",
            "h_speed_z_best", "h_l3f_z_avg3", "h_early_pos_avg3", "jk1y_win_rate", "tr1y_win_rate", "h_prize_avg_diff"]
FEATURES = (BASE_FEATURES + HORSE_FEATURES + PEOPLE_FEATURES + CLASS_FEATURES + PACE_FEATURES + COMBO_FEATURES
            + [c + "_rel" for c in RELATIVE] + [c + "_rk" for c in RELATIVE])


def _prior_rates(df, key, prefix, k, days=None):
    """key(말/기수/조교사)별로 해당 날짜 이전까지의 출전수·승률·입상률.

    k 는 표본이 적을 때 평균 쪽으로 당기는 강도, days 를 주면 최근 그 일수 동안만 센다.
    """
    cols = ["n", "w", "t"]
    tmp = pd.DataFrame({
        key: df[key], "date": df["date"],
        "n": df["ord"].notna().astype(int), "w": df["win"].fillna(0), "t": df["top3"].fillna(0),
    }).dropna(subset=[key])
    daily = tmp.groupby([key, "date"], as_index=False).sum().sort_values("date").reset_index(drop=True)
    cum = daily.groupby(key)[cols].cumsum()
    prior = cum - daily[cols]
    if days:
        # 기준일(date - days) 이전까지의 누적을 빼서 최근 구간만 남긴다.
        old = pd.concat([daily[[key, "date"]], cum], axis=1)
        left = daily[[key]].assign(cut=daily["date"] - pd.Timedelta(days=days))
        old = pd.merge_asof(left, old, left_on="cut", right_on="date", by=key, allow_exact_matches=False)
        prior = prior - old[cols].fillna(0).to_numpy()
    win_avg, top3_avg = df["win"].mean(), df["top3"].mean()
    daily[f"{prefix}_starts"] = prior["n"]
    daily[f"{prefix}_win_rate"] = (prior["w"] + win_avg * k) / (prior["n"] + k)
    daily[f"{prefix}_top3_rate"] = (prior["t"] + top3_avg * k) / (prior["n"] + k)
    return df.merge(daily.drop(columns=cols), on=[key, "date"], how="left")


def _prior_mean(df, keys, col):
    """keys 로 묶은 그룹 안에서, 현재 행 이전 col 값들의 평균 (없으면 NaN)."""
    prev = df.groupby(keys, sort=False)[col].shift(1)
    by = [df[k] for k in keys]
    total = prev.fillna(0).groupby(by, sort=False).cumsum()
    count = prev.notna().astype(int).groupby(by, sort=False).cumsum()
    return total / count.where(count > 0)


def _last3_mean(g, col):
    return pd.concat([g[col].shift(i) for i in (1, 2, 3)], axis=1).mean(axis=1)


def _prior_best(df, g, col):
    best = g[col].shift(1).fillna(-np.inf).groupby(df["hrNo"], sort=False).cummax()
    return best.where(best > -np.inf)


def _zscore(df, value):
    """경마장·거리별 평균 대비 표준점수."""
    grp = value.groupby([df["meet"], df["rcDist"]])
    return (value - grp.transform("mean")) / grp.transform("std")


def build_features(df):
    """성적 표(+ ord 가 비어 있는 출전 예정 행)를 받아 특성을 붙여 돌려준다."""
    df = df.sort_values(["date", "meet", "rcNo", "chulNo"]).reset_index(drop=True)
    df["race_id"] = (df["meet"].astype(str) + "_" + df["date"].dt.strftime("%Y%m%d")
                     + "_" + df["rcNo"].astype(int).astype(str))
    df["field_size"] = df.groupby("race_id")["chulNo"].transform("size")
    df["win"] = (df["ord"] == 1).astype(float).where(df["ord"].notna())
    df["top3"] = (df["ord"] <= 3).astype(float).where(df["ord"].notna())
    # 0 = 1착, 1 = 꼴찌
    span = (df["field_size"] - 1).clip(lower=1)
    df["fin_pct"] = (df["ord"] - 1) / span
    df["speed_z"] = _zscore(df, df["rcDist"] / df["rcTime"])
    # 그날 그 경마장의 주로가 전체적으로 빨랐는지 느렸는지(주로 상태)를 빼 준다.
    df["speed_z"] -= df.groupby(["meet", "date"])["speed_z"].transform("mean")
    # 구간 기록: 값이 클수록 빠르도록 부호를 뒤집는다.
    df["s1f_z"] = -_zscore(df, df["s1f_time"])
    df["l3f_z"] = -_zscore(df, df["l3f_time"])
    df["early_pos"] = (df["s1f_ord"] - 1) / span
    df["gain"] = df["early_pos"] - df["fin_pct"]  # 초반 위치에서 얼마나 올라왔는가

    df = _prior_rates(df, "hrNo", "h", k=5)
    df = _prior_rates(df, "jkNo", "jk", k=30)
    df = _prior_rates(df, "trNo", "tr", k=30)
    df = _prior_rates(df, "jkNo", "jk1y", k=30, days=365)
    df = _prior_rates(df, "trNo", "tr1y", k=30, days=365)

    g = df.groupby("hrNo", sort=False)
    for c in ["fin_pct", "speed_z"]:
        df[f"h_{c}_last"] = g[c].shift(1)
        df[f"h_{c}_avg3"] = _last3_mean(g, c)
        df[f"h_{c}_avg"] = _prior_mean(df, ["hrNo"], c)
    df["h_speed_z_best"] = _prior_best(df, g, "speed_z")
    df["h_days_since"] = g["date"].diff().dt.days
    df["h_prev_wg"] = g["wgHr"].shift(1)
    df["h_prev_wg_chg"] = g["wg_chg"].shift(1)
    df["budam_diff"] = df["wgBudam"] - g["wgBudam"].shift(1)
    df["dist_diff"] = df["rcDist"] - g["rcDist"].shift(1)

    df["class_diff"] = df["class_level"] - g["class_level"].shift(1)
    df["prize_diff"] = df["prize"] - g["prize"].shift(1)
    df["h_prize_avg_diff"] = df["prize"] - _prior_mean(df, ["hrNo"], "prize")

    df["h_early_pos_avg3"] = _last3_mean(g, "early_pos")
    df["h_s1f_z_avg3"] = _last3_mean(g, "s1f_z")
    df["h_l3f_z_avg3"] = _last3_mean(g, "l3f_z")
    df["h_l3f_z_best"] = _prior_best(df, g, "l3f_z")
    df["h_gain_avg3"] = _last3_mean(g, "gain")

    gd = df.groupby(["hrNo", "rcDist"], sort=False)
    df["h_dist_starts"] = gd.cumcount()
    df["h_dist_fin_pct_avg"] = _prior_mean(df, ["hrNo", "rcDist"], "fin_pct")

    prev_jk = g["jkNo"].shift(1)
    df["same_jockey"] = (df["jkNo"] == prev_jk).astype(float).where(prev_jk.notna())
    df["hj_starts"] = df.groupby(["hrNo", "jkNo"], sort=False).cumcount()
    df["hj_fin_pct_avg"] = _prior_mean(df, ["hrNo", "jkNo"], "fin_pct")

    by_race = df.groupby("race_id")
    for c in RELATIVE:
        df[c + "_rel"] = df[c] - by_race[c].transform("mean")
        df[c + "_rk"] = (by_race[c].rank(ascending=False) - 1) / span
    return df
