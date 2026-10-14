"""1착 확률·3착 이내 확률 모델을 학습하고 검증 결과를 출력한다.

    python train.py

가장 최근 20% 기간을 검증용으로 떼어 평가한 뒤, 전체 데이터로 다시 학습해 models/ 에 저장한다.
"""
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from kra.api import ROOT
from kra.data import load_results
from kra.features import FEATURES, build_features

MODELS = ROOT / "models"
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=50,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1)


def top_pick(va, score, lowest=False):
    """경주마다 score 가 가장 좋은 말 1두를 골랐을 때의 (1착 적중률, 3착내 적중률, 경주수)."""
    v = va.dropna(subset=[score])
    by_race = v.groupby("race_id")[score]
    picks = v.loc[by_race.idxmin() if lowest else by_race.idxmax()]
    return picks["win"].mean(), picks["top3"].mean(), len(picks)


def main():
    df = build_features(load_results())
    dates = sorted(df["date"].unique())
    cut = dates[int(len(dates) * 0.8)]
    tr, va = df[df["date"] < cut], df[df["date"] >= cut].copy()
    print(f"학습 {len(tr):,}행 ({tr['race_id'].nunique():,}경주) ~{pd.Timestamp(cut):%Y-%m-%d} "
          f"/ 검증 {len(va):,}행 ({va['race_id'].nunique():,}경주)")

    MODELS.mkdir(exist_ok=True)
    for target in ["win", "top3"]:
        m = lgb.train(PARAMS, lgb.Dataset(tr[FEATURES], tr[target]), 3000,
                      valid_sets=[lgb.Dataset(va[FEATURES], va[target])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        p = m.predict(va[FEATURES], num_iteration=m.best_iteration)
        va[f"p_{target}"] = p
        base = log_loss(va[target], [tr[target].mean()] * len(va))
        print(f"\n[{target}] 트리 {m.best_iteration}개 | logloss {log_loss(va[target], p):.4f} "
              f"(평균만 찍을 때 {base:.4f}) | AUC {roc_auc_score(va[target], p):.4f}")
        imp = pd.Series(m.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
        print("  중요 특성:", ", ".join(imp.index[:8]))
        final = lgb.train(PARAMS, lgb.Dataset(df[FEATURES], df[target]), m.best_iteration)
        # LightGBM 의 save_model 은 윈도우에서 한글 경로에 쓰지 못한다.
        (MODELS / f"{target}.txt").write_text(final.model_to_string(), encoding="utf-8")

    print("\n경주마다 1두를 골랐을 때 (검증 기간)")
    rows = [("모델 1순위", top_pick(va, "p_win")),
            ("인기 1위(최저 단승배당)", top_pick(va, "winOdds", lowest=True))]
    for name, (w, t, n) in rows:
        print(f"  {name:<20} 1착 {w:6.1%} | 3착내 {t:6.1%} | {n:,}경주")
    print(f"  {'무작위':<20} 1착 {(1 / va.groupby('race_id').size()).mean():6.1%}")
    print(f"\n모델 저장: {MODELS}")


if __name__ == "__main__":
    main()
