import scipy.stats as stats
from tqdm import tqdm
import numpy as np
import pandas as pd


def calculate_pearson(x, y) -> tuple[float, float]:
    """Pearson 상관관계 p-값 계산"""
    try:
        if np.std(x) == 0 or np.std(y) == 0:
            return 0.0, 0.0
        statistic, p_value = stats.pearsonr(x, y)
        return float(statistic), float(p_value)  # type: ignore

    except Exception as e:
        print(f"⚠️ pearson_test 에러: {e}")
        return 0.0, 1.0


def find_comovement_pairs(pivot, max_lag=6, min_nonzero=3) -> pd.DataFrame:
    """
    상호 상관 분석으로
    공행성 쌍을 탐색하는 함수

    Args:
        max_lag: 최대 지연 기간 (1~max_lag개월까지 탐색)
        corr_threshold: 상관계수 임계값 (절댓값 기준)
    """
    items = pivot.index.to_list()
    months = pivot.columns.to_list()
    n_months = len(months)

    results = []

    print(f"설정값:")
    print(f"- 최대 지연 기간: {max_lag}개월")

    for i, leader in tqdm(enumerate(items), desc="공행성 쌍 탐색"):
        leader_id, leader_hs4 = leader

        x = pivot.loc[leader].values.astype(float)

        for follower in items:
            if follower == leader:
                continue

            follower_id, follower_hs4 = follower

            y = pivot.loc[follower].values.astype(float)
            if np.count_nonzero(y) < min_nonzero:
                continue
            ccf_raw = np.correlate(x, y, mode="full") / n_months
            best_lag_idx = np.argmax(np.abs(ccf_raw))
            best_lag = best_lag_idx - (n_months - 1)

            if abs(best_lag) <= max_lag and best_lag < -1:
                corr, p_value = calculate_pearson(
                    x[:best_lag],
                    y[-best_lag:],
                )
                if abs(corr) >= 0.45 and p_value < 0.01:
                    results.append(
                        {
                            "선행품목": leader_id,
                            "후행품목": follower_id,
                            "선행품목hs4": leader_hs4,
                            "후행품목hs4": follower_hs4,
                            "최적지연기간": abs(best_lag),
                            "상관계수": ccf_raw[best_lag_idx],
                        }
                    )

    pairs = pd.DataFrame(results)

    if len(pairs) > 0:
        print(f"\n📊 검정 결과:")
        print(f"- 총 발견된 공행성 쌍: {len(pairs)}")

    return pairs


if __name__ == "__main__":
    import pandas as pd
    from get_data_table import VALUE_PIVOUT, get_base_data
    from differencing import zscore_normalize, log_difference_transform
    from sklearn.preprocessing import robust_scale

    data = get_base_data()
    pivot = VALUE_PIVOUT(data)
    pivot = log_difference_transform(pivot)

    data_array = pivot.values
    scaled_array = robust_scale(data_array, axis=1)
    pivot_scaled = pd.DataFrame(scaled_array, index=pivot.index, columns=pivot.columns)

    pairs = find_comovement_pairs(pivot, max_lag=12, min_nonzero=12)
    print(pairs.head())
    pairs.to_csv("datas/comovement_pairs.csv", index=False)
    print("공행성 쌍 Saved.")
