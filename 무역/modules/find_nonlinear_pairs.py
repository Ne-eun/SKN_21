import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from tqdm import tqdm


def find_nonlinear_comovement_pairs(pivot, mi_threshold=0.1, max_lag=12):
    """
    시계열 데이터프레임에서 상호 정보량을 사용하여
    비선형 공행성 쌍을 추출합니다.

    Args:
        pivot (pd.DataFrame): 날짜가 컬럼, 품목이 인덱스인 시계열 데이터프레임.
        mi_threshold (float): 상호 정보량(MI) 임계값.
    """
    # 품목 목록 추출
    items = pivot.index.to_list()
    months = pivot.columns.to_list()
    n_months = len(months)

    results = []

    # 모든 품목 쌍에 대해 반복
    for i, leader in tqdm(enumerate(items), desc="공행성 쌍 탐색"):
        leader_id, leader_hs4 = leader
        x = pivot.loc[leader].values.astype(float).reshape(-1, 1)

        for follower in items:
            if follower == leader:
                continue

            follower_id, follower_hs4 = follower
            y = pivot.loc[follower].values.astype(float)

            best_lag = 0
            best_mi = 0.0

            for lag in range(1, max_lag + 1):
                if n_months <= lag:
                    continue

                mi_value = mutual_info_regression(x[:-lag], y[lag:], random_state=42)[0]

                if mi_value > best_mi:
                    best_lag = lag
                    best_mi = mi_value

            if best_mi >= mi_threshold:
                results.append(
                    {
                        "선행품목": leader_id,
                        "후행품목": follower_id,
                        "선행품목hs4": leader_hs4,
                        "후행품목hs4": follower_hs4,
                        "MI": best_mi,
                        "최적지연기간": best_lag,
                    }
                )

    return results


if __name__ == "__main__":
    import pandas as pd
    from get_data_table import VALUE_PIVOUT, get_base_data
    from differencing import log_difference_transform
    from sklearn.preprocessing import robust_scale

    data = get_base_data()
    pivot = VALUE_PIVOUT(data)
    pivot = log_difference_transform(pivot)

    data_array = pivot.values
    scaled_array = robust_scale(data_array, axis=1)
    pivot_scaled = pd.DataFrame(scaled_array, index=pivot.index, columns=pivot.columns)

    # 비선형 공행성 쌍 탐색
    nonlinear_pairs = find_nonlinear_comovement_pairs(
        pivot_scaled, mi_threshold=0.25, max_lag=12
    )
    nonlinear_pairs_df = pd.DataFrame(nonlinear_pairs)
    print(nonlinear_pairs_df)
    nonlinear_pairs_df.to_csv("./datas/nonlinear_comovement_pairs.csv", index=False)
