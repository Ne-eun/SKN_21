import numpy as np
import pandas as pd
from tqdm import tqdm


def calculate_values(leaders, followers, t, lag):
    b_t = followers[t]
    b_t_1 = followers[t - 1]
    a_t_lag = leaders[t - lag]

    b_ma3 = np.mean(followers[t - 2 : t + 1])  # 3개월 이동평균
    b_std3 = np.std(followers[t - 2 : t + 1])  # 3개월 표준편차

    a_ma3 = np.mean(leaders[t - lag - 2 : t - lag + 1])
    a_std3 = np.std(leaders[t - lag - 2 : t - lag + 1])

    # 비율 특성
    b_growth = (followers[t] - followers[t - 1]) / max(followers[t - 1], 1)
    a_growth = (leaders[t - lag] - leaders[t - lag - 1]) / max(leaders[t - lag - 1], 1)
    return {
        "현재총무역량": b_t,
        "직전달총무역량": b_t_1,
        "선행품목lag무역량": a_t_lag,
        "후행품목3개월이동평균": b_ma3,
        "후행품목3개월표준편차": b_std3,
        "선행품목3개월이동평균": a_ma3,
        "선행품목3개월표준편차": a_std3,
        "후행품목성장률": b_growth,
        "선행품목성장률": a_growth,
    }


def calculate_weight_quantity(leaders, followers, t, lag):
    return {"현재총무역량무게": followers[t], "선행품목lag무역량무게": leaders[t - lag]}


def get_property(data, row):
    선행품목_hs4_encoded = data.query(
        "item_id == @row.선행품목 and hs4 == @row.선행품목hs4"
    )
    후행품목_hs4_encoded = data.query(
        "item_id == @row.후행품목 and hs4 == @row.후행품목hs4"
    )
    return {
        "최대상관계수": row.상관계수,
        "최적지연기간": row.최적지연기간,
        "후행품목hs4_encoded": 후행품목_hs4_encoded["hs4_encoded"].values[0],
        "선행품목hs4_encoded": 선행품목_hs4_encoded["hs4_encoded"].values[0],
    }


def create_train_set(
    value_pivot, weight_quantity_pivot, pairs, raw_data
) -> tuple[pd.DataFrame, pd.Series]:
    months = value_pivot.columns.to_list()
    features = []
    targets = []

    for row in tqdm(pairs.itertuples(index=False), desc="feature 생성"):
        leader = (row.선행품목, row.선행품목hs4)
        follower = (row.후행품목, row.후행품목hs4)
        lag = row.최적지연기간

        if leader not in value_pivot.index or follower not in value_pivot.index:
            continue
        # 월별 무역량
        leader_values = value_pivot.loc[leader].values
        follower_values = value_pivot.loc[follower].values

        # 월별 무역량의 중량
        leader_weight = weight_quantity_pivot.loc[leader].values
        follower_weight = weight_quantity_pivot.loc[follower].values

        for t in range(lag + 2, len(months) - 1):
            calculated_values_features = calculate_values(
                leader_values, follower_values, t, lag
            )
            property_features = get_property(raw_data, row)
            weight_quantity_features = calculate_weight_quantity(
                leader_weight, follower_weight, t, lag
            )

            features.append(
                {
                    **calculated_values_features,
                    **property_features,
                    **weight_quantity_features,
                }
            )
            targets.append(follower_values[t + 1])
    return (pd.DataFrame(features), pd.Series(targets))


if __name__ == "__main__":
    import pandas as pd
    from get_data_table import VALUE_PIVOUT, QUANTITY_WEIGHT_PIVOUT, get_base_data

    pairs = pd.read_csv("./datas/comovement_pairs.csv")
    data = get_base_data()
    value_pivot = VALUE_PIVOUT(data)
    value_pivot = value_pivot.drop(columns=[pd.to_datetime("2025-07-01")])

    weight_quantity_pivot = QUANTITY_WEIGHT_PIVOUT(data)
    weight_quantity_pivot = weight_quantity_pivot.drop(
        columns=[pd.to_datetime("2025-07-01")]
    )

    X_test, y_test = create_train_set(value_pivot, weight_quantity_pivot, pairs, data)
    X_test.to_csv("./datas/X_test.csv", index=False)
    y_test.to_csv("./datas/y_test.csv", index=False)
    print("Test set created and saved.")
