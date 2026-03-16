import pandas as pd
import numpy as np
from modules.create_features import (
    calculate_values,
    calculate_weight_quantity,
    get_property,
)


def create_submit(
    value_pivot, weight_quantity_pivot, pairs, raw_data, model, scaler
) -> pd.DataFrame:
    months = value_pivot.columns.to_list()
    result = []

    for row in pairs.itertuples(index=False):
        leader = (row.선행품목, row.선행품목hs4)
        follower = (row.후행품목, row.후행품목hs4)
        lag = row.최적지연기간

        if leader not in value_pivot.index or follower not in value_pivot.index:
            continue

        # 월별 무역량
        leader_values = value_pivot.loc[leader].values.astype(float)
        follower_values = value_pivot.loc[follower].values.astype(float)

        # 월별 무역량의 중량
        leader_weight = weight_quantity_pivot.loc[leader].values.astype(float)
        follower_weight = weight_quantity_pivot.loc[follower].values.astype(float)

        t = len(months) - 1  # 마지막 달에 대한 예측
        calculated_values_features = calculate_values(
            leader_values, follower_values, t, lag
        )
        property_features = get_property(raw_data, row)
        weight_quantity_features = calculate_weight_quantity(
            leader_weight, follower_weight, t, lag
        )

        X_test = {
            **calculated_values_features,
            **property_features,
            **weight_quantity_features,
        }

        X_test = np.array([list(X_test.values())])
        X_test = scaler.transform(X_test)
        X_test = pd.DataFrame(
            X_test,
            columns=list(calculated_values_features.keys())
            + list(property_features.keys())
            + list(weight_quantity_features.keys()),
        )
        y_pred = model.predict(X_test)[0]

        y_pred = max(0.0, float(y_pred))
        y_pred = int(round(y_pred))

        result.append(
            {
                "leading_item_id": row.선행품목,
                "following_item_id": row.후행품목,
                "value": y_pred,
            }
        )

    return pd.DataFrame(result)
