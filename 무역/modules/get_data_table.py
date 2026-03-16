import pandas as pd
from modules.claculator import value_per


def get_unit_value(row) -> float:
    if row["quantity_weight"] == 0:
        return 0
    if row["value"] == 0:
        return 0
    return row["value"] / row["quantity_weight"]


def get_base_data() -> pd.DataFrame:
    raw = pd.read_csv("modules/train.csv")
    raw["quantity_weight"] = raw.apply(
        lambda row: max(row["quantity"], 1) * max(row["weight"], 1), axis=1
    )
    raw["unit_value"] = raw.apply(get_unit_value, axis=1)
    raw["value_per_weight"] = raw.apply(
        lambda row: value_per(row["value"], row["quantity_weight"]), axis=1
    )
    raw["value_per_quantity"] = raw.apply(
        lambda row: value_per(row["value"], row["quantity"]), axis=1
    )
    target_mean = raw.groupby(["hs4", "item_id"])["unit_value"].mean()
    raw["hs4_encoded"] = raw.apply(
        lambda row: target_mean.loc[(row["hs4"], row["item_id"])], axis=1
    )
    raw["ym"] = pd.to_datetime(
        raw["year"].astype(str) + "-" + raw["month"].astype(str).str.zfill(2)
    )
    raw["ym_seq"] = raw.apply(
        lambda row: pd.to_datetime(
            (row["ym"] + pd.Timedelta(days=int(row["seq"] - 1)))
        ).strftime("%Y-%m-%d"),
        axis=1,
    )
    return raw


def VALUE_PIVOUT(data: pd.DataFrame) -> pd.DataFrame:
    """월별 총 무역량 피벗 테이블 생성"""
    result = data.groupby(["item_id", "hs4", "ym"], as_index=False)["value"].sum()

    # item_id × ym 피벗 (월별 총 무역량 매트릭스 생성)
    result = result.pivot(
        index=["item_id", "hs4"], columns="ym", values="value"
    ).fillna(0.0)
    return result


def QUANTITY_WEIGHT_PIVOUT(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """월별 총 무역량(수량*중량) 피벗 테이블 생성"""
    result = data.groupby(["item_id", "hs4", "ym"], as_index=False)[
        "quantity_weight"
    ].sum()

    result = result.pivot(
        index=["item_id", "hs4"], columns="ym", values="quantity_weight"
    ).fillna(0.0)
    return result
