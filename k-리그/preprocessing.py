import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import tqdm


def get_field_zone(x, y):
    zone_x = "left" if x < 0.5 else "right"
    zone_y = "low" if y < 0.34 else ("high" if y > 0.66 else "mid")
    return f"{zone_x}_{zone_y}"


def preprocessing(df: pd.DataFrame, filename=None) -> pd.DataFrame:
    # 정규화
    field_x, field_y = 105, 68
    df["nor_start_x"] = df["start_x"] / field_x
    df["nor_start_y"] = df["start_y"] / field_y
    df["nor_end_x"] = df["end_x"] / field_x
    df["nor_end_y"] = df["end_y"] / field_y

    # 이동 거리
    df["dx"] = (df["end_x"] - df["start_x"]) / field_x
    df["dy"] = (df["end_y"] - df["start_y"]) / field_y
    df["movement_distance"] = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2)

    # 필드 구역
    df["field_zone"] = df.apply(
        lambda row: get_field_zone(row["nor_start_x"], row["nor_start_y"]), axis=1
    )

    # 시간 차이
    df["duration"] = df.time_seconds.diff().fillna(0)

    # 공의 속도
    df["ball_velocity"] = df["movement_distance"] / (df["duration"] + 1e-6)

    # 팀
    df["is_home"] = df.is_home.apply(lambda x: 1 if x else 0)

    # 이벤트 직전 이동 속도
    df["prev_ball_velocity"] = df.groupby("player_id")["ball_velocity"].shift(1)

    # 골대와의 거리
    df["goal_distance"] = np.sqrt(
        (df["nor_start_x"] - 1) ** 2 + (df["nor_start_y"] - 0.5) ** 2
    )

    # 선수별 이벤트 성공률
    df["success_rate"] = df.groupby("player_id")["result_name"].transform(
        lambda x: (x == "Successful").mean()
    )

    # 누적 이동 거리 기반 피로도
    df["cumulative_movement_distance"] = df.groupby("player_id")[
        "movement_distance"
    ].cumsum()

    if filename:
        df.to_csv(filename, index=False)
        print(f"Saved preprocessed data to {filename}")
    return df
