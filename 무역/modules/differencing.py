import numpy as np
import pandas as pd


def log_difference_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    주어진 무역 데이터프레임에 대해 로그 차분 변환을 수행합니다.

    매개변수:
    df (pd.DataFrame): 품목을 행(index)으로, 날짜를 열(columns)로 하는 무역 데이터프레임.

    반환값:
    pd.DataFrame: 로그 차분이 적용된 데이터프레임.
    """

    log_transformed_data = np.log1p(df)  # ln(V_t + 𝜖) 변환

    # 로그 차분 계산: ln(V_t + 𝜖) - ln(V_{t-1} + 𝜖)
    # np.diff는 axis=1 (열 방향)으로 차분하면 컬럼 개수가 1개 줄어듦
    diff_array = np.diff(log_transformed_data, axis=1)

    # DataFrame 생성 (첫 번째 날짜는 차분 불가능하므로 제외)
    diff_data = pd.DataFrame(
        diff_array,
        index=df.index,  # 품목 인덱스 유지
        columns=df.columns[1:],  # 첫 날짜 제외한 컬럼 사용
    )
    return diff_data


def zscore_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    품목별로 z-score 정규화를 수행합니다.

    매개변수:
    df (pd.DataFrame): 품목을 행(index)으로, 날짜를 열(columns)로 하는 무역 데이터프레임.

    반환값:
    pd.DataFrame: z-score 정규화된 데이터프레임.
    """
    # 각 행(품목)별로 평균과 표준편차 계산
    mean = df.mean(axis=1)
    std = df.std(axis=1)

    # z-score 계산: (x - μ) / σ
    # 표준편차가 0인 경우(모든 값이 같은 경우) 0으로 설정
    zscore_df = df.sub(mean, axis=0).div(std.replace(0, 1), axis=0)

    return zscore_df


if __name__ == "__main__":
    from get_data_table import VALUE_PIVOUT, get_base_data

    data = get_base_data()
    pivot = VALUE_PIVOUT(data)

    diff_data = log_difference_transform(pivot)
    print(diff_data.isna().sum())
