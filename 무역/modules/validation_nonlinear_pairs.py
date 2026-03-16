import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np


def validate_nonlinear_pairs(
    nonlinear_pairs_df, pivot_df, window=12, max_lag=6, alpha=0.05, min_corr_std=0.15
):
    """
    비선형 공행성 쌍 목록에 대해 롤링 윈도우 안정성과 그랜저 인과성을 검증합니다.

    Args:
        nonlinear_pairs_df (pd.DataFrame): 2차 추출(MI)된 쌍 목록 (선행품목, 후행품목 컬럼 필수).
        pivot_df (pd.DataFrame): 품목 ID가 인덱스, 날짜가 컬럼인 원 무역량 데이터.
        window (int): 롤링 윈도우 기간 (개월).
        max_lag (int): 그랜저 인과성 최대 지연 기간.
        alpha (float): 통계적 유의성 임계값 (P-value).
        min_corr_std (float): 롤링 상관계수 표준편차 최대 허용치 (안정성 기준).

    Returns:
        pd.DataFrame: 최종 검증을 통과한 비선형 공행성 쌍 목록.
    """
    # 1. 데이터 정상화 (로그 차분) - 안정성과 인과성 분석의 기반
    # axis=1 (컬럼 방향)으로 차분, 첫 번째 컬럼(NaN)은 제거
    df_stationary = np.log(pivot_df).diff(axis=1).iloc[:, 1:]

    final_pairs = []

    for _, row in nonlinear_pairs_df.iterrows():
        leader = row["선행품목"]
        follower = row["후행품목"]

        try:
            # -------------------------------------------
            # A. 롤링 윈도우 안정성 필터 (Rolling Window Analysis)
            # -------------------------------------------
            # 정상화된 데이터 간의 롤링 상관계수 계산
            rolling_corr = (
                df_stationary.loc[leader]
                .rolling(window=window)
                .corr(df_stationary.loc[follower])
            )
            std_dev = rolling_corr.std()

            # 안정성 검증: 관계의 변동성이 낮아야 함
            if std_dev >= min_corr_std:
                continue  # 불안정한 쌍은 제외

            # -------------------------------------------
            # B. 그랜저 인과성 검증 (Granger Causality Test)
            # -------------------------------------------
            # 테스트 데이터 준비 (컬럼: follower, leader / 행: 시간)
            test_data = pd.DataFrame(
                {
                    "follower": df_stationary.loc[follower].values,
                    "leader": df_stationary.loc[leader].values,
                }
            )

            # 그랜저 인과성 테스트 (leader가 follower를 예측하는지)
            results = grangercausalitytests(
                test_data[["follower", "leader"]], max_lag, verbose=False
            )

            # F-test의 최소 P-value 추출
            min_p_value = min(
                [results[lag][0]["ssr_ftest"][1] for lag in range(1, max_lag + 1)]
            )

            # 인과성 검증: 통계적으로 유의미한 선후행 관계가 존재해야 함
            if min_p_value < alpha:
                row["Rolling_Corr_Std"] = std_dev
                row["Granger_PValue"] = min_p_value
                final_pairs.append(row)

        except Exception as e:
            # 데이터 누락, 길이 불일치 등의 예외 발생 시 해당 쌍 건너뛰기
            continue

    return pd.DataFrame(final_pairs)
