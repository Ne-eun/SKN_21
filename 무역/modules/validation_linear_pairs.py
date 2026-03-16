import pandas as pd
from statsmodels.tsa.stattools import coint, grangercausalitytests
import os
import contextlib


# -------------------------------------------------------------
# 1. 공적분 검증 필터 (장기 균형 관계 확인)
# -------------------------------------------------------------
def filter_by_cointegration(pairs_df, pivot_df, alpha=0.05):
    """
    공적분 테스트를 사용하여 장기적 균형 관계가 없는 쌍을 필터링합니다.
    pivot_df: 원본 무역량 데이터 (품목이 인덱스, 날짜가 컬럼)
    """
    valid_pairs = []
    for _, row in pairs_df.iterrows():

        x = pivot_df.loc[(row.선행품목, row.선행품목hs4)].values
        y = pivot_df.loc[(row.후행품목, row.후행품목hs4)].values

        # coint 함수: t-통계량과 p-value 반환
        _, p_value, _ = coint(x, y, autolag="aic")

        if p_value < alpha:
            row["공적분p"] = p_value
            valid_pairs.append(row)

    return pd.DataFrame(valid_pairs)


# -------------------------------------------------------------
# 2. 롤링 윈도우 안정성 필터 (시간적 안정성 확인)
# -------------------------------------------------------------
def filter_by_rolling_correlation(
    candidate_pairs_df, pivot_df, window=12, min_corr_std=0.2
):
    """
    롤링 윈도우 상관계수 분석을 사용하여 관계 안정성이 낮은 쌍을 필터링합니다.
    pivot_df: 원본 무역량 데이터 (품목이 인덱스, 날짜가 컬럼) -> 로그 차분 필요
    """
    stable_pairs = []

    for _, row in candidate_pairs_df.iterrows():
        lag = row["최적지연기간"]

        x = pivot_df.loc[(row.선행품목, row.선행품목hs4)].values
        y = pivot_df.loc[(row.후행품목, row.후행품목hs4)].values

        # 롤링 상관계수 계산
        rolling_corr = (
            pd.Series(x[:-lag]).rolling(window=window).corr(pd.Series(y[lag:]))
        )

        # 롤링 상관계수의 표준편차(변동성) 계산
        std_dev = rolling_corr.std()

        if std_dev < min_corr_std:
            row["롤링상관계수표준편차"] = std_dev
            stable_pairs.append(row)

    return pd.DataFrame(stable_pairs)


# -------------------------------------------------------------
# 3. 그랜저 인과성 필터 (선후행 관계 유의성 확인)
# -------------------------------------------------------------
def filter_by_granger_causality(candidate_pairs_df, pivot_df, max_lag=12, alpha=0.05):
    """
    그랜저 인과성 테스트를 사용하여 선후행 관계가 유의미하지 않은 쌍을 필터링합니다.
    pivot_df: 원본 무역량 데이터 (품목이 인덱스, 날짜가 컬럼) -> 로그 차분 필요
    """
    causal_pairs = []

    for _, row in candidate_pairs_df.iterrows():
        # statsmodels는 컬럼이 변수(품목), 행이 시간(날짜)인 형태를 선호합니다.
        # 데이터: [[follower], [leader]] 형태로 그랜저 테스트 수행
        follower_values = pivot_df.loc[(row.선행품목, row.선행품목hs4)].values
        leader_values = pivot_df.loc[(row.후행품목, row.후행품목hs4)].values

        # 상수값 체크: 표준편차가 0이거나 매우 작으면 건너뜀
        # 그랜저 테스트는 변동성이 있는 시계열에만 적용 가능
        if follower_values.std() < 1e-10 or leader_values.std() < 1e-10:
            continue

        test_data = pd.DataFrame(
            {
                "follower": follower_values,
                "leader": leader_values,
            }
        )

        try:
            # 테스트 결과에서 P-value 추출
            # F-test 결과의 P-value를 사용
            with open(os.devnull, "w") as f:  # 계산 결과 출력 억제
                with contextlib.redirect_stdout(f):  # 출력 억제 2
                    results = grangercausalitytests(
                        test_data[["follower", "leader"]], max_lag
                    )

            # 최적 lag의 P-value 중 가장 작은 값 사용
            min_p_value = min(
                [results[lag][0]["ssr_ftest"][1] for lag in range(1, max_lag + 1)]
            )

            if min_p_value < alpha:
                row["그랜저인과성p"] = min_p_value
                causal_pairs.append(row)
        except Exception as e:
            # 그랜저 테스트 실패 시 해당 쌍은 건너뜀 (상수값 외 다른 문제 대비)
            continue

    return pd.DataFrame(causal_pairs)


# -------------------------------------------------------------
# 4. 최종 검증 통합 실행
# -------------------------------------------------------------
def get_final_validation(pairs, pivot, preprocessed_pivot):
    """
    세 가지 필터를 순차적으로 적용하여 최종 공행성 쌍 목록을 추출합니다.
    """

    # # 1. 공적분 검증: 장기 균형 관계 확인 (가장 엄격한 필터 중 하나)
    # print("1. 공적분 검증 시작...")
    # validated_coint = filter_by_cointegration(pairs, pivot, alpha=0.1)
    # print(f"   -> 공적분 통과 쌍: {len(validated_coint)}개")

    # # 2. 롤링 윈도우 분석: 시간적 안정성 확인 (분산이 낮은 쌍 선별)
    # print("2. 롤링 윈도우 분석 시작...")
    # validated_rolling = filter_by_rolling_correlation(
    #     validated_coint, preprocessed_pivot, window=12, min_corr_std=0.25
    # )
    # print(f"   -> 안정성 통과 쌍: {len(validated_rolling)}개")

    # 3. 그랜저 인과성 검증: 선후행 관계 유의성 확인
    print("3. 그랜저 인과성 검증 시작...")
    final_pairs = filter_by_granger_causality(
        pairs, preprocessed_pivot, max_lag=12, alpha=0.05
    )
    print(f"   -> 최종 선후행 쌍: {len(final_pairs)}개")

    return final_pairs


if __name__ == "__main__":
    import pandas as pd
    from get_data_table import VALUE_PIVOUT, get_base_data
    from differencing import log_difference_transform
    from sklearn.preprocessing import robust_scale

    data = get_base_data()
    pivot = VALUE_PIVOUT(data)
    preprocessed_pivot = log_difference_transform(pivot)
    pairs = pd.read_csv("./datas/comovement_pairs.csv")

    valid_pairs = get_final_validation(pairs, pivot, preprocessed_pivot)
    valid_pairs.to_csv("./datas/validated_linear_comovement_pairs.csv", index=False)
    print(valid_pairs.head())
