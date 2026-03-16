import scipy.stats as stats
from scipy.signal import find_peaks
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
import numpy as np
import pandas as pd


def calculate_pearson(x, y) -> tuple[float, float]:
    """Pearson 상관관계 p-값 계산"""
    try:
        if np.std(x) == 0 or np.std(y) == 0:
            return 0.0, 0.0
        statistic, p_value = stats.pearsonr(x, y)
        return float(statistic), float(p_value)
    except Exception as e:
        print(f"⚠️ pearson_test 에러: {e}")
        return 0.0, 1.0


def calculate_multiple_correlations(x, y) -> dict:
    """다양한 상관계수 계산 (선형 + 비선형 관계 포착)"""
    results = {}

    # Pearson (선형 관계)
    pearson_corr, pearson_p = calculate_pearson(x, y)
    results["pearson_corr"] = pearson_corr
    results["pearson_p"] = pearson_p

    # Spearman (단조 관계)
    try:
        # 상수 배열 체크 (Spearman도 동일한 문제 발생)
        if len(np.unique(x)) <= 1 or len(np.unique(y)) <= 1:
            results["spearman_corr"] = 0.0
            results["spearman_p"] = 1.0
        else:
            spearman_result = spearmanr(x, y)
            results["spearman_corr"] = float(spearman_result[0])  # type: ignore
            results["spearman_p"] = float(spearman_result[1])  # type: ignore
    except:
        results["spearman_corr"] = 0.0
        results["spearman_p"] = 1.0

    # Mutual Information (비선형 관계)
    try:
        if len(x) > 3:  # 최소 데이터 필요
            mi = mutual_info_regression(x.reshape(-1, 1), y, random_state=42)[0]
            results["mi_score"] = float(mi)
        else:
            results["mi_score"] = 0.0
    except:
        results["mi_score"] = 0.0

    return results


def validate_with_rolling_window(x, y, window=6) -> dict:
    """Rolling window 상관계수로 안정성 검증"""
    if len(x) < window:
        return {"mean_corr": 0.0, "std_corr": 1.0, "stable": False}

    correlations = []
    for i in range(len(x) - window + 1):
        try:
            x_window = x[i : i + window]
            y_window = y[i : i + window]

            # 상수 배열 체크 (고유값이 1개 이하면 상수 배열)
            if len(np.unique(x_window)) <= 1 or len(np.unique(y_window)) <= 1:
                continue

            # 표준편차 체크 (추가 안전장치)
            if np.std(x_window) < 1e-10 or np.std(y_window) < 1e-10:
                continue

            corr_result = stats.pearsonr(x_window, y_window)
            corr = float(corr_result[0])  # type: ignore
            if not np.isnan(corr):
                correlations.append(corr)
        except:
            continue

    if len(correlations) == 0:
        return {"mean_corr": 0.0, "std_corr": 1.0, "stable": False}

    mean_corr = np.mean(correlations)
    std_corr = np.std(correlations)

    # 안정성: 평균이 높고 표준편차가 낮음
    stable = abs(mean_corr) >= 0.4 and std_corr < 0.25

    return {
        "mean_corr": float(mean_corr),
        "std_corr": float(std_corr),
        "stable": stable,
    }


def find_multiple_lags(ccf_raw, n_months, max_lag=6, prominence=0.2):
    """CCF에서 여러 유의미한 지연 찾기"""
    lags = np.arange(-n_months + 1, n_months)

    # 피크 탐지
    peaks, properties = find_peaks(
        np.abs(ccf_raw), prominence=prominence, distance=2  # 최소 2개월 간격
    )

    valid_lags = []
    for peak in peaks:
        lag = lags[peak]
        # 선행 관계만 (lag > 0 = y가 x보다 뒤처짐)
        if 0 < lag <= max_lag:
            valid_lags.append({"lag": int(lag), "ccf_value": float(ccf_raw[peak])})

    # CCF 값 기준 내림차순 정렬
    valid_lags.sort(key=lambda x: abs(x["ccf_value"]), reverse=True)

    return valid_lags


def check_sufficient_variation(series, min_cv=0.1):
    """충분한 변동성이 있는지 확인 (변동계수 기반)"""
    mean_val = np.mean(series)
    std_val = np.std(series)

    if abs(mean_val) < 1e-8:
        return False

    cv = std_val / abs(mean_val)
    return cv >= min_cv


def find_comovement_pairs_improved(
    pivot,
    max_lag=12,
    min_nonzero=12,
    pearson_threshold=0.45,
    spearman_threshold=0.4,
    mi_threshold=0.05,
    min_data_points=20,
    use_multiple_testing_correction=True,
    exclude_same_hs4=False,
) -> pd.DataFrame:
    """
    개선된 공행성 쌍 탐색 함수

    개선사항:
    1. 비선형 관계 포착 (Spearman, MI)
    2. 다중 지연 효과 탐지
    3. Rolling window 안정성 검증
    4. 다중 검정 보정
    5. 충분한 변동성 확인
    """
    items = pivot.index.to_list()
    months = pivot.columns.to_list()
    n_months = len(months)

    results = []
    p_values_for_correction = []

    print(f"설정값:")
    print(f"- 최대 지연 기간: {max_lag}개월")
    print(f"- Pearson 임계값: {pearson_threshold}")
    print(f"- Spearman 임계값: {spearman_threshold}")
    print(f"- MI 임계값: {mi_threshold}")
    print(f"- 최소 데이터 포인트: {min_data_points}")

    for i, leader in tqdm(enumerate(items), desc="공행성 쌍 탐색"):
        leader_id, leader_hs4 = leader
        x = pivot.loc[leader].values.astype(float)

        # 선행 품목의 변동성 확인
        if not check_sufficient_variation(x):
            continue

        for follower in items:
            if follower == leader:
                continue

            follower_id, follower_hs4 = follower

            # 동일 HS4 제외 옵션
            if exclude_same_hs4 and leader_hs4 == follower_hs4:
                continue

            y = pivot.loc[follower].values.astype(float)

            # 후행 품목 검증
            if np.count_nonzero(y) < min_nonzero:
                continue

            if not check_sufficient_variation(y):
                continue

            # CCF 계산 (표준화된 상호상관)
            ccf_raw = np.correlate(x, y, mode="full") / n_months

            # 여러 유의미한 지연 찾기
            valid_lags = find_multiple_lags(
                ccf_raw, n_months, max_lag=max_lag, prominence=0.15
            )

            # 각 유의미한 지연에 대해 검증
            for lag_info in valid_lags[:3]:  # 상위 3개까지만
                lag = lag_info["lag"]

                # 충분한 데이터 포인트 확인
                if n_months - lag < min_data_points:
                    continue

                # 지연 적용
                x_lagged = x[:-lag]
                y_lagged = y[lag:]

                # 다양한 상관계수 계산
                corr_results = calculate_multiple_correlations(x_lagged, y_lagged)

                # Rolling window 안정성 검증
                stability = validate_with_rolling_window(
                    x_lagged, y_lagged, window=min(6, len(x_lagged) // 3)
                )

                # 임계값 검증 (OR 조건: 하나라도 강한 관계면 포함)
                pearson_pass = (
                    abs(corr_results["pearson_corr"]) >= pearson_threshold
                    and corr_results["pearson_p"] < 0.01
                )
                spearman_pass = (
                    abs(corr_results["spearman_corr"]) >= spearman_threshold
                    and corr_results["spearman_p"] < 0.01
                )
                mi_pass = corr_results["mi_score"] >= mi_threshold

                if (pearson_pass or spearman_pass or mi_pass) and stability["stable"]:
                    result = {
                        "선행품목": leader_id,
                        "후행품목": follower_id,
                        "선행품목hs4": leader_hs4,
                        "후행품목hs4": follower_hs4,
                        "최적지연기간": lag,
                        "ccf_value": lag_info["ccf_value"],
                        "pearson_corr": corr_results["pearson_corr"],
                        "pearson_p": corr_results["pearson_p"],
                        "spearman_corr": corr_results["spearman_corr"],
                        "spearman_p": corr_results["spearman_p"],
                        "mi_score": corr_results["mi_score"],
                        "rolling_mean_corr": stability["mean_corr"],
                        "rolling_std_corr": stability["std_corr"],
                        "stable": stability["stable"],
                    }
                    results.append(result)
                    p_values_for_correction.append(corr_results["pearson_p"])

    if len(results) == 0:
        print("발견된 공행성 쌍이 없습니다.")
        return pd.DataFrame()

    pairs = pd.DataFrame(results)

    # 다중 검정 보정
    if use_multiple_testing_correction and len(p_values_for_correction) > 0:
        print(f"\n다중 검정 보정 수행 중...")
        _, corrected_pvals, _, _ = multipletests(
            p_values_for_correction,
            method="fdr_bh",  # Benjamini-Hochberg FDR
            alpha=0.05,
        )
        pairs["pearson_p_corrected"] = corrected_pvals

        # 보정된 p-value로 필터링
        pairs = pairs[pairs["pearson_p_corrected"] < 0.05]
        print(f"보정 후 남은 쌍: {len(pairs)}")

    # 중복 제거 (동일 쌍의 여러 지연 중 가장 강한 것만)
    pairs = pairs.sort_values("pearson_corr", key=abs, ascending=False)
    pairs = pairs.drop_duplicates(subset=["선행품목", "후행품목"], keep="first")

    print(f"\n📊 최종 결과:")
    print(f"- 총 발견된 공행성 쌍: {len(pairs)}")
    print(f"- Pearson 기반: {(pairs['pearson_corr'].abs() >= pearson_threshold).sum()}")
    print(
        f"- Spearman 기반: {(pairs['spearman_corr'].abs() >= spearman_threshold).sum()}"
    )
    print(f"- MI 기반: {(pairs['mi_score'] >= mi_threshold).sum()}")

    return pairs


if __name__ == "__main__":
    import pandas as pd
    from get_data_table import VALUE_PIVOUT, get_base_data
    from differencing import log_difference_transform
    from sklearn.preprocessing import MinMaxScaler

    # 데이터 로드
    data = get_base_data()
    pivot = VALUE_PIVOUT(data)
    pivot = log_difference_transform(pivot)

    # MinMax Scaling (상대적 변화 패턴 유지)
    scaler = MinMaxScaler()
    data_array = pivot.values
    scaled_array = scaler.fit_transform(data_array.T).T  # 품목별 스케일링
    pivot_scaled = pd.DataFrame(scaled_array, index=pivot.index, columns=pivot.columns)

    # 개선된 공행성 쌍 탐색
    pairs = find_comovement_pairs_improved(
        pivot_scaled,
        max_lag=12,
        min_nonzero=12,
        pearson_threshold=0.45,
        spearman_threshold=0.4,
        mi_threshold=0.05,
        min_data_points=20,
        use_multiple_testing_correction=True,
        exclude_same_hs4=False,  # True로 설정시 동일 HS4 제외
    )

    print("\n탐지 된 공행성쌍:")
    print(len(pairs))

    # 저장
    pairs.to_csv("datas/improved_comovement_pairs.csv", index=False)
    print("\n✅ 개선된 공행성 쌍 저장 완료: datas/improved_comovement_pairs.csv")
