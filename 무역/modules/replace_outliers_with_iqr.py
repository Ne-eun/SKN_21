def replace_outliers_with_q3(row, threshold=1.5):
    """
    각 행(item_id)에 대해 IQR 기반 이상치를 Q1, Q3 값으로 대체
    """
    Q1 = row.quantile(0.25)
    Q3 = row.quantile(0.75)
    IQR = Q3 - Q1

    # 이상치 마스크 생성
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    lower_outlier_mask = row < lower_bound
    upper_outlier_mask = row > upper_bound

    # 이상치를 Q3, Q1 값으로 대체
    row_processed = row.copy()
    row_processed[lower_outlier_mask] = Q1
    row_processed[upper_outlier_mask] = Q3

    return row_processed
