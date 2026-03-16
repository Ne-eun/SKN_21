from re import X
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_cv_indices(cv, X_length, ax=None):
    """TimeSeriesSplit의 분할을 시각화하는 함수"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    n_splits = cv.get_n_splits()

    for ii, (tr, tt) in enumerate(cv.split(range(X_length))):
        # Training set
        ax.fill_between(
            tr,
            [ii] * len(tr),
            [ii + 0.4] * len(tr),
            alpha=0.6,
            color="blue",
            label="Training" if ii == 0 else "",
        )

        # Test set
        ax.fill_between(
            tt,
            [ii] * len(tt),
            [ii + 0.4] * len(tt),
            alpha=0.6,
            color="red",
            label="Test" if ii == 0 else "",
        )

    ax.set_ylabel("CV Fold")
    ax.set_xlabel("Time Index")
    ax.set_title(f"TimeSeriesSplit Cross-Validation ({n_splits} folds)")
    ax.legend(loc="upper right")
    ax.set_ylim(-0.5, n_splits - 0.5)

    return ax


def prepare_time_series_data(pivot) -> tuple[list, pd.DataFrame]:
    """
    피벗 데이터를 시계열 분석을 위해 준비

    Returns:
    - time_index: 시간 인덱스 (월별 데이터)
    - item_data: 각 품목별 시계열 데이터
    """
    # 날짜 컬럼을 시간 순으로 정렬
    time_columns = sorted(pivot.columns)
    pivot_sorted = pivot[time_columns]

    print(f"📅 시계열 데이터 기간:")
    print(f"  시작: {time_columns[0]}")
    print(f"  종료: {time_columns[-1]}")
    print(f"  총 기간: {len(time_columns)}개월")

    return time_columns, pivot_sorted


def create_time_series_datasets(pivot):
    """
    TimeSeriesSplit를 사용하여 훈련/테스트 데이터셋 생성

    Returns:
    - datasets: [(X_train, X_test, y_train, y_test, train_dates, test_dates), ...]
    """
    time_index, pivot_data = prepare_time_series_data(pivot)

    FOLD_SIZE = 3
    tscv = TimeSeriesSplit(n_splits=FOLD_SIZE)
    n_timepoints: int = len(time_index)
    splited_range = tscv.split(np.arange(n_timepoints))

    print(f"📊 교차검증 설정:")
    print(f"  총 시점 수: {n_timepoints}")
    print(f"  분할 수: {tscv.get_n_splits()}")
    print(f"  최소 훈련 크기: {n_timepoints // (tscv.get_n_splits() + 1)}")

    datasets: list[dict] = []
    for fold, (train_idx, test_idx) in enumerate(splited_range):
        # 날짜 정보
        train_dates = [time_index[i] for i in train_idx]
        test_dates = [time_index[i] for i in test_idx]

        train_set = pivot_data.iloc[train_idx]
        test_set = pivot_data.iloc[test_idx]

        datasets.append(
            {
                "fold": fold + 1,
                "train_set": train_set,
                "test_set": test_set,
                "train_dates": train_dates,
                "test_dates": test_dates,
                "train_idx": train_idx,
                "test_idx": test_idx,
            }
        )

        print(
            f"  Fold {fold + 1}: 훈련({len(train_dates)}개월) → 테스트({len(test_dates)}개월)"
        )
    return datasets
