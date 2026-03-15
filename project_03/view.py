from re import escape
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score

st.set_page_config(
    page_title="박내은 - 최종 모델 리포트",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
        .header {
            text-align: center;
        }
        .box {
            margin: 0 auto;
        }
    </style>
    <div class="header">
        <h1 style='margin: 0;'>은행 이탈고객 예측 모델 리포트 - 박내은</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

st.subheader("Feature Engineering")
feature_df = pd.DataFrame(
    {
        "특성": [
            "상품 당 평균 잔고(product_mean_balance)",
            "연봉 대비 계좌 잔고(balance_to_salary)",
            "연령 가중치(risk_age_rank)",
        ],
        "특성 생성 방법": ["계좌 잔고 / 상품 갯수", "잔고 / 연봉", "연령대 구간화"],
        "비고": [
            "",
            "",
            "30대 이하, 30대, 40대, 50대, 60대 이상",
        ],
    },
)
st.dataframe(feature_df, hide_index=True)

st.divider()

st.subheader("Preprocessing")
st.markdown(":orange-badge[⚠️ XGBoost, LightGBM 모델 특성상 스케일링을 적용하지 않음]")
encoding_df = pd.DataFrame(
    {
        "특성": ["도시(country)", "성별(sex)", "연령 가중치(risk_age_rank)"],
        "인코딩 방법": ["원핫 인코딩", "라벨 인코딩", "원핫 인코딩"],
        "비고": ["", "남성을 1로 가중치를 부여 하고자 함", ""],
    }
)
st.dataframe(encoding_df, hide_index=True)
st.markdown(
    """
        <h4 style="font-size: 1.4rem">Data Splitting Strategy</h4>
        <p>KFold를 초기 시도했으나 데이터 양 부족으로 인한 성능 저하</br> 최종 Hold-out 데이터셋으로 학습 진행</p>
    """,
    unsafe_allow_html=True,
)

st.divider()


st.subheader("Models")
st.markdown(
    """
        <ul>
            <li>앙상블(LightGBM + XGBoost) 두 모델의 평균치로 최종 예측</li>
            <li>Target Data(이탈/유지) 불균형을 잡기위해 class 가중치를 다르게 설정</li>
        </ul>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)
with col1:
    st.markdown(
        """
        <style>
            .model_box {
                padding: 20px;
                border: 1px solid #666;
                border-radius: 20px;
                margin-bottom: 40px;
            }
        </style>
        <div class="model_box">
            <h4>LightGBM의 장점</h4>
            <ul>
                <li>빠른 학습 속도</li>
                <li>메모리 효율성</li>
                <li>범주형 변수 자동 처리</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="model_box">
            <h4>XGBoost의 장점</h4>
            <ul>
                <li>높은 예측 성능</li>
                <li>정규화 기능</li>
                <li>과적합 방지</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
parameter_df = pd.DataFrame(
    {
        "모델": [
            "lightgbm",
            "lightgbm",
            "lightgbm",
            "lightgbm",
            "lightgbm",
            "lightgbm",
            "lightgbm",
            "lightgbm",
            "xgboost",
            "xgboost",
            "xgboost",
            "xgboost",
            "xgboost",
            "xgboost",
            "xgboost",
            "xgboost",
            "xgboost",
        ],
        "파라미터": [
            "random_state",
            "force_col_wise",
            "max_depth",
            "num_leaves",
            "n_estimators",
            "learning_rate",
            "scale_pos_weight",
            "colsample_bytree",
            "n_estimators",
            "max_depth",
            "learning_rate",
            "scale_pos_weight",
            "random_state",
            "objective",
            "tree_method",
            "reg_alpha",
            "reg_lambda",
        ],
        "사용한 값": [
            "42",
            "True",
            "8",
            "34",
            "120",
            "0.01",
            "5",
            "0.8",
            "700",
            "4",
            "0.01",
            "4",
            "42",
            "binary:logistic",
            "approx",
            "0.1",
            "1.0",
        ],
    }
)
parameter_df["모델"] = parameter_df["모델"].mask(parameter_df["모델"].duplicated(), "")
st.dataframe(parameter_df, hide_index=True)

st.divider()

st.subheader("Post processing")
st.markdown(
    """
    <ul>
        <li>Recall 0.7이상 목표: 이탈 고객을 놓치지 않는 것이 중요</li>
        <li>Precision 0.5이상 목표: 너무 많은 고객을 이탈로 분류하지 않도록 조정</li>
        <li>최종 임계값 0.47 적용</li>
    </ul>
    """,
    unsafe_allow_html=True,
)

threshold_comparison = pd.DataFrame(
    {
        "임계값": [0.50, 0.47],
        "Precision": [0.5357, 0.5017],
        "Recall": [0.7383, 0.7772],
        "F1-Score": [0.6209, 0.6098],
        "Accuracy": [0.8144, 0.7952],
    }
)

st.dataframe(threshold_comparison, hide_index=True)
st.divider()

st.subheader("ROC AUC Curve")

script_dir = Path(__file__).parent
y_test = pd.read_csv(script_dir / "assets" / "y_test.csv")
ensemble_test_pred = pd.read_csv(script_dir / "assets" / "ensemble_test_pred.csv")
y_valid = pd.read_csv(script_dir / "assets" / "y_valid.csv")
ensemble_valid_pred = pd.read_csv(script_dir / "assets" / "ensemble_valid_pred.csv")

fig, ax = plt.subplots(figsize=(8, 6))

v_rfc_roc = roc_auc_score(y_valid, ensemble_valid_pred)
v_fpr, v_tpr, _ = roc_curve(y_valid, ensemble_valid_pred)

t_rfc_roc = roc_auc_score(y_test, ensemble_test_pred)
t_fpr, t_tpr, _ = roc_curve(y_test, ensemble_test_pred)

ax.plot(v_fpr, v_tpr, label=f"Validation Set (AUC = {v_rfc_roc:.4f})")
ax.plot(t_fpr, t_tpr, label=f"Test Set (AUC = {t_rfc_roc:.4f})")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)

st.subheader("최종 성능 요약")
final_metrics = pd.DataFrame(
    {
        "Metric": ["ROC AUC", "Accuracy", "Precision", "Recall", "F1-Score"],
        "Score": [0.8671, 0.8004, 0.4987, 0.7706, 0.6055],
    }
)
st.dataframe(final_metrics, hide_index=True)

st.divider()
