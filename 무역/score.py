"""
Dacon AI빅데이터 분석 경진대회
평가 함수 구현
"""

import numpy as np
import pandas as pd


def _validate_input(answer_df, submission_df):
    """입력 데이터 유효성 검증"""
    # ① 컬럼 개수·이름 일치 여부
    if len(answer_df.columns) != len(submission_df.columns) or not all(
        answer_df.columns == submission_df.columns
    ):
        raise ValueError(
            "The columns of the answer and submission dataframes do not match."
        )

    # ② 필수 컬럼에 NaN 존재 여부
    if submission_df.isnull().values.any():
        raise ValueError("The submission dataframe contains missing values.")

    # ③ pair 중복 여부
    pairs = list(
        zip(submission_df["leading_item_id"], submission_df["following_item_id"])
    )
    if len(pairs) != len(set(pairs)):
        raise ValueError(
            "The submission dataframe contains duplicate (leading_item_id, following_item_id) pairs."
        )


def comovement_f1(answer_df, submission_df):
    """공행성쌍 F1 계산"""
    ans = answer_df[["leading_item_id", "following_item_id"]].copy()
    sub = submission_df[["leading_item_id", "following_item_id"]].copy()

    ans["pair"] = list(zip(ans["leading_item_id"], ans["following_item_id"]))
    sub["pair"] = list(zip(sub["leading_item_id"], sub["following_item_id"]))

    G = set(ans["pair"])
    P = set(sub["pair"])

    tp = len(G & P)
    fp = len(P - G)
    fn = len(G - P)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return f1, precision, recall, tp, fp, fn


def comovement_nmae(answer_df, submission_df, eps=1e-6):
    """전체 U = G ∪ P에 대한 clipped NMAE 계산"""
    ans = answer_df[["leading_item_id", "following_item_id", "value"]].copy()
    sub = submission_df[["leading_item_id", "following_item_id", "value"]].copy()

    ans["pair"] = list(zip(ans["leading_item_id"], ans["following_item_id"]))
    sub["pair"] = list(zip(sub["leading_item_id"], sub["following_item_id"]))

    G = set(ans["pair"])
    P = set(sub["pair"])
    U = G | P

    ans_val = dict(zip(ans["pair"], ans["value"]))
    sub_val = dict(zip(sub["pair"], sub["value"]))

    errors = []
    tp_errors = []  # 실제 매칭된 쌍들의 오차만 별도 분석

    for pair in U:
        if pair in G and pair in P:
            # 정수 변환(반올림)
            y_true = int(round(float(ans_val[pair])))
            y_pred = int(round(float(sub_val[pair])))
            rel_err = abs(y_true - y_pred) / (abs(y_true) + eps)
            rel_err = min(rel_err, 1.0)  # 오차 100% 이상은 100%로 간주
            tp_errors.append(rel_err)
        else:
            rel_err = 1.0  # FN, FP는 오차 100%
        errors.append(rel_err)

    return np.mean(errors) if errors else 1.0, tp_errors


def comovement_score(answer_df, submission_df):
    """최종 대회 점수 계산"""
    _validate_input(answer_df, submission_df)
    f1, precision, recall, tp, fp, fn = comovement_f1(answer_df, submission_df)
    nmae_full, tp_errors = comovement_nmae(answer_df, submission_df, 1e-6)
    S2 = 1 - nmae_full
    score = 0.6 * f1 + 0.4 * S2

    return {
        "final_score": score,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "nmae": nmae_full,
        "S2": S2,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tp_errors": tp_errors,
    }


def validate_submission_with_test(target_row, submission_df, pairs_df):
    """
    Test 데이터를 기반으로 submission을 검증하는 함수

    Parameters:
    - target_row: 2025년 7월 무역량 데이터 (pandas.DataFrame, columns=item_id, value)
    - submission_df: 제출 데이터 (columns: leading_item_id, following_item_id, value)
    - pairs_df: 발견된 공행성 쌍 데이터 (선행품목, 선행품목hs4, 후행품목, 후행품목hs4, 최적지연기간 등)
    Returns:
    - validation_results: 검증 결과 딕셔너리
    """

    print("=" * 70)
    print("🔍 Test 데이터 기반 Submission 검증")
    print("=" * 70)

    # 1. 기본 정보 출력
    print(f"📊 기본 정보:")
    print(f"  • Test 데이터 (2025년 7월): {len(target_row)}개 품목")
    print(f"  • Submission 예측 쌍: {len(submission_df)}개")

    # 2. Test 데이터에서 실제로 거래가 있는 품목들로 답안 생성
    # (실제 대회에서는 정답 파일이 제공되지만, 여기서는 test 데이터를 활용)
    active_items = target_row[target_row["value"] > 0]["item_id"].tolist()
    # 3. 가상의 정답 데이터 생성 (실제 공행성 쌍 기반)
    answer_pairs = []

    if pairs_df is not None and len(pairs_df) > 0:
        # 발견된 공행성 쌍 중에서 following_item이 활성 품목인 것들만 선택
        for _, row in pairs_df.iterrows():
            if hasattr(row, "후행품목"):
                following_item = row.후행품목
            else:
                continue

            if following_item in active_items:
                values = target_row.groupby("item_id")["value"].sum()
                actual_value = int(values[following_item])
                answer_pairs.append(
                    {
                        "leading_item_id": row.선행품목,
                        "following_item_id": following_item,
                        "value": actual_value,
                    }
                )

    if len(answer_pairs) == 0:
        print("❌ 검증할 수 있는 정답 쌍이 없습니다.")
        return None

    answer_df = pd.DataFrame(answer_pairs)

    # 4. 대회 평가식 적용
    try:
        results = comovement_score(answer_df, submission_df)

        print(f"\n📋 대회 평가 결과:")
        print(f"  🏆 최종 점수: {results['final_score']:.4f}")
        print(f"     ├─ F1-Score (60%): {results['f1_score']:.4f}")
        print(f"     └─ S2 (40%): {results['S2']:.4f} (1-NMAE)")

        print(f"\n📈 상세 지표:")
        print(f"  • F1-Score: {results['f1_score']:.4f}")
        print(f"  • Precision: {results['precision']:.4f}")
        print(f"  • Recall: {results['recall']:.4f}")
        print(f"  • NMAE: {results['nmae']:.4f}")

        print(f"\n🎯 매칭 분석:")
        print(f"  • True Positive (TP): {results['tp']}개 (정확히 매칭된 쌍)")
        print(f"  • False Positive (FP): {results['fp']}개 (잘못 예측한 쌍)")
        print(f"  • False Negative (FN): {results['fn']}개 (놓친 쌍)")

        # 5. TP 오차 분석
        if len(results["tp_errors"]) > 0:
            tp_errors = np.array(results["tp_errors"])
            print(f"\n📊 매칭된 쌍들의 예측 정확도:")
            print(f"  • 평균 상대오차: {np.mean(tp_errors):.4f}")
            print(f"  • 중간값 상대오차: {np.median(tp_errors):.4f}")
            print(f"  • 최소 상대오차: {np.min(tp_errors):.4f}")
            print(f"  • 최대 상대오차: {np.max(tp_errors):.4f}")
            print(f"  • 10% 이내 정확도: {(tp_errors <= 0.1).mean()*100:.1f}%")
            print(f"  • 20% 이내 정확도: {(tp_errors <= 0.2).mean()*100:.1f}%")

        # 6. 성능 등급 분류
        score = results["final_score"]
        if score >= 0.8:
            grade = "🥇 우수 (0.8+)"
        elif score >= 0.6:
            grade = "🥈 양호 (0.6-0.8)"
        elif score >= 0.4:
            grade = "🥉 보통 (0.4-0.6)"
        else:
            grade = "⚠️ 개선필요 (<0.4)"

        print(f"\n🏅 성능 등급: {grade}")

        # 7. 개선 제안
        print(f"\n💡 개선 제안:")
        if results["f1_score"] < 0.6:
            print("  🔸 F1-Score 개선 방안:")
            print("    - 더 많은 공행성 쌍 발견 (threshold 낮추기)")
            print("    - 통계적 검정 조건 완화")
            print("    - 앙상블 모델로 robustness 향상")

        if results["nmae"] > 0.3:
            print("  🔸 NMAE 개선 방안:")
            print("    - 예측 모델 성능 향상 (XGBoost, RandomForest)")
            print("    - 더 많은 특성 엔지니어링")
            print("    - 이상치 제거 및 후처리 개선")

        if results["precision"] < results["recall"]:
            print("  🔸 False Positive 감소:")
            print("    - 더 보수적인 공행성 기준 적용")
            print("    - 모델 confidence threshold 조정")
        elif results["recall"] < results["precision"]:
            print("  🔸 False Negative 감소:")
            print("    - 공행성 탐색 범위 확대")
            print("    - 더 완화된 필터링 조건")

        # 결과 반환
        results["validation_details"] = {
            "test_active_items": len(active_items),
            "answer_pairs": len(answer_df),
            "submission_pairs": len(submission_df),
            "grade": grade,
        }

        return results

    except Exception as e:
        print(f"❌ 검증 중 오류 발생: {e}")
        return None


def print_validation_summary(test_data, submission_df, pairs_df):

    # Test 데이터와 Submission 데이터를 사용하여 검증
    validation_results = validate_submission_with_test(
        target_row=test_data,  # 2025년 7월 실제 무역량 데이터
        submission_df=submission_df,  # 우리가 예측한 제출 데이터
        pairs_df=pairs_df,  # 발견한 공행성 쌍 데이터
    )

    """검증 결과 요약 출력"""
    if validation_results is not None:
        print("\n" + "=" * 80)
        print("🔥 최종 검증 결과 요약")
        print("=" * 80)

        final_score = validation_results["final_score"]
        f1_score = validation_results["f1_score"]
        nmae = validation_results["nmae"]

        print(f"🏆 대회 최종 점수: {final_score:.4f}")
        print(f"   ├─ 공행성 쌍 매칭 (F1): {f1_score:.4f} (가중치 60%)")
        print(f"   └─ 예측값 정확도 (S2): {1-nmae:.4f} (가중치 40%)")

        # 예상 순위 추정 (대략적)
        if final_score >= 0.85:
            rank_estimate = "🥇 상위 5% 예상"
        elif final_score >= 0.75:
            rank_estimate = "🥈 상위 10% 예상"
        elif final_score >= 0.65:
            rank_estimate = "🥉 상위 25% 예상"
        elif final_score >= 0.50:
            rank_estimate = "📊 중위권 예상"
        else:
            rank_estimate = "⚠️ 하위권 예상"

        print(f"\n📊 예상 성과: {rank_estimate}")

        # 주요 개선 포인트 강조
        if f1_score < 0.6:
            print("\n🔥 긴급 개선 필요: F1-Score가 낮습니다!")
            print("   → 더 많은 공행성 쌍을 발견해야 합니다.")

        if nmae > 0.4:
            print("\n🔥 긴급 개선 필요: 예측 정확도가 낮습니다!")
            print("   → 모델 성능 향상이 필요합니다.")

        # 다음 단계 제안
        print(f"\n🚀 다음 단계 제안:")
        print(f"1️⃣ 공행성 탐색 개선 - 더 완화된 조건으로 재실행")
        print(f"2️⃣ 앙상블 모델 적용 - XGBoost + RandomForest")
        print(f"3️⃣ 특성 엔지니어링 고도화 - 시계열 특성 추가")
        print(f"4️⃣ 후처리 최적화 - 예측값 범위 제한 및 스무딩")

    else:
        print("❌ 검증을 수행할 수 없습니다. 데이터를 확인해주세요.")

    print(f"\n✨ 검증 완료! 결과를 바탕으로 모델을 개선해보세요.")
