from app.config import A_VALUE, B_VALUE, REVIEW_WEIGHT, CAUTION_WEIGHT, CONVENIENCE_WEIGHT
import numpy as np
import json
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def compute_composite_score(row, review_weight, caution_weight, convenience_weight):
    try:
        base = row['final_score']
        review_val = float(row['review'])
        review_adjust = review_weight * (np.log(review_val + 50) / np.log(1000))
        pos = row.get('caution_배달가능', 0) + row.get('caution_예약가능', 0) + row.get('caution_포장가능', 0)
        neg = row.get('caution_배달불가', 0) + row.get('caution_예약불가', 0) + row.get('caution_포장불가', 0)
        conv_cols = [col for col in row.index if col.startswith("conv_") and col != "conv_편의시설 정보 없음"]
        conv_mean = np.mean([row[col] for col in conv_cols]) if conv_cols and any(row[col] for col in conv_cols) else 0
        conv_adjust = convenience_weight * conv_mean
        return base + review_adjust + caution_weight * (pos - neg) + conv_adjust
    
    except Exception as e:
        logger.error(f"compute_composite_score 오류: {e}", exc_info=True)
        raise e


def sigmoid_transform(x, a, b):
    try:
        return 5 * (1 / (1 + np.exp(-a * (x - b))))
    
    except Exception as e:
        logger.error(f"sigmoid_transform 오류: {e}", exc_info=True)
        raise

def generate_recommendations(data_filtered: pd.DataFrame, stacking_reg, model_features: list, user_id: str, scaler) -> dict:
    try:
        # 필요한 모든 피처가 있는지 확인하고 없으면 추가 (0으로 채움)
        for feature in model_features:
            if feature not in data_filtered.columns:
                data_filtered[feature] = 0

        # 인덱스 초기화
        data_filtered = data_filtered.reset_index(drop=True)

        # 학습 시 사용한 피처 순서대로 DataFrame 생성
        X_pred = data_filtered[model_features].copy()
        # 스케일러를 사용하여 피처 스케일링 적용
        X_pred_scaled = pd.DataFrame(scaler.transform(X_pred), columns=X_pred.columns)

        # 예측 수행
        data_filtered['predicted_score'] = stacking_reg.predict(X_pred_scaled)
        data_filtered['final_score'] = data_filtered['score']

        # 유의사항 관련 컬럼이 없으면 0으로 채움
        for col in ['caution_배달가능', 'caution_예약가능', 'caution_포장가능',
                    'caution_배달불가', 'caution_예약불가', 'caution_포장불가']:
            if col not in data_filtered.columns:
                data_filtered[col] = 0

        data_filtered['review'] = pd.to_numeric(data_filtered['review'], errors='coerce')

        # composite_score 계산
        data_filtered['composite_score'] = data_filtered.apply(
            lambda row: compute_composite_score(row, REVIEW_WEIGHT, CAUTION_WEIGHT, CONVENIENCE_WEIGHT), axis=1
        )
        data_filtered['composite_score'] = data_filtered['composite_score'].apply(
            lambda x: sigmoid_transform(x, A_VALUE, B_VALUE)
        )

        # composite_score 기준 오름차순 정렬 후 상위 15개 추천 추출
        recommendations_all = data_filtered.sort_values(by='composite_score', ascending=False)
        top15 = recommendations_all[['category_id', 'restaurant_id', 'score', 'predicted_score', 'composite_score']].head(15).copy()
        
        # 추천 결과 추출 후, 필요한 컬럼을 정수로 변환
        top15['category_id'] = top15['category_id'].astype(int)
        top15['restaurant_id'] = top15['restaurant_id'].astype(int)
        top15['score'] = top15['score'].astype(float)  # 만약 score가 정수여야 한다면

        top15['predicted_score'] = top15['predicted_score'].round(3)
        top15['composite_score'] = top15['composite_score'].round(3)

        # OrderedDict을 사용하여 user_id가 첫 번째 키로 오도록 구성
        # 각 레코드를 OrderedDict로 변환하여 필드 순서 보장
        # 사용자 정보와 추천 결과를 딕셔너리로 구성
        result_dict = {
            "user": user_id,  # 앞에서 입력받은 사용자 ID
            "recommendations": json.loads(top15.to_json(orient='records', force_ascii=False))
        }

        # 딕셔너리를 JSON 문자열로 변환 (들여쓰기 적용)
        result_json = json.dumps(result_dict, ensure_ascii=False, indent=4)
        logger.info("추천 모델 결과가 산출 완료되었습니다. JSON으로 변환합니다.")
        return result_json
    
    except Exception as e:
        logger.error(f"generate_recommendations 오류: {e}", exc_info=True)
        raise e