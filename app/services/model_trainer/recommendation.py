# app/services/model_trainer/recommendation.py

from app.setting import A_VALUE, B_VALUE, REVIEW_WEIGHT, CAUTION_WEIGHT, CONVENIENCE_WEIGHT
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

def generate_recommendations(data_filtered: pd.DataFrame, stacking_reg, model_features: list, user_id: str, scaler, user_features: pd.DataFrame = None) -> dict:
    """
    사용자 ID와 식당 데이터를 기반으로 개인화된 추천 생성
    
    Args:
        data_filtered: 필터링된 식당 데이터 DataFrame
        stacking_reg: 학습된, 적재된 스태킹 모델
        model_features: 모델 학습에 사용된 특성 목록
        user_id: 사용자 ID
        scaler: 특성 스케일링에 사용된 스케일러
        user_features: 전처리된 사용자 특성 데이터 DataFrame (없으면 일반 추천 제공)
        
    Returns:
        dict: 추천 결과를 담은 JSON 문자열
    """
    try:
        # 사용자 데이터가 있는 경우 개인화 적용
        user_row = None
        if user_features is not None:
            user_row = user_features[user_features['user_id'] == user_id]
            
            if not user_row.empty:
                logger.info(f"사용자 ID {user_id}의 개인화 추천 생성 중...")
                
                # 1. 가격 필터링: 사용자 max_price 이하인 식당만 선택
                if 'price' in data_filtered.columns and 'max_price' in user_row.columns:
                    max_price = user_row['max_price'].values[0]
                    if max_price > 0:
                        logger.debug(f"사용자 최대 가격 {max_price}원 이하의 식당으로 필터링")
                        data_filtered = data_filtered[data_filtered['price'] <= max_price].copy()
                
                # 2. 카테고리 보너스 점수 초기화
                data_filtered['category_bonus'] = 0.0
                
                # 3. 사용자 선호 카테고리에 보너스 점수 부여
                for i in range(1, 13):
                    category_col = f"category_{i}"
                    if category_col in user_row.columns and user_row[category_col].values[0] == 1:
                        # 사용자 선호 카테고리와 일치하는 식당에 보너스
                        data_filtered.loc[data_filtered['category_id'] == i, 'category_bonus'] = 0.3
                        logger.debug(f"카테고리 {i}에 기본 보너스 0.3 적용")
                        
                        # EDA에서 중요도가 높은 카테고리에 추가 보너스
                        if i in [4, 7, 9, 10]:
                            data_filtered.loc[data_filtered['category_id'] == i, 'category_bonus'] += 0.2
                            logger.debug(f"중요 카테고리 {i}에 추가 보너스 0.2 적용")
            else:
                logger.warning(f"사용자 ID {user_id}에 대한 정보가 없습니다. 기본 추천을 제공합니다.")

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
        
        # 사용자 선호도 보너스 적용 (사용자 데이터가 있고 개인화가 적용된 경우)
        if user_row is not None and not user_row.empty and 'category_bonus' in data_filtered.columns:
            logger.info("사용자 선호도 보너스 점수 적용 중...")
            
            # 카테고리 보너스 적용
            data_filtered['composite_score'] += data_filtered['category_bonus']
            
            # 사용자의 completed_reservations 값이 높을수록 예약 가능한 식당에 가중치
            if 'completed_reservations' in user_row.columns:
                completed_reservations = user_row['completed_reservations'].values[0]
                if completed_reservations > 3:  # 예약 경험이 많은 사용자
                    logger.debug(f"예약 경험이 많은 사용자({completed_reservations}회)에게 예약 가능 식당 가중치 부여")
                    data_filtered.loc[data_filtered['caution_예약가능'] == 1, 'composite_score'] += 0.2
            
            # 사용자의 like_to_reservation_ratio가 높을수록 인기 식당에 가중치
            if 'like_to_reservation_ratio' in user_row.columns:
                ratio = user_row['like_to_reservation_ratio'].values[0]
                if ratio > 2.0:  # 찜을 많이 하는 사용자
                    logger.debug(f"찜 대비 예약 비율이 높은 사용자({ratio})에게 인기 식당 가중치 부여")
                    # 리뷰가 많은 식당에 추가 보너스 (로그 스케일)
                    data_filtered['popularity_bonus'] = 0.15 * (np.log(data_filtered['review'] + 1) / np.log(1000))
                    data_filtered['composite_score'] += data_filtered['popularity_bonus']
        
        # 최종 점수 시그모이드 변환
        data_filtered['composite_score'] = data_filtered['composite_score'].apply(
            lambda x: sigmoid_transform(x, A_VALUE, B_VALUE)
        )

        # composite_score 기준 내림차순 정렬 후 상위 15개 추천 추출
        recommendations_all = data_filtered.sort_values(by='composite_score', ascending=False)
        top15 = recommendations_all[['category_id', 'restaurant_id', 'score', 'predicted_score', 'composite_score']].head(15).copy()
        
        # 추천 결과 추출 후, 필요한 컬럼을 정수로 변환
        top15['category_id'] = top15['category_id'].astype(int)
        top15['restaurant_id'] = top15['restaurant_id'].astype(int)
        top15['score'] = top15['score'].astype(float)

        top15['predicted_score'] = top15['predicted_score'].round(3)
        top15['composite_score'] = top15['composite_score'].round(3)

        # 사용자 정보와 추천 결과를 딕셔너리로 구성
        result_dict = {
            "user": user_id,
            "recommendations": json.loads(top15.to_json(orient='records', force_ascii=False))
        }

        # 딕셔너리를 JSON 문자열로 변환 (들여쓰기 적용)
        result_json = json.dumps(result_dict, ensure_ascii=False, indent=4)
        logger.info("추천 모델 결과가 산출 완료되었습니다. JSON으로 변환합니다.")
        return result_json
    
    except Exception as e:
        logger.error(f"generate_recommendations 오류: {e}", exc_info=True)
        raise e