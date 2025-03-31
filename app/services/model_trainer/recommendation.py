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
    신규 사용자의 경우 사용자 특성 없이도 카테고리 기반 추천 제공
    
    Args:
        data_filtered: 필터링된 식당 데이터 DataFrame
        stacking_reg: 학습된, 적재된 스태킹 모델
        model_features: 모델 학습에 사용된 특성 목록
        user_id: 사용자 ID
        scaler: 특성 스케일링에 사용된 스케일러
        user_features: 전처리된 사용자 특성 데이터 DataFrame (없으면 신규 사용자로 간주)
        
    Returns:
        dict: 추천 결과를 담은 JSON 문자열
    """
    try:
        # 사용자 유형 확인 (기존/신규)
        is_new_user = True
        user_row = None
        
        # 사용자 ID를 문자열로 변환
        user_id_str = str(user_id)
        logger.info(f"사용자 ID(str): {user_id_str}의 추천 생성 시작")
        
        if user_features is not None and not user_features.empty:
            # 디버깅을 위한 사용자 ID 목록과 타입 로깅
            logger.debug(f"사용자 특성 데이터 크기: {user_features.shape}")
            sample_ids = user_features['user_id'].head(5).values
            logger.debug(f"사용자 ID 샘플(상위 5개): {sample_ids}")
            logger.debug(f"사용자 ID 타입: {type(user_features['user_id'].iloc[0])}")
            
            # 모든 ID를 문자열로 변환하여 비교
            user_features['user_id_str'] = user_features['user_id'].astype(str)
            
            # 변환된 ID로 검색
            matching_rows = user_features[user_features['user_id_str'] == user_id_str]
            
            if not matching_rows.empty:
                is_new_user = False
                user_row = matching_rows.iloc[0:1]  # 첫 번째 일치 행만 사용
                logger.info(f"ID {user_id_str}의 사용자 데이터 찾음 - 개인화 추천 생성")
                
                # 가격 필터링 (기존 사용자만)
                if 'max_price' in user_row.columns and 'price' in data_filtered.columns:
                    max_price = user_row['max_price'].values[0]
                    if max_price > 0:
                        logger.debug(f"사용자 최대 가격 {max_price}원 이하의 식당으로 필터링")
                        data_filtered = data_filtered[data_filtered['price'] <= max_price].copy()
            else:
                logger.info(f"ID {user_id_str}의 사용자 데이터를 찾을 수 없음 - 카테고리 기반 기본 추천 생성")
        else:
            logger.info("사용자 특성 데이터가 없음 - 카테고리 기반 기본 추천 생성")
        
        # 카테고리 보너스 점수 초기화 (기존/신규 사용자 모두 적용)
        data_filtered['category_bonus'] = 0.0
        
        # 선호 카테고리 보너스 적용 로직
        if not is_new_user and user_row is not None:
            # 기존 사용자: 선호 카테고리 데이터 기반 보너스
            for i in range(1, 13):
                category_col = f"category_{i}"
                if category_col in user_row.columns and user_row[category_col].values[0] == 1:
                    data_filtered.loc[data_filtered['category_id'] == i, 'category_bonus'] = 0.3
                    logger.debug(f"카테고리 {i}에 기본 보너스 0.3 적용")
                    
                    if i in [4, 7, 9, 10]:  # 중요 카테고리
                        data_filtered.loc[data_filtered['category_id'] == i, 'category_bonus'] += 0.2
                        logger.debug(f"중요 카테고리 {i}에 추가 보너스 0.2 적용")
        else:
            # 신규 사용자: 필터링된 모든 식당은 사용자가 선택한 카테고리에 해당
            # 모든 식당에 동일한 카테고리 보너스 부여
            logger.info(f"신규 사용자용 카테고리 기반 추천 생성")
            data_filtered['category_bonus'] = 0.3

        # 모델 예측을 위한 피처 준비 (기존/신규 사용자 모두 동일)
        for feature in model_features:
            if feature not in data_filtered.columns:
                data_filtered[feature] = 0

        data_filtered = data_filtered.reset_index(drop=True)
        X_pred = data_filtered[model_features].copy()
        X_pred_scaled = pd.DataFrame(scaler.transform(X_pred), columns=X_pred.columns)

        # 모델 예측 수행
        data_filtered['predicted_score'] = stacking_reg.predict(X_pred_scaled)
        data_filtered['final_score'] = data_filtered['score']

        # 유의사항 관련 컬럼 확인
        for col in ['caution_배달가능', 'caution_예약가능', 'caution_포장가능',
                    'caution_배달불가', 'caution_예약불가', 'caution_포장불가']:
            if col not in data_filtered.columns:
                data_filtered[col] = 0

        data_filtered['review'] = pd.to_numeric(data_filtered['review'], errors='coerce')

        # 기본 점수 계산 (기존/신규 사용자 모두 동일)
        data_filtered['composite_score'] = data_filtered.apply(
            lambda row: compute_composite_score(row, REVIEW_WEIGHT, CAUTION_WEIGHT, CONVENIENCE_WEIGHT), axis=1
        )
        
        # 카테고리 보너스 적용
        data_filtered['composite_score'] += data_filtered['category_bonus']
        
        # 기존 사용자만을 위한 추가 개인화 점수
        if not is_new_user and 'completed_reservations' in user_row.columns:
            completed_reservations = user_row['completed_reservations'].values[0]
            if completed_reservations > 3:
                logger.debug(f"예약 경험이 많은 사용자에게 예약 가능 식당 가중치 부여")
                data_filtered.loc[data_filtered['caution_예약가능'] == 1, 'composite_score'] += 0.2
            
            if 'like_to_reservation_ratio' in user_row.columns:
                ratio = user_row['like_to_reservation_ratio'].values[0]
                if ratio > 2.0:
                    logger.debug(f"찜 대비 예약 비율이 높은 사용자에게 인기 식당 가중치 부여")
                    data_filtered['popularity_bonus'] = 0.15 * (np.log(data_filtered['review'] + 1) / np.log(1000))
                    data_filtered['composite_score'] += data_filtered['popularity_bonus']
        
        # 신규 사용자를 위한 추가 처리: 리뷰 수에 약간의 가중치
        elif is_new_user:
            # 신규 사용자는 인기 있는(리뷰가 많은) 식당에 약간의 가중치
            logger.debug("신규 사용자를 위해 리뷰가 많은 식당에 추가 가중치 부여")
            data_filtered['popularity_bonus'] = 0.1 * (np.log(data_filtered['review'] + 1) / np.log(1000))
            data_filtered['composite_score'] += data_filtered['popularity_bonus']
        
        # 최종 점수 시그모이드 변환
        data_filtered['composite_score'] = data_filtered['composite_score'].apply(
            lambda x: sigmoid_transform(x, A_VALUE, B_VALUE)
        )

        # composite_score 기준 내림차순 정렬
        recommendations_all = data_filtered.sort_values(by='composite_score', ascending=False)

        # 중복 제거를 위해 restaurant_id를 기준으로 첫 번째 레코드만 유지
        recommendations_all = recommendations_all.drop_duplicates(subset=['restaurant_id'], keep='first')

        # 상위 15개 추천 추출
        top15 = recommendations_all[['category_id', 'restaurant_id', 'score', 'predicted_score', 'composite_score']].head(15).copy()
        
        # 결과 포맷팅
        top15['category_id'] = top15['category_id'].astype(int)
        top15['restaurant_id'] = top15['restaurant_id'].astype(int)
        top15['score'] = top15['score'].astype(float)
        top15['predicted_score'] = top15['predicted_score'].round(3)
        top15['composite_score'] = top15['composite_score'].round(3)

        # 결과 딕셔너리 생성
        result_dict = {
            "user": user_id,
            "is_new_user": is_new_user,  # 신규 사용자 여부 표시 (옵션)
            "recommendations": json.loads(top15.to_json(orient='records', force_ascii=False))
        }

        # JSON 문자열 변환
        result_json = json.dumps(result_dict, ensure_ascii=False, indent=4)
        logger.info(f"{'신규' if is_new_user else '기존'} 사용자 추천 결과 생성 완료")
        return result_json
    
    except Exception as e:
        logger.error(f"generate_recommendations 오류: {e}", exc_info=True)
        raise e