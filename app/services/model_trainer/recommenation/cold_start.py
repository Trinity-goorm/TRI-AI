# app/services/model_trainer/recommendation/cold_start.py

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def enhance_cold_start_recommendations(data_filtered, user_id, user_features_df=None):
    """
    신규 사용자를 위한 추천 로직을 강화하는 함수
    
    Args:
        data_filtered: 필터링된 식당 데이터
        user_id: 사용자 ID
        user_features_df: 사용자 특성 데이터 (옵션)
        
    Returns:
        pd.DataFrame: 향상된 추천 점수가 포함된 데이터프레임
    """
    try:
        logger.debug(f"신규 사용자 {user_id}를 위한 강화된 추천 로직 적용")
        
        # 1. 카테고리 다양성 강화
        # 각 카테고리별 고르게 추천하도록 조정
        category_counts = data_filtered['category_id'].value_counts()
        total_restaurants = len(data_filtered)
        
        # 다양성 점수 계산 (희소 카테고리에 높은 점수)
        diversity_score = 1 - (category_counts / total_restaurants)
        category_diversity_map = diversity_score.to_dict()
        
        # 다양성 점수 적용 (기존 category_diversity_bonus 강화)
        data_filtered['enhanced_diversity_bonus'] = data_filtered['category_id'].map(
            category_diversity_map
        ).fillna(0) * 0.15  # 다양성 가중치 증가
        
        # 2. 사용자 선호도 분석 (있는 경우)
        user_preferred_category = None
        if user_features_df is not None:
            # 문자열로 변환하여 비교
            user_id_str = str(user_id)
            user_features_df['user_id_str'] = user_features_df['user_id'].astype(str)
            user_data = user_features_df[user_features_df['user_id_str'] == user_id_str]
            
            if not user_data.empty:
                # 선호 카테고리 정보 추출
                preferred_cols = [col for col in user_data.columns if col.startswith('preferred_category')]
                for col in preferred_cols:
                    if col in user_data.columns and not pd.isna(user_data[col].iloc[0]):
                        user_preferred_category = user_data[col].iloc[0]
                        break
        
        # 선호 카테고리 보너스 적용
        if user_preferred_category is not None:
            # 선호 카테고리 식당에 가중치 부여
            data_filtered['preferred_category_bonus'] = 0.0
            data_filtered.loc[data_filtered['category_id'] == user_preferred_category, 'preferred_category_bonus'] = 0.4
        
        # 3. 인기도 기반 보너스 (신규 사용자용)
        # 리뷰 수 기반 인기도 - 로그 스케일링으로 극단값 완화
        max_review = data_filtered['review'].max()
        if max_review > 0:
            data_filtered['popularity_bonus'] = (
                np.log1p(data_filtered['review']) / np.log1p(max_review)
            ) * 0.2
        else:
            data_filtered['popularity_bonus'] = 0
        
        # 4. 운영 시간 보너스
        if 'duration_hours' in data_filtered.columns:
            # 운영 시간이 긴 식당 가중치
            max_duration = data_filtered['duration_hours'].max()
            if max_duration > 0:
                data_filtered['duration_bonus'] = (
                    data_filtered['duration_hours'] / max_duration
                ) * 0.1
            else:
                data_filtered['duration_bonus'] = 0
        else:
            data_filtered['duration_bonus'] = 0
        
        # 5. 편의시설 보너스
        convenience_cols = [col for col in data_filtered.columns if col.startswith('conv_') 
                           and col != 'conv_편의시설 정보 없음']
        
        if convenience_cols:
            # 편의시설 수 기반 보너스
            data_filtered['convenience_bonus'] = data_filtered[convenience_cols].sum(axis=1) * 0.05
        else:
            data_filtered['convenience_bonus'] = 0
        
        # 6. 모든 보너스 합산하여 최종 점수에 추가
        cold_start_bonus = (
            data_filtered.get('enhanced_diversity_bonus', 0) +
            data_filtered.get('preferred_category_bonus', 0) +
            data_filtered.get('popularity_bonus', 0) +
            data_filtered.get('duration_bonus', 0) +
            data_filtered.get('convenience_bonus', 0)
        )
        
        # 기존 composite_score에 추가
        data_filtered['cold_start_bonus'] = cold_start_bonus
        data_filtered['composite_score'] += cold_start_bonus
        
        logger.debug(f"신규 사용자 추천 강화 완료: 평균 보너스 점수 {cold_start_bonus.mean():.4f}")
        return data_filtered
        
    except Exception as e:
        logger.error(f"신규 사용자 추천 강화 중 오류: {e}", exc_info=True)
        return data_filtered