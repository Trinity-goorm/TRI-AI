# app/services/model_trainer/recommendation/diversity.py

import pandas as pd
import logging

logger = logging.getLogger(__name__)

def calculate_category_diversity_bonus(data_filtered):
    """
    카테고리별 다양성을 고려한 보너스 점수 계산
    
    Args:
        data_filtered (pd.DataFrame): 식당 데이터
    
    Returns:
        pd.DataFrame: 다양성 보너스가 추가된 데이터프레임
    """
    try:
        # 카테고리별 식당 수 계산
        category_counts = data_filtered['category_id'].value_counts()
        total_restaurants = len(data_filtered)
        
        # 카테고리별 희소성 계산 (희소한 카테고리에 더 높은 보너스)
        diversity_bonus = 1 - (category_counts / total_restaurants)
        
        # 보너스 점수 매핑
        category_bonus_map = diversity_bonus.to_dict()
        
        # 각 식당의 카테고리에 따라 보너스 점수 할당
        data_filtered['category_diversity_bonus'] = data_filtered['category_id'].map(category_bonus_map).fillna(0)
        
        return data_filtered
    
    except Exception as e:
        logger.error(f"calculate_category_diversity_bonus 오류: {e}", exc_info=True)
        return data_filtered