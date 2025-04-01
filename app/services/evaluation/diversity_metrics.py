# app/services/evaluation/diversity_metrics.py

import numpy as np
import pandas as pd
import scipy.stats
import logging

logger = logging.getLogger(__name__)

def evaluate_recommendation_diversity(recommendations, user_history=None, restaurant_data=None):
    """
    추천 결과의 다양성을 평가하는 함수
    
    Args:
        recommendations: 사용자별 추천 결과 딕셔너리 {user_id: [restaurant_ids]}
        user_history: 사용자 과거 상호작용 데이터 (옵션)
        restaurant_data: 식당 메타데이터 (옵션)
    
    Returns:
        dict: 다양성 평가 결과
    """
    try:
        diversity_metrics = {}
        
        # 추천 결과를 DataFrame으로 변환
        recommendations_list = []
        for user_id, rest_ids in recommendations.items():
            for rank, rest_id in enumerate(rest_ids):
                recommendations_list.append({
                    'user_id': user_id,
                    'restaurant_id': rest_id,
                    'rank': rank + 1
                })
        
        recs_df = pd.DataFrame(recommendations_list)
        if recs_df.empty:
            logger.warning("추천 결과가 없어 다양성 평가를 진행할 수 없습니다.")
            return {
                "category_diversity": 0,
                "user_coverage": 0,
                "item_coverage": 0,
                "novelty": 0,
                "serendipity": 0,
                "intra_list_similarity": 0
            }
        
        # 1. 카테고리 다양성 평가
        if restaurant_data is not None and 'category_id' in restaurant_data.columns:
            # 식당별 카테고리 매핑
            restaurant_categories = restaurant_data.set_index('restaurant_id')['category_id'].to_dict()
            
            # 추천된 식당에 카테고리 매핑
            recs_df['category_id'] = recs_df['restaurant_id'].map(restaurant_categories)
            
            # 카테고리 엔트로피 계산
            if 'category_id' in recs_df.columns and not recs_df['category_id'].isna().all():
                category_counts = recs_df['category_id'].value_counts(normalize=True)
                category_entropy = scipy.stats.entropy(category_counts)
                category_diversity = 1 - (1 / (1 + category_entropy))  # 0~1 범위로 정규화
                diversity_metrics['category_diversity'] = category_diversity
            else:
                diversity_metrics['category_diversity'] = 0
        else:
            diversity_metrics['category_diversity'] = 0
        
        # 2. 사용자 커버리지 (전체 사용자 중 추천을 받은 사용자 비율)
        if user_history is not None:
            all_users = user_history['user_id'].unique()
            recommended_users = set(recs_df['user_id'].unique())
            user_coverage = len(recommended_users) / len(all_users) if len(all_users) > 0 else 0
            diversity_metrics['user_coverage'] = user_coverage
        else:
            diversity_metrics['user_coverage'] = 1.0  # 사용자 히스토리가 없으면 100%로 가정
        
        # 3. 아이템 커버리지 (전체 식당 중 추천된 식당 비율)
        if restaurant_data is not None:
            all_restaurants = restaurant_data['restaurant_id'].unique()
            recommended_restaurants = set(recs_df['restaurant_id'].unique())
            item_coverage = len(recommended_restaurants) / len(all_restaurants) if len(all_restaurants) > 0 else 0
            diversity_metrics['item_coverage'] = item_coverage
        else:
            diversity_metrics['item_coverage'] = 0
        
        # 4. 새로움 (Novelty) - 인기 있는 아이템보다 덜 알려진 아이템 추천 정도
        if user_history is not None and restaurant_data is not None:
            # 식당별 인기도 (상호작용 횟수) 계산
            restaurant_popularity = user_history['restaurant_id'].value_counts()
            total_interactions = len(user_history)
            
            # 각 식당의 희소성 = -log(인기도)
            restaurant_rarity = restaurant_popularity.map(
                lambda x: -np.log(x / total_interactions) if x > 0 else 0
            )
            
            # 추천된 식당의 평균 희소성 (높을수록 새로운 항목을 추천)
            recommended_rarity = []
            for rest_id in recs_df['restaurant_id'].unique():
                if rest_id in restaurant_rarity:
                    recommended_rarity.append(restaurant_rarity[rest_id])
            
            novelty = np.mean(recommended_rarity) if recommended_rarity else 0
            
            # 0~1 범위로 정규화 (높을수록 새로움이 큼)
            max_possible_novelty = -np.log(1 / total_interactions) if total_interactions > 0 else 1
            normalized_novelty = novelty / max_possible_novelty if max_possible_novelty > 0 else 0
            
            diversity_metrics['novelty'] = normalized_novelty
        else:
            diversity_metrics['novelty'] = 0
        
        # 5. 세렌디피티 (Serendipity) - 의외성, 예상치 못한 발견
        if user_history is not None and restaurant_data is not None:
            # 사용자별 과거 상호작용 카테고리
            user_categories = {}
            for user_id in recs_df['user_id'].unique():
                user_hist = user_history[user_history['user_id'] == user_id]
                user_rest_ids = user_hist['restaurant_id'].tolist()
                
                # 해당 식당들의 카테고리 추출
                user_rest_categories = [
                    restaurant_categories.get(rest_id) 
                    for rest_id in user_rest_ids 
                    if rest_id in restaurant_categories
                ]
                
                user_categories[user_id] = set(user_rest_categories)
            
            # 각 사용자의 추천 중 새로운 카테고리 비율 계산
            serendipity_scores = []
            
            for user_id in recs_df['user_id'].unique():
                if user_id not in user_categories:
                    continue
                    
                user_recs = recs_df[recs_df['user_id'] == user_id]
                
                # 추천된 식당들의 카테고리
                rec_categories = [
                    restaurant_categories.get(rest_id) 
                    for rest_id in user_recs['restaurant_id'] 
                    if rest_id in restaurant_categories
                ]
                
                # 사용자가 과거에 경험하지 않은 카테고리 비율
                new_categories = [cat for cat in rec_categories if cat not in user_categories[user_id]]
                serendipity = len(new_categories) / len(rec_categories) if rec_categories else 0
                
                serendipity_scores.append(serendipity)
            
            # 전체 사용자의 평균 세렌디피티
            diversity_metrics['serendipity'] = np.mean(serendipity_scores) if serendipity_scores else 0
        else:
            diversity_metrics['serendipity'] = 0
        
        # 6. 추천 목록 내 유사성 (낮을수록 다양한 추천)
        if restaurant_data is not None and 'category_id' in restaurant_data.columns:
            intra_similarities = []
            
            for user_id in recs_df['user_id'].unique():
                user_recs = recs_df[recs_df['user_id'] == user_id]
                
                # 사용자에게 추천된 식당의 카테고리 목록
                rec_categories = [
                    restaurant_categories.get(rest_id) 
                    for rest_id in user_recs['restaurant_id'] 
                    if rest_id in restaurant_categories
                ]
                
                # 카테고리 쌍 간의 유사성 계산 (동일 카테고리면 1, 다르면 0)
                similarities = []
                for i in range(len(rec_categories)):
                    for j in range(i+1, len(rec_categories)):
                        if rec_categories[i] is not None and rec_categories[j] is not None:
                            sim = 1 if rec_categories[i] == rec_categories[j] else 0
                            similarities.append(sim)
                
                # 사용자별 평균 유사성
                user_similarity = np.mean(similarities) if similarities else 0
                intra_similarities.append(user_similarity)
            
            # 전체 사용자의 평균 유사성 (낮을수록 다양함)
            avg_intra_similarity = np.mean(intra_similarities) if intra_similarities else 0
            
            # 다양성 지표로 변환 (1 - 유사성)
            diversity_metrics['intra_list_similarity'] = 1 - avg_intra_similarity
        else:
            diversity_metrics['intra_list_similarity'] = 0
        
        logger.info(f"추천 다양성 평가 완료: {diversity_metrics}")
        return diversity_metrics
    
    except Exception as e:
        logger.error(f"추천 다양성 평가 중 오류: {e}", exc_info=True)
        return {
            "category_diversity": 0,
            "user_coverage": 0,
            "item_coverage": 0,
            "novelty": 0,
            "serendipity": 0,
            "intra_list_similarity": 0
        }