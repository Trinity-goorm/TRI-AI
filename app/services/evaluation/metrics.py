# app/services/evaluation/metrics.py

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. 평점 예측 기반 지표: MAE, RMSE
def calculate_rating_metrics(y_true, y_pred):
    """
    평점 예측 성능 지표 계산
    
    Args:
        y_true: 실제 평점 값
        y_pred: 예측 평점 값
    
    Returns:
        dict: MAE, RMSE 지표
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    return {
        'MAE': mae,
        'RMSE': rmse
    }

# 2. 랭킹 기반 지표: Precision@K, Recall@K, NDCG@K, Hit Rate@K
def precision_at_k(recommended_items, relevant_items, k):
    """
    Precision@K 계산
    
    Args:
        recommended_items: 추천 아이템 리스트
        relevant_items: 관련 있는(실제로 상호작용한) 아이템 리스트
        k: 상위 K개 아이템 고려
    
    Returns:
        float: Precision@K 값
    """
    if len(recommended_items) == 0:
        return 0.0
    
    # 상위 K개 아이템만 고려
    recommended_k = recommended_items[:k]
    
    # 관련 있는 아이템 중 추천된 것의 수
    relevant_and_recommended = len(set(recommended_k) & set(relevant_items))
    
    return relevant_and_recommended / min(k, len(recommended_items))

def recall_at_k(recommended_items, relevant_items, k):
    """
    Recall@K 계산
    
    Args:
        recommended_items: 추천 아이템 리스트
        relevant_items: 관련 있는(실제로 상호작용한) 아이템 리스트
        k: 상위 K개 아이템 고려
    
    Returns:
        float: Recall@K 값
    """
    if len(relevant_items) == 0:
        return 0.0
    
    # 상위 K개 아이템만 고려
    recommended_k = recommended_items[:k]
    
    # 관련 있는 아이템 중 추천된 것의 수
    relevant_and_recommended = len(set(recommended_k) & set(relevant_items))
    
    return relevant_and_recommended / len(relevant_items)

def ndcg_at_k(recommended_items, relevant_items, k):
    """
    NDCG@K (Normalized Discounted Cumulative Gain) 계산
    
    Args:
        recommended_items: 추천 아이템 리스트
        relevant_items: 관련 있는(실제로 상호작용한) 아이템 리스트
        k: 상위 K개 아이템 고려
    
    Returns:
        float: NDCG@K 값
    """
    if len(relevant_items) == 0:
        return 0.0
    
    # 상위 K개 아이템만 고려
    recommended_k = recommended_items[:k]
    
    # DCG 계산 (Discounted Cumulative Gain)
    dcg = 0
    for i, item in enumerate(recommended_k):
        # 추천 아이템이 관련 있는 경우 1, 아니면 0
        rel = 1 if item in relevant_items else 0
        # i+1을 사용하는 이유: 인덱스는 0부터 시작하지만 순위는 1부터 시작
        dcg += rel / np.log2(i + 2)  # log_2(rank + 1)
    
    # 이상적인 추천 순서 생성 (관련 아이템이 먼저 추천됨)
    ideal_ordering = sorted(recommended_k, key=lambda x: 1 if x in relevant_items else 0, reverse=True)
    
    # IDCG 계산 (Ideal DCG)
    idcg = 0
    for i, item in enumerate(ideal_ordering):
        rel = 1 if item in relevant_items else 0
        idcg += rel / np.log2(i + 2)
    
    # IDCG가 0이면 NDCG는 0
    if idcg == 0:
        return 0.0
    
    return dcg / idcg

def hit_rate_at_k(recommended_items, relevant_items, k):
    """
    Hit Rate@K 계산
    
    Args:
        recommended_items: 추천 아이템 리스트
        relevant_items: 관련 있는(실제로 상호작용한) 아이템 리스트
        k: 상위 K개 아이템 고려
    
    Returns:
        float: Hit Rate@K 값 (1 또는 0)
    """
    # 상위 K개 아이템만 고려
    recommended_k = recommended_items[:k]
    
    # 관련 아이템 중 하나라도 추천 목록에 있으면 1, 아니면 0
    return 1.0 if len(set(recommended_k) & set(relevant_items)) > 0 else 0.0

# 모든 랭킹 지표를 종합적으로 계산하는 함수
def calculate_ranking_metrics(recommendations_dict, test_interactions, k_values=[5, 10]):
    """
    사용자별 추천 결과에 대한 랭킹 지표 계산
    
    Args:
        recommendations_dict: 사용자별 추천 아이템 딕셔너리 {user_id: [restaurant_id, ...]}
        test_interactions: 테스트 데이터 (사용자가 실제로 상호작용한 아이템)
        k_values: 평가할 K 값 리스트
    
    Returns:
        dict: 계산된 모든 랭킹 지표
    """
    # 사용자별 관련 아이템 딕셔너리 구성
    user_relevant_items = {}
    for _, row in test_interactions.iterrows():
        user_id = row['user_id']
        restaurant_id = row['restaurant_id']
        
        if user_id not in user_relevant_items:
            user_relevant_items[user_id] = []
        
        user_relevant_items[user_id].append(restaurant_id)
    
    # 결과 저장을 위한 딕셔너리
    results = {}
    
    for k in k_values:
        precision_sum = 0
        recall_sum = 0
        ndcg_sum = 0
        hit_rate_sum = 0
        user_count = 0
        
        for user_id, recommended_items in recommendations_dict.items():
            if user_id in user_relevant_items:
                relevant_items = user_relevant_items[user_id]
                
                precision_sum += precision_at_k(recommended_items, relevant_items, k)
                recall_sum += recall_at_k(recommended_items, relevant_items, k)
                ndcg_sum += ndcg_at_k(recommended_items, relevant_items, k)
                hit_rate_sum += hit_rate_at_k(recommended_items, relevant_items, k)
                user_count += 1
        
        if user_count > 0:
            results[f'Precision@{k}'] = precision_sum / user_count
            results[f'Recall@{k}'] = recall_sum / user_count
            results[f'NDCG@{k}'] = ndcg_sum / user_count
            results[f'Hit_Rate@{k}'] = hit_rate_sum / user_count
        else:
            results[f'Precision@{k}'] = 0.0
            results[f'Recall@{k}'] = 0.0
            results[f'NDCG@{k}'] = 0.0
            results[f'Hit_Rate@{k}'] = 0.0
    
    return results

def calculate_segment_performance(recommendations_dict, test_interactions, k_values=[5, 10, 15]):
    """
    사용자 세그먼트별 추천 시스템 성능 평가
    
    Args:
        recommendations_dict: 사용자별 추천 아이템 딕셔너리 {user_id: [restaurant_id, ...]}
        test_interactions: 테스트 데이터 (사용자가 실제로 상호작용한 아이템)
        k_values: 평가할 K 값 리스트
    
    Returns:
        dict: 사용자 세그먼트별 성능 지표
    """
    # 사용자 세그먼트 정의 함수
    def categorize_user(user_id, interactions):
        """
        사용자 세그먼트 분류
        
        Args:
            user_id: 사용자 ID
            interactions: 사용자 상호작용 데이터
        
        Returns:
            str: 사용자 세그먼트 ('new', 'active', 'inactive')
        """
        user_interactions = interactions[interactions['user_id'] == user_id]
        
        if len(user_interactions) == 0:
            return 'new'
        elif len(user_interactions) > 10:
            return 'active'
        else:
            return 'inactive'
    
    # 사용자별 관련 아이템 딕셔너리 구성
    user_relevant_items = {}
    for _, row in test_interactions.iterrows():
        user_id = row['user_id']
        restaurant_id = row['restaurant_id']
        
        if user_id not in user_relevant_items:
            user_relevant_items[user_id] = []
        
        user_relevant_items[user_id].append(restaurant_id)
    
    # 세그먼트별 성능 저장 딕셔너리
    segment_metrics = {
        'new': {f'Precision@{k}': [] for k in k_values} | 
               {f'Recall@{k}': [] for k in k_values} | 
               {f'NDCG@{k}': [] for k in k_values} | 
               {f'Hit_Rate@{k}': [] for k in k_values},
        'active': {f'Precision@{k}': [] for k in k_values} | 
                  {f'Recall@{k}': [] for k in k_values} | 
                  {f'NDCG@{k}': [] for k in k_values} | 
                  {f'Hit_Rate@{k}': [] for k in k_values},
        'inactive': {f'Precision@{k}': [] for k in k_values} | 
                    {f'Recall@{k}': [] for k in k_values} | 
                    {f'NDCG@{k}': [] for k in k_values} | 
                    {f'Hit_Rate@{k}': [] for k in k_values}
    }
    
    # 각 사용자별 세그먼트 성능 계산
    for user_id, recommended_items in recommendations_dict.items():
        # 사용자 세그먼트 분류
        segment = categorize_user(user_id, test_interactions)
        
        # 해당 사용자의 관련 아이템
        if user_id not in user_relevant_items:
            continue
        
        relevant_items = user_relevant_items[user_id]
        
        # K 값별 성능 계산
        for k in k_values:
            segment_metrics[segment][f'Precision@{k}'].append(
                precision_at_k(recommended_items, relevant_items, k)
            )
            segment_metrics[segment][f'Recall@{k}'].append(
                recall_at_k(recommended_items, relevant_items, k)
            )
            segment_metrics[segment][f'NDCG@{k}'].append(
                ndcg_at_k(recommended_items, relevant_items, k)
            )
            segment_metrics[segment][f'Hit_Rate@{k}'].append(
                hit_rate_at_k(recommended_items, relevant_items, k)
            )
    
    # 세그먼트별 평균 성능 계산
    segment_performance = {}
    for segment, metrics in segment_metrics.items():
        segment_performance[segment] = {}
        for metric, values in metrics.items():
            segment_performance[segment][metric] = (
                np.mean(values) if values else 0.0
            )
    
    return segment_performance