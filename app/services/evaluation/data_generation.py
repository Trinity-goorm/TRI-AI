# app/services/evaluation/data_generation.py

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def create_test_interactions(globals_dict):
    """
    더 현실적인 사용자-식당 상호작용 테스트 데이터 생성
    
    Args:
        globals_dict: 전역 변수 딕셔너리
    
    Returns:
        pd.DataFrame: 상호작용 테스트 데이터
    """
    try:
        df_model = globals_dict.get("df_model")
        user_features_df = globals_dict.get("user_features_df")
        
        if df_model is None or user_features_df is None:
            logger.warning("테스트 데이터 생성에 필요한 데이터가 부족합니다.")
            return pd.DataFrame(columns=['user_id', 'restaurant_id', 'score'])
        
        # 모든 사용자와 식당 데이터 가져오기
        all_users = user_features_df['user_id'].tolist()
        all_restaurants = df_model['restaurant_id'].unique().tolist()
        
        # 사용자별 프로필 특성 (있다면) 활용
        user_preferences = {}
        
        if 'preferred_category' in user_features_df.columns:
            for _, user_row in user_features_df.iterrows():
                user_id = user_row['user_id']
                preferred_category = user_row.get('preferred_category')
                user_preferences[user_id] = preferred_category
        
        # 테스트 상호작용 생성
        test_interactions = []
        
        for user_id in all_users[:100]:  # 테스트를 위해 100명 사용자만 선택
            # 사용자별 특성에 맞는 식당 선택
            user_preferred_category = user_preferences.get(user_id)
            
            if user_preferred_category:
                # 사용자 선호 카테고리가 있으면 해당 카테고리의 식당들 중에서 선택
                category_restaurants = df_model[
                    df_model['category_id'] == user_preferred_category
                ]['restaurant_id'].tolist()
                
                if category_restaurants:
                    # 선호 카테고리 식당 선택 (60%)
                    preferred_count = max(1, int(np.random.randint(3, 6) * 0.6))
                    preferred_restaurants = np.random.choice(
                        category_restaurants, 
                        size=min(preferred_count, len(category_restaurants)),
                        replace=False
                    )
                    
                    # 기타 랜덤 식당 선택 (40%)
                    other_restaurants = list(set(all_restaurants) - set(category_restaurants))
                    other_count = np.random.randint(1, 3)
                    other_selected = np.random.choice(
                        other_restaurants,
                        size=min(other_count, len(other_restaurants)),
                        replace=False
                    )
                    
                    selected_restaurants = np.concatenate([preferred_restaurants, other_selected])
                else:
                    # 선호 카테고리 식당이 없으면 랜덤 선택
                    selected_restaurants = np.random.choice(
                        all_restaurants, 
                        size=np.random.randint(3, 6),
                        replace=False
                    )
            else:
                # 사용자 선호도 정보가 없으면 랜덤 선택
                selected_restaurants = np.random.choice(
                    all_restaurants, 
                    size=np.random.randint(3, 6),
                    replace=False
                )
            
            # 선택된 식당에 대한 가상 평점 생성 (3.0 ~ 5.0)
            for restaurant_id in selected_restaurants:
                # 선호 카테고리 식당은 더 높은 평점 확률
                if user_preferred_category and restaurant_id in category_restaurants:
                    score = np.random.uniform(4.0, 5.0)  # 선호 카테고리 식당은 평점 높게
                else:
                    score = np.random.uniform(3.0, 5.0)  # 기타 식당은 일반적인 분포
                
                test_interactions.append({
                    'user_id': user_id,
                    'restaurant_id': restaurant_id,
                    'score': round(score, 1)  # 소수점 첫째 자리까지
                })
        
        return pd.DataFrame(test_interactions)
    
    except Exception as e:
        logger.error(f"테스트 데이터 생성 중 오류: {e}", exc_info=True)
        return pd.DataFrame(columns=['user_id', 'restaurant_id', 'score'])

def create_stratified_train_test_split(df_model, test_ratio=0.2, random_state=42):
    """
    사용자와 식당 분포를 보존하는 층화 훈련/테스트 분할
    
    Args:
        df_model (DataFrame): 분할할 데이터프레임
        test_ratio (float): 테스트 데이터 비율 (0~1)
        random_state (int): 랜덤 시드
    
    Returns:
        tuple: (훈련 데이터프레임, 테스트 데이터프레임)
    """
    try:
        from sklearn.model_selection import GroupShuffleSplit
        
        logger.debug("층화 데이터 분할 시작...")
        
        # 필수 컬럼 확인
        required_cols = ['user_id', 'restaurant_id']
        for col in required_cols:
            if col not in df_model.columns:
                logger.error(f"데이터에 필수 컬럼이 없습니다: {col}")
                raise ValueError(f"데이터에 필수 컬럼이 없습니다: {col}")
        
        # 사용자별 데이터 분할
        users = df_model['user_id'].unique()
        logger.info(f"총 {len(users)}명의 사용자 데이터에 대해 분할 수행")
        
        # 사용자별 활동 수준 파악 (상호작용 횟수 기준)
        user_activity = df_model.groupby('user_id').size()
        low_activity = user_activity[user_activity <= 5].index.tolist()
        medium_activity = user_activity[(user_activity > 5) & (user_activity <= 20)].index.tolist()
        high_activity = user_activity[user_activity > 20].index.tolist()
        
        logger.debug(f"저활동 사용자: {len(low_activity)}명, 중활동: {len(medium_activity)}명, 고활동: {len(high_activity)}명")
        
        # 각 활동 그룹별로 층화 분할
        train_indices = []
        test_indices = []
        
        for activity_group in [low_activity, medium_activity, high_activity]:
            if not activity_group:
                continue
                
            # 해당 활동 그룹의 사용자 데이터만 선택
            group_data = df_model[df_model['user_id'].isin(activity_group)]
            group_users = group_data['user_id'].unique()
            
            # 사용자를 훈련/테스트로 분할
            np.random.seed(random_state)
            test_size = int(len(group_users) * test_ratio)
            test_users = np.random.choice(group_users, size=test_size, replace=False)
            train_users = np.array([u for u in group_users if u not in test_users])
            
            # 각 사용자별 데이터 인덱스 수집
            for user in train_users:
                user_indices = group_data[group_data['user_id'] == user].index.tolist()
                
                # 활동량이 많은 사용자는 일부 데이터만 훈련에 사용
                if user in high_activity and len(user_indices) > 30:
                    np.random.shuffle(user_indices)
                    train_indices.extend(user_indices[:30])  # 최대 30개 상호작용만 사용
                else:
                    train_indices.extend(user_indices)
            
            for user in test_users:
                user_indices = group_data[group_data['user_id'] == user].index.tolist()
                
                # 테스트용 사용자도 최대 10개 상호작용만 테스트에 사용
                if len(user_indices) > 10:
                    np.random.shuffle(user_indices)
                    test_indices.extend(user_indices[:10])
                else:
                    test_indices.extend(user_indices)
        
        # 훈련/테스트 데이터프레임 생성
        train_df = df_model.loc[train_indices].copy()
        test_df = df_model.loc[test_indices].copy()
        
        # 균형 확인
        logger.info(f"훈련 데이터: {len(train_df)} 상호작용, {train_df['user_id'].nunique()} 사용자")
        logger.info(f"테스트 데이터: {len(test_df)} 상호작용, {test_df['user_id'].nunique()} 사용자")
        
        # 점수(score) 컬럼이 없는 경우 가상의 점수 생성 (평가를 위해)
        if 'score' not in test_df.columns:
            logger.warning("테스트 데이터에 'score' 컬럼이 없어 가상 점수를 생성합니다")
            
            # 카테고리별 평균 점수 계산 (카테고리 정보가 있는 경우)
            if 'category_id' in test_df.columns and 'category_id' in train_df.columns:
                category_avg_scores = train_df.groupby('category_id')['score'].mean().to_dict()
                default_score = train_df['score'].mean()
                
                # 카테고리 기반 가상 점수 할당
                test_df['score'] = test_df['category_id'].map(
                    lambda x: category_avg_scores.get(x, default_score)
                )
                
                # 약간의 무작위성 추가 (±0.5)
                np.random.seed(random_state)
                test_df['score'] += np.random.uniform(-0.5, 0.5, size=len(test_df))
                test_df['score'] = test_df['score'].clip(1, 5)  # 1~5 범위로 제한
            else:
                # 카테고리 정보가 없으면 3~5 사이 무작위 점수
                test_df['score'] = np.random.uniform(3, 5, size=len(test_df))
        
        logger.debug("층화 데이터 분할 완료")
        return train_df, test_df
        
    except Exception as e:
        logger.error(f"층화 데이터 분할 중 오류 발생: {e}", exc_info=True)
        # 기본 분할 방식 적용
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df_model, test_size=test_ratio, random_state=random_state)
        return train_df, test_df