import numpy as np
import pandas as pd
import logging
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

def build_hybrid_recommender(df_ratings, df_restaurants):
    """
    협업 필터링과 콘텐츠 기반 추천을 결합한 하이브리드 추천 모델 구축
    
    Args:
        df_ratings: 사용자-식당 평점 데이터 (user_id, restaurant_id, score 컬럼 필요)
        df_restaurants: 식당 메타데이터 (restaurant_id, category_id 등 특성 포함)
    
    Returns:
        callable: 하이브리드 추천 함수
    """
    try:
        logger.info("하이브리드 추천 모델 구축 시작")
        
        # 1. 협업 필터링 (사용자-아이템 매트릭스 기반)
        logger.debug("협업 필터링 모델 구축 중...")
        # 평점 데이터 확인
        if df_ratings.empty or 'score' not in df_ratings.columns:
            raise ValueError("평점 데이터가 비어있거나 필수 컬럼이 없습니다")
            
        # 사용자-아이템 평점 매트릭스 생성
        user_item_matrix = df_ratings.pivot_table(
            index='user_id', 
            columns='restaurant_id', 
            values='score', 
            fill_value=0
        )
        
        # 협업 필터링 유사도 계산
        # 메모리 관리를 위해 실제 구현 시 이 부분을 최적화할 수 있음
        cf_user_similarity = cosine_similarity(user_item_matrix)
        cf_user_similarity = pd.DataFrame(
            cf_user_similarity,
            index=user_item_matrix.index,
            columns=user_item_matrix.index
        )
        
        # 아이템 유사도 계산
        cf_item_similarity = cosine_similarity(user_item_matrix.T)
        cf_item_similarity = pd.DataFrame(
            cf_item_similarity,
            index=user_item_matrix.columns,
            columns=user_item_matrix.columns
        )
        
        logger.debug(f"협업 필터링 모델 구축 완료: {user_item_matrix.shape[0]}명의 사용자, {user_item_matrix.shape[1]}개의 식당")
        
        # 2. 콘텐츠 기반 필터링
        logger.debug("콘텐츠 기반 필터링 모델 구축 중...")
        # 식당 메타 데이터 준비
        if df_restaurants.empty:
            raise ValueError("식당 메타데이터가 비어있습니다")
        
        # 콘텐츠 기반 필터링을 위한 특성 선택
        content_features = ['category_id']
        
        # 편의 시설, 주의사항 등의 특성 추가
        convenience_cols = [col for col in df_restaurants.columns if col.startswith('conv_')]
        caution_cols = [col for col in df_restaurants.columns if col.startswith('caution_')]
        
        content_features.extend(convenience_cols)
        content_features.extend(caution_cols)
        
        # 모든 특성이 존재하는지 확인
        valid_features = [f for f in content_features if f in df_restaurants.columns]
        
        # 유효한 특성이 없으면 카테고리만 사용
        if not valid_features:
            if 'category_id' in df_restaurants.columns:
                valid_features = ['category_id']
            else:
                raise ValueError("콘텐츠 기반 필터링에 사용할 특성이 없습니다")
        
        # 중복 제거된 식당 데이터 준비
        restaurant_features = df_restaurants.drop_duplicates('restaurant_id')
        restaurant_features = restaurant_features.set_index('restaurant_id')
        
        # 범주형 변수 원-핫 인코딩
        categorical_features = ['category_id']
        for feature in categorical_features:
            if feature in restaurant_features.columns:
                # 원-핫 인코딩
                dummies = pd.get_dummies(restaurant_features[feature], prefix=feature)
                restaurant_features = pd.concat([
                    restaurant_features.drop(feature, axis=1),
                    dummies
                ], axis=1)
        
        # 콘텐츠 기반 유사도 계산
        # 콘텐츠 특성 선택
        content_cols = [col for col in restaurant_features.columns 
                      if any(col.startswith(f"{feature}_") for feature in categorical_features) 
                      or col in valid_features]
        
        if not content_cols:
            logger.warning("콘텐츠 특성이 없어 기본 특성 사용")
            content_cols = restaurant_features.columns[:5]  # 첫 5개 컬럼 사용
        
        # 유사도 계산
        content_similarity = cosine_similarity(restaurant_features[content_cols])
        content_similarity = pd.DataFrame(
            content_similarity,
            index=restaurant_features.index,
            columns=restaurant_features.index
        )
        
        logger.debug(f"콘텐츠 기반 모델 구축 완료: {len(restaurant_features)}개 식당, {len(content_cols)}개 특성")
        
        # 3. 하이브리드 추천 함수
        def hybrid_recommend(user_id, n=15, alpha=0.7):
            """
            하이브리드 방식으로 식당 추천
            
            Args:
                user_id: 사용자 ID
                n: 추천할 식당 수
                alpha: 협업 필터링 가중치 (0~1), 1-alpha는 콘텐츠 기반 가중치
                
            Returns:
                list: 추천된 식당 ID 리스트
            """
            try:
                # 사용자 ID가 문자열이면 정수로 변환 시도
                if isinstance(user_id, str):
                    try:
                        user_id = int(user_id)
                    except ValueError:
                        pass
                
                # A. 협업 필터링 점수 계산
                cf_scores = {}
                
                # 사용자가 평점 매트릭스에 있는 경우 (기존 사용자)
                if user_id in cf_user_similarity.index:
                    # 1. 유사 사용자 기반 추천
                    similar_users = cf_user_similarity[user_id].sort_values(ascending=False).index[1:11]  # 자신 제외 상위 10명
                    
                    # 2. 유사 사용자들의 평점 가중 평균 계산
                    user_ratings = user_item_matrix.loc[user_id]
                    similar_users_ratings = user_item_matrix.loc[similar_users]
                    user_similarities = cf_user_similarity[user_id].loc[similar_users]
                    
                    # 아직 평가하지 않은 식당만 추천 대상
                    unrated_items = user_ratings[user_ratings == 0].index
                    
                    for item in unrated_items:
                        # 유사 사용자들 중 해당 식당을 평가한 사용자들만 사용
                        item_ratings = similar_users_ratings[item]
                        relevant_users = item_ratings[item_ratings > 0].index
                        
                        if len(relevant_users) > 0:
                            # 유사도 가중 평균 계산
                            relevant_similarities = user_similarities.loc[relevant_users]
                            relevant_ratings = item_ratings.loc[relevant_users]
                            
                            if relevant_similarities.sum() > 0:
                                cf_scores[item] = (relevant_similarities * relevant_ratings).sum() / relevant_similarities.sum()
                            else:
                                cf_scores[item] = relevant_ratings.mean()
                    
                    # 3. 아이템 기반 협업 필터링 추가
                    # 사용자가 이미 평가한 식당
                    rated_items = user_ratings[user_ratings > 0].index
                    
                    for item in unrated_items:
                        if item not in cf_scores and item in cf_item_similarity.columns:
                            # 이미 평가한 식당과의 유사성 기반 점수 계산
                            item_similarities = cf_item_similarity[item].loc[rated_items]
                            item_ratings = user_ratings.loc[rated_items]
                            
                            if item_similarities.sum() > 0:
                                cf_scores[item] = (item_similarities * item_ratings).sum() / item_similarities.sum()
                else:
                    # 신규 사용자는 협업 필터링 점수 없음
                    logger.debug(f"사용자 {user_id}는 협업 필터링 데이터가 없습니다")
                
                # B. 콘텐츠 기반 점수 계산
                cb_scores = {}
                
                # 1. 사용자가 이미 평가한 식당이 있는 경우
                user_data = df_ratings[df_ratings['user_id'] == user_id]
                
                if not user_data.empty:
                    # 평점이 높은 순으로 사용자가 평가한 식당 정렬
                    user_favorites = user_data.sort_values('score', ascending=False)
                    top_restaurants = user_favorites.head(5)['restaurant_id'].tolist()
                    
                    # 이미 평가한 식당과 유사한 식당 추천
                    for rest_id in top_restaurants:
                        if rest_id in content_similarity.index:
                            similar_restaurants = content_similarity[rest_id].sort_values(ascending=False)
                            
                            for similar_id, similarity in similar_restaurants.items():
                                if similar_id != rest_id:  # 자기 자신 제외
                                    if similar_id not in cb_scores:
                                        cb_scores[similar_id] = 0
                                    
                                    # 평가한 식당의 평점과 유사도를 곱하여 점수 계산
                                    rest_score = user_data[user_data['restaurant_id'] == rest_id]['score'].iloc[0]
                                    cb_scores[similar_id] += similarity * rest_score
                else:
                    # 2. 사용자 평가 데이터가 없는 경우 (신규 사용자)
                    # 전체 평균 평점으로 인기 식당 추천
                    popular_restaurants = df_ratings.groupby('restaurant_id')['score'].agg(['mean', 'count'])
                    popular_restaurants = popular_restaurants[popular_restaurants['count'] >= 5]  # 최소 5개 이상 평가
                    popular_restaurants['popularity'] = popular_restaurants['mean'] * np.log1p(popular_restaurants['count'])
                    
                    for rest_id, row in popular_restaurants.sort_values('popularity', ascending=False).head(20).iterrows():
                        cb_scores[rest_id] = row['popularity']
                
                # C. 하이브리드 점수 계산
                hybrid_scores = {}
                
                # 모든 식당 ID 수집
                all_restaurant_ids = set(list(cf_scores.keys()) + list(cb_scores.keys()))
                
                for rest_id in all_restaurant_ids:
                    # 협업 필터링 점수 (없으면 0)
                    cf_score = cf_scores.get(rest_id, 0)
                    
                    # 콘텐츠 기반 점수 (없으면 0)
                    cb_score = cb_scores.get(rest_id, 0)
                    
                    # 하이브리드 점수 계산 (알파 가중 평균)
                    if cf_score > 0 and cb_score > 0:
                        # 둘 다 점수가 있으면 가중 평균
                        hybrid_scores[rest_id] = alpha * cf_score + (1 - alpha) * cb_score
                    elif cf_score > 0:
                        # 협업 필터링 점수만 있으면 그대로 사용
                        hybrid_scores[rest_id] = cf_score
                    elif cb_score > 0:
                        # 콘텐츠 기반 점수만 있으면 그대로 사용
                        hybrid_scores[rest_id] = cb_score
                
                # 이미 평가한 식당 제외
                rated_items = df_ratings[df_ratings['user_id'] == user_id]['restaurant_id'].tolist()
                for item in rated_items:
                    if item in hybrid_scores:
                        del hybrid_scores[item]
                
                # 점수 기준 상위 n개 식당 추천
                recommended_items = sorted(
                    hybrid_scores.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:n]
                
                # 식당 ID만 추출
                recommended_ids = [rest_id for rest_id, _ in recommended_items]
                
                # 추천 결과가 부족하면 인기 식당으로 보충
                if len(recommended_ids) < n:
                    needed = n - len(recommended_ids)
                    
                    # 인기 식당 계산
                    popular_rest = df_ratings.groupby('restaurant_id')['score'].mean().sort_values(ascending=False)
                    
                    # 이미 추천한 식당과 평가한 식당 제외
                    excluded_ids = set(recommended_ids + rated_items)
                    additional_ids = [
                        rest_id for rest_id in popular_rest.index 
                        if rest_id not in excluded_ids
                    ][:needed]
                    
                    recommended_ids.extend(additional_ids)
                
                logger.debug(f"사용자 {user_id}에게 {len(recommended_ids)}개 식당 하이브리드 추천 생성")
                return recommended_ids
                
            except Exception as e:
                logger.error(f"하이브리드 추천 생성 중 오류: {e}", exc_info=True)
                
                # 오류 발생 시 인기 식당 기반 추천으로 대체
                popular_rest = df_ratings.groupby('restaurant_id')['score'].mean().sort_values(ascending=False)
                return popular_rest.head(n).index.tolist()
        
        logger.info("하이브리드 추천 모델 구축 완료")
        return hybrid_recommend
        
    except Exception as e:
        logger.error(f"하이브리드 추천 모델 구축 중 오류: {e}", exc_info=True)
        # 오류 발생 시 기본 추천 함수 반환
        def fallback_recommend(user_id, n=15, **kwargs):
            # 평점 기준 인기 식당 추천
            popular_rest = df_ratings.groupby('restaurant_id')['score'].mean().sort_values(ascending=False)
            return popular_rest.head(n).index.tolist()
            
        return fallback_recommend


def generate_hybrid_recommendations(df_ratings, df_restaurants, user_id, n=15, alpha=0.7):
    """
    하이브리드 추천 모델을 사용하여 추천 생성
    
    Args:
        df_ratings: 사용자-식당 평점 데이터
        df_restaurants: 식당 메타데이터
        user_id: 추천 대상 사용자 ID
        n: 추천할 식당 수
        alpha: 협업 필터링 가중치 (0~1)
        
    Returns:
        dict: 추천 결과 딕셔너리
    """
    try:
        # 하이브리드 추천 모델 구축
        hybrid_recommend = build_hybrid_recommender(df_ratings, df_restaurants)
        
        # 추천 생성
        recommended_items = hybrid_recommend(user_id, n=n, alpha=alpha)
        
        # 추천 식당 정보 수집
        recommendations = []
        
        for i, rest_id in enumerate(recommended_items):
            # 식당 정보 추출
            rest_data = df_restaurants[df_restaurants['restaurant_id'] == rest_id]
            if rest_data.empty:
                continue
                
            # 카테고리 ID
            category_id = int(rest_data['category_id'].iloc[0]) if 'category_id' in rest_data.columns else -1
            
            # 점수 정보
            if 'score' in rest_data.columns:
                score = float(rest_data['score'].mean())
            else:
                # 평점 데이터에서 점수 가져오기
                rest_ratings = df_ratings[df_ratings['restaurant_id'] == rest_id]
                score = float(rest_ratings['score'].mean()) if not rest_ratings.empty else 4.0
            
            # 추천 식당 정보
            recommendations.append({
                "category_id": category_id,
                "restaurant_id": int(rest_id),
                "score": score,
                "predicted_score": score,
                "composite_score": 5.0 - (i * 0.15)  # 순위에 따라 점수 부여 (5.0~2.75)
            })
        
        # 결과 구성
        is_new_user = len(df_ratings[df_ratings['user_id'] == user_id]) == 0
        
        result_dict = {
            "user": int(user_id) if isinstance(user_id, (int, float)) else user_id,
            "is_new_user": is_new_user,
            "recommendations": recommendations
        }
        
        return result_dict
        
    except Exception as e:
        logger.error(f"하이브리드 추천 생성 중 오류: {e}", exc_info=True)
        # 기본 결과 반환
        return {
            "user": int(user_id) if isinstance(user_id, (int, float)) else user_id,
            "is_new_user": True,
            "recommendations": []
        }