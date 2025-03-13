import pymysql
import json
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, date
from app.config import RESTAURANTS_DIR, USER_DIR
from app.config.queries import (
    RESTAURANT_QUERY,
    USER_QUERY,
    USER_PREFERENCES_QUERY,
    USER_PREFERENCE_CATEGORIES_QUERY,
    LIKES_QUERY,
    RESERVATIONS_QUERY
)

# SSH 터널링이 필요한 경우에만 import
try:
    import sshtunnel
    has_sshtunnel = True
except ImportError:
    has_sshtunnel = False

logger = logging.getLogger("data_sync")

load_dotenv()

class DateTimeEncoder(json.JSONEncoder):
    """날짜/시간 객체를 JSON으로 직렬화하기 위한 커스텀 인코더"""
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super().default(obj)

def convert_bytes_to_str(data):
    """데이터에서 bytes 타입을 문자열로 변환"""
    if isinstance(data, bytes):
        return data.decode('utf-8', errors='replace')
    elif isinstance(data, dict):
        return {key: convert_bytes_to_str(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_bytes_to_str(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(convert_bytes_to_str(item) for item in data)
    else:
        return data

def convert_datetime(data):
    """데이터에서 날짜/시간 객체를 문자열로 변환"""
    if isinstance(data, (datetime, date)):
        return data.isoformat()
    elif isinstance(data, dict):
        return {key: convert_datetime(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_datetime(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(convert_datetime(item) for item in data)
    else:
        return data

def process_and_save_data(data, filepath, prefix):
    """데이터 전처리 및 파일 저장을 처리하는 헬퍼 함수"""
    try:
        # 바이트 데이터를 문자열로 변환
        data = convert_bytes_to_str(data)
        # 날짜/시간 객체를 문자열로 변환
        data = convert_datetime(data)
        
        # JSON으로 저장
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"{prefix} 저장 완료: {len(data)}개 항목")
        return True
    except Exception as e:
        logger.error(f"{prefix} 저장 중 오류: {e}")
        return False

def fetch_data_from_rds():
    """RDS에서 식당 데이터와 사용자 관련 데이터를 가져와 JSON 파일로 저장"""
    # 환경 변수로 SSH 터널링 사용 여부 결정
    use_ssh_tunnel = os.environ.get('USE_SSH_TUNNEL', 'false').lower() == 'true'
    
    # SSH 터널링이 필요하지만 패키지가 설치되지 않은 경우
    if use_ssh_tunnel and not has_sshtunnel:
        logger.error("SSH 터널링이 필요하지만 'sshtunnel' 패키지가 설치되지 않았습니다. 'pip install sshtunnel'을 실행해주세요.")
        return False
    
    try:
        logger.info(f"RDS에서 데이터 가져오기 시작 {'(SSH 터널링 사용)' if use_ssh_tunnel else ''}")
        
        # SSH 터널링을 사용하는 경우
        if use_ssh_tunnel:
            return fetch_data_using_ssh_tunnel()
        # 직접 연결하는 경우 (같은 VPC 내부 등)
        else:
            return fetch_data_direct_connection()
    
    except Exception as e:
        logger.error(f"데이터 가져오기 오류: {str(e)}", exc_info=True)
        return False

def fetch_data_direct_connection():
    """직접 RDS에 연결하여 데이터 가져오기"""
    try:
        # RDS 연결 설정
        connection = pymysql.connect(
            host=os.environ.get('RDS_HOST', 'localhost'),
            port=int(os.environ.get('RDS_PORT', 3306)),
            user=os.environ.get('RDS_USER', 'username'),
            password=os.environ.get('RDS_PASSWORD', 'password'),
            database=os.environ.get('RDS_DATABASE', 'database'),
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor,
            connect_timeout=30
        )
        
        try:
            # 이하 데이터 처리 로직
            return process_data(connection)
            
        finally:
            connection.close()
    
    except Exception as e:
        logger.error(f"직접 연결 오류: {str(e)}", exc_info=True)
        return False

def fetch_data_using_ssh_tunnel():
    """SSH 터널을 통해 RDS에 연결하여 데이터 가져오기"""
    try:
        with sshtunnel.SSHTunnelForwarder(
            (os.environ.get('SSH_HOST'), int(os.environ.get('SSH_PORT', 22))),
            ssh_username=os.environ.get('SSH_USER'),
            ssh_password=os.environ.get('SSH_PASSWORD') if os.environ.get('SSH_PASSWORD') else None,
            ssh_pkey=os.environ.get('SSH_KEY_PATH') if os.environ.get('SSH_KEY_PATH') else None,
            remote_bind_address=(os.environ.get('RDS_HOST'), int(os.environ.get('RDS_PORT', 3306)))
        ) as tunnel:
            logger.info(f"SSH 터널 설정 완료 (local port: {tunnel.local_bind_port})")
            
            # RDS 연결 설정 (localhost와 터널링된 포트 사용)
            connection = pymysql.connect(
                host='127.0.0.1',  # 항상 로컬호스트, 터널을 통해 접속
                port=tunnel.local_bind_port,  # 터널이 할당한 로컬 포트
                user=os.environ.get('RDS_USER', 'username'),
                password=os.environ.get('RDS_PASSWORD', 'password'),
                database=os.environ.get('RDS_DATABASE', 'database'),
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor,
                connect_timeout=30
            )
            
            try:
                # 이하 데이터 처리 로직
                return process_data(connection)
                
            finally:
                connection.close()
    
    except Exception as e:
        logger.error(f"SSH 터널링 연결 오류: {str(e)}", exc_info=True)
        return False

def process_data(connection):
    """데이터베이스 연결을 받아 데이터를 처리하고 파일로 저장"""
    # 타임스탬프 생성 (모든 파일에 동일한 타임스탬프 사용)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 레스토랑 관련 데이터 처리
    success_restaurant = process_restaurant_data(connection, timestamp)
    
    # 2. 사용자 관련 데이터 처리 시도 (실패해도 전체 프로세스는 계속 진행)
    try:
        success_user = process_user_data(connection, timestamp)
        if success_user:
            logger.info("사용자 데이터 동기화 완료")
        else:
            logger.warning("일부 사용자 데이터 동기화 실패")
    except Exception as e:
        logger.warning(f"사용자 데이터 동기화 중 오류 발생: {e}")
        success_user = False
    
    # 레스토랑 데이터만 성공적으로 동기화되어도 전체 프로세스 성공으로 간주
    return success_restaurant

def process_restaurant_data(connection, timestamp):
    """레스토랑 관련 데이터 처리 및 저장"""
    try:
        # 디렉토리는 config에서 자동으로 생성됨
        restaurant_dir = Path(RESTAURANTS_DIR)
        
        with connection.cursor() as cursor:
            # 식당 데이터 쿼리 및 저장
            cursor.execute(RESTAURANT_QUERY)
            restaurant_data = cursor.fetchall()
            
            # 데이터 처리 및 저장
            rest_filepath = restaurant_dir / f"restaurant_data_{timestamp}.json"
            success = process_and_save_data(
                restaurant_data, 
                rest_filepath, 
                "식당 데이터"
            )
            
            if success:
                # 오래된 파일 정리 (최신 3개만 유지)
                cleanup_old_files(str(restaurant_dir), "restaurant_data_", 3)
                return True
            else:
                return False
            
    except Exception as e:
        logger.error(f"레스토랑 데이터 처리 오류: {str(e)}", exc_info=True)
        return False

def process_user_data(connection, timestamp):
    """사용자 관련 데이터 처리 및 저장 (개별 쿼리별 오류 처리)"""
    # 디렉토리는 config에서 자동으로 생성됨
    user_dir = Path(USER_DIR)
    
    # 적어도 하나의 쿼리가 성공했는지 추적
    success_count = 0
    total_queries = 5  # 총 실행할 쿼리 수
    
    with connection.cursor() as cursor:
        # 1. 사용자 기본 정보 쿼리 및 저장
        try:
            cursor.execute(USER_QUERY)
            user_data = cursor.fetchall()
            
            user_filepath = user_dir / f"user_data_{timestamp}.json"
            if process_and_save_data(user_data, user_filepath, "사용자 기본 정보"):
                success_count += 1
                # 파일이 생성되었다면 정리 로직도 실행
                cleanup_old_files(str(user_dir), "user_data_", 3)
        except Exception as e:
            logger.warning(f"사용자 기본 정보 처리 실패: {e}")
        
        # 2. 사용자 가격 범위 선호도 데이터 쿼리 및 저장
        try:
            cursor.execute(USER_PREFERENCES_QUERY)
            user_preferences_data = cursor.fetchall()
            
            pref_filepath = user_dir / f"user_preferences_{timestamp}.json"
            if process_and_save_data(user_preferences_data, pref_filepath, "사용자 가격 범위 선호도 데이터"):
                success_count += 1
                # 파일이 생성되었다면 정리 로직도 실행
                cleanup_old_files(str(user_dir), "user_preferences_", 3)
        except Exception as e:
            logger.warning(f"사용자 가격 범위 선호도 데이터 처리 실패: {e}")
        
        # 3. 사용자 카테고리 선호도 데이터 쿼리 및 저장
        try:
            cursor.execute(USER_PREFERENCE_CATEGORIES_QUERY)
            user_preference_categories_data = cursor.fetchall()
            
            cat_pref_filepath = user_dir / f"user_preference_categories_{timestamp}.json"
            if process_and_save_data(user_preference_categories_data, cat_pref_filepath, "사용자 카테고리 선호도 데이터"):
                success_count += 1
                # 파일이 생성되었다면 정리 로직도 실행
                cleanup_old_files(str(user_dir), "user_preference_categories_", 3)
        except Exception as e:
            logger.warning(f"사용자 카테고리 선호도 데이터 처리 실패: {e}")
        
        # 4. 찜 데이터 쿼리 및 저장
        try:
            cursor.execute(LIKES_QUERY)
            likes_data = cursor.fetchall()
            
            likes_filepath = user_dir / f"likes_{timestamp}.json"
            if process_and_save_data(likes_data, likes_filepath, "찜 데이터"):
                success_count += 1
                # 파일이 생성되었다면 정리 로직도 실행
                cleanup_old_files(str(user_dir), "likes_", 3)
        except Exception as e:
            logger.warning(f"찜 데이터 처리 실패: {e}")
        
        # 5. 예약 데이터 쿼리 및 저장
        try:
            cursor.execute(RESERVATIONS_QUERY)
            reservations_data = cursor.fetchall()
            
            reservations_filepath = user_dir / f"reservations_{timestamp}.json"
            if process_and_save_data(reservations_data, reservations_filepath, "예약 데이터"):
                success_count += 1
                # 파일이 생성되었다면 정리 로직도 실행
                cleanup_old_files(str(user_dir), "reservations_", 3)
        except Exception as e:
            logger.warning(f"예약 데이터 처리 실패: {e}")
    
    # 일부 쿼리라도 성공했으면 일부 성공으로 간주
    if success_count > 0:
        logger.info(f"사용자 데이터 동기화: {success_count}/{total_queries} 쿼리 성공")
        return True
    else:
        logger.warning("모든 사용자 데이터 쿼리 실패")
        return False

def cleanup_old_files(directory, prefix, keep_count):
    """특정 접두사를 가진 파일 중 최신 파일 n개만 유지"""
    try:
        dir_path = Path(directory)
        files = [f for f in dir_path.glob(f"{prefix}*.json")]
        
        # 파일의 생성 시간에 따라 정렬
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # 유지할 파일 수보다 많으면 오래된 파일 삭제
        if len(files) > keep_count:
            for old_file in files[keep_count:]:
                old_file.unlink()
                logger.info(f"오래된 파일 삭제: {old_file}")
    
    except Exception as e:
        logger.error(f"파일 정리 중 오류: {str(e)}", exc_info=True)