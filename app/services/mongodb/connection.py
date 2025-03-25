# app/servies/mongodb/connection.py

import os
import logging
from dotenv import load_dotenv
from pymongo import MongoClient

# SSH 터널링이 필요한 경우에만 import
try:
    import sshtunnel
    has_sshtunnel = True
except ImportError:
    has_sshtunnel = False

logger = logging.getLogger("mongodb.connection")
load_dotenv()

def get_mongodb_connection():
    """SSH 터널링 설정에 따라 MongoDB 연결 생성"""
    # 환경 변수로 SSH 터널링 사용 여부 결정
    use_ssh_tunnel = os.environ.get('USE_SSH_TUNNEL', 'true').lower() == 'true'
    
    # SSH 터널링이 필요하지만 패키지가 설치되지 않은 경우
    if use_ssh_tunnel and not has_sshtunnel:
        logger.error("SSH 터널링이 필요하지만 'sshtunnel' 패키지가 설치되지 않았습니다. 'pip install sshtunnel'을 실행해주세요.")
        raise ImportError("sshtunnel 패키지가 필요합니다.")
    
    # SSH 터널링을 사용하는 경우
    if use_ssh_tunnel:
        return get_mongodb_via_ssh()
    # 직접 연결하는 경우
    else:
        return get_mongodb_direct()

def get_mongodb_direct():
    """직접 MongoDB에 연결"""
    # MongoDB 연결 정보
    mongo_host = os.environ.get('MONGO_HOST')
    mongo_port = int(os.environ.get('MONGO_PORT', 27017))
    mongo_user = os.environ.get('MONGO_USER')
    mongo_password = os.environ.get('MONGO_PASSWORD')
    mongo_database = os.environ.get('MONGO_DATABASE')
    
    # 필수 환경 변수 확인
    if not all([mongo_host, mongo_user, mongo_password, mongo_database]):
        logger.error("MongoDB 연결에 필요한 환경 변수가 설정되지 않았습니다.")
        raise ValueError("MongoDB 연결 설정이 부족합니다.")
    
    # MongoDB 연결 문자열 생성
    mongo_uri = f"mongodb://{mongo_user}:{mongo_password}@{mongo_host}:{mongo_port}/{mongo_database}"
    
    # MongoDB 클라이언트 연결
    client = MongoClient(mongo_uri)
    db = client[mongo_database]
    logger.info(f"MongoDB {mongo_database} 직접 연결 성공")
    
    return client, db

def get_mongodb_via_ssh():
    """SSH 터널을 통해 MongoDB에 연결"""
    # SSH 연결 정보
    ssh_host = os.environ.get('SSH_HOST')
    ssh_port = int(os.environ.get('SSH_PORT', 22))
    ssh_user = os.environ.get('SSH_USER')
    ssh_password = os.environ.get('SSH_PASSWORD')
    ssh_key_path = os.environ.get('SSH_KEY_PATH')
    
    # MongoDB 연결 정보
    mongo_host = os.environ.get('MONGO_HOST')
    mongo_port = int(os.environ.get('MONGO_PORT', 27017))
    mongo_user = os.environ.get('MONGO_USER')
    mongo_password = os.environ.get('MONGO_PASSWORD')
    mongo_database = os.environ.get('MONGO_DATABASE')
    
    # 필수 환경 변수 확인
    if not all([ssh_host, ssh_user, mongo_host, mongo_user, mongo_password, mongo_database]):
        logger.error("SSH 터널링 또는 MongoDB 연결에 필요한 환경 변수가 설정되지 않았습니다.")
        raise ValueError("SSH 또는 MongoDB 연결 설정이 부족합니다.")
        
    # SSH 인증 방식 확인 (비밀번호 또는 키 파일)
    if not ssh_password and not ssh_key_path:
        logger.error("SSH 연결을 위한 비밀번호 또는 키 파일 경로가 필요합니다.")
        raise ValueError("SSH 인증 정보가 부족합니다.")
    
    # SSH 터널 설정
    tunnel = sshtunnel.SSHTunnelForwarder(
        (ssh_host, ssh_port),
        ssh_username=ssh_user,
        ssh_password=ssh_password if ssh_password else None,
        ssh_pkey=ssh_key_path if ssh_key_path else None,
        remote_bind_address=(mongo_host, mongo_port)
    )
    
    # 터널 시작
    tunnel.start()
    logger.info(f"SSH 터널 설정 완료 (local port: {tunnel.local_bind_port})")
    
    # MongoDB 연결 문자열 생성 (localhost와 터널링된 포트 사용)
    mongo_uri = f"mongodb://{mongo_user}:{mongo_password}@127.0.0.1:{tunnel.local_bind_port}/{mongo_database}"
    
    try:
        # MongoDB 클라이언트 연결
        client = MongoClient(mongo_uri)
        db = client[mongo_database]
        logger.info(f"MongoDB {mongo_database} SSH 터널 연결 성공")
        
        # 터널과 클라이언트를 함께 반환 (종료를 위해)
        return client, db, tunnel
    except Exception as e:
        # 오류 발생 시 터널 종료
        tunnel.stop()
        raise e