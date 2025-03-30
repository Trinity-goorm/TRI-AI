# app/servies/mongodb/data_converter.py

import json
import logging
import numpy as np
from datetime import datetime, date
from pathlib import Path

logger = logging.getLogger(__name__)

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

def convert_numpy_types(data):
    """NumPy 타입을 Python 네이티브 타입으로 변환"""
    if isinstance(data, dict):
        return {k: convert_numpy_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_types(v) for v in data]
    elif isinstance(data, np.integer):  # NumPy 정수형
        return int(data)
    elif isinstance(data, np.floating):  # NumPy 실수형
        return float(data)
    elif isinstance(data, np.ndarray):  # NumPy 배열
        return convert_numpy_types(data.tolist())
    elif isinstance(data, (datetime, date)):  # 날짜/시간 객체
        return data.isoformat()
    else:
        return data

def process_and_save_data(data, filepath, prefix):
    """데이터 전처리 및 파일 저장을 처리하는 헬퍼 함수"""
    try:
        # 바이트 데이터를 문자열로 변환
        data = convert_bytes_to_str(data)
        # 날짜/시간 객체와 NumPy 타입을 변환
        data = convert_datetime(data)
        data = convert_numpy_types(data)
        
        # 디렉토리 확인 및 생성
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # JSON으로 저장
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"{prefix} 저장 완료: {len(data)}개 항목")
        return True
    except Exception as e:
        logger.error(f"{prefix} 저장 중 오류: {e}")
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