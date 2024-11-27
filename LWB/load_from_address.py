import time
import requests
import pandas as pd

# Kakao API Geocoding 함수
def geocode_kakao(address, api_key):
    """
    Kakao Local API를 사용해 주소를 위도, 경도로 변환하는 함수
    :param address: 도로명 주소
    :param api_key: Kakao REST API Key
    :return: (latitude, longitude) 좌표 튜플
    """
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = {"Authorization": f"KakaoAK {api_key}"}
    params = {"query": address}
    
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code == 200:
        result = response.json()
        if result['documents']:
            x = float(result['documents'][0]['x'])  # 경도 (longitude)
            y = float(result['documents'][0]['y'])  # 위도 (latitude)
            return y, x
        else:
            return None, None  # 결과 없음
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None, None

import sys

# year = sys.argv[1]
address_df = pd.read_csv('school.csv', encoding="utf-8")

# 결과 저장할 리스트
coordinates = []

for idx, row in enumerate(address_df.values):

    # API Key 설정
    kakao_api_key = "d7485f66690e24c0228af322bd00163c"  # Kakao REST API 키를 입력하세요 # 수연씨

    # 지오코딩 실행
    
    lat, lng = geocode_kakao(row[1], kakao_api_key)
    coordinates.append({'school_name': row[0] ,'address': row[1], 'latitude': lat, 'longitude': lng})
    
    # 진행 상황 출력
    if (idx + 1) % 10 == 0:
        print(f"{idx+1}/{address_df.shape[0]} addresses processed.")

        # 결과를 데이터프레임으로 변환
        coordinates_df = pd.DataFrame(coordinates)

        # 결과 저장
        coordinates_df.to_csv("school-location.csv", index=False)

    
    # API 호출 제한을 피하기 위해 딜레이 추가 (필요시)
    time.sleep(0.3)  # 0.2초 대기 (초당 5건 요청)

# 결과를 데이터프레임으로 변환
coordinates_df = pd.DataFrame(coordinates)

# 결과 저장
coordinates_df.to_csv("school-location.csv", index=False)
