import os
import warnings

import pandas as pd
import requests
import streamlit as st

warnings.filterwarnings("ignore")
st.set_page_config(layout="wide")


def get_api_key() -> str | None:
    """
    API 키를 가져오는 헬퍼 함수.
    우선순위:
    1) st.secrets["FLIGHTAWARE_API_KEY"]
    2) 환경변수 FLIGHTAWARE_API_KEY
    3) 화면에서 입력
    """

    key = "hcmdwPPBAYf5KBdAyxZT0zirFGmAtp4C"

    # 2) 환경변수
    if not key:
        key = os.getenv("FLIGHTAWARE_API_KEY")

    # 3) 화면 입력
    if not key:
        key = st.text_input(
            "FlightAware AeroAPI Key",
            value="",
            type="password",
            help="FlightAware AeroAPI 계정에서 발급받은 API Key를 입력하세요.",
        )

    if not key:
        st.warning("API Key를 입력하면 데이터를 조회할 수 있습니다.")
        return None

    return key


def parse_extra_params(raw: str) -> dict:
    """
    'key1=val1&key2=val2' 형태의 문자열을 dict로 파싱.
    """
    params: dict[str, str] = {}
    raw = raw.strip()
    if not raw:
        return params

    for pair in raw.split("&"):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            continue
        k, v = pair.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k:
            params[k] = v
    return params


def call_flightaware_api(resource_id: str, resource_type: str, api_key: str, params: dict) -> pd.DataFrame:
    """
    FlightAware AeroAPI 호출 후 DataFrame으로 변환.
    GET /airports/{id}/flights 또는 GET /operators/{id}/flights
    GET /history/airports/{id}/flights/departures
    페이지네이션을 자동으로 처리하여 모든 페이지를 가져옵니다.
    """
    BASE_URL = "https://aeroapi.flightaware.com/aeroapi"
    
    # 리소스 타입에 따라 다른 URL 구조 사용
    if resource_type == "history":
        url = f"{BASE_URL}/history/airports/{resource_id}/flights/departures"
    elif resource_type == "aircraft_types":
        url = f"{BASE_URL}/aircraft/types/{resource_id}"
    elif resource_type == "schedules":
        # schedules는 날짜가 URL 경로에 포함됨
        date_start = params.pop("date_start", None)
        date_end = params.pop("date_end", None)
        if not date_start or not date_end:
            raise RuntimeError("schedules 엔드포인트는 date_start와 date_end 파라미터가 필요합니다. 예: id=DUMMY&date_start=2025-01-01&date_end=2025-01-02")
        # 날짜를 그대로 URL 경로에 포함 (requests가 자동으로 인코딩 처리)
        url = f"{BASE_URL}/schedules/{date_start}/{date_end}"
        resource_id = None  # schedules는 resource_id를 사용하지 않음
    else:
        url = f"{BASE_URL}/{resource_type}/{resource_id}/flights"
    
    headers = {
        "x-apikey": api_key
    }
    
    # aircraft_types와 schedules는 특별 처리
    if resource_type == "aircraft_types":
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        
        if resp.status_code != 200:
            error_msg = f"HTTP {resp.status_code} 오류 발생\n\n"
            error_msg += f"호출한 URL: {url}\n\n"
            
            try:
                error_data = resp.json()
                if isinstance(error_data, dict):
                    if "detail" in error_data:
                        error_msg += f"**상세 오류:** {error_data['detail']}\n\n"
                    if "reason" in error_data:
                        error_msg += f"**오류 유형:** {error_data['reason']}\n\n"
                    if "title" in error_data:
                        error_msg += f"**오류 제목:** {error_data['title']}\n\n"
            except:
                pass
            
            if resp.status_code == 404:
                error_msg += "404 오류: 요청한 항공기 타입을 찾을 수 없습니다.\n"
                error_msg += "ICAO 항공기 타입 코드가 올바른지 확인하세요.\n\n"
                error_msg += f"서버 응답: {resp.text[:500]}"
            else:
                error_msg += f"서버 응답: {resp.text[:500]}"
            
            raise RuntimeError(error_msg)
        
        try:
            data = resp.json()
        except Exception as e:
            raise RuntimeError(f"JSON 파싱 오류: {e}\n원본 응답: {resp.text[:500]}")
        
        # 단일 객체를 리스트로 변환하여 DataFrame 생성
        if isinstance(data, dict):
            return pd.json_normalize([data], sep="_")
        return pd.DataFrame([data]) if data else pd.DataFrame()
    
    # schedules는 페이지네이션 처리 (flights 배열 반환 가능)
    if resource_type == "schedules":
        all_data = []
        current_url = url
        current_params = params.copy()
        page_count = 0
        max_pages = 100  # API 제한: 최대 40페이지
        
        # 페이지네이션 처리
        while current_url and page_count < max_pages:
            # 디버깅을 위해 실제 호출 URL 로깅 (첫 페이지만)
            if page_count == 0:
                if current_params:
                    full_url = f"{current_url}?{'&'.join([f'{k}={v}' for k, v in current_params.items()])}"
                else:
                    full_url = current_url
                
                # curl 명령어 생성
                curl_cmd = f'curl -X GET "{full_url}" \\\n -H "Accept: application/json; charset=UTF-8" \\\n -H "x-apikey: {api_key}"'
                st.info(f"**실제 CURL 명령어:**\n```bash\n{curl_cmd}\n```")
            
            resp = requests.get(current_url, headers=headers, params=current_params, timeout=30)
            
            if resp.status_code != 200:
                error_msg = f"HTTP {resp.status_code} 오류 발생\n\n"
                if page_count == 0:
                    error_msg += f"호출한 URL: {full_url}\n\n"
                else:
                    error_msg += f"호출한 URL: {current_url}\n\n"
                
                # JSON 응답 파싱 시도
                try:
                    error_data = resp.json()
                    if isinstance(error_data, dict):
                        if "detail" in error_data:
                            error_msg += f"**상세 오류:** {error_data['detail']}\n\n"
                        if "reason" in error_data:
                            error_msg += f"**오류 유형:** {error_data['reason']}\n\n"
                        if "title" in error_data:
                            error_msg += f"**오류 제목:** {error_data['title']}\n\n"
                except:
                    pass
                
                if resp.status_code == 400:
                    error_msg += "400 오류: 잘못된 요청\n"
                    error_msg += "가능한 원인:\n"
                    error_msg += "1. 날짜 범위가 유효하지 않습니다 (과거 3개월 ~ 미래 1년, 최대 3주 범위)\n"
                    error_msg += "2. 파라미터 형식이 잘못되었을 수 있습니다\n"
                    error_msg += "3. 필수 파라미터가 누락되었을 수 있습니다\n\n"
                    error_msg += f"서버 응답: {resp.text[:500]}"
                elif resp.status_code == 404:
                    error_msg += "404 오류: 요청한 리소스를 찾을 수 없습니다.\n"
                    error_msg += "가능한 원인:\n"
                    error_msg += "1. 날짜 형식이 잘못되었을 수 있습니다\n"
                    error_msg += "2. API 키에 해당 리소스에 대한 접근 권한이 없을 수 있습니다\n\n"
                    error_msg += f"서버 응답: {resp.text[:500]}"
                elif resp.status_code == 401:
                    error_msg += "401 오류: 인증 실패\n"
                    error_msg += "API 키가 유효하지 않거나 만료되었을 수 있습니다.\n\n"
                    error_msg += f"서버 응답: {resp.text[:500]}"
                else:
                    error_msg += f"서버 응답: {resp.text[:500]}"
                
                raise RuntimeError(error_msg)
            
            try:
                data = resp.json()
            except Exception as e:
                raise RuntimeError(f"JSON 파싱 오류: {e}\n원본 응답: {resp.text[:500]}")
            
            # schedules 응답 처리
            if isinstance(data, dict):
                # schedules API는 "scheduled" 키 사용
                if "scheduled" in data:
                    scheduled = data["scheduled"]
                    if isinstance(scheduled, list):
                        if scheduled:
                            all_data.extend(scheduled)
                            if page_count == 0:
                                st.info(f"첫 페이지: {len(scheduled)}개 스케줄 조회됨 (총 {len(all_data)}개)")
                    else:
                        st.warning(f"응답의 'scheduled' 키가 리스트가 아닙니다. 타입: {type(scheduled)}")
                # 하위 호환성을 위해 schedules 키도 확인
                elif "schedules" in data:
                    schedules = data["schedules"]
                    if isinstance(schedules, list):
                        if schedules:
                            all_data.extend(schedules)
                            if page_count == 0:
                                st.info(f"첫 페이지: {len(schedules)}개 스케줄 조회됨 (총 {len(all_data)}개)")
                # flights 키도 확인 (일부 응답 형식)
                elif "flights" in data:
                    flights = data["flights"]
                    if isinstance(flights, list):
                        if flights:
                            all_data.extend(flights)
                            if page_count == 0:
                                st.info(f"첫 페이지: {len(flights)}개 항공편 조회됨 (총 {len(all_data)}개)")
                else:
                    # 응답에 scheduled, schedules, flights 키가 없는 경우
                    if page_count == 0:
                        st.warning(f"응답에 'scheduled', 'schedules' 또는 'flights' 키가 없습니다. 응답 키: {list(data.keys())}")
                        st.json(data)  # 디버깅을 위해 JSON 표시
            elif isinstance(data, list):
                # 응답이 직접 리스트인 경우
                if data:
                    all_data.extend(data)
                    if page_count == 0:
                        st.info(f"첫 페이지: {len(data)}개 항공편 조회됨 (총 {len(all_data)}개)")
                # 리스트는 페이지네이션 정보가 없으므로 종료
                break
            
            # 다음 페이지 확인 (딕셔너리 응답인 경우)
            if isinstance(data, dict):
                if "links" in data and isinstance(data["links"], dict):
                    next_link = data["links"].get("next", "")
                    if next_link:
                        # 다음 페이지 URL이 절대 URL인 경우
                        if next_link.startswith("http"):
                            current_url = next_link
                            current_params = {}  # URL에 이미 파라미터가 포함되어 있음
                        else:
                            # 상대 URL인 경우
                            current_url = f"{BASE_URL}{next_link}"
                            current_params = {}
                        page_count += 1
                        continue
                elif "num_pages" in data:
                    # num_pages가 있고 현재 페이지가 마지막이 아니면 다음 페이지 요청
                    num_pages = data.get("num_pages", 1)
                    if page_count + 1 < num_pages and page_count + 1 < max_pages:
                        # 다음 페이지 파라미터 추가
                        current_params = params.copy()
                        current_params["page"] = page_count + 1
                        page_count += 1
                        continue
            
            # 다음 페이지가 없으면 종료
            break
        
        # 모든 페이지 데이터를 DataFrame으로 변환
        if not all_data:
            return pd.DataFrame()
        
        # 페이지네이션 결과 요약
        if page_count > 0:
            st.success(f"✅ 총 {page_count + 1}페이지에서 {len(all_data)}개 스케줄 조회 완료 (페이지당 평균: {len(all_data) // (page_count + 1) if page_count > 0 else len(all_data)}개)")
        
        # 중첩된 구조를 펼치기 위해 normalize 사용
        return pd.json_normalize(all_data, sep="_")
    
    # 페이지네이션이 필요한 엔드포인트 처리
    all_data = []
    current_url = url
    current_params = params.copy()
    page_count = 0
    max_pages = 100  # API 제한: 최대 40페이지
    
    # 페이지네이션 처리
    while current_url and page_count < max_pages:
        # 디버깅을 위해 실제 호출 URL 로깅 (첫 페이지만)
        if page_count == 0:
            if current_params:
                full_url = f"{current_url}?{'&'.join([f'{k}={v}' for k, v in current_params.items()])}"
            else:
                full_url = current_url
        
        resp = requests.get(current_url, headers=headers, params=current_params, timeout=30)

        if resp.status_code != 200:
            error_msg = f"HTTP {resp.status_code} 오류 발생\n\n"
            if page_count == 0:
                error_msg += f"호출한 URL: {full_url}\n\n"
            else:
                error_msg += f"호출한 URL: {current_url}\n\n"
            
            # JSON 응답 파싱 시도
            try:
                error_data = resp.json()
                if isinstance(error_data, dict):
                    if "detail" in error_data:
                        error_msg += f"**상세 오류:** {error_data['detail']}\n\n"
                    if "reason" in error_data:
                        error_msg += f"**오류 유형:** {error_data['reason']}\n\n"
                    if "title" in error_data:
                        error_msg += f"**오류 제목:** {error_data['title']}\n\n"
            except:
                pass
            
            if resp.status_code == 400:
                error_msg += "400 오류: 잘못된 요청\n"
                error_msg += "가능한 원인:\n"
                error_msg += "1. 날짜 범위가 유효하지 않습니다 (과거 10일 이내만 조회 가능)\n"
                error_msg += "2. 파라미터 형식이 잘못되었을 수 있습니다\n"
                error_msg += "3. 필수 파라미터가 누락되었을 수 있습니다\n\n"
                error_msg += "💡 **참고:** FlightAware API는 과거 10일 이내의 데이터만 조회할 수 있습니다.\n\n"
                error_msg += f"서버 응답: {resp.text[:500]}"
            elif resp.status_code == 404:
                error_msg += "404 오류: 요청한 리소스를 찾을 수 없습니다.\n"
                error_msg += "가능한 원인:\n"
                error_msg += "1. Resource ID가 잘못되었을 수 있습니다\n"
                error_msg += "2. API 키에 해당 리소스에 대한 접근 권한이 없을 수 있습니다\n\n"
                error_msg += f"서버 응답: {resp.text[:500]}"
            elif resp.status_code == 401:
                error_msg += "401 오류: 인증 실패\n"
                error_msg += "API 키가 유효하지 않거나 만료되었을 수 있습니다.\n\n"
                error_msg += f"서버 응답: {resp.text[:500]}"
            else:
                error_msg += f"서버 응답: {resp.text[:500]}"
            
            raise RuntimeError(error_msg)

        try:
            data = resp.json()
        except Exception as e:
            raise RuntimeError(f"JSON 파싱 오류: {e}\n원본 응답: {resp.text[:500]}")

        # 응답 데이터 수집
        if isinstance(data, dict):
            # history API는 departures 키 사용
            if "departures" in data:
                departures = data["departures"]
                if departures:
                    all_data.extend(departures)
                    if page_count == 0:
                        st.info(f"첫 페이지: {len(departures)}개 항공편 조회됨 (총 {len(all_data)}개)")
            
            # 일반 API는 flights 키 사용
            elif "flights" in data:
                flights = data["flights"]
                if flights:
                    all_data.extend(flights)
                    if page_count == 0:
                        st.info(f"첫 페이지: {len(flights)}개 항공편 조회됨 (총 {len(all_data)}개)")
            
            # 다음 페이지 확인
            if "links" in data and isinstance(data["links"], dict):
                next_link = data["links"].get("next", "")
                if next_link:
                    # 다음 페이지 URL이 절대 URL인 경우
                    if next_link.startswith("http"):
                        current_url = next_link
                        current_params = {}  # URL에 이미 파라미터가 포함되어 있음
                    else:
                        # 상대 URL인 경우
                        current_url = f"{BASE_URL}{next_link}"
                        current_params = {}
                    page_count += 1
                    continue
            elif "num_pages" in data:
                # num_pages가 있고 현재 페이지가 마지막이 아니면 다음 페이지 요청
                num_pages = data.get("num_pages", 1)
                if page_count + 1 < num_pages and page_count + 1 < max_pages:
                    # 다음 페이지 파라미터 추가
                    current_params = params.copy()
                    current_params["page"] = page_count + 1
                    page_count += 1
                    continue
        
        # 다음 페이지가 없으면 종료
        break
    
    # 모든 페이지 데이터를 DataFrame으로 변환
    if not all_data:
        return pd.DataFrame()
    
    # 페이지네이션 결과 요약
    if page_count > 0:
        st.success(f"✅ 총 {page_count + 1}페이지에서 {len(all_data)}개 항공편 조회 완료 (페이지당 평균: {len(all_data) // (page_count + 1)}개)")
    
    # 중첩된 구조를 펼치기 위해 normalize 사용
    return pd.json_normalize(all_data, sep="_")


def main():
    st.title("FlightAware AeroAPI - Operators Flights")
    st.markdown(
        """
        **FlightAware AeroAPI**를 사용하여 특정 Operator의 항공편 데이터를 조회합니다.
        
        - **엔드포인트**: `GET /operators/{id}/flights`
        - **문서**: [FlightAware AeroAPI Documentation](https://www.flightaware.com/aeroapi/portal/documentation#get-/operators/-id-/flights)
        """
    )

    st.divider()

    api_key = "hcmdwPPBAYf5KBdAyxZT0zirFGmAtp4C"
    if not api_key:
        st.stop()

    col1, col2 = st.columns([1, 2])
    
    with col1:
        resource_type = st.selectbox(
            "리소스 타입",
            options=["history", "airports", "operators", "aircraft_types", "schedules"],
            index=0,
            help="history: 과거 출발 항공편, airports: 공항, operators: 항공사, aircraft_types: 항공기 타입 정보, schedules: 스케줄",
        )
    
    with col2:
        raw_query = st.text_input(
            "쿼리문",
            value="",
            placeholder="예) id=ICN&start=2025-01-01T00:00:00Z&end=2025-01-02T00:00:00Z",
            help="`id=VALUE&key=value&key2=value2` 형식으로 입력하세요. id는 필수입니다.",
        )
    
    # 리소스 타입에 따라 다른 예시 표시
    if resource_type == "history":
        st.info("💡 **예시 쿼리문 (history):**\n- `id=ICN&start=2025-12-01T00:00:00Z&end=2025-12-01T23:59:59Z` - 공항 출발 항공편 (최대 24시간 범위)\n\n⚠️ **주의:** History API는 **2011-01-01부터 현재로부터 15일 전까지**의 데이터를 조회할 수 있으며, **최대 1일(24시간) 범위**만 가능합니다.")
    elif resource_type == "aircraft_types":
        st.info("💡 **예시 쿼리문 (aircraft_types):**\n- `id=B738` - ICAO 항공기 타입 코드 (예: B738, A320, B777 등)\n\n항공기 타입 정보(설명, 제조사, 엔진 타입 등)를 조회합니다.")
    elif resource_type == "schedules":
        st.info("""
        💡 **예시 쿼리문 (schedules):**
        
        **입력 형식:** 쿼리문에 `date_start`와 `date_end`를 포함하면 자동으로 `/schedules/{date_start}/{date_end}` 형식으로 변환됩니다.
        

        
        **필터 옵션 추가 (쿼리 파라미터로 전달):**
        - `date_start=2025-12-19T00:00:00Z&date_end=2025-12-19T23:59:59Z&origin=ICN` 
        - `date_start=2025-12-19T00:00:00Z&date_end=2025-12-19T23:59:59Z&destination=ICN` 

        
        ⚠️ **주의:** 
        - **과거 3개월부터 미래 1년까지**의 데이터 조회 가능
        - date_start와 date_end 사이는 **최대 3주**까지 가능
        - 날짜 형식: `2025-12-19T00:00:00Z` 또는 `2025-12-19`
        - `date_start`와 `date_end`는 URL 경로로 변환, 나머지(`origin`, `destination`, `airline` 등)는 쿼리 파라미터(?key=value)로 전달
        """)
    else:
        st.info("💡 **예시 쿼리문:**\n- `id=ICN&start=2025-01-01T00:00:00Z&end=2025-01-02T00:00:00Z` - 공항/항공사 ID와 날짜 범위\n.")

    # 디버그 모드 옵션 추가
    show_debug = st.checkbox("디버그 정보 표시", help="호출하는 URL을 표시합니다.")
    
    if st.button("데이터 조회", type="primary"):
        if not raw_query:
            st.error("쿼리문을 입력해주세요.")
            return
        
        # 쿼리문 파싱
        params = parse_extra_params(raw_query)
        
        # schedules는 id가 필요 없고 date_start, date_end가 필요
        if resource_type == "schedules":
            if "date_start" not in params or "date_end" not in params:
                st.error("schedules 엔드포인트는 'date_start'와 'date_end' 파라미터가 필요합니다. 예: date_start=2025-01-01&date_end=2025-01-02")
                return
            resource_id = "DUMMY"  # schedules는 resource_id를 사용하지 않지만 함수 시그니처를 위해 더미 값 사용
        else:
            # id 파라미터 추출 (필수)
            if "id" not in params:
                st.error("쿼리문에 'id' 파라미터가 필요합니다. 예: id=ICN&start=...")
                return
            resource_id = params.pop("id")
        
        # resource_type이 쿼리문에 있으면 제거 (UI에서 선택한 것을 우선)
        params.pop("resource_type", None)
        
        # 디버그 정보 표시
        if show_debug:
            BASE_URL = "https://aeroapi.flightaware.com/aeroapi"
            if resource_type == "history":
                url = f"{BASE_URL}/history/airports/{resource_id}/flights/departures"
            elif resource_type == "aircraft_types":
                url = f"{BASE_URL}/aircraft/types/{resource_id}"
            elif resource_type == "schedules":
                date_start = params.get("date_start", "")
                date_end = params.get("date_end", "")
                url = f"{BASE_URL}/schedules/{date_start}/{date_end}"
            else:
                url = f"{BASE_URL}/{resource_type}/{resource_id}/flights"
            if params and resource_type != "schedules":
                full_url = f"{url}?{'&'.join([f'{k}={v}' for k, v in params.items()])}"
            else:
                full_url = url
            st.code(f"호출 URL: {full_url}", language="text")
            st.code(f"Headers: x-apikey: {api_key[:10]}...", language="text")

        with st.spinner("FlightAware AeroAPI에서 데이터를 가져오는 중입니다..."):
            try:
                df = call_flightaware_api(resource_id, resource_type, api_key, params)
            except Exception as e:
                st.error(f"API 호출 중 오류가 발생했습니다:\n\n{e}")
                st.info("💡 **문제 해결 팁:**\n- FlightAware AeroAPI 문서를 확인하여 올바른 쿼리문 형식을 사용하고 있는지 확인하세요.\n- API 키가 유효하고 해당 리소스에 대한 접근 권한이 있는지 확인하세요.\n- 디버그 정보를 활성화하여 호출하는 URL을 확인하세요.")
                return

        st.success(f"조회 완료! 행 수: {len(df)}")
        
        if len(df) > 0:

            st.write(df)
            df.to_parquet("future_schedules_arrival_1231_ICN.parquet")


            # 확실히 취소된 항공편 제외
            df = df[~df["status"].isin(["Diverted", "Scheduled / Delayed", "Delayed", "Cancelled"])]

            # General Aviation 제거거
            df = df[df["type"]=="Airline"]

            # status가 Unknown일 때, 동일 Flight_Number가 22시간 내에 있으면 해당편은 캔슬편으로 간주하여 제거
            time_col = None
            for col in ["scheduled_out", "actual_out", "estimated_out", "scheduled_off", "actual_off"]:
                if col in df.columns:
                    time_col = col
                    break
            
            if time_col:
                # 시간 컬럼을 datetime으로 변환
                df[time_col + "_datetime"] = pd.to_datetime(df[time_col], errors='coerce')
                
                # status가 Unknown인 행 찾기
                mask_unknown_status = df["status"].str.upper() == "UNKNOWN"
                unknown_indices = df[mask_unknown_status].index.tolist()
                
                # 제거할 인덱스 저장
                indices_to_remove = []
                
                for idx in unknown_indices:
                    current_ident = df.loc[idx, "ident"]
                    current_time = df.loc[idx, time_col + "_datetime"]
                    
                    if pd.isna(current_ident) or pd.isna(current_time):
                        continue
                    
                    # 동일한 ident를 가진 다른 행들 찾기
                    same_ident_mask = (df["ident"] == current_ident) & (df.index != idx)
                    same_ident_df = df[same_ident_mask]
                    
                    if len(same_ident_df) > 0:
                        # 22시간 내에 있는 항공편이 있는지 확인
                        for other_idx in same_ident_df.index:
                            other_time = df.loc[other_idx, time_col + "_datetime"]
                            if pd.isna(other_time):
                                continue
                            
                            time_diff = abs((current_time - other_time).total_seconds() / 3600)
                            if time_diff <= 22:
                                # 22시간 내에 동일한 ident가 있으면 Unknown 제거
                                indices_to_remove.append(idx)
                                break
                
                # Unknown 행 제거
                if indices_to_remove:
                    df = df.drop(indices_to_remove)
                    st.info(f"Status Unknown 제거: {len(indices_to_remove)}개 항공편 제거됨")

            # 화물기 타입 & 화물전용 항공사 제외
            cargo_aircraft_types = [
                "B712F", "B732F", "B733F", "B734F", "B73F",
                "B742F", "B743F", "B744", "B744F", "B748", "B748F",
                "B752F", "B753F",
                "B762F", "B763F", "B76F",
                "B77L", "B77F",
                "A300F", "A30B", "A310F",
                "A320P2F", "A321P2F",
                "A332F", "A333F", "A33F", "A35KF",
                "AT72F", "AT75F", "AT76F",
                "DH8F", "SF34F", "SF58F",
                "AN12", "AN26", "AN124", "AN225"
            ] + ["74F", "73F"]


            cargo_airlines = [
            "FX", "5X", "PO", "CV", "CK", "RU", "M7", "LD", "K4", "NC",
            "KZ", "L8", "ES", "QY", "3S",
            "KJ", "YG", "GI", "W8", "MP", "C8",
            "CF", "O3", "RH", "7L", "2Y", "P3", "3V",
            "4M", "UC", "M6", "T5", "N8"
            ]

            df = df[~df["operator_iata"].isin(cargo_airlines)]
            df = df[~df["aircraft_type"].isin(cargo_aircraft_types)]



            # Terminal 매핑
            df["terminal_origin"] = (
                df["terminal_origin"]
                .fillna(
                    df.groupby("operator_iata")["terminal_origin"]
                    .transform(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
                )
                .fillna("UNKNOWN")
            )



            st.write(f"필터링 후 행 수: {len(df)}")
            st.write(df)

            st.write(df["terminal_origin"].value_counts())
            st.write(df.groupby(["terminal_origin", "operator_iata"]).size())
            st.write(df[df["terminal_origin"]=="UNKNOWN"].groupby(["operator_iata", "aircraft_type"]).size())

        else:
            st.info("조회된 데이터가 없습니다.")


if __name__ == "__main__":
    main()

# df=pd.read_parquet("future_schedules_1231_ICN.parquet")
# df["actual_ident_icao"] = df["actual_ident_icao"].fillna(df["ident_icao"])
# df["actual_ident_iata"] = df["actual_ident_iata"].fillna(df["ident_iata"])

# # 코드쉐어편 제거
# df = df.drop_duplicates(subset=["actual_ident_iata"], keep="first")

# # st.write(df)
# # st.write(len(df))

# # 항공편 제거
# st.write("Schedules")
# df_pax = df[df["seats_cabin_coach"]>0]
# df_pax["iata_code"] =df_pax["actual_ident_iata"].str[:2]
# st.write(df_pax)
# st.write(len(df_pax))
# st.write(df_pax["iata_code"].value_counts())


# st.write("History")
# df=pd.read_parquet("history_departures_1202.parquet")
# st.write(df)



df=pd.read_parquet("future_schedules_1231_ICN.parquet")
st.write("future_schedules_1231_ICN")
df

df=pd.read_parquet("future_schedules_arrival_1231_ICN.parquet")
st.write("future_schedules_arrival_1231_ICN")
df

st.write("history_departures_1202")
df=pd.read_parquet("history_departures_1202.parquet")
df

