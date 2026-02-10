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
    1) st.secrets["AVIATION_EDGE_API_KEY"]
    2) 환경변수 AVIATION_EDGE_API_KEY
    3) 화면에서 입력
    """
    key = None

    # 1) Streamlit secrets
    try:
        key = st.secrets.get("AVIATION_EDGE_API_KEY")
    except Exception:
        key = None

    # 2) 환경변수
    if not key:
        key = os.getenv("AVIATION_EDGE_API_KEY")

    # 3) 화면 입력
    # key = st.text_input(
    #     "Aviation Edge API Key",
    #     value=key or "",
    #     type="password",
    #     help="Aviation Edge 계정에서 발급받은 API Key 를 입력하세요.",
    # )
    

    if not key:
        st.warning("API Key 를 입력하면 데이터를 조회할 수 있습니다.")
        return None

    return key


def test_endpoint(BASE_URL: str, endpoint: str, api_key: str) -> tuple[bool, str, int]:
    """
    엔드포인트가 존재하는지 테스트.
    Returns: (존재 여부, 응답 메시지, 상태 코드)
    """
    url = f"{BASE_URL}/{endpoint}"
    params = {"key": api_key}
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            return True, "엔드포인트가 존재합니다.", resp.status_code
        elif resp.status_code == 404:
            return False, f"404 오류: 엔드포인트를 찾을 수 없습니다.\n응답: {resp.text[:200]}", resp.status_code
        else:
            return False, f"HTTP {resp.status_code} 오류\n응답: {resp.text[:200]}", resp.status_code
    except Exception as e:
        return False, f"연결 오류: {str(e)}", 0


def parse_extra_params(raw: str) -> dict:
    """
    'key1=val1&key2=val2' 형태의 문자열을 dict 로 파싱.
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


def call_aviation_edge(BASE_URL: str, endpoint: str, api_key: str, extra_params: dict) -> pd.DataFrame:
    """
    Aviation Edge API 호출 후 DataFrame 으로 변환.
    """
    url = f"{BASE_URL}/{endpoint}"
    params = {"key": api_key}
    params.update(extra_params)

    # 디버깅을 위해 실제 호출 URL 로깅
    full_url = f"{url}?{'&'.join([f'{k}={v}' for k, v in params.items()])}"
    
    resp = requests.get(url, params=params, timeout=30)

    if resp.status_code != 200:
        error_msg = f"HTTP {resp.status_code} 오류 발생\n\n"
        error_msg += f"호출한 URL: {full_url}\n\n"
        
        if resp.status_code == 404:
            error_msg += "404 오류: 요청한 엔드포인트를 찾을 수 없습니다.\n"
            error_msg += "가능한 원인:\n"
            error_msg += "1. 엔드포인트 이름이 잘못되었을 수 있습니다 (대소문자, 오타 확인)\n"
            error_msg += "2. API URL 구조가 변경되었을 수 있습니다\n"
            error_msg += "3. 해당 엔드포인트가 더 이상 지원되지 않을 수 있습니다\n\n"
            error_msg += f"서버 응답: {resp.text[:500]}"
        else:
            error_msg += f"서버 응답: {resp.text[:500]}"
        
        raise RuntimeError(error_msg)

    try:
        data = resp.json()
    except Exception as e:
        raise RuntimeError(f"JSON 파싱 오류: {e}\n원본 응답: {resp.text[:500]}")

    # 대부분의 엔드포인트는 list[dict] 형태로 반환
    if isinstance(data, list):
        if not data:
            return pd.DataFrame()
        # flights 관련 엔드포인트와 timetable, seatConfiguration은 하위 필드를 컬럼으로 펼쳐서 보여주기 위해 normalize 사용
        if endpoint in {"flights", "timetable", "flightsHistory", "flightsFuture", "seatConfiguration"}:
            return pd.json_normalize(data, sep="_")
        return pd.DataFrame(data)

    # 그 외에는 그대로 한 번 보여주고, DataFrame 으로도 변환 시도
    st.info("응답 형식이 리스트가 아니라서 JSON 원본도 함께 표시합니다.")
    st.json(data)
    return pd.json_normalize(data)



def main():
    api_key = "e3a33f-9e28aa"
    BASE_URL = "https://aviation-edge.com/v2/public"


    st.title("Aviation Edge API 뷰어")
    st.markdown(
        """
        **Aviation Edge** 에서 제공하는 여러 테이블(엔드포인트)을 직접 쿼리해서
        `pandas.DataFrame` 형태로 확인할 수 있는 페이지입니다.

        - **엔드포인트 예시**
          - `airlineDatabase` : 항공사 테이블
          - `airportDatabase` : 공항 테이블
          - `airplaneDatabase` : 항공기 테이블
          - `seatConfiguration` : 좌석 구성 테이블 (이름 확인 필요)
          - `routes` : 노선 테이블
          - `timetable` : 스케줄/타임테이블
        
        📚 **공식 문서:** [Aviation Edge API Documentation](https://aviation-edge.com/developers/documentation/)
        """
    )

    st.divider()

    col1, col2 = st.columns([2, 3])

    with col1:
        endpoint = st.selectbox(
            "엔드포인트(테이블) 선택",
            options=[
                "airlineDatabase",
                "airportDatabase",
                "airplaneDatabase",
                "cityDatabase",
                "countryDatabase",

                "timetable",
                "flightsHistory",
                "flightsFuture",

                "routes",
                "seatConfiguration",
            ],
            index=0,
            help="Aviation Edge 문서에 나온 public 엔드포인트 이름입니다.",
        )

    with col2:
        raw_params = st.text_input(
            "추가 쿼리 파라미터 (선택)",
            value="",
            placeholder="예) codeIataAirline=KE&country=South Korea",
            help="`key=value&key2=value2` 형식으로 입력하면 됩니다. 비워두면 전체 레코드를 요청합니다(요금제에 따라 제한될 수 있음).",
        )
    st.info("과거 출발스케쥴 flightsHistory |  code=ICN&type=departure&status=active&date_from=2025-12-02&date_to=2025-12-02")
    st.info("과거 도착스케쥴 flightsHistory |  code=ICN&type=arrival&status=landed&date_from=2025-12-02&date_to=2025-12-02")
    st.info("과거 출발스케쥴 flightsFuture | type=departure&iataCode=ICN&date=2025-12-31")
    
    # 엔드포인트 테스트 기능 추가
    with st.expander("🔍 엔드포인트 존재 여부 테스트", expanded=False):
        st.write("선택한 엔드포인트가 실제로 존재하는지 확인합니다.")
        if st.button("현재 엔드포인트 테스트", key="test_endpoint"):
            with st.spinner("테스트 중..."):
                exists, message, status_code = test_endpoint(BASE_URL, endpoint, api_key)
                if exists:
                    st.success(f"✅ {message}")
                else:
                    st.error(f"❌ {message}")
                    st.info("💡 **팁:** 엔드포인트 이름이 다를 수 있습니다. 아래 후보들을 시도해보세요.")
        
        # seatConfiguration 관련 가능한 이름 후보들
        if endpoint == "seatConfiguration":
            st.write("**가능한 엔드포인트 이름 후보:**")
            possible_names = [
                "seatConfiguration",
                "seatConfig", 
                "seatMap",
                "seatDatabase",
                "airplaneSeatConfiguration",
                "seatconfiguration",  # 소문자
            ]
            if st.button("가능한 이름들 테스트", key="test_possible_names"):
                st.write("테스트 중...")
                results = []
                for name in possible_names:
                    exists, message, status_code = test_endpoint(BASE_URL, name, api_key)
                    results.append({
                        "이름": name,
                        "존재": "✅" if exists else "❌",
                        "상태": status_code,
                        "메시지": message[:100]
                    })
                results_df = pd.DataFrame(results)
                st.write(results_df)
    
    # seatConfiguration 쿼리 예시 표시
    if endpoint == "seatConfiguration":
        st.info("💡 **seatConfiguration 쿼리 예시:**")
        st.info("특정 항공사: `codeIataAirline=KE` 또는 `airlineIataCode=KE`")
        st.info("특정 항공기 타입: `aircraftType=Boeing 777`")
        st.info("조합: `codeIataAirline=KE&aircraftType=Boeing 777`")
        st.warning("⚠️ **주의:** 엔드포인트 이름이 정확하지 않을 수 있습니다. 위의 '엔드포인트 존재 여부 테스트'를 사용하여 확인하세요.")

    # 디버그 모드 옵션 추가
    show_debug = st.checkbox("디버그 정보 표시", help="호출하는 URL을 표시합니다.")
    if st.button("데이터 조회", type="primary"):
        extra_params = parse_extra_params(raw_params)
        
        # 디버그 정보 표시
        if show_debug:
            url = f"{BASE_URL}/{endpoint}"
            params = {"key": api_key}
            params.update(extra_params)
            full_url = f"{url}?{'&'.join([f'{k}={v}' for k, v in params.items()])}"
            st.code(f"호출 URL: {full_url}", language="text")

        with st.spinner("Aviation Edge API 에서 데이터를 가져오는 중입니다..."):
            try:
                df = call_aviation_edge(BASE_URL, endpoint, api_key, extra_params)
            except Exception as e:
                st.error(f"API 호출 중 오류가 발생했습니다:\n\n{e}")
                st.info("💡 **문제 해결 팁:**\n- Aviation Edge API 문서를 확인하여 올바른 엔드포인트 이름을 사용하고 있는지 확인하세요.\n- API 키가 유효하고 해당 엔드포인트에 대한 접근 권한이 있는지 확인하세요.\n- 디버그 정보를 활성화하여 호출하는 URL을 확인하세요.")
                return

        st.success(f"조회 완료! 행 수: {len(df)}")
        st.write(df)

        if endpoint == "flightsHistory":
            st.write("코드쉐어편 제거 및 전처리")

            #전처리
            st.info("""
                전처리 내용
                1) codeshare 편 제거
                2) 터미널 값 채우기: 동일한 flight_iataNumber를 가진 다른 행에서 터미널 값 찾기
                3) 중복 제거: 동일한 항공편번호가 바로뒤 22시간내 나타날 경우, 이전 항공편 제거(단 unknown일 경우만만)
                """)
            
            # 1) 코드쉐어편 제거
            if "codeshared_airline_name" in df.columns:
                df = df[df["codeshared_airline_name"].isna()]
            
            # 날짜 컬럼 생성
            if "arrival_scheduledTime" in df.columns:
                df["arrival_schedule_date"] = df["arrival_scheduledTime"].str[:10]
            if "departure_scheduledTime" in df.columns:
                df["departure_schedule_date"] = df["departure_scheduledTime"].str[:10]
            
            # 2) 터미널 값 채우기: 동일한 flight_iataNumber를 가진 다른 행에서 터미널 값 찾기
            if "flight_iataNumber" in df.columns:
                # departure_terminal 채우기
                if "departure_terminal" in df.columns:
                    # 동일한 flight_iataNumber를 가진 행들 중 departure_terminal이 있는 값으로 채우기
                    terminal_map_dep = df[df["departure_terminal"].notna()].groupby("flight_iataNumber")["departure_terminal"].first()
                    df["departure_terminal"] = df["departure_terminal"].fillna(
                        df["flight_iataNumber"].map(terminal_map_dep)
                    )
                
                # arrival_terminal 채우기
                if "arrival_terminal" in df.columns:
                    # 동일한 flight_iataNumber를 가진 행들 중 arrival_terminal이 있는 값으로 채우기
                    terminal_map_arr = df[df["arrival_terminal"].notna()].groupby("flight_iataNumber")["arrival_terminal"].first()
                    df["arrival_terminal"] = df["arrival_terminal"].fillna(
                        df["flight_iataNumber"].map(terminal_map_arr)
                    )
            
            # 남은 Null 값은 "UNKNOWN"으로 채우기
            if "arrival_terminal" in df.columns:
                df["arrival_terminal"] = df["arrival_terminal"].fillna("UNKNOWN")
            if "departure_terminal" in df.columns:
                df["departure_terminal"] = df["departure_terminal"].fillna("UNKNOWN")
            
            # 3) 중복 제거: 동일한 날, 동일한 flight_iataNumber, 동일한 flight_icaoNumber, 8시간 내
            # departure와 arrival 모두 처리
            dedup_configs = [
                {
                    "type": "departure",
                    "schedule_date_col": "departure_schedule_date",
                    "scheduled_time_col": "departure_scheduledTime",
                    "datetime_col": "departure_datetime"
                },
                {
                    "type": "arrival",
                    "schedule_date_col": "arrival_schedule_date",
                    "scheduled_time_col": "arrival_scheduledTime",
                    "datetime_col": "arrival_datetime"
                }
            ]
            
            for config in dedup_configs:
                required_cols = ["flight_iataNumber", "flight_icaoNumber", config["scheduled_time_col"]]
                if not all(col in df.columns for col in required_cols):
                    continue
                
                # datetime 변환
                df[config["datetime_col"]] = pd.to_datetime(df[config["scheduled_time_col"]], errors='coerce')
                
                # 날짜별로 정렬
                df = df.sort_values(config["datetime_col"])
                
                # 중복 제거를 위한 마스크 생성
                mask_to_keep = pd.Series(True, index=df.index)
                
                # schedule_date와 flight_iataNumber, flight_icaoNumber로 그룹화
                for (date, iata, icao), group in df.groupby([config["schedule_date_col"], "flight_iataNumber", "flight_icaoNumber"]):
                    if len(group) > 1:
                        # 그룹 내에서 시간 차이 계산
                        group_indices = group.index.tolist()
                        group_times = group[config["datetime_col"]]
                        
                        # 8시간 내에 있는 항공편 찾기
                        for i, idx1 in enumerate(group_indices):
                            if not mask_to_keep[idx1]:  # 이미 제거 대상으로 표시된 경우 스킵
                                continue
                            
                            time1 = group_times.loc[idx1]
                            if pd.isna(time1):
                                continue
                            
                            # 같은 그룹 내 다른 항공편들과 비교
                            for j, idx2 in enumerate(group_indices):
                                if i >= j or not mask_to_keep[idx2]:
                                    continue
                                
                                time2 = group_times.loc[idx2]
                                if pd.isna(time2):
                                    continue
                                
                                time_diff = abs((time1 - time2).total_seconds() / 3600)  # 시간 차이 (시간 단위)
                                
                                if time_diff <= 22:
                                    # 22시간 내에 있으면, 더 나중 시간의 항공편만 남기기
                                    # 단, 제거하는 편은 status가 "active"가 아니고 "landed"도 아닌 것만 제거
                                    if "status" in df.columns:
                                        status1 = df.loc[idx1, "status"]
                                        status2 = df.loc[idx2, "status"]
                                        
                                        if time1 < time2:
                                            # idx1을 제거하려고 할 때, status가 active나 landed가 아니면 제거
                                            if status1 not in ["active", "landed"]:
                                                mask_to_keep[idx1] = False
                                            # idx1이 active나 landed면 idx2를 확인 (idx2가 active나 landed가 아니면 제거)
                                            elif status2 not in ["active", "landed"]:
                                                mask_to_keep[idx2] = False
                                        else:
                                            # idx2를 제거하려고 할 때, status가 active나 landed가 아니면 제거
                                            if status2 not in ["active", "landed"]:
                                                mask_to_keep[idx2] = False
                                            # idx2가 active나 landed면 idx1을 확인 (idx1이 active나 landed가 아니면 제거)
                                            elif status1 not in ["active", "landed"]:
                                                mask_to_keep[idx1] = False
                                    else:
                                        # status 컬럼이 없으면 기존 로직대로 처리
                                        if time1 < time2:
                                            mask_to_keep[idx1] = False
                                        else:
                                            mask_to_keep[idx2] = False
                
                # 중복 제거 전 행 수
                before_dedup = len(df)
                df = df[mask_to_keep]
                after_dedup = len(df)
                
                if before_dedup != after_dedup:
                    st.success(f"✅ {config['type'].capitalize()} 중복 제거 완료: {before_dedup}개 → {after_dedup}개 (제거된 항공편: {before_dedup - after_dedup}개)")


            #결과보기 결과보기 
            st.write(df)
            st.write(f"총 길이 : {len(df)}")
            st.write("arrival_schedule_date 그룹화")
            st.write(df.groupby(["arrival_schedule_date", "arrival_terminal"]).size().unstack())
            st.write("departure_schedule_date 그룹화")
            st.write(df.groupby(["departure_schedule_date", "departure_terminal"]).size().unstack())
            st.write(df.groupby(["departure_schedule_date", "airline_iataCode", "departure_terminal"]).size().unstack())

        if endpoint == "flightsFuture":
            st.write("코드쉐어편 제거")
            # 1) 코드쉐어편 제거
            if "codeshared_airline_name" in df.columns:
                df = df[df["codeshared_airline_name"].isna()]
                st.success(f"코드쉐어편 제거 행 수: {len(df)}")


                st.write("departure_terminal 그룹화")
                st.write(df.groupby(["departure_terminal"]).size())
                st.write("arrival_terminal 그룹화")
                st.write(df.groupby(["arrival_terminal"]).size())

        if endpoint == "airplaneDatabase":
            st.write("항공기 데이터베이스 분석")
            
            if len(df) > 0:
                st.write("### 기본 통계")
                st.write(f"총 항공기 수: {len(df)}")
                
                # 주요 컬럼이 있는 경우 그룹화 표시
                if "aircraftName" in df.columns:
                    st.write("### 항공기명별 분포")
                    st.write(df.groupby("aircraftName").size().sort_values(ascending=False))
                
                if "aircraftType" in df.columns:
                    st.write("### 항공기 타입별 분포")
                    st.write(df.groupby("aircraftType").size().sort_values(ascending=False))
                
                if "airline_iataCode" in df.columns:
                    st.write("### 항공사별 항공기 수")
                    st.write(df.groupby("airline_iataCode").size().sort_values(ascending=False))
                
                if "airline_icaoCode" in df.columns:
                    st.write("### 항공사(ICAO)별 항공기 수")
                    st.write(df.groupby("airline_icaoCode").size().sort_values(ascending=False))

        if endpoint == "seatConfiguration":
            st.write("좌석 구성 데이터베이스 분석")
            
            if len(df) > 0:
                st.write("### 기본 통계")
                st.write(f"총 좌석 구성 수: {len(df)}")
                
                # 주요 컬럼이 있는 경우 그룹화 표시
                airline_cols = [col for col in df.columns if 'airline' in col.lower() and ('iata' in col.lower() or 'icao' in col.lower() or 'name' in col.lower())]
                for col in airline_cols:
                    if col in df.columns and df[col].notna().any():
                        st.write(f"### 항공사별 좌석 구성 수 ({col})")
                        st.write(df.groupby(col).size().sort_values(ascending=False))
                
                aircraft_cols = [col for col in df.columns if 'aircraft' in col.lower() and ('type' in col.lower() or 'name' in col.lower() or 'model' in col.lower())]
                for col in aircraft_cols:
                    if col in df.columns and df[col].notna().any():
                        st.write(f"### 항공기별 좌석 구성 수 ({col})")
                        st.write(df.groupby(col).size().sort_values(ascending=False))
                
                # 총 좌석 수 통계
                total_seat_cols = [col for col in df.columns if 'total' in col.lower() and 'seat' in col.lower()]
                if total_seat_cols:
                    st.write("### 총 좌석 수 통계")
                    for col in total_seat_cols:
                        if df[col].dtype in ['int64', 'float64']:
                            st.write(f"**{col}**:")
                            st.write(f"- 평균: {df[col].mean():.1f}석")
                            st.write(f"- 최소: {df[col].min()}석")
                            st.write(f"- 최대: {df[col].max()}석")
                            st.write(f"- 중앙값: {df[col].median():.1f}석")
                
                # 클래스 구성 정보 분석 (class_configuration 관련 컬럼)
                class_cols = [col for col in df.columns if 'class' in col.lower() and 'configuration' in col.lower()]
                if class_cols:
                    st.write("### 클래스 구성 정보")
                    for col in class_cols:
                        if df[col].dtype == 'object':
                            st.write(f"**{col}** 샘플:")
                            # 처음 몇 개만 표시
                            sample_data = df[col].dropna().head(5)
                            for idx, val in sample_data.items():
                                st.code(f"{idx}: {val}", language="json")
                
                # 좌석 피치(Seat Pitch) 분석
                pitch_cols = [col for col in df.columns if 'pitch' in col.lower()]
                if pitch_cols:
                    st.write("### 좌석 피치(Seat Pitch) 통계")
                    for col in pitch_cols:
                        if df[col].dtype in ['int64', 'float64']:
                            st.write(f"**{col}**:")
                            st.write(df[col].describe())
                        elif df[col].dtype == 'object':
                            st.write(f"**{col}** 분포:")
                            st.write(df.groupby(col).size().sort_values(ascending=False))
                
                # 좌석 폭(Seat Width) 분석
                width_cols = [col for col in df.columns if 'width' in col.lower() and 'seat' in col.lower()]
                if width_cols:
                    st.write("### 좌석 폭(Seat Width) 통계")
                    for col in width_cols:
                        if df[col].dtype in ['int64', 'float64']:
                            st.write(f"**{col}**:")
                            st.write(df[col].describe())
                        elif df[col].dtype == 'object':
                            st.write(f"**{col}** 분포:")
                            st.write(df.groupby(col).size().sort_values(ascending=False))
                
                # 좌석 배치(Seat Arrangement) 분석
                arrangement_cols = [col for col in df.columns if 'arrangement' in col.lower()]
                if arrangement_cols:
                    st.write("### 좌석 배치(Seat Arrangement) 분포")
                    for col in arrangement_cols:
                        if col in df.columns and df[col].notna().any():
                            st.write(f"**{col}**:")
                            st.write(df.groupby(col).size().sort_values(ascending=False))
                
                # 기타 좌석 관련 컬럼
                other_seat_cols = [col for col in df.columns 
                                 if ('seat' in col.lower() or 'amenit' in col.lower() or 'special' in col.lower())
                                 and col not in total_seat_cols + pitch_cols + width_cols + arrangement_cols + class_cols]
                if other_seat_cols:
                    st.write("### 기타 좌석 관련 정보")
                    for col in other_seat_cols:
                        if df[col].dtype in ['int64', 'float64']:
                            st.write(f"**{col}** 통계:")
                            st.write(df[col].describe())
                        elif df[col].dtype == 'object':
                            # 객체 타입인 경우 샘플만 표시
                            unique_count = df[col].nunique()
                            if unique_count <= 20:
                                st.write(f"**{col}** 분포:")
                                st.write(df.groupby(col).size().sort_values(ascending=False))
                            else:
                                st.write(f"**{col}** (고유값 {unique_count}개, 샘플):")
                                st.write(df[col].dropna().head(10))


if __name__ == "__main__":
    main()
