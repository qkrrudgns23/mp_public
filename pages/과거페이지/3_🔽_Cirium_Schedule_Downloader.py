import streamlit as st
from utils.cirium import *
from utils.universal_masterplan_new import *


st.set_page_config(layout="wide")

path = r"C:\Users\qkrru\Desktop\바탕 화면\creative_code\DMK_레포지토리\cirium_database"

def process_schedule(airport, conn, table_name):
    # query
    query = f"""
    SELECT 
        operating_carrier_id,
        operating_carrier_iata,
        flight_number,
        departure_station_code_iata,
        arrival_station_code_iata,
        departure_terminal,
        arrival_terminal,
        passenger_departure_time_local,
        passenger_arrival_time_local,
        flight_distance,
        total_seats

    FROM {table_name}
    WHERE ((departure_station_code_iata = '{airport}')
        OR (arrival_station_code_iata = '{airport}')
        )
        AND is_codeshare = 0
    """
#         AND DATE(passenger_departure_time_local) = DATE('2024-09-11')

    df = pd.read_sql_query(query, conn)
    # processing
    df[f'passenger_departure_time_local']=pd.to_datetime(df[f'passenger_departure_time_local'])
    df[f'passenger_arrival_time_local']=pd.to_datetime(df[f'passenger_arrival_time_local'])
    df['total_seats']=df['total_seats'].astype(float)
    df['flight_number']=df['operating_carrier_iata']+df['flight_number'].astype(str)

    df[f'departure_terminal']=df[f'departure_terminal'].fillna('UNKNOWN')
    df[f'arrival_terminal']=df[f'arrival_terminal'].fillna('UNKNOWN')

    df=df.rename({
        'departure_station_code_iata':'departure_airport_iata',
        'arrival_station_code_iata':'arrival_airport_iata',
        'total_seats':'total_seat_count',
        'passenger_departure_time_local':'scheduled_gate_departure_local',
        'passenger_arrival_time_local':'scheduled_gate_arrival_local'
    }, axis=1)
    df["year"] = df["scheduled_gate_departure_local"].dt.year
    return df






# for airport in ["ICN", "IST", "NRT", "BLR","MNL", "SGN", "BKK","DXB","CGK","PER","DAC","GRU","MVD"]:

# 공항 코드 입력
airport_input = st.text_input(
    "Enter airport codes (comma-separated)",
    value="TPE",
    help="예: TPE, ICN 또는 TPE,ICN"
)

# 입력된 값을 리스트로 변환
if airport_input:
    airport_list = [code.strip().upper() for code in airport_input.split(',') if code.strip()]
else:
    airport_list = ["TPE"]

if len(airport_list) > 0:
    st.write(f"**공항 목록:** {', '.join(airport_list)}")

# 다운로드 시작 버튼
if st.button("🚀 다운로드 시작", type="primary"):
    if len(airport_list) == 0:
        st.error("공항 코드를 입력해주세요.")
    else:
        st.info(f"📥 총 {len(airport_list)}개 공항 다운로드를 시작합니다: {', '.join(airport_list)}")
        
        conn = connect_cirium()
        all_results = []
        
        for idx, airport in enumerate(airport_list, 1):
            st.info(f"🔄 [{idx}/{len(airport_list)}] {airport} 공항 처리 시작...")
            
            try:
                # History 처리
                st.write(f"{airport} History Dataset")
                df_history = process_schedule(airport=airport, conn=conn, table_name="schedule_history")
                df_history.to_parquet(path + "/" + f"{airport}_schedule_history.parquet")
                df_history = pd.read_parquet(path + "/" + f"{airport}_schedule_history.parquet")

                # Future 처리
                st.write(f"{airport} Future Dataset")
                df_future = process_schedule(airport=airport, conn=conn, table_name="schedule")
                df_future.to_parquet(path + "/" + f"{airport}_schedule.parquet")
                df_future = pd.read_parquet(path + "/" + f"{airport}_schedule.parquet")

                # 데이터 결합 및 처리
                df_combined = pd.concat([df_history, df_future])
                df_final = df_combined.drop_duplicates(keep='first')  # 중복 중 첫 번째만 남김
                df_final["primary_usage"] = "Passenger"

                df_final["flight_io"]=np.where(df_final["departure_airport_iata"]==airport,"d","a")
                df_final["dep/arr_airport"]=np.where(df_final["flight_io"]=="d",df_final["arrival_airport_iata"], df_final["departure_airport_iata"])

                df_final["scheduled_gate_local"]=np.where(df_final["flight_io"]=="d",df_final["scheduled_gate_departure_local"], df_final["scheduled_gate_arrival_local"])
                df_final["terminal"]=np.where(df_final["flight_io"]=="d",df_final["departure_terminal"], df_final["arrival_terminal"])


                df_carriers=pd.read_parquet('data/raw/carrier/cirium_carrier_ref.parquet') 
                df_final=pd.merge(df_final, df_carriers[['operating_carrier_id','operating_carrier_name']], on='operating_carrier_id')


                df_airports=pd.read_parquet('data/raw/airport/cirium_airport_ref.parquet') 
                df_final=pd.merge(df_final, df_airports[['airport_id','country_code',"country_name","region_name"]], left_on='dep/arr_airport', right_on='airport_id')

                df_final=df_final.drop(['airport_id'], axis=1)
                selected_country = df_airports[df_airports['airport_id']==airport]['country_code'].values[0]
                df_final['International/Domestic'] = np.where(df_final['country_code'] == selected_country, 'domestic', 'international')


                df_final.to_parquet(path + "/" + f"{airport}_schedule_ready_.parquet")
                
                all_results.append((airport, df_final))
                st.success(f"✅ [{idx}/{len(airport_list)}] {airport} 공항 처리 완료!")
                
            except Exception as e:
                st.error(f"❌ [{idx}/{len(airport_list)}] {airport} 공항 처리 중 오류 발생: {str(e)}")
        
        # 모든 처리 완료
        st.success(f"🎉 모든 공항({len(airport_list)}개) 다운로드 완료!")
        
        # 마지막 처리된 공항의 데이터 표시
        if len(all_results) > 0:
            last_airport, df = all_results[-1]
            st.write(f"### 마지막 처리된 공항: {last_airport}")
            st.dataframe(df)