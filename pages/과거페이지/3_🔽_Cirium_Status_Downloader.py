import streamlit as st
from utils.cirium import *
from utils.universal_masterplan import *


st.set_page_config(layout="wide")
conn=connect_cirium()





container = st.container(border=True)
with container:
    st.subheader("CIRIUM")
    airport_list_str = st.text_input(
            "**3Letter Airport Code**",
            value="ICN",
            key="select_airport_",
        )
    airport_list_str = airport_list_str.upper()
    airport_list = airport_list_str.split(',')
    airport_list = [airport_code.strip() for airport_code in airport_list]
    st.write(str(airport_list))

    if st.button(
        "**Download Flight Data(Cirium)**",
        use_container_width=True,
        key="downloader",
    ):
        for idx, airport_code in enumerate(airport_list):
            try : 
                import os 
                st.write(f"{airport_code} start!!!")
                file_list = [file.split('_')[0] for file in os.listdir(f'../../cirium_database')]
                if airport_code in file_list : 
                    st.write(f"{airport_code} already done")
                else : 
                    st.warning("Start Download!!")
                    airport_df_orig = query_flight_extended_total_year(airport_code, conn)
                    airport_df_orig.to_parquet( f'../../cirium_database/{airport_code}_gd_table_orig.parquet')
                    st.write(f"{airport_code} finished!!! ({idx}/{len(airport_list)}), Length : {len(airport_df_orig)}")

            except Exception as e :
                st.write(e)


    if st.button(
        "**Process Flight Data(Cirium)**",
        use_container_width=True,
        key="processor",
    ):
        for airport_code in airport_list:
            try : 
                import os 
                st.write(f"{airport_code} start!!!")
                file_list = [file.split('_')[0] for file in os.listdir(f'../../cirium_database')]
                if airport_code + "_gd_table" in file_list : 
                    st.write(f"{airport_code} already done")
                else : 
                    airport_df_orig = pd.read_parquet( f'../../cirium_database/{airport_code}_gd_table_orig.parquet')
                    airport_df = process_flight_extended_total_year(airport_code, airport_df_orig)
                    airport_df.to_parquet( f'../../cirium_database/{airport_code}_gd_table.parquet')
                    st.write(f'{airport_code} process finished!!!')
                    st.write(f'df length : {len(airport_df)}')
                    st.write(airport_df['scheduled_gate_local'].dt.year.value_counts())
                st.divider()
            except Exception as e :
                st.write(e)


container3 = st.container(border=True)
with container3:
    st.subheader("OAG")

    sc_tab, lf_tab = st.tabs(["SCHEDULE","LOAD-FACTOR"])
    with sc_tab:
        uploaded_files = st.file_uploader("OAG Schedule Processor", type="csv", accept_multiple_files=True)
        airport_code = st.text_input("Type airport iata code")
        airport_code = airport_code.upper()
        if uploaded_files:
            dfs = []
            for file in uploaded_files:
                df = pd.read_csv(file)
                dfs.append(df)
            
            combined_df = pd.concat(dfs, ignore_index=True)
            
            if st.button("스케쥴 가공하기") and airport_code:
                combined_df = process_oag_schedule(combined_df, airport_code)
                combined_df["terminal"]=combined_df["terminal"].astype(str)
                combined_df.to_parquet(f"{airport_code}_OAG_SC_Processed.parquet")
                st.success("가공 완료!")
                st.divider()
                st.write("가공된것 불러온결과!!!")
                combined_df = pd.read_parquet(f"{airport_code}_OAG_SC_Processed.parquet")
                combined_df["year"] = combined_df["scheduled_gate_local"].dt.year
                st.write(combined_df["scheduled_gate_local"].dt.year.value_counts())
                st.write(combined_df.groupby(["year"])["total_seat_count"].agg("sum"))

    with lf_tab:
        uploaded_files = st.file_uploader("OAG Load-Factor Processor", type="csv", accept_multiple_files=True)

        c1,c2,c3=st.columns(3)
        with c1:
            airport_code = st.text_input("Type airport iata code", key="load_factor_processor")
            airport_code = airport_code.upper()
        with c2:
            start_year = int(st.number_input(
                "**Start Year**",
                value=2014,
                min_value=2014, # 최소 2018년 이후부터 분석(시리움에서 2018이전데이터 못받음)
                key="Start Year",
            ))
        with c3:
            end_year = int(st.number_input(
                "**End Year**",
                value=2024,
                key="End Year",
            ))
        if uploaded_files:
            dfs = []
            for file in uploaded_files:
                df = pd.read_csv(file, usecols=range(77,96))
                dfs.append(df)
            
            df_load_factor_orig = pd.concat(dfs, ignore_index=True)
            usecols=["Op. Al. (Dominant)","Origin","Destination","Op. Al. (Leg 1)","Gateway 1","Gateway 2","Estimated Pax","Timeseries.7"]
            df_load_factor_orig=df_load_factor_orig[usecols]


            if st.button("로드펙터 가공하기") and airport_code:
                df_schedule_orig=pd.read_parquet(f"{airport_code}_OAG_SC_Processed.parquet")

                load_factor_df = process_oad_load_factor(df_schedule_orig, df_load_factor_orig, airport_code=airport_code, start_year=start_year, end_year=end_year)
                load_factor_df.to_parquet(f"{airport_code}_OAG_LF_Processed.parquet")
                st.success("가공 완료!")
                st.divider()
                st.write("가공된것 불러온결과!!!")
                load_factor_df = pd.read_parquet(f"{airport_code}_OAG_LF_Processed.parquet")
                st.write(load_factor_df)


container2 = st.container(border=True)
with container2:
    st.subheader("REF DATA")
    if st.button(
        "**Download Airport Data(Cirium)**",
        use_container_width=True,
        key="downloader_airport",
    ):
        df_airports = process_airports(conn)
        df_airports.to_parquet('df_airports.parquet')
        st.dataframe(df_airports)



    if st.button(
        "**Download Airport Data(Open Data)**",
        use_container_width=True,
        key="downloader_airport_opendata",
    ):
        process_opendata()


    if st.button(
        "**Process Peak Calculation(Cirium)**",
        use_container_width=True,
        key="peak_calculation_cirium",
    ):
        airport_peak_df=[]
        for idx,airport_code in enumerate(airport_list):
            st.write(idx)
            try:
                df_airport=pd.read_parquet( f'../../cirium_database/{airport_code}_gd_table.parquet')
                df_airport['scheduled_gate_date'] = df_airport['scheduled_gate_local'].dt.date
                df_airport['scheduled_gate_hour'] = df_airport['scheduled_gate_local'].dt.hour
                df_airport['scheduled_gate_minute'] = df_airport['scheduled_gate_local'].dt.minute
                df_airport['scheduled_gate_month'] = df_airport['scheduled_gate_local'].dt.month

                for year in range(2018,2025):
                    try:
                        df_year=df_airport[df_airport["scheduled_gate_local"].dt.year==year]
                        if len(df_year)<10:
                            pass
                        else:
                            H_Factor, D_Factor, PHF, SUM = H_D(df_year, default_cols=["movement"])
                            airport_peak_df+=[[airport_code, year, H_Factor, D_Factor, PHF, SUM]]
                    except:
                        st.write(airport_code, year, "/ERROR!!")
            except:
                st.write(airport_code, "/ERROR!!")
        airport_peak_df=pd.DataFrame(airport_peak_df, columns=["airport_iata_code","year","h_factor","d_factor","phf","movement"])
        st.dataframe(airport_peak_df)
        airport_peak_df.to_parquet("data/raw/peak/airport_peak_df_cirium.parquet")



    if st.button(
        "**Process Peak Calculation(OAG)**",
        use_container_width=True,
        key="peak_calculation_oag",
    ):
        airport_list=["TIV","TGD","DVO","AHB","KWI","UGC"]
        airport_peak_df=[]
        for idx,airport_code in enumerate(airport_list):
            st.write(airport_code, "완료")
            try:
                df_airport=pd.read_parquet( f'{airport_code}_OAG_SC_Processed.parquet')
                df_airport['scheduled_gate_date'] = df_airport['scheduled_gate_local'].dt.date
                df_airport['scheduled_gate_hour'] = df_airport['scheduled_gate_local'].dt.hour
                df_airport['scheduled_gate_minute'] = df_airport['scheduled_gate_local'].dt.minute
                df_airport['scheduled_gate_month'] = df_airport['scheduled_gate_local'].dt.month

                for year in range(2018,2025):
                    try:
                        df_year=df_airport[df_airport["scheduled_gate_local"].dt.year==year]
                        if len(df_year)<10:
                            pass
                        else:
                            H_Factor, D_Factor, PHF, SUM = H_D(df_year, default_cols=["movement"])
                            airport_peak_df+=[[airport_code, year, H_Factor, D_Factor, PHF, SUM]]
                    except Exception as e:
                        st.write(airport_code, year, "/ERROR!!")
                        st.write(e)
            except Exception as e:
                st.write(airport_code, "/ERROR!!")
                st.write(e)
        airport_peak_df=pd.DataFrame(airport_peak_df, columns=["airport_iata_code","year","h_factor","d_factor","phf","movement"])
        st.dataframe(airport_peak_df)
        airport_peak_df.to_parquet("data/raw/peak/airport_peak_df_oag.parquet")



    if st.button(
        "**Process Cities Data**",
        use_container_width=True,
        key="downloader_cities_data",
    ):

        aci_airport=pd.read_csv('data/raw/airport/aci_airport.csv', encoding='cp949')
        df_airport=pd.read_parquet("data/raw/airport/cirium_airport_ref.parquet")
        df_cities= pd.read_csv('data/raw/forecast/worldcities.csv', encoding="cp949")
        

        for radius_distance in [30, 100, 200]: ## 300, 500, 1000
            def haversine_vectorized(lat1, lon1, lat2_array, lon2_array):
                R = 6371  # Earth's radius in kilometers
                # Convert all latitudes/longitudes to radians
                lat1_rad = np.radians(lat1)
                lon1_rad = np.radians(lon1)
                lat2_rad = np.radians(lat2_array)
                lon2_rad = np.radians(lon2_array)
                # Compute differences
                dlat = lat2_rad - lat1_rad
                dlon = lon2_rad - lon1_rad
                # Haversine formula
                a = np.sin(dlat/2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2.0)**2
                c = 2 * np.arcsin(np.sqrt(a))
                distances = R * c
                return distances

            # 새로운 컬럼 생성

            df_airport[f'distance_{radius_distance}km'] = 0

            # 각 공항별로 계산
            for idx, airport in df_airport.iterrows():
                if idx//1000==idx/1000:
                    st.write(idx, "/", len(df_airport), "/", radius_distance)
                # 현재 공항의 위치 좌표
                curr_lat = airport['lat']
                curr_lon = airport['lon']
                
                # 현재 공항 기준으로 거리 계산
                distances = haversine_vectorized(curr_lat, curr_lon,
                                            df_cities['lat'].values,
                                            df_cities['lng'].values)
                
                # 반경 내 도시들 필터링
                cities_in_range = df_cities[(distances <= radius_distance) & (df_cities["population"] > 0)]
                
                # 해당 반경 내 총 인구수 계산
                total_population = cities_in_range['population'].sum()
                
                # 결과를 데이터프레임에 저장
                df_airport.at[idx, f'distance_{radius_distance}km'] = total_population

            # population을 정수형으로 변환
            df_airport[f'distance_{radius_distance}km'] = df_airport[f'distance_{radius_distance}km'].astype(int)

            st.dataframe(df_airport)



        aci_airport_year=aci_airport[aci_airport["year"]==2019]

        df_airport = pd.merge(df_airport, aci_airport_year, left_on="airport_id",right_on="airport_code_iata", how="left")
        df_airport.to_parquet("data/raw/df_airports.parquet")

