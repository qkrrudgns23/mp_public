import pandas as pd
import psycopg2
import numpy as np
from datetime import datetime
import streamlit as st

def process_flight_extended(airport, date, conn, acdm_mapping, flight_io):
    # query
    query = f"""
    SELECT 
        operating_carrier_id,
        operating_carrier_name,
        operating_carrier_iata,
        flight_number,
        tail_number,
        departure_airport_iata,
        arrival_airport_iata,
        {flight_io}_terminal,
        {flight_io}_gate,
        scheduled_gate_{flight_io}_local,
        actual_gate_{flight_io}_local,
        actual_runway_{flight_io}_local,
        total_seat_count
    FROM flights_extended
    WHERE (({flight_io}_airport_id = '{airport}') OR ({flight_io}_airport_iata = '{airport}'))
        AND DATE(scheduled_gate_{flight_io}_local) = DATE('{date}')
        AND total_seat_count > 0
        AND is_cancelled IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    
    # processing
    df[f'scheduled_gate_{flight_io}_local']=pd.to_datetime(df[f'scheduled_gate_{flight_io}_local'])
    df[f'actual_gate_{flight_io}_local']=pd.to_datetime(df[f'actual_gate_{flight_io}_local'])
    df['total_seat_count']=df['total_seat_count'].astype(float)
    df['flight_number']=df['operating_carrier_iata']+df['flight_number'].astype(str)
    df[f'{flight_io}_terminal']=df[f'{flight_io}_terminal'].fillna('UNKNOWN')
    df=df.rename({
        f'scheduled_gate_{flight_io}_local': acdm_mapping[f'scheduled_gate_{flight_io}_local'],
        f'actual_gate_{flight_io}_local' : acdm_mapping[f'actual_gate_{flight_io}_local'],
        f'actual_runway_{flight_io}_local' : acdm_mapping[f'actual_runway_{flight_io}_local'],
    }, axis=1)
    return df

def make_ground_time_col(df):
    df = df.sort_values(['aircraft_serial_number', 'actual_gate_local','actual_runway_local'])
    df['io_num'] = (df['flight_io'] == 'd').astype(int)
    df['is_turnaround'] = 1
    
    for name, col in zip(
        ['standing_time', 'ground_time'],
        ['actual_gate_local', 'actual_runway_local']
    ):
        if name in ['standing_time', 'ground_time']:
            df[name] = df.groupby('aircraft_serial_number').apply(lambda x: 
                (x[col].shift(-1) - x[col]) * 
                ((x['io_num'] == 0) & (x['io_num'].shift(-1) == 1))
            ).reset_index(level=0, drop=True)
            df.loc[df['flight_io'] == 'd', name] = pd.NaT
            df[name] = df.groupby('aircraft_serial_number')[name].ffill(limit=1)
            df[name] = df[name].dt.total_seconds()//60
        
        if name == 'standing_time':  # actual_gate_local 처리할 때 is_turnaround도 계산
            df['is_turnaround'] = df.groupby('aircraft_serial_number').apply(lambda x:
                ((x['io_num'] == 0) & (x['io_num'].shift(-1) == 1) & 
                (x['gate'] == x['gate'].shift(-1))).astype(int)
            ).reset_index(level=0, drop=True)
            df.loc[df['flight_io'] == 'd', 'is_turnaround'] = None
            df['is_turnaround'] = df.groupby('aircraft_serial_number')['is_turnaround'].ffill(limit=1)

    df = df.drop('io_num', axis=1)
    return df

def query_flight_extended_total_year(airport, conn):
    # query
    query = f"""
    SELECT 
        flight_id,
        operating_carrier_id,
        operating_carrier_name,
        operating_carrier_iata,
        flight_number,
        tail_number,
        fleet_aircraft_id, 
        departure_terminal,
        arrival_terminal,
        departure_gate,
        arrival_gate,

        departure_airport_id,
        arrival_airport_id,
        departure_airport_iata,
        arrival_airport_iata,
        scheduled_gate_departure_local,
        scheduled_gate_arrival_local,
        actual_gate_departure_local,
        actual_gate_arrival_local,
        actual_runway_departure_local,
        actual_runway_arrival_local,

        actual_taxi_out_time,
        actual_taxi_in_time,
        gate_departure_delay,
        gate_arrival_delay,
        baggage_claim,
        actual_flight_duration,
        actual_block_time,
        aircraft_type,
        aircraft_type_series,
        aircraft_code_iata,
        aircraft_code_icao,
        aircraft_serial_number,

        total_seat_count,
        is_cancelled,
        is_diverted

    FROM flights_extended
    WHERE ((departure_airport_id = '{airport}')
        OR (arrival_airport_id = '{airport}')
        OR (departure_airport_iata = '{airport}')
        OR (arrival_airport_iata = '{airport}')
        )
    """

    query_add = f"""
    SELECT 
        flight_id,
        aircraft_market_sector,
        aircraft_class,
        aircraft_market_group,
        aircraft_family, 
        primary_usage,
        flight_distance_km
    FROM tracked_utilization
    WHERE ((departure_airport_id = '{airport}')
        OR (arrival_airport_id = '{airport}')
        OR (departure_airport_iata = '{airport}')
        OR (tracked_arrival_airport_iata = '{airport}')
        )
    """
    #         AND flown_total_seats > 0


    df_orig = pd.read_sql_query(query, conn); df_add = pd.read_sql_query(query_add, conn)
    df_orig = pd.merge(df_orig,df_add, on='flight_id', how='left')
    print('quary finished!!!')
    return df_orig


def fill_ref_col(df, fill_col, ref_cols, method='mode', add_e=False):
    df['ref_col'] = df[ref_cols].astype(str).agg('_'.join, axis=1)
    if method=='mode':
        mapping = df[df[fill_col].notna()].groupby('ref_col')[fill_col].agg(
            lambda x: x.value_counts().index[0] if len(x) > 0 else None)
    elif method=='first':
        mapping = df[df[fill_col].notna()].groupby('ref_col')[fill_col].first()
    if add_e==True:
        mapping = mapping.apply(lambda x: f"{x} (E)" if pd.notna(x) and ' (E)' not in str(x) else x)
    df[fill_col] = df[fill_col].fillna(df['ref_col'].map(mapping))
    df = df.drop('ref_col', axis=1)
    return df


def fill_ref_cols(df, fill_cols, ref_cols, method='mode', add_e=False):
    """
    여러 컬럼을 한번에 채우는 함수
    
    Parameters:
    df: DataFrame
    fill_cols: list - 채우고자 하는 컬럼들의 리스트
    ref_cols: list - 참조할 컬럼들의 리스트
    method: str - 'mode' 또는 'first'
    add_e: bool - True일 경우 값 뒤에 " (E)" 추가
    """
    df['ref_col'] = df[ref_cols].astype(str).agg('_'.join, axis=1)
    
    if method == 'mode':
        mapping = df[df[fill_cols].notna().any(axis=1)].groupby('ref_col')[fill_cols].agg(
            lambda x: x.dropna().value_counts().index[0] if len(x.dropna()) > 0 else None)
    elif method == 'first':
        mapping = df[df[fill_cols].notna().any(axis=1)].groupby('ref_col')[fill_cols].first()
    
    if add_e:
        for col in mapping.columns:
            mapping[col] = mapping[col].apply(lambda x: f"{x} (E)" if pd.notna(x) and ' (E)' not in str(x) else x)
    
    for fill_col in fill_cols:
        map=mapping[fill_col]
        valid_map = {k: v for k, v in map.items() if k != "None"}
        df[fill_col] = df[fill_col].fillna(df['ref_col'].map(valid_map))
        
    df = df.drop('ref_col', axis=1)
    return df



def process_flight_extended_total_year(airport, df_orig):
    df = df_orig[(df_orig["is_cancelled"]!=1)&(df_orig["is_diverted"]!=1)].copy()
    st.write(len(df))





    df['actual_taxi_out_time'] = df['actual_taxi_out_time'].fillna(df.groupby('departure_gate')['actual_taxi_out_time'].transform('mean'))//1 # Assumption
    df['actual_taxi_out_time'] = df['actual_taxi_out_time'].fillna(df['actual_taxi_out_time'].mean())//1 # Assumption
    df['actual_taxi_in_time'] = df['actual_taxi_in_time'].fillna(df.groupby('arrival_gate')['actual_taxi_in_time'].transform('mean'))//1 # Assumption
    df['actual_taxi_in_time'] = df['actual_taxi_in_time'].fillna(df['actual_taxi_in_time'].mean())//1 # Assumption

    df['actual_gate_departure_local'] = df['actual_gate_departure_local'].fillna(df['actual_runway_departure_local']-pd.to_timedelta(df['actual_taxi_out_time'], unit='m')) # Assumption
    df['actual_gate_arrival_local'] = df['actual_gate_arrival_local'].fillna(df['actual_runway_arrival_local']+pd.to_timedelta(df['actual_taxi_in_time'], unit='m')) # Assumption
    df['gate_departure_delay'] = df['gate_departure_delay'].fillna((df['actual_gate_departure_local']-df['scheduled_gate_departure_local']).dt.total_seconds()//60) # Assumption
    df['gate_arrival_delay'] = df['gate_arrival_delay'].fillna((df['actual_gate_arrival_local']-df['scheduled_gate_arrival_local']).dt.total_seconds()//60) # Assumption
    # processing
    dep_mask = ((df['departure_airport_id']==airport) | (df['departure_airport_iata']==airport))
    df['origin_airport']=airport
    df['dep/arr_airport']=np.where(dep_mask, 
                                    df['arrival_airport_id'].fillna(df['arrival_airport_iata']),
                                    df['departure_airport_id'].fillna(df['departure_airport_iata']),
                                    )
    df['flight_io']=np.where(dep_mask, 'd', 'a')
    df['terminal']=np.where(dep_mask, df['departure_terminal'], df['arrival_terminal'])
    df['baggage_claim']=np.where(dep_mask, None, df['baggage_claim'])
    df['gate']=np.where(dep_mask, df['departure_gate'], df['arrival_gate'])
    df['scheduled_gate_local']=np.where(dep_mask, df['scheduled_gate_departure_local'], df['scheduled_gate_arrival_local'])
    df['actual_gate_local']=np.where(dep_mask, df['actual_gate_departure_local'], df['actual_gate_arrival_local'])
    df['actual_runway_local']=np.where(dep_mask, df['actual_runway_departure_local'], df['actual_runway_arrival_local'])
    df['actual_taxi_time']=np.where(dep_mask, df['actual_taxi_out_time'], df['actual_taxi_in_time'])
    df['gate_delay']=np.where(dep_mask, df['gate_departure_delay'], df['gate_arrival_delay'])


    # scheduled_gate_local과 actual_gate_local 상호 보완
    df['scheduled_gate_local'] = df['scheduled_gate_local'].fillna(df['actual_gate_local']) # Assumption
    df['actual_gate_local'] = df['actual_gate_local'].fillna(df['scheduled_gate_local']) # Assumption

    # flight_io에 따른 actual_runway_local 계산
    taxi_delta = pd.Timedelta(minutes=1) * df['actual_taxi_time']
    arrival_mask = (df['flight_io'] == 'a')& (df['actual_runway_local'].isna())
    departure_mask = (df['flight_io'] == 'd')& (df['actual_runway_local'].isna())
    df.loc[arrival_mask, 'actual_runway_local'] = df.loc[arrival_mask, 'actual_gate_local'] - taxi_delta[arrival_mask] # Assumption
    df.loc[departure_mask, 'actual_runway_local'] = df.loc[departure_mask, 'actual_gate_local'] + taxi_delta[departure_mask] # Assumption




    df_aircraft=pd.read_parquet('data/raw/aircraft/cirium_aircraft_ref.parquet') 
    df_airports=pd.read_parquet('data/raw/airport/cirium_airport_ref.parquet') 
    df_aircraft = df_aircraft[['aircraft_id','aircraft_start_of_life_date','aircraft_age_months','operating_maximum_takeoff_weight_lb','maximum_landing_weight_lb','maximum_payload_lb']]
    df = add_aircraft_info(df, df_aircraft)
    st.write("aircraft info finished!!!")

    df=pd.merge(df, df_airports[['airport_id','country_code',"country_name","region_name"]], left_on='dep/arr_airport', right_on='airport_id')
    selected_country = df_airports[df_airports['airport_id']==airport]['country_code'].values[0]
    df['International/Domestic'] = np.where(df['country_code'] == selected_country, 'domestic', 'international')
    st.write("Int/Dom & Region/Country finished!!!")


    df['movement']=1
    df['actual_runway_month']=df['actual_runway_local'].dt.year.astype(str) + df['actual_runway_local'].dt.month.astype(str)
    st.write("basic fillna finished!!!")
    st.write('---')



    # operating_carrier_iata
    before_carrier = df["operating_carrier_iata"].isna().sum()
    df = fill_ref_cols(df, fill_cols=['operating_carrier_iata'], ref_cols=['tail_number'], method='mode')
    after_carrier = df["operating_carrier_iata"].isna().sum()
    st.write(f"carrier before/after : {before_carrier}/{after_carrier}")
    df = fill_ref_cols(df, fill_cols=['operating_carrier_iata'], ref_cols=['aircraft_serial_number'], method='mode')
    after_carrier = df["operating_carrier_iata"].isna().sum()
    st.write(f"carrier before/after : {before_carrier}/{after_carrier}")
    df["operating_carrier_iata"] = df["operating_carrier_iata"].fillna(df["operating_carrier_id"].str.replace("*",""))
    df['carrier_flight'] = df['operating_carrier_iata'] + df['flight_number'].astype(str)
    after_carrier = df["operating_carrier_iata"].isna().sum()
    st.write(f"carrier before/after : {before_carrier}/{after_carrier}")
    st.write('---')


    # terminal
    before_terminal = df["terminal"].isna().sum()
    df = fill_ref_cols(df, fill_cols=['terminal'], ref_cols=['operating_carrier_iata','flight_number','actual_runway_month'], method='mode', add_e=False)
    df = fill_ref_cols(df, fill_cols=['terminal'], ref_cols=['tail_number','actual_runway_month'], method='mode', add_e=False)
    df = fill_ref_cols(df, fill_cols=['terminal'], ref_cols=['operating_carrier_iata','actual_runway_month'], method='mode', add_e=False)
    after_terminal = df["terminal"].isna().sum()
    st.write(f"terminal before/after : {before_terminal}/{after_terminal}")
    df['terminal'] = df['terminal'].fillna('UNKNOWN') # Assumption
    st.write('---')


    # flight_distance_km
    before_flight_distance_km = df["flight_distance_km"].isna().sum()
    df = fill_ref_cols(df, fill_cols=["flight_distance_km"], ref_cols=["departure_airport_id","arrival_airport_id"], method='mode', add_e=False)
    after_flight_distance_km = df["flight_distance_km"].isna().sum()
    st.write(f"flight_distance_km before/after : {before_flight_distance_km}/{after_flight_distance_km}")
    st.write('---')



    before_mtow = df["operating_maximum_takeoff_weight_lb"].isna().sum()
    df = fill_ref_cols(df, fill_cols=['operating_maximum_takeoff_weight_lb'], ref_cols=['aircraft_family'], method='mode', add_e=False)
    df = fill_ref_cols(df, fill_cols=['maximum_landing_weight_lb'], ref_cols=['aircraft_family'], method='mode', add_e=False)
    df = fill_ref_cols(df, fill_cols=['maximum_payload_lb'], ref_cols=['aircraft_family'], method='mode', add_e=False)
    df = fill_ref_cols(df, fill_cols=['operating_maximum_takeoff_weight_lb'], ref_cols=['aircraft_market_group'], method='mode', add_e=False)
    df = fill_ref_cols(df, fill_cols=['maximum_landing_weight_lb'], ref_cols=['aircraft_market_group'], method='mode', add_e=False)
    df = fill_ref_cols(df, fill_cols=['maximum_payload_lb'], ref_cols=['aircraft_market_group'], method='mode', add_e=False)
    df = fill_ref_cols(df, fill_cols=['operating_maximum_takeoff_weight_lb'], ref_cols=['aircraft_class'], method='mode', add_e=False)
    df = fill_ref_cols(df, fill_cols=['maximum_landing_weight_lb'], ref_cols=['aircraft_class'], method='mode', add_e=False)
    df = fill_ref_cols(df, fill_cols=['maximum_payload_lb'], ref_cols=['aircraft_class'], method='mode', add_e=False)
    after_mtow = df["operating_maximum_takeoff_weight_lb"].isna().sum()
    st.write(f"MTOW before/after : {before_mtow}/{after_mtow}")
    st.write('---')

    # flight
    before_aircraft_code_icao = df["aircraft_code_icao"].isna().sum()
    flight_category_cols = [
                    'aircraft_market_sector',
                    'aircraft_class',
                    'aircraft_market_group',
                    'aircraft_family',
                    'aircraft_type',
                    'aircraft_type_series',
                    'aircraft_code_iata',
                    'aircraft_code_icao',
                    ]
    
    aircraft_key = df[flight_category_cols].value_counts().reset_index()
    aircraft_key = aircraft_key.drop_duplicates(subset=['aircraft_code_icao'])
    df = fill_ref_cols(df, fill_cols=["aircraft_code_icao"], ref_cols=["operating_carrier_iata","flight_number"], method='mode', add_e=False)
    after_aircraft_code_icao = df["aircraft_code_icao"].isna().sum()
    st.write(f"aircraft_code_icao before/after : {before_aircraft_code_icao}/{after_aircraft_code_icao}")
    st.write('---')




    # primary_usage / total_seat_count
    before_primary_usage = df["primary_usage"].isna().sum()
    before_total_seat_count = df["total_seat_count"].isna().sum()
    df.loc[(df['primary_usage'].isna()) & (df['total_seat_count'] > 0), 'primary_usage'] = 'Passenger'
    df.loc[(df['primary_usage'] == 'Freight/Cargo'), 'total_seat_count'] = 0
    df = fill_ref_cols(df, fill_cols=['primary_usage'], ref_cols=['tail_number'], method='mode', add_e=False)
    df = fill_ref_cols(df, fill_cols=['total_seat_count'], ref_cols=['tail_number'], method='first')

    after_primary_usage = df["primary_usage"].isna().sum()
    after_total_seat_count = df["total_seat_count"].isna().sum()
    st.write(f"1차 : primary_usage before/after : {before_primary_usage}/{after_primary_usage}")
    st.write(f"1차 : total_seat_count before/after : {before_total_seat_count}/{after_total_seat_count}")



    df = fill_ref_cols(df, fill_cols=['primary_usage'], ref_cols=['dep/arr_airport','operating_carrier_iata','flight_number'], method='mode', add_e=False)
    df = fill_ref_cols(df, fill_cols=['primary_usage'], ref_cols=['dep/arr_airport','operating_carrier_iata','aircraft_type_series'], method='mode', add_e=False)
    df = fill_ref_cols(df, fill_cols=['total_seat_count'], ref_cols=['primary_usage','aircraft_type_series'], method='mode', add_e=False)
    df = fill_ref_cols(df, fill_cols=['total_seat_count'], ref_cols=['primary_usage','aircraft_type'], method='mode', add_e=False)
    df = fill_ref_cols(df, fill_cols=['total_seat_count'], ref_cols=['primary_usage','operating_carrier_iata','flight_number'], method='mode', add_e=False)
    df.loc[(df['primary_usage'].isna()) & (df['total_seat_count'] > 0), 'primary_usage'] = 'Passenger'


    df = fill_ref_cols(df, fill_cols=['total_seat_count'], ref_cols=['primary_usage','aircraft_type_series'], method='mode', add_e=False)
    df = fill_ref_cols(df, fill_cols=['total_seat_count'], ref_cols=['primary_usage','aircraft_type'], method='mode', add_e=False)
    df = fill_ref_cols(df, fill_cols=['total_seat_count'], ref_cols=['primary_usage','operating_carrier_iata','flight_number'], method='mode', add_e=False)

    df.loc[(df['primary_usage'] == 'Freight/Cargo'), 'total_seat_count'] = 0
    after_primary_usage = df["primary_usage"].isna().sum()
    after_total_seat_count = df["total_seat_count"].isna().sum()
    st.write(f"2차 : primary_usage before/after : {before_primary_usage}/{after_primary_usage}")
    st.write(f"2차 : total_seat_count before/after : {before_total_seat_count}/{after_total_seat_count}")


    df = fill_ref_cols(df, fill_cols=['total_seat_count'], ref_cols=['primary_usage','aircraft_family'], method='mode', add_e=False)
    df = fill_ref_cols(df, fill_cols=['total_seat_count'], ref_cols=['primary_usage','aircraft_market_group'], method='mode', add_e=False)
    df.loc[(df['primary_usage'].isna()) & (df['total_seat_count'] > 0), 'primary_usage'] = 'Passenger'

    st.write(df["primary_usage"].value_counts())
    after_primary_usage = df["primary_usage"].isna().sum()
    after_total_seat_count = df["total_seat_count"].isna().sum()
    st.write(f"3차 : primary_usage before/after : {before_primary_usage}/{after_primary_usage}")
    st.write(f"3차 : total_seat_count before/after : {before_total_seat_count}/{after_total_seat_count}")
    st.write('---')

    df.loc[(df['primary_usage'].isna()) & (df['total_seat_count'] == 0), 'primary_usage'] = 'UNKNOWN'
    df['primary_usage'] = df['primary_usage'].fillna('UNKNOWN') # Assumption


    # ground / standing time
    df = make_ground_time_col(df)



    # final column filter
    df = df[[
        'flight_id',
        'origin_airport', 
        'dep/arr_airport',
        'flight_distance_km',
        'total_seat_count', 
        'flight_io',
        'movement',
        'baggage_claim',

        'scheduled_gate_local',
        'actual_gate_local',
        'actual_runway_local',

        'operating_carrier_id', 
        'operating_carrier_name',
        'operating_carrier_iata', 
        'flight_number', 
        'tail_number',
        'aircraft_serial_number', 
        'terminal', 
        'gate', 

        'actual_taxi_time',
        'actual_block_time',
        'gate_delay',
        'actual_flight_duration',
        'standing_time',
        'ground_time',
        'is_turnaround',

        'aircraft_market_sector',
        'aircraft_class',
        'aircraft_market_group',
        'aircraft_family', 
        'aircraft_type', 
        'aircraft_type_series',
        'aircraft_code_iata', 
        'aircraft_code_icao',
        'primary_usage',

        'aircraft_start_of_life_date',
        'aircraft_age_months',
        'operating_maximum_takeoff_weight_lb',
        'maximum_landing_weight_lb',
        'maximum_payload_lb',

        "country_code",
        "country_name",
        "region_name",
        "International/Domestic",
    ]]
    st.write(df[df["total_seat_count"].isna()])
    return df

def process_schedule(airport, date, conn, acdm_mapping, flight_io):
    # query
    query = f"""
    SELECT 
        operating_carrier_id,
        operating_carrier_iata,
        flight_number,
        departure_station_code_iata,
        arrival_station_code_iata,
        {flight_io}_terminal,
        passenger_{flight_io}_time_local,
        total_seats
    FROM schedule
    WHERE ({flight_io}_station_code_iata = '{airport}')
        AND DATE(passenger_{flight_io}_time_local) = DATE('{date}')
        AND total_seats > 0
        AND is_codeshare = 0
    """
    df = pd.read_sql_query(query, conn)

    # processing
    df[f'passenger_{flight_io}_time_local']=pd.to_datetime(df[f'passenger_{flight_io}_time_local'])
    df['total_seats']=df['total_seats'].astype(float)
    df['flight_number']=df['operating_carrier_iata']+df['flight_number'].astype(str)
    df[f'{flight_io}_terminal']=df[f'{flight_io}_terminal'].fillna('UNKNOWN')
    df=df.rename({
        f'passenger_{flight_io}_time_local': acdm_mapping[f'passenger_{flight_io}_time_local'],
        'departure_station_code_iata':'departure_airport_iata',
        'arrival_station_code_iata':'arrival_airport_iata',
        'total_seats':'total_seat_count'
    }, axis=1)
    return df

def process_opendata():
    import requests
    urls = [
        'https://davidmegginson.github.io/ourairports-data/airports.csv',
        'https://davidmegginson.github.io/ourairports-data/runways.csv',
        'https://davidmegginson.github.io/ourairports-data/navaids.csv'
    ]

    file_names = [
        'data/raw/airport/airports(open_data).csv',
        'data/raw/airport/runways(open_data).csv',
        'data/raw/airport/navaids(open_data).csv'
    ]

    for url, file_name in zip(urls, file_names):
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_name, 'wb') as f:
                f.write(response.content)
            st.write(f'Successfully downloaded {file_name}')
        else:
            st.write(f'Failed to download {url}')


def process_airports(conn):
    # query
    query = f"""
    SELECT airport_id, airport_iata, airport_icao, country_code, country_name, region_name, lat_long, elevation_feet, name, city
    FROM airports
    WHERE is_active = 1
    """
    df = pd.read_sql_query(query, conn)  
    

    conditions = [
        df['name'].str.lower().str.contains('airport'),
        df['name'].str.lower().str.contains('aerodrome'),
        df['name'].str.lower().str.contains('airfield'),
        df['name'].str.lower().str.contains('helip'),
    ]

    choices = [
        'airport',
        'aerodrome',
        'airfield',
        'helipad',
    ]

    df['category'] = np.select(conditions, choices, default='others')

    # WKB 문자열을 위도/경도로 변환하는 함수
    def convert_wkb_to_latlong(wkb_str):
        import binascii
        from shapely import wkb
        binary_data = binascii.unhexlify(wkb_str)
        point = wkb.loads(binary_data)
        return pd.Series({'lat': point.y, 'lon': point.x})

    df[['lat', 'lon']] = df['lat_long'].apply(convert_wkb_to_latlong)
    return df

def process_aircraft_configurations(conn):
    # query
    query = f"""
    SELECT *
    FROM aircraft_configurations
    """
    df = pd.read_sql_query(query, conn)  
    return df

def process_aircraft(conn):
    # query
    query = f"""
    SELECT *
    FROM aircraft
    """
    df = pd.read_sql_query(query, conn)  
    return df

def process_carriers(conn):
    # query
    query = f"""
    SELECT carrier_id, name
    FROM carriers
    WHERE is_active = 1
    """
    df = pd.read_sql_query(query, conn)  
    # processing
    df=df.rename({
        'name': 'operating_carrier_name',
        'carrier_id':'operating_carrier_id',
    }, axis=1)    
    return df

def add_airport_info(df, df_airports):
    # add country, region information
    for flight_io in ['departure','arrival']:
        df_io_airports=df_airports.copy()
        df_io_airports.columns =[f'{flight_io}_' + str(col) for col in df_airports.columns]
        df=pd.merge(df, df_io_airports, left_on=f'{flight_io}_airport_iata', right_on=f'{flight_io}_airport_id', how='left')
        df=df.drop(f'{flight_io}_airport_id', axis=1)
    # add intermational/Domestic information
    df['International/Domestic'] = np.where(df['departure_country_code'] == df['arrival_country_code'], 'domestic', 'international')
    return df

def add_aircraft_info(df, df_aircraft):
    df=pd.merge(df, df_aircraft, left_on=f'fleet_aircraft_id', right_on=f'aircraft_id', how='left')
    df=df.drop(['aircraft_id'], axis=1)
    return df

def add_carrier_info(df, df_carriers):
    df=pd.merge(df, df_carriers, on='operating_carrier_id', how='left')
    return df

@st.fragment
def connect_cirium():
    redshift_username = "wcummekqqmoh" #"rknzizulmxbu"
    redshift_password = "3+CyF0@hx0V_4y]E" #"qU5#|0Ni%f3U~Zk5"
    redshift_hostname = "alto.sky.cirium.com"
    redshift_port = 5439
    redshift_database = "ciriumsky"

    # Redshift에 연결
    conn = psycopg2.connect(
        host=redshift_hostname,
        port=redshift_port,
        dbname=redshift_database,
        user=redshift_username,
        password=redshift_password,
        sslmode="require"  # SSL 연결 강제
    )
    return conn

@st.fragment
def managing_cirium_data(airport, selected_date, flight_io, conn):
    # get today info
    today = str(datetime.now())[:10]
    acdm_mapping={
        f'passenger_departure_time_local':'SOBT', # from schedule table
        'scheduled_gate_departure_local':'SOBT',
        'actual_gate_departure_local':'AOBT',
        'actual_runway_departure_local':'ATOT',
        f'passenger_arrival_time_local':'SIBT', # from schedule table
        'scheduled_gate_arrival_local':'SIBT',
        'actual_gate_arrival_local':'AIBT',
        'actual_runway_arrival_local':'ALDT',
        }
    flight_io_dt=acdm_mapping[f'scheduled_gate_{flight_io}_local']
    print(today, selected_date)
    if today > selected_date :
        print('actual data')
        df_flight_extended=process_flight_extended(airport, selected_date, conn, acdm_mapping, flight_io)
        df_airports=pd.read_parquet('data/raw/cirium_airport_ref.parquet') # df_airports=process_airports(conn)
        df_flight_extended=add_airport_info(df_flight_extended, df_airports)
        is_actual=True
        return df_flight_extended, flight_io_dt, is_actual
    elif today <= selected_date :
        print('schedule data')
        df_schedule=process_schedule(airport, selected_date, conn, acdm_mapping, flight_io)
        df_airports=pd.read_parquet('data/raw/cirium_airport_ref.parquet') # df_airports=process_airports(conn)
        df_carriers=pd.read_parquet('data/raw/carrier/cirium_carrier_ref.parquet') # df_carriers=process_carriers(conn)
        df_schedule=add_airport_info(df_schedule, df_airports)
        df_schedule=add_carrier_info(df_schedule, df_carriers)
        is_actual=False
        return df_schedule, flight_io_dt, is_actual
    



# General Aircraft Name
# Carrier Alliance
# Aircraft Manufacturer
# Aircraft Max Take Off Weights (t)
def process_oag_schedule(df_oag, airport_code):

    """
    [Dimensions]
    Carrier : Carrier Code / Carrier Name / Flight No / Carrier Alliance

    Origin & Destination : Dep Airport Code / Dep IATA Country Code / Dep IATA Country Name / Dep Region Code / Dep Region Name / Dep Terminal >> arrival vice versa / International/Domestic

    Days & Time : Local Dep Time / Local Arr Time / Local Arr Day / Flying Time / Ground Time 

    Equipment : General Aircraft Code / General Aircraft Name / Specific Aircraft Code / Equipment Group / Seats / First Seats / Business Seats / Economy Seats / Aircraft Range(km) / Aircraft Manufacturer / Aircraft Max Take Off Weight (t)
    
    Distance : GCD (km)
    Service Type : Mainlin/Low Cost
    Other : Freight Class

    [Metrics]
    FreightTons (Total)

    From : start date >> To : End Data
    * Use Time Series >> check

    Selected Flight Type : Operating Flights
    Show by : Operating Carrier
    Stops : Non_stop >> check

    Origin & Destination : origin : ex) ICN
    Direction : Two Way
    """



    import pandas as pd
    # General Aircraft Name
    # Carrier Alliance
    # Aircraft Manufacturer
    # Aircraft Max Take Off Weights (t)
    rename_dict = {
        "GCD (km)":"flight_distance_km",
        "Seats":"total_seat_count",
        "First Seats":"first_seat_count",
        "Business Seats":"business_seat_count",
        "Economy Seats":"economy_seat_count",
        "Carrier Code":"operating_carrier_id",
        "Carrier Name":"operating_carrier_name",
        "Flight No":"flight_number",
        "Equipment Group":"aircraft_class",
        "General Aircraft Code":"aircraft_type",
        "Specific Aircraft Code":"aircraft_code_iata",
        "General Aircraft Name":"aircraft_name",
        "Carrier Alliance":"alliance",
        "Aircraft Manufacturer":"manufacturer",

        }

    df_oag = df_oag.rename(rename_dict, axis=1)
    df_oag["primary_usage"] = np.where(df_oag["total_seat_count"].fillna(0)>0, "Passenger", "Cargo")

    df_oag["operating_carrier_iata"]=df_oag["operating_carrier_id"].copy()
    df_oag["flight_id"]=None
    df_oag["baggage_claim"]=None
    df_oag["tail_number"]=None
    df_oag["aircraft_serial_number"]=None
    df_oag["gate"]=None
    df_oag["actual_taxi_time"]=None
    df_oag["actual_block_time"]=None
    df_oag["gate_delay"]=None
    df_oag["actual_flight_duration"]=None
    df_oag["standing_time"]=None
    df_oag["is_turnaround"]=None
    df_oag["aircraft_market_group"]=None
    df_oag["aircraft_family"]=None
    df_oag["aircraft_type_series"]=None
    df_oag["aircraft_code_icao"]=None
    df_oag["actual_gate_local"]=None
    df_oag["actual_runway_local"]=None


    df_oag["aircraft_market_sector"]="Commercial"
    df_oag["origin_airport"]=airport_code
    df_oag["dep/arr_airport"] = np.where(df_oag["Dep Airport Code"]==airport_code, df_oag["Arr Airport Code"], df_oag["Dep Airport Code"])
    df_oag["country_code"] = np.where(df_oag["Dep Airport Code"]==airport_code, df_oag["Arr IATA Country Code"], df_oag["Dep IATA Country Code"])
    df_oag["country_name"] = np.where(df_oag["Dep Airport Code"]==airport_code, df_oag["Arr IATA Country Name"], df_oag["Dep IATA Country Name"])
    df_oag["region_code"] = np.where(df_oag["Dep Airport Code"]==airport_code, df_oag["Arr Region Code"], df_oag["Dep Region Code"])
    df_oag["region_name"] = np.where(df_oag["Dep Airport Code"]==airport_code, df_oag["Arr Region Name"], df_oag["Dep Region Name"])
    df_oag["terminal"] = np.where(df_oag["Dep Airport Code"]==airport_code, df_oag["Dep Terminal"], df_oag["Arr Terminal"])
    df_oag["flight_io"] = np.where(df_oag["Dep Airport Code"]==airport_code, "d","a")
    df_oag["movement"]=1

    df_oag["Local Dep Time"] = df_oag["Local Dep Time"].astype(str).str.zfill(4)
    df_oag["Local Dep Time"] = pd.to_datetime(df_oag["Time series"].astype(str) + ' ' + df_oag["Local Dep Time"].str[:2] + ":" + df_oag["Local Dep Time"].str[2:])
    df_oag["Local Arr Day"] = df_oag["Local Arr Day"].map(lambda x: '0' if x=='P' else x).astype(float)
    df_oag["Local Arr Time"] = df_oag["Local Arr Time"].astype(str).str.zfill(4)
    df_oag["Local Arr Time"] = pd.to_datetime(df_oag["Time series"].astype(str) + ' ' + df_oag["Local Arr Time"].str[:2] + ":" + df_oag["Local Arr Time"].str[2:]) + pd.to_timedelta(df_oag["Local Arr Day"], unit="d")
    df_oag["scheduled_gate_local"] = np.where(df_oag["Dep Airport Code"]==airport_code, df_oag["Local Dep Time"], df_oag["Local Arr Time"])

    # 문자열을 분으로 변환
    df_oag['ground_time'] = df_oag['Ground Time'].str.split(':').apply(lambda x: int(x[0]) * 60 + int(x[1]))
    df_oag['block_time'] = df_oag['Flying Time'].str.split(':').apply(lambda x: int(x[0]) * 60 + int(x[1]))

    df_oag["operating_maximum_takeoff_weight_lb"]=df_oag["Aircraft Max Take Off Weight (t)"].astype(float).fillna(0)*2000
    df_oag["maximum_payload_lb"]=df_oag["FreightTons (Total)"].astype(float).fillna(0)*2000
    df_oag["International/Domestic"] = df_oag["International/Domestic"].str.lower()

    replacement_dict = {
        "J": "Regional Jets",
        "JN": "Narrowbody Jets", 
        "JW": "Widebody Jets",
        "RJ": "Regional Jets",
        "T": "Narrowbody Jets"
    }

    df_oag["aircraft_class"] = df_oag["aircraft_class"].map(replacement_dict)
    df_oag = df_oag[[
        "flight_id",
        "origin_airport",
        "dep/arr_airport",
        "flight_distance_km",
        "total_seat_count",
        
        ########## ADD COl ###########
        "first_seat_count",
        "business_seat_count",
        "economy_seat_count",
        "aircraft_name",
        "alliance",
        "manufacturer",
        ########## ADD COl ###########


        "operating_maximum_takeoff_weight_lb",
        "maximum_payload_lb",

        "flight_io",
        "movement",
        "baggage_claim",
        "scheduled_gate_local",
        "actual_gate_local",
        "actual_runway_local",
        "operating_carrier_id",
        "operating_carrier_name",
        "operating_carrier_iata",
        "flight_number",
        "tail_number",
        "aircraft_serial_number",
        "terminal",
        "gate",

        "actual_taxi_time",
        "actual_block_time",
        "gate_delay",
        "actual_flight_duration",
        "standing_time",

        "ground_time",
        "is_turnaround",


        "aircraft_market_sector",
        "aircraft_class",

        "aircraft_market_group",
        "aircraft_family",
        "aircraft_type",
        "aircraft_type_series",
        "aircraft_code_iata",
        "aircraft_code_icao",
        "primary_usage",

        "country_code",
        "country_name",
        "region_code",
        "region_name",
        "International/Domestic"
        ""

    ]]
    st.write(df_oag)
    return df_oag



def process_oad_load_factor(df_schedule_orig, df_load_factor_orig, airport_code, start_year, end_year):
    """
    Segment and Load Factor Report 
    Timeseries : check
    Period : Month
    Selected Flight Type : Operating Flights
    Show by : Operating Carrier

    Airport 1 : ex) ICN
    Diriection : Two Way

    """

    # filter year
    load_factor_df=[]
    for year in range(start_year, end_year+1):
        st.subheader(year, " START!!!")
        df_schedule=df_schedule_orig[df_schedule_orig["scheduled_gate_local"].dt.year==year]
        load_factor_year_df=df_schedule.groupby(["flight_io","dep/arr_airport"])["total_seat_count"].agg("sum").reset_index()
        load_factor_year_df["index"]=load_factor_year_df["flight_io"]+"_"+load_factor_year_df["dep/arr_airport"]

        df_load_factor=df_load_factor_orig[(df_load_factor_orig["Op. Al. (Dominant)"].notna())&(df_load_factor_orig["Timeseries.7"].astype(str).str[:4].astype(float)==year)]





        # #####################################################################3
        df_tr=df_load_factor[df_load_factor["Gateway 1"]==airport_code]
        st.write(int(df_tr["Estimated Pax"].sum()), f"원본TR")
        df_tr["Estimated Pax"]=df_tr["Estimated Pax"]/2 *1.0 # Calibration

        df_tr["Destination"]=np.where(df_tr["Gateway 2"].notna(), df_tr["Gateway 2"], df_tr["Destination"])
        df_tr_arr=df_tr[["Op. Al. (Dominant)","Origin","Estimated Pax"]].rename({"Op. Al. (Dominant)":"operating_carrier_iata","Origin":"dep/arr_airport"},axis=1)
        df_tr_dep=df_tr[["Op. Al. (Dominant)","Destination","Estimated Pax"]].rename({"Op. Al. (Dominant)":"operating_carrier_iata","Destination":"dep/arr_airport"},axis=1)

        df_tr_arr["flight_io"]="a"
        df_tr_dep["flight_io"]="d"
        df_tr=pd.concat([df_tr_arr, df_tr_dep])
        df_tr["Origin"]=airport_code
        df_tr=df_tr.groupby(["flight_io","dep/arr_airport"])["Estimated Pax"].agg("sum").reset_index()
        df_tr["index"]=df_tr["flight_io"]+"_"+df_tr["dep/arr_airport"]

        load_factor_year_df = pd.merge(load_factor_year_df, df_tr[["index","Estimated Pax"]], on="index",how="left")
        load_factor_year_df = load_factor_year_df.rename({"Estimated Pax":"tr_pax"},axis=1)
        st.write(int(load_factor_year_df["tr_pax"].sum()),"1차 TR")
        # #####################################################################3
        df_od=df_load_factor[(df_load_factor["Origin"]==airport_code)|(df_load_factor["Destination"]==airport_code)]
        st.write(int(df_od["Estimated Pax"].sum()), "원본OD")
        df_od["Estimated Pax"]=df_od["Estimated Pax"]*1.0 # Calibration

        df_od["Destination"]=np.where((df_od["Gateway 1"].notna() & (df_od["Origin"]==airport_code)),df_od["Gateway 1"],  df_od["Destination"])

        df_od["Origin"]=np.where((df_od["Gateway 2"].notna() & (df_od["Destination"]==airport_code)),df_od["Gateway 2"],  
                                np.where(df_od["Gateway 1"].notna(),
                                            df_od["Gateway 1"],
                                            df_od["Origin"]))



        df_od["flight_io"]=np.where(df_od["Origin"]==airport_code,"d","a")
        df_od["dep/arr_airport"]=np.where(df_od["Origin"]==airport_code,df_od["Destination"], df_od["Origin"])
        df_od=df_od.rename({"Op. Al. (Dominant)":"operating_carrier_iata"},axis=1)
        df_od=df_od[["flight_io","dep/arr_airport","operating_carrier_iata","Estimated Pax"]]
        df_od = df_od.groupby(["flight_io","dep/arr_airport"])["Estimated Pax"].agg("sum").reset_index()
        df_od["index"]=df_od["flight_io"]+"_"+df_od["dep/arr_airport"]



        load_factor_year_df = pd.merge(load_factor_year_df, df_od[["index","Estimated Pax"]], on="index",how="left")
        load_factor_year_df = load_factor_year_df.rename({"Estimated Pax":"od_pax"},axis=1)
        st.write(int(load_factor_year_df["od_pax"].sum()),"1차 OD")
        # #####################################################################3

        load_factor_year_df["total_pax"]=load_factor_year_df["od_pax"]+load_factor_year_df["tr_pax"]
        load_factor_year_df["load_factor"] = load_factor_year_df["total_pax"]/load_factor_year_df["total_seat_count"]

        load_factor_year_df["load_factor"] = load_factor_year_df["load_factor"].clip(lower=0.6, upper=0.95)

        load_factor_year_df["od_lf"]=load_factor_year_df["load_factor"]*(load_factor_year_df["od_pax"])/(load_factor_year_df["od_pax"]+load_factor_year_df["tr_pax"])
        load_factor_year_df["tr_lf"]=load_factor_year_df["load_factor"]*(load_factor_year_df["tr_pax"])/(load_factor_year_df["od_pax"]+load_factor_year_df["tr_pax"])
        load_factor_year_df['od_lf'] = load_factor_year_df['od_lf'].fillna(load_factor_year_df['od_lf'].mean())
        load_factor_year_df['tr_lf'] = load_factor_year_df['tr_lf'].fillna(load_factor_year_df['tr_lf'].mean())
        load_factor_year_df["load_factor"]=load_factor_year_df["od_lf"]+load_factor_year_df["tr_lf"]

        load_factor_year_df["od_pax"]=load_factor_year_df["od_lf"]*load_factor_year_df["total_seat_count"]
        load_factor_year_df["tr_pax"]=load_factor_year_df["tr_lf"]*load_factor_year_df["total_seat_count"]
        load_factor_year_df["total_pax"]=load_factor_year_df["od_pax"]+load_factor_year_df["tr_pax"]
        # load_factor_year_df
        st.write(int(load_factor_year_df["od_pax"].sum()), "3차 OD")
        st.write(int(load_factor_year_df["tr_pax"].sum()),"3차 TR")

        st.write("총여객", int(load_factor_year_df["od_pax"].sum()+load_factor_year_df["tr_pax"].sum()))
        if load_factor_year_df["total_pax"].sum()==0:
            load_factor_year_df["load_factor"]=0.77
            load_factor_year_df["od_lf"]=0.77
            load_factor_year_df["tr_lf"]=0
            load_factor_year_df["Assumed"]="Yes : 0.77"

        load_factor_year_df["year"]=year
        load_factor_df+=[load_factor_year_df]

    load_factor_df=pd.concat(load_factor_df)
    load_factor_df["origin"]=airport_code
    st.dataframe(load_factor_df)
    return load_factor_df[["origin","flight_io","dep/arr_airport","load_factor","od_lf","tr_lf","year"]]