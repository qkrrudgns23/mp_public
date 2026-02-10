import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, time, date
import warnings
from utils.cirium import connect_cirium, managing_cirium_data
warnings.filterwarnings('ignore')

# 전역 랜덤 시드 설정
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

st.set_page_config(
    page_title="MNL Analysis",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.write("data processing")

path = r"C:\Users\qkrru\Desktop\바탕 화면\현행업무\● 진행_사업_수주\마닐라_컨설팅_솔루션_사업\260102 항공사 재배치자료\260102_Flexa_Reassignment_Report\Analysis_excel_data\Flight_Data_Source & Results.xlsx"
df = pd.read_excel(path, sheet_name="Flight Data Source")

st.write(df)



df_airport=pd.read_parquet("data/raw/airport/cirium_airport_ref.parquet")[["airport_id","country_code","country_name","region_name"]]
df_carrier=pd.read_parquet("data/raw/carrier/cirium_carrier_ref.parquet")
df_carrier

df = pd.merge(df, df_airport, left_on="dep/arr_airport", right_on="airport_id", how="left")
df = pd.merge(df, df_carrier, left_on="operating_carrier_iata", right_on="operating_carrier_id", how="left")
df["International/Domestic"] = np.where(df["country_code"] == "PH", "domestic", "international")

# parquet 저장 시 문제 될 수 있는 object 타입 컬럼들을 모두 문자열로 통일
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str)


# actual_block / scheduled_block 후처리
# -1, +1 문자 제거 후 4자리 zfill
df["scheduled_block"] = (
    df["scheduled_block"]
    .astype(str)
    .str.replace(r"[\+\-]1$", "", regex=True)  # 끝에 붙은 -1, +1 제거
    .str.strip()
    .str.zfill(4)
)
df["actual_block"] = (
    df["actual_block"]
    .astype(str)
    .str.replace(r"[\+\-]1$", "", regex=True)  # 끝에 붙은 -1, +1 제거
    .str.strip()
    .str.zfill(4)
)

# HHMM → HH:MM 형태로 변환
sch_block_time = (
    df["scheduled_block"].str.slice(0, 2) + ":" + df["scheduled_block"].str.slice(2, 4)
)
act_block_time = (
    df["actual_block"].str.slice(0, 2) + ":" + df["actual_block"].str.slice(2, 4)
)

# sch_date + scheduled_block → sch_datetime
df["scheduled_gate_local"] = pd.to_datetime(
    df["sch_date"].astype(str) + " " + sch_block_time,
    errors="coerce",
)

# act_data + actual_block → act_datetime
df["actual_gate_local"] = pd.to_datetime(
    df["act_date"].astype(str) + " " + act_block_time,
    errors="coerce",
)

# scheduled_gate_local / actual_gate_local 에서 유효하지 않은 값(NaT) 제거하여
# 이후 날짜 비교 시 datetime.date 와 float 섞여서 생기는 오류 방지
st.write(len(df))
df = df[~df["scheduled_gate_local"].isna()].copy()
st.write(len(df))



df.to_parquet("mnl_1208_1214_gd.parquet")
st.write(df)