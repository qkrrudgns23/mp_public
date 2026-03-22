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

# Global random seed setting
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

st.set_page_config(
    page_title="MNL Analysis",
    layout="wide",
    initial_sidebar_state="collapsed"
)

if not st.session_state.get("authenticated", False):
    st.warning("Please log in from the Home page to access this section.")
    st.stop()

st.write("data processing")

path = r"C:\Users\qkrru\Desktop\desktop\Current work\● progress_business_Orders\Manila_consulting_solution_business\260102 Airline relocation data\260102_Flexa_Reassignment_Report\Analysis_excel_data\Flight_Data_Source & Results.xlsx"
df = pd.read_excel(path, sheet_name="Flight Data Source")

st.write(df)



df_airport=pd.read_parquet("data/raw/airport/cirium_airport_ref.parquet")[["airport_id","country_code","country_name","region_name"]]
df_carrier=pd.read_parquet("data/raw/carrier/cirium_carrier_ref.parquet")
df_carrier

df = pd.merge(df, df_airport, left_on="dep/arr_airport", right_on="airport_id", how="left")
df = pd.merge(df, df_carrier, left_on="operating_carrier_iata", right_on="operating_carrier_id", how="left")
df["International/Domestic"] = np.where(df["country_code"] == "PH", "domestic", "international")

# parquet There may be problems when saving object Unify all type columns as strings
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str)


# actual_block / scheduled_block Post-processing
# -1, +1 4 digits after character removal zfill
df["scheduled_block"] = (
    df["scheduled_block"]
    .astype(str)
    .str.replace(r"[\+\-]1$", "", regex=True)  # attached to the end -1, +1 eliminate
    .str.strip()
    .str.zfill(4)
)
df["actual_block"] = (
    df["actual_block"]
    .astype(str)
    .str.replace(r"[\+\-]1$", "", regex=True)  # attached to the end -1, +1 eliminate
    .str.strip()
    .str.zfill(4)
)

# HHMM → HH:MM convert to form
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

# scheduled_gate_local / actual_gate_local Invalid value in(NaT) by removing
# When comparing later dates datetime.date and float Avoid mixing errors
st.write(len(df))
df = df[~df["scheduled_gate_local"].isna()].copy()
st.write(len(df))



df.to_parquet("mnl_1208_1214_gd.parquet")
st.write(df)