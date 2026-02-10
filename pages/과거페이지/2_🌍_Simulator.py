import streamlit as st
from utils.universal_simulator_new2 import *
from utils.cirium import *
st.set_page_config(layout="wide")
apply_css()
return_dict = {}


# df1 = pd.read_parquet("cirium/" + f"BAH_schedule_ready_.parquet")
# df2 = pd.read_parquet("cirium/" + f"BAH_schedule_ready.parquet")
# df_count1 = (df1["scheduled_gate_local"].dt.date).value_counts().sort_index()
# df_count1
# df_count1 = df_count1.rename("new")
# df_count2 = (df2["scheduled_gate_local"].dt.date).value_counts().sort_index()
# df_count2 = df_count2.rename("old")

# st.write(pd.concat([df_count1, df_count2], axis=1))



return_dict = select_airport(return_dict)
# _ = run(return_dict)
