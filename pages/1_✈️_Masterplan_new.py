import streamlit as st
from utils.masterplan import *

st.set_page_config(
    page_title="MASTERPLANNER",
    layout="wide",
    initial_sidebar_state="collapsed"  # Keep sidebar collapsed
)

if not st.session_state.get("authenticated", False):
    st.warning("Please log in from the Home page to access this section.")
    st.stop()

ms=MasterplanInput()
ms.apply_css()




select_airport_tab, airport_overview, masterplan_tab = st.tabs(["**✅ SELECT AIRPORT**","**✅ PROFILER**","**☑️ MASTERPLAN**"])
with select_airport_tab:
    ms.select_airport_block()
with airport_overview :
    ms.airport_basic_info()
with masterplan_tab:
    ms.masterplan_func()

