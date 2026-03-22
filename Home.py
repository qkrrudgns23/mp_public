import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import json
import os

st.set_page_config(
    page_title="HOME",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Login Credentials (Recommended to change to environment variable)
LOGIN_CREDENTIALS = {
    "admin": "admin123",  # ID: admin, PW: admin123
}

# Full-screen dark theme for Home
st.markdown("""
<style>
    /* sidebar transparent */
    [data-testid="stSidebar"] { background: transparent !important; }
    [data-testid="stSidebar"] > div:first-child { background: transparent !important; }
    
    .stApp { background: #000000 !important; padding-top: 0 !important; }
    [data-testid="stAppViewContainer"] { background: #000000 !important; padding: 0 !important; overflow: visible !important; }
    [data-testid="stHeader"], header[data-testid="stHeader"] { display: none !important; height: 0 !important; min-height: 0 !important; padding: 0 !important; margin: 0 !important; overflow: hidden !important; border: none !important; }
    .block-container { padding: 0 !important; max-width: 100% !important; overflow: visible !important; min-height: 100vh !important; margin: 0 !important; }
    [data-testid="stAppViewContainer"] > section, [data-testid="stAppViewContainer"] main { padding-top: 0 !important; margin-top: 0 !important; }
    div[data-testid="stVerticalBlock"] { padding-top: 0 !important; }
    .block-container iframe, div[data-testid="stEmbeddedFrameBlock"] iframe { position: fixed !important; top: -120px !important; left: 0 !important; width: 100vw !important; height: calc(100vh + 120px) !important; min-height: calc(100vh + 120px) !important; display: block !important; border: none !important; z-index: 0 !important; }
    
    /* Login / Logout - bottom right */
    div[data-testid="column"]:has(form),
    div[data-testid="column"]:has(button) {
        position: fixed !important;
        bottom: 24px !important;
        right: 24px !important;
        top: auto !important;
        z-index: 2147483647 !important;
        min-width: 200px !important;
        background: transparent !important;
        padding: 16px !important;
        border: none !important;
        border-radius: 8px !important;
    }
    div[data-testid="column"]:has(form) form,
    div[data-testid="column"]:has(button) form {
        border: none !important;
    }
    
    /* Title */
    div[data-testid="column"]:has(form) p,
    div[data-testid="column"]:has(button) p {
        color: rgba(255,255,255,0.9) !important;
        font-weight: 300 !important;
        letter-spacing: 0.08em !important;
        font-size: 0.75rem !important;
        margin-bottom: 10px !important;
    }
    
    /* Input fields */
    div[data-testid="column"]:has(form) input {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 6px !important;
        color: rgba(255,255,255,0.95) !important;
        padding: 6px 10px !important;
        font-size: 0.8rem !important;
    }
    div[data-testid="column"]:has(form) input:focus {
        border-color: rgba(255,255,255,0.25) !important;
    }
    
    /* Labels */
    div[data-testid="column"]:has(form) label {
        color: rgba(255,255,255,0.5) !important;
        font-size: 0.75rem !important;
    }
    
    /* Login form: Enter button hidden (Enter Submission possible with key, button required on form) */
    div[data-testid="column"]:has(form) button {
        position: absolute !important;
        left: -9999px !important;
        width: 1px !important;
        height: 1px !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }
    
    /* Logout button */
    div[data-testid="column"]:has(button):not(:has(form)) button {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        color: rgba(255,255,255,0.9) !important;
        border-radius: 6px !important;
        font-size: 0.8rem !important;
    }
    div[data-testid="column"]:has(button):not(:has(form)) button:hover {
        background: rgba(255,255,255,0.12) !important;
        border-color: rgba(255,255,255,0.2) !important;
    }
</style>
""", unsafe_allow_html=True)

# Reset session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# Login check
if not st.session_state.authenticated:
    # Place the globe first (full screen) + Login form overlay
    _, col_right = st.columns([5, 1])
    with col_right:
        with st.form("login_form"):
            login_id = st.text_input("Username", key="login_id", placeholder="ID")
            login_pw = st.text_input("Password", key="login_pw", type="password", placeholder="••••••••")
            submitted = st.form_submit_button("Login")
            if submitted:
                if login_id and login_pw and LOGIN_CREDENTIALS.get(login_id) == login_pw:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
    df_airport = pd.read_parquet("data/raw/airport/cirium_airport_ref.parquet")
    df_airport["airport_name"] = df_airport["name"] + " (" + df_airport["airport_id"] + ")"
    airports_data = [
        {"lat": float(row["lat"]), "lon": float(row["lon"]), "name": str(row["airport_name"])}
        for _, row in df_airport.iterrows()
    ]
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(_script_dir, "data", "home_globe.html")
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    html_content = html_content.replace("__AIRPORTS_JSON__", json.dumps(airports_data, ensure_ascii=False))
    components.html(html_content, height=2000, scrolling=False)
else:
    # Upon successful login
    df_airport = pd.read_parquet("data/raw/airport/cirium_airport_ref.parquet")
    df_airport["airport_name"] = df_airport["name"] + " (" + df_airport["airport_id"] + ")"
    airports_data = [
        {"lat": float(row["lat"]), "lon": float(row["lon"]), "name": str(row["airport_name"])}
        for _, row in df_airport.iterrows()
    ]
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(_script_dir, "data", "home_globe.html")
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    html_content = html_content.replace("__AIRPORTS_JSON__", json.dumps(airports_data, ensure_ascii=False))
    components.html(html_content, height=2000, scrolling=False)

    # logout button (Bottom right, same location as login box)
    _, col_right = st.columns([5, 1])
    with col_right:
        st.markdown("Signed in")
        if st.button("Sign Out"):
            st.session_state.authenticated = False
            st.rerun()
