import streamlit as st
from utils.masterplan import *
# 📕📗📘📙📒📓📚📖
# --- CSS ---

st.set_page_config(
    page_title="HOME",
    layout="wide",
    initial_sidebar_state="collapsed"  # 사이드바 접힌 상태 유지
)



# 사용 예시
set_bg_image(image_path="data/image/MP_right.png")



df_airport=pd.read_parquet("data/raw/airport/cirium_airport_ref.parquet")
df_airport["airport_name"] = df_airport["name"] + " (" + df_airport["airport_id"] + ")"
fig = go.Figure()
st.markdown("<h1 style='color : white; '>Enter Your Airport Here!!<h1>", unsafe_allow_html=True)
# st.title("Enter your Airport")

# dst_airport (파란색 점들 추가)
fig.add_trace(go.Scattergeo(
    lon=df_airport["lon"],
    lat=df_airport["lat"],
    mode='markers',
    marker=dict(
        size=1.5,
        color='#00FF00'
    ),
    hovertext=df_airport["airport_name"],  # 여기에 공항 이름 컬럼 사용
    hoverinfo='text'

))

# geo 객체를 사용하여 지구본 설정
fig.update_geos(
    projection_type='orthographic',
    showland=True,
    showcountries=True,
    showocean=True,
    showcoastlines=True,
    bgcolor='rgba(0,0,0,0)', 
    landcolor='rgb(42, 35, 35)',
    oceancolor = "#007aff",


    # landcolor='rgb(20, 20, 20)',
    # oceancolor='rgb(10, 30, 60)',
    # countrycolor='rgba(255,255,255,0.1)',  # 국가 경계선 은은하게
    # showframe=False,


)
fig.update_layout(
height=1700,
width=1700,
paper_bgcolor='rgba(0,0,0,0)',  # 페이퍼 배경을 투명하게
plot_bgcolor='rgba(0,0,0,0)' ,   # 플롯 배경을 투명하게
margin=dict(l=0, r=0, t=0, b=0),  # 모든 마진을 0으로 설정
showlegend=False  # 범례 없애기
)


if st.button("**Explore Your Airport→**", type="primary"):
    st.switch_page("pages/1_✈️_Masterplan.py")

st.markdown(
    '<span style="color:#FFFFFF">🟢 <b>Accessible Airport</b></span>',
    unsafe_allow_html=True
)
st.plotly_chart(fig, config={'scrollZoom': False}, use_container_width=True)

st.title("")
st.title("")
st.image("data/image/who_we_are.svg", use_column_width=True)
st.title("")
st.title("")
st.title("")
st.title("")
st.image("data/image/for_whom_1.svg", use_column_width=True)
st.title("")
st.title("")
st.title("")
st.title("")
st.title("")
st.title("")
st.title("")
st.image("data/image/for_whom_2.svg", use_column_width=True)
st.title("")
st.title("")
st.title("")
st.title("")
st.title("")
st.image("data/image/5_things.svg")
st.title("")
st.title("")
st.title("")
st.image("data/image/data_source.svg")
st.title("")
st.title("")
st.title("")
st.image("data/image/flexa_samsung.svg")
st.title("")
st.title("")
st.title("")
st.title("")
st.image("data/image/price_policy.svg")
st.title("")
st.title("")
st.title("")
st.title("")
st.image("data/image/qr_linkedin.svg")










