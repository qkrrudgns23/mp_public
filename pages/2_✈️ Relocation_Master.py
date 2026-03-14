import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations, product
import hashlib
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, time, date
import warnings
warnings.filterwarnings('ignore')
import os
import sys
import zipfile
import io
# Streamlit 페이지에서도 프로젝트 루트를 path에 넣어 utils 등 import 가능하게 함
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)  # pages/ 의 부모 = 프로젝트 루트
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from utils.masterplan import MasterplanInput
try:
    from utils.result_storage.storage import save_result, load_result, list_saved_results, delete_result
except ImportError:
    save_result = load_result = list_saved_results = delete_result = None
_build_scenario_report_html = None
try:
    from utils.generate_scenario_report_rev2 import build_report_html as _build_scenario_report_html
except Exception as _e:
    # fallback: 파일 경로로 직접 로드 (Streamlit 등에서 sys.path 불일치 시)
    try:
        import importlib.util
        _g = os.path.join(_PROJECT_ROOT, "utils", "generate_scenario_report_rev2.py")
        if os.path.isfile(_g):
            _spec = importlib.util.spec_from_file_location("generate_scenario_report_rev2", _g)
            _mod = importlib.util.module_from_spec(_spec)
            if _PROJECT_ROOT not in sys.path:
                sys.path.insert(0, _PROJECT_ROOT)
            _spec.loader.exec_module(_mod)
            _build_scenario_report_html = getattr(_mod, "build_report_html", None)
        else:
            pass
    except Exception as _e_fb:
        pass
    if _build_scenario_report_html is None:
        import logging
        logging.getLogger(__name__).debug("generate_scenario_report_rev2 import failed: %s", _e)

def _percentile_rank_value(sorted_desc, p):
    """Percentile을 보간하지 않고, 해당 순위의 실제 값을 반환.
    sorted_desc: 내림차순 정렬된 1D 배열 (최대값이 인덱스 0).
    p: 0~100 퍼센타일. 반환: (0-based 인덱스, 해당 위치의 값)."""
    n = len(sorted_desc)
    if n == 0:
        return 0, np.nan
    rank_from_bottom = int(np.ceil(p / 100.0 * n))  # 1-based, 작은 쪽부터
    idx = n - rank_from_bottom
    idx = max(0, min(idx, n - 1))
    return idx, float(sorted_desc[idx])


st.set_page_config(
    page_title="Relocation Master",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# if not st.session_state.get("authenticated", False):
#     st.warning("Please log in from the Home page to access this section.")
#     st.stop()


# apply_css 함수 정의 (의존성 제거) — 라이트/다크 모드 대응
def apply_css():
    css = """
    <style>
    /* 배경 — config.toml 테마 사용, 강제 검정 제거 */
    [data-testid="stAppViewContainer"] { padding: 0 !important; overflow: visible !important; }
    .block-container { background: transparent !important; padding: 1.5rem 2.5rem 1.5rem 2.5rem !important; max-width: 100% !important; overflow: visible !important; margin: 0 auto !important; position: relative !important; z-index: 1 !important; }
    div[data-testid="stVerticalBlock"] { background: transparent !important; }
    [data-testid="stExpander"], [data-testid="stExpander"] > div { background: transparent !important; }
    [data-testid="stTab"] { background: transparent !important; }
    
    /* 사이드바·우측 영역 — config.toml 테마 사용 */
    
    /* 글씨체: config.toml의 pretendard와 동일하게 유지 (홈페이지와 통일) */
    h3 { padding-top: 30px; }
    body, p, h1, h2, h3, h4, h5, h6, textarea {
        font-size: 20px;
    }

    /* Card container — 투명하게 (배경 통일) */
    .card {
        border: 1px solid rgba(148, 163, 184, 0.12);
        background: rgba(15, 18, 25, 0.4);
        border-radius: 10px;
        padding: 20px 22px;
        margin: 14px 0 18px 0;
        box-shadow: none;
    }
    .card h3, .card h4, .card h5, .card h6 {
        margin: 0 0 14px 0;
        font-size: 0.95rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        color: #e2e8f0 !important;
        text-transform: none;
    }
    .section-title {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 16px;
        padding-bottom: 10px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.12);
        font-size: 0.9rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        color: #cbd5e1 !important;
    }
    .section-title::before {
        content: "";
        width: 3px;
        height: 14px;
        background: #4da6ff;
        border-radius: 2px;
    }

    /* Light fallback */
    @media (prefers-color-scheme: light) {
        .stApp:not([data-theme="dark"]) .card {
            background: rgba(255,255,255,0.9) !important;
            border-color: rgba(15, 23, 42, 0.08) !important;
        }
        .stApp:not([data-theme="dark"]) .section-title { color: #334155 !important; }
        .stApp:not([data-theme="dark"]) .section-title::before { background: #007aff; }
    }
    /* Dark mode: system preference */
    @media (prefers-color-scheme: dark) {
        .card {
            background: rgba(15, 18, 25, 0.4) !important;
            border-color: rgba(148, 163, 184, 0.12) !important;
            color: #e2e8f0 !important;
        }
        .card h3, .card h4, .card h5, .card h6 { color: #e2e8f0 !important; }
        .card p, .card div { color: #cbd5e1 !important; }
    }
    /* Dark mode: Streamlit theme */
    [data-theme="dark"] .card,
    .stApp[data-theme="dark"] .card {
        background: rgba(15, 18, 25, 0.4) !important;
        border-color: rgba(148, 163, 184, 0.12) !important;
        color: #e2e8f0 !important;
    }
    [data-theme="dark"] .card h3, [data-theme="dark"] .card h4,
    .stApp[data-theme="dark"] .card h3, .stApp[data-theme="dark"] .card h4 { color: #e2e8f0 !important; }
    /* Scenario comparison card view (dark) — div/h4만 덮어쓰고 span(차이/개선 색) 유지 */
    @media (prefers-color-scheme: dark) {
        .relocation-card { background: rgba(15, 18, 25, 0.5) !important; border-color: rgba(80, 90, 100, 0.4) !important; box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important; }
        .relocation-card h4, .relocation-card > div { color: #e2e8f0 !important; }
        .relocation-card div div { color: #cbd5e1 !important; }
        .relocation-card .card-muted { color: #94a3b8 !important; }
        .relocation-card [style*="border"] { border-color: #475569 !important; }
    }
    [data-theme="dark"] .relocation-card,
    .stApp[data-theme="dark"] .relocation-card {
        background: rgba(15, 18, 25, 0.5) !important; border-color: rgba(80, 90, 100, 0.4) !important;
    }
    [data-theme="dark"] .relocation-card h4, [data-theme="dark"] .relocation-card > div,
    .stApp[data-theme="dark"] .relocation-card h4, .stApp[data-theme="dark"] .relocation-card > div { color: #e2e8f0 !important; }
    /* Streamlit dark theme (배경이 어두우면 body에 클래스 추가됨) */
    body.streamlit-dark .card {
        background: rgba(15, 18, 25, 0.4) !important; border-color: rgba(148, 163, 184, 0.12) !important; color: #e2e8f0 !important;
    }
    body.streamlit-dark .card h3, body.streamlit-dark .card h4 { color: #f1f5f9 !important; }
    body.streamlit-dark .relocation-card {
        background: rgba(15, 18, 25, 0.5) !important; border-color: rgba(80, 90, 100, 0.4) !important;
    }
    body.streamlit-dark .relocation-card h4, body.streamlit-dark .relocation-card > div { color: #e2e8f0 !important; }
    
    /* number_input 공통 — Time Unit, Num of locations, Num of Scenarios 등 통일 */
    [data-testid="stNumberInput"],
    div[data-testid="stNumberInput"] > div,
    div.row-widget.stNumberInput,
    div.row-widget.stNumberInput > div {
        background: transparent !important;
        background-color: transparent !important;
    }
    [data-testid="stNumberInput"] input,
    [data-testid="stNumberInput"] button,
    div.row-widget.stNumberInput input,
    div.row-widget.stNumberInput button {
        background: transparent !important;
        background-color: transparent !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        color: rgba(255,255,255,0.95) !important;
        font-size: 0.95rem !important;
        padding: 6px 10px !important;
    }
    /* +/- 버튼 아이콘 보이게 */
    [data-testid="stNumberInput"] button,
    div.row-widget.stNumberInput button {
        opacity: 1 !important;
    }
    [data-testid="stNumberInput"] button svg,
    [data-testid="stNumberInput"] button path,
    [data-testid="stNumberInput"] button span,
    div.row-widget.stNumberInput button svg,
    div.row-widget.stNumberInput button path,
    div.row-widget.stNumberInput button span {
        fill: rgba(255,255,255,0.9) !important;
        stroke: rgba(255,255,255,0.9) !important;
        color: rgba(255,255,255,0.9) !important;
    }
    [data-testid="stNumberInput"] label,
    [data-testid="stNumberInput"] p,
    div.row-widget.stNumberInput label,
    div.row-widget.stNumberInput p {
        color: rgba(203,213,225,0.9) !important;
        font-size: 0.9rem !important;
    }
    </style>
    <script>
    (function() {
        function isDarkBg(el) {
            if (!el) return false;
            var s = window.getComputedStyle(el);
            var bg = s.backgroundColor || s.background;
            var m = bg.match(/rgba?\\(\\s*(\\d+)\\s*,\\s*(\\d+)\\s*,\\s*(\\d+)/);
            if (m) { var r=+m[1],g=+m[2],b=+m[3]; return (r+g+b) < 400; }
            if (bg.indexOf('rgb(0') === 0 || bg === 'transparent') {
                var p = el.parentElement; return p ? isDarkBg(p) : false;
            }
            return false;
        }
        function check() {
            var root = document.body || document.documentElement;
            var main = document.querySelector('[data-testid="stAppViewContainer"]') || document.querySelector('.main') || document.body;
            if (isDarkBg(main)) root.classList.add('streamlit-dark');
            else root.classList.remove('streamlit-dark');
        }
        if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', check);
        else check();
        setTimeout(check, 500);
    })();
    </script>
    """
    return st.markdown(css, unsafe_allow_html=True)


def apply_button_css():
    """Relocation Master 버튼(Load, Delete, Run, Save 등) 공통 스타일 - 작은 크기"""
    btn_css = """
    <style>
    div.stButton > button,
    div.stButton > button *,
    div[data-testid="stButton"] > button,
    div[data-testid="stButton"] > button *,
    div.row-widget.stButton > button,
    div.row-widget.stButton > button *,
    [data-testid="stHorizontalBlock"] > div:nth-child(2) button,
    [data-testid="stHorizontalBlock"] > div:nth-child(2) button *,
    [data-testid="stHorizontalBlock"] > div:nth-child(3) button,
    [data-testid="stHorizontalBlock"] > div:nth-child(3) button * {
        font-size: 15px !important;
        padding: 2px 5px !important;
        height: 24px !important;
        min-height: 24px !important;
        min-width: auto !important;
        line-height: 1.2 !important;
    }
    div.stButton,
    div[data-testid="stButton"] {
        font-size: 15px !important;
    }
    /* Streamlit button (kind 속성) + 다운로드 버튼 */
    button[kind="primary"],
    button[kind="secondary"],
    button[kind="primary"] *,
    button[kind="secondary"] * {
        font-size: 15px !important;
    }
    </style>
    """
    return st.markdown(btn_css, unsafe_allow_html=True)


# 전역 랜덤 시드 설정
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# 전역 category_dict 정의 (원본 컬럼명 기반, mean, sigma, min_max_clip 통합)
CATEGORY_DICT = {
    "departure movement": {
        "agg_col": "movement",
        "flight_io": "d",
        "mean": 0,
        "sigma": 0,
        "min_max_clip": [-1, 1],
    },
    "arrival movement": {
        "agg_col": "movement",
        "flight_io": "a",
        "mean": 0,
        "sigma": 0,
        "min_max_clip": [-1, 1],
    },
    "departure total_seat_count": {
        "agg_col": "total_seat_count",
        "flight_io": "d",
        "mean": 0,
        "sigma": 0,
        "min_max_clip": [-1, 1],
    },
    "arrival total_seat_count": {
        "agg_col": "total_seat_count",
        "flight_io": "a",
        "mean": 0,
        "sigma": 0,
        "min_max_clip": [-1, 1],
    },
    "departure total_pax": {
        "agg_col": "total_pax",
        "flight_io": "d",
        "mean": 0,
        "sigma": 0,
        "min_max_clip": [-1, 1],
    },
    "arrival total_pax": {
        "agg_col": "total_pax",
        "flight_io": "a",
        "mean": 20,
        "sigma": 6,
        "min_max_clip": [3, 50],
    },
    "departure od_pax": {
        "agg_col": "od_pax",
        "flight_io": "d",
        "mean": -120,
        "sigma": 26,
        "min_max_clip": [-360, -37],
    },
    "arrival od_pax": {
        "agg_col": "od_pax",
        "flight_io": "a",
        "mean": 20,
        "sigma": 6,
        "min_max_clip": [3, 50],
    },
    "departure tr_pax": {
        "agg_col": "tr_pax",
        "flight_io": "d",
        "mean": -50,
        "sigma": 15,
        "min_max_clip": [-80, -37],
    },
    "arrival tr_pax": {
        "agg_col": "tr_pax",
        "flight_io": "a",
        "mean": 20,
        "sigma": 6,
        "min_max_clip": [3, 50],
    },
}

# 그룹화할 공통 컬럼 리스트 (공유 변수) - 전역적으로 사용
BASE_COL_OPTIONS = [
    "terminal_carrier",
    "operating_carrier_name",
    "Carrier_Destination",
    "flight_number",
    "terminal_carrier_int/dom",
    "terminal_iata",
    "terminal_iata_int/dom",
    "terminal_iata_int/dom_dest",
    # 추가 기본 컬럼들
    "flight_io",
    "terminal",
    "operating_carrier_iata",
]

# Summary by Group 탭 목록 (label, column). 리포트/Streamlit에서 탭으로 표시
SUMMARY_BY_GROUP_TABS = [
    ("Terminal Carrier", "terminal_carrier"),
    ("Destination", "dep/arr_airport"),
    ("Operating Carrier (IATA)", "operating_carrier_iata"),
    ("A/C Code", "A/C Code"),
    ("International/Domestic", "International/Domestic"),
]



def stable_unit_seed(unit: str) -> int:
    """Stable hash-based seed for a unit string (FNV-1a like)"""
    h = 2166136261
    for ch in unit:
        h ^= ord(ch)
        h *= 16777619
        h &= 0xFFFFFFFF
    return RANDOM_SEED ^ h


def render_relocation_schematic(loc_names: list, fixed_per_loc: dict, moving_units: list):
    import plotly.graph_objects as go

    n = max(2, len(loc_names))
    width = 200 + n * 280
    
    # Calculate dynamic height based on content
    box_height_per_item = 28
    allocate_items = moving_units if moving_units else ["(none)"]
    allocate_box_height = max(80, 40 + len(allocate_items) * box_height_per_item)
    
    # Find max location box height
    max_loc_items = 0
    for name in loc_names:
        items = fixed_per_loc.get(name, [])
        max_loc_items = max(max_loc_items, len(items) if items else 1)
    max_loc_box_height = max(60, 30 + max_loc_items * box_height_per_item)
    
    # Calculate total needed height with padding
    arrow_space = 120
    top_margin = 40
    bottom_margin = 40
    height = top_margin + allocate_box_height + arrow_space + max_loc_box_height + bottom_margin

    fig = go.Figure()

    # Positions
    x_gap = width / (n + 1)
    # allocate box in the middle-top
    allocate_x = width / 2
    allocate_y = height - top_margin - allocate_box_height/2

    # Draw allocate box with rounded corners (using path approximation)
    def add_rounded_rect(x0, y0, x1, y1, radius, color, fill_color, width):
        # Approximate rounded rectangle with path
        r = radius
        path = f"M {x0+r},{y0} L {x1-r},{y0} Q {x1},{y0} {x1},{y0+r} L {x1},{y1-r} Q {x1},{y1} {x1-r},{y1} L {x0+r},{y1} Q {x0},{y1} {x0},{y1-r} L {x0},{y0+r} Q {x0},{y0} {x0+r},{y0} Z"
        fig.add_shape(type="path", path=path, line=dict(color=color, width=width), fillcolor=fill_color)
    
    allocate_box_width = 200
    add_rounded_rect(allocate_x-allocate_box_width/2, allocate_y-allocate_box_height/2, 
                     allocate_x+allocate_box_width/2, allocate_y+allocate_box_height/2,
                     8, "#007aff", "#dbeafe", 3)
    fig.add_annotation(x=allocate_x, y=allocate_y+allocate_box_height/2-15, text="Allocate", showarrow=False, font=dict(size=18, color="#007aff", weight="bold"))
    # Draw each item as a small box with rounded corners
    start_y = allocate_y + allocate_box_height/2 - 45
    for idx, item in enumerate(allocate_items):
        item_y = start_y - idx * box_height_per_item
        item_width = 160
        add_rounded_rect(allocate_x-item_width/2, item_y-12, allocate_x+item_width/2, item_y+12,
                        6, "#007aff", "#ffffff", 1.5)
        fig.add_annotation(x=allocate_x, y=item_y, text=item, showarrow=False, font=dict(size=14, color="#1f2937"), align="center")

    # Draw location boxes and arrows
    for i, name in enumerate(loc_names, start=1):
        x = i * x_gap
        # box
        items = fixed_per_loc.get(name, [])
        items = items if items else ["(none)"]
        loc_box_height = max(60, 30 + len(items) * box_height_per_item)
        y = bottom_margin + loc_box_height/2
        loc_box_width = 180
        add_rounded_rect(x-loc_box_width/2, y-loc_box_height/2, x+loc_box_width/2, y+loc_box_height/2,
                         8, "#6b7280", "#f3f4f6", 2)
        # title
        fig.add_annotation(x=x, y=y+loc_box_height/2-15, text=name, showarrow=False, font=dict(size=18, color="#4b5563", weight="bold"))
        # Draw each item as a small box with rounded corners
        start_y = y + loc_box_height/2 - 35
        for idx, item in enumerate(items):
            item_y = start_y - idx * box_height_per_item
            item_width = 140
            add_rounded_rect(x-item_width/2, item_y-12, x+item_width/2, item_y+12,
                             6, "#6b7280", "#ffffff", 1.5)
            fig.add_annotation(x=x, y=item_y, text=item, showarrow=False, font=dict(size=14, color="#1f2937"), align="center")

        # arrow from allocate to each location
        fig.add_annotation(x=x, y=y+loc_box_height/2, ax=allocate_x, ay=allocate_y-allocate_box_height/2,
                           xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=2.5, arrowcolor="#666")

    fig.update_xaxes(visible=False, range=[0, width])
    fig.update_yaxes(visible=False, range=[0, height])
    # Remove height limit to show all items; 배경 투명도 높임 (다크/라이트 테마 모두 자연스럽게)
    fig.update_layout(
        height=height+60, width=None, margin=dict(l=10, r=10, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig

def create_normal_dist_col_by_airline(
    df,
    ref_col,
    new_col,
    mean,
    sigma,
    min_max_clip,
    relocation_unit="terminal_carrier", 
    unit="m",
    iteration=1,
    datetime=False,
):
    """
    항공사별로 독립적인 정규분포 랜덤값을 생성하여 새 컬럼을 추가
    """
    # Check if the mean is within the clipping range
    assert (
        min_max_clip[0] <= mean <= min_max_clip[1]
    ), "mean 값이 clipping 범위를 넘어서고 있습니다 >> min~max clipping 범위 내로 mean값을 재설정해주세요"
    
    # 결과를 저장할 배열 초기화
    df = df.copy()
    df[new_col] = 0.0
    # 항공사별로 독립적인 랜덤 분포 생성
    for unique in df[relocation_unit].unique():
        unique_mask = df[relocation_unit] == unique
        unique_count = unique_mask.sum()
        
        if unique_count > 0:
            # 항공사 이름을 안정적인 시드로 활용 (런타임/세션 영향 제거)
            rng = np.random.default_rng(np.random.PCG64(stable_unit_seed(str(unique))))
            
            # 해당 항공사 승객들에 대한 랜덤값 생성
            random_arr = rng.normal(mean, sigma, size=unique_count)
            
            # Resample values outside the clipping range up to "iteration" times
            for _ in range(iteration):
                out_of_range_indices = np.where(
                    (random_arr < min_max_clip[0]) | (random_arr > min_max_clip[1])
                )
                if len(out_of_range_indices[0]) > 0:
                    random_arr[out_of_range_indices] = rng.normal(
                        mean, sigma, size=len(out_of_range_indices[0])
                    )
            
            # Add the generated values to the reference column and create the new column
            if datetime == False:
                df.loc[unique_mask, new_col] = df.loc[unique_mask, ref_col] + random_arr
            elif datetime == True:
                # Convert the random values to timedelta if handling datetime
                timedelta_arr = pd.to_timedelta(random_arr, unit=unit)
                timedelta_arr = timedelta_arr.round("S")
                df.loc[unique_mask, new_col] = df.loc[unique_mask, ref_col] + timedelta_arr
    
    return df

def calculate_statistics(df, time_column="SHOW", unit_min=15):
    """show_profile과 동일한 방식으로 통계값 계산"""
    
    # 시간 단위로 그룹화
    df[time_column] = pd.to_datetime(df[time_column])
    df['hour'] = df[time_column].dt.hour
    df['minute'] = df[time_column].dt.minute
    df['n_min'] = round(df['hour'] + (df['minute']//unit_min) * (unit_min/60), 2)
    
    # 시간대별 승객 수 집계
    hourly_counts = df.groupby('n_min').size()
    
    if len(hourly_counts) == 0:
        return {'97.5%': 0, '95%': 0, '90%': 0, 'Mean': 0}
    stats = {
        '97.5%': hourly_counts.quantile(0.975),
        '95%': hourly_counts.quantile(0.95),
        '90%': hourly_counts.quantile(0.90),
        'Mean': hourly_counts.mean()
    }
    return stats

def _compute_avg_seat(filtered_data, dep_arr):
    """aggregate base 상관없이 전체좌석수합/전체항공편합. dep_arr: Dep|Arr|Both."""
    if filtered_data is None or len(filtered_data) == 0:
        return None
    if "total_seat_count" not in filtered_data.columns:
        return None
    if dep_arr == "Dep":
        subset = filtered_data[filtered_data["flight_io"].astype(str).str.lower().str.strip() == "d"] if "flight_io" in filtered_data.columns else filtered_data
    elif dep_arr == "Arr":
        subset = filtered_data[filtered_data["flight_io"].astype(str).str.lower().str.strip() == "a"] if "flight_io" in filtered_data.columns else filtered_data
    else:
        subset = filtered_data
    if len(subset) == 0:
        return None
    total_seats = float(subset["total_seat_count"].sum())
    total_flights = float(subset["movement"].sum()) if "movement" in subset.columns else float(len(subset))
    if total_flights <= 0:
        return None
    return round(total_seats / total_flights, 2)


def make_count_df(df, start_date, end_date, time_col, group, buffer_day=True, freq_min=1):
    """시간 단위(freq_min 분)로 SHOW를 집계하는 함수 (빈 슬롯 포함)."""
    df_copied = df.copy()

    # 전체 타임라인 생성 (freq_min 분 간격)
    if buffer_day:
        time_range = pd.date_range(
            start=start_date - pd.to_timedelta(1, unit="d"),
            end=end_date + pd.to_timedelta(2, unit="d"),
            freq=f"{freq_min}T",
        )
    else:
        time_range = pd.date_range(
            start=start_date,
            end=end_date + pd.Timedelta(days=1),
            freq=f"{freq_min}T",
        )[:-1]

    time_range_df = pd.DataFrame(time_range, columns=["Time"])

    # SHOW(or time_col)을 freq_min 분 단위로 절단
    df_copied[time_col] = df_copied[time_col].dt.floor(f"{freq_min}T")

    # 그룹별 카운트
    count_df = df_copied.groupby([time_col, group]).size().reset_index(name="index")
    count_df.columns = ["Time", group, "index"]

    # 전체 타임라인과 매칭 (없는 슬롯은 0으로 채움)
    count_df = pd.merge(time_range_df, count_df, on="Time", how="left")
    count_df["index"] = count_df["index"].fillna(0)
    count_df[group] = count_df[group].fillna("")

    # 그룹별 순위 계산
    ranking_df = count_df.groupby(group)["index"].sum().sort_values(ascending=False)
    ranking_order = ranking_df.index.tolist()

    count_df_complete = count_df
    return count_df_complete, ranking_order


def compute_capacity_constraint_delay_hours(
    values,
    capacity,
    slot_duration_minutes=60,
    return_overflow_per_slot=False,
):
    """
    용량 제약으로 인한 총 지연시간(시간)을 계산합니다.
    캐스케이딩 오버플로우 로직: 이전 시간대의 미처리 분은 다음 시간대로 넘어가고,
    각 오버플로우된 단위는 1 슬롯(예: 1시간)만큼 지연된 것으로 계산합니다.

    예시: 05시(12편), 06시(13편), 07시(4편), 용량 10
    - 05시: 12편 수신, 10편 처리, 2편 overflow → 2 delay-hours
    - 06시: 13+2=15편 수신, 10편 처리, 5편 overflow → 5 delay-hours
    - 07시: 4+5=9편 수신, 9편 처리, 0 overflow
    - 총 지연시간: 2+5 = 7 delay-hours

    Parameters
    ----------
    values : array-like
        시간 순서대로 정렬된 수요값 (편수 또는 승객수 등).
    capacity : numeric
        시간당 용량.
    slot_duration_minutes : int, default 60
        각 슬롯의 길이(분). 지연시간을 시간 단위로 변환할 때 사용.
    return_overflow_per_slot : bool, default False
        True이면 각 슬롯별 overflow 리스트도 반환.

    Returns
    -------
    total_delay_hours : float
        총 지연시간(시간 단위).
    overflow_per_slot : list, optional
        return_overflow_per_slot=True일 때만 반환. 슬롯별 overflow 값 리스트.
    """
    values = np.asarray(values, dtype=float)
    if capacity <= 0 or len(values) == 0:
        total_hours = 0.0
        overflow_list = [0.0] * len(values) if return_overflow_per_slot else None
        return (total_hours, overflow_list) if return_overflow_per_slot else total_hours

    hours_per_slot = slot_duration_minutes / 60.0
    overflow = 0.0
    total_delay_hours = 0.0
    overflow_per_slot = [] if return_overflow_per_slot else None

    for v in values:
        total_demand = overflow + float(v)
        processed = min(total_demand, capacity)
        overflow = max(0.0, total_demand - capacity)
        total_delay_hours += overflow * hours_per_slot
        if return_overflow_per_slot:
            overflow_per_slot.append(overflow)

    if return_overflow_per_slot:
        return total_delay_hours, overflow_per_slot
    return total_delay_hours


def _delay_stats_from_overflow(overflow_per_slot, capacity, slot_duration_minutes):
    """overflow_per_slot로부터 max_delay_hours, p95_delay_hours 계산. FIFO 가정."""
    if not overflow_per_slot or capacity <= 0:
        return 0.0, 0.0
    hours_per_slot = slot_duration_minutes / 60.0
    delay_counts = {}  # delay_slots -> count
    max_slots = 0
    for ovf in overflow_per_slot:
        if ovf <= 0:
            continue
        n_batches = int(np.ceil(ovf / capacity))
        max_slots = max(max_slots, n_batches)
        for k in range(1, n_batches + 1):
            batch_size = min(capacity, ovf - (k - 1) * capacity)
            if batch_size > 0:
                delay_counts[k] = delay_counts.get(k, 0) + int(batch_size)
    if not delay_counts:
        return 0.0, 0.0
    total_units = sum(delay_counts.values())
    cum = 0
    p95_slots = max_slots
    for k in sorted(delay_counts.keys()):
        cum += delay_counts[k]
        if cum >= total_units * 0.95:
            p95_slots = k
            break
    max_delay_hours = max_slots * hours_per_slot
    p95_delay_hours = p95_slots * hours_per_slot
    return max_delay_hours, p95_delay_hours


def _fmt_delay_hours_min(delay_hours):
    """예: 2.3 -> '2.3' (숫자만)"""
    if delay_hours is None or (isinstance(delay_hours, float) and np.isnan(delay_hours)):
        return "0.0"
    h = float(delay_hours)
    return f"{h:.1f}"


def show_bar(df, ranking_order, group, capa_df=None, max_y=None):
    """show_profile과 동일한 바차트 생성 함수"""
    
    # 상위 그룹만 선택 (최대 10개)
    top_groups = ranking_order[:10]
    df_filtered = df[df[group].isin(top_groups)]
    
    # 피벗 테이블 생성
    pivot_df = df_filtered.pivot(index='Time', columns=group, values='index').fillna(0)
    
    # 바차트 생성
    fig = px.bar(
        pivot_df,
        x=pivot_df.index,
        y=pivot_df.columns,
        title=f"Time Series by {group}",
        barmode='stack'
    )
    
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Count",
        height=750
    )
    
    if max_y:
        fig.update_layout(yaxis_range=[0, max_y])
    
    st.plotly_chart(fig, use_container_width=True)
    
    return pivot_df

def scenario_fingerprint(mapping: dict, index: int) -> str:
    """Create scenario id from index only (no hash)."""
    return f"Scenario_{index:03d}"


class RelocationSetting:
    """Relocation 설정을 관리하는 클래스"""
    
    def __init__(self, df_orig):
        """초기화 - 원본 데이터프레임 저장"""
        self.df_orig = df_orig
        self.df_filtered = None
        self.start_date = None
        self.end_date = None
        self.unit_min = 60
        self.selected_agg_col = None
        self.selected_agg_cols = []
        self.relocation_unit = None
        self.selected_metrics = set()
        self.loc_count = 4
        self.max_sample_count = 50
        
        # 결과 변수들
        self.loc_names = []
        self.fixed_per_loc = {}
        self.moving_unit = []
        self.allowed_targets_map = {}
    
    def render_relocation_settings_tab(self):
        """Relocation Settings 탭 렌더링"""
        filter_col, puff_group_col = st.columns([0.35,0.65])
        
        with filter_col:
            self._render_date_method()

        with puff_group_col:
            self.relocation_unit = st.selectbox(
                "**Relocation Unit**",
                options=BASE_COL_OPTIONS + ["Custom Group"],
                key="relocation_unit_mnl",
                help="Relocation 분석을 위한 단위를 설정합니다."
            )
            if self.relocation_unit == "Custom Group":
                self._render_custom_group()

    def _render_date_method(self):
        """Date & Method 섹션 렌더링"""

        
        # Select Date Range
        min_date = self.df_orig["scheduled_gate_local"].dt.date.min() + pd.Timedelta(days=1)
        max_date = self.df_orig["scheduled_gate_local"].dt.date.max()
        date_range = st.date_input(
            "**Filter Data**",
            value=(min_date, min_date + pd.Timedelta(days=6)),
            min_value=min_date,
            max_value=max_date,
            key="date_range_mnl"
        )
        if isinstance(date_range, tuple):
            self.start_date, self.end_date = [pd.to_datetime(d) for d in date_range]
        else:
            self.start_date = pd.to_datetime(date_range)
            self.end_date = self.start_date
        
        # 기본 날짜 필터
        df_filtered = self.df_orig[
            (self.df_orig["scheduled_gate_local"].dt.date >= self.start_date.date()) & 
            (self.df_orig["scheduled_gate_local"].dt.date <= self.end_date.date())
        ]

        # 추가 필터 UI (Select Date Range 바로 아래)
        with st.expander("🔍 Filters", expanded=False):
            filter_cols = st.multiselect(
                "Filter columns",
                options=list(df_filtered.columns),
                default=[c for c in ["terminal", "operating_carrier_iata", "International/Domestic"] if c in df_filtered.columns],
                help="Choose columns to filter by (only selected values will be kept).",
                key="date_method_filter_cols",
            )

            for col in filter_cols:
                uniques = sorted(df_filtered[col].dropna().unique().tolist())
                selected = st.multiselect(
                    f"Values for `{col}`",
                    options=uniques,
                    default=uniques,
                    key=f"date_method_filter_{col}",
                )
                if len(selected) > 0:
                    df_filtered = df_filtered[df_filtered[col].isin(selected)]

            self.df_filtered = df_filtered

        # Edit puff groups (below Additional Filters, not nested)
        with st.expander("➕ Add Flights", expanded=False):
            df_puffed, group_col = _render_puff_editor(self)
            time_candidates = [c for c in ["SHOW", "scheduled_gate_local"] if c in df_puffed.columns]
            if group_col is not None and len(time_candidates) > 0:
                time_col = time_candidates[0]
                df_puffed[time_col] = pd.to_datetime(df_puffed[time_col])
                df_puffed["Time_Hour"] = df_puffed[time_col].dt.floor("H")
                agg_df = (
                    df_puffed.groupby(["Time_Hour", group_col])
                    .size()
                    .reset_index(name="count")
                    .sort_values("Time_Hour")
                )
                st.markdown("#### Puff result – Stacked Bar (1-hour bins)")
                fig = px.bar(agg_df, x="Time_Hour", y="count", color=group_col, barmode="stack")
                fig.update_layout(xaxis_title="Time (hourly)", yaxis_title="Count", hovermode="x unified", height=600)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Time column (`SHOW` or `scheduled_gate_local`) not found – stacked bar is skipped.")
            self.df_filtered = df_puffed
        st.caption(f"**→Original / Filtered:** {self.df_orig.shape} / {self.df_filtered.shape}")

        # Setting Unit minutes
        c1, c2 =st.columns([0.4,0.6])
        with c1:
            self.unit_min = st.number_input(
                "**Time Unit (minutes)**", 
                value=60, 
                min_value=5, 
                max_value=60, 
                step=5,
                key="unit_min_mnl"
            )
        
        with c2:
            # Select Aggregate base (원본 컬럼명 직접 사용)
            agg_options = []
            if "total_seat_count" in self.df_filtered.columns:
                agg_options.append("total_seat_count")
            if "movement" in self.df_filtered.columns:
                agg_options.append("movement")
            if "total_pax" in self.df_filtered.columns:
                agg_options.append("total_pax")
            if "od_pax" in self.df_filtered.columns:
                agg_options.append("od_pax")
            if "tr_pax" in self.df_filtered.columns:
                agg_options.append("tr_pax")
            
            # 원본 컬럼명을 직접 사용 (변환 없음) — 멀티 선택
            self.selected_agg_cols = st.multiselect(
                "**Aggregate base**",
                options=agg_options,
                default=agg_options[:2] if agg_options else [],
                key="agg_base_multiselect",
                help="여러 개 선택 시 Run analysis에서 모든 선택 항목에 대해 분석합니다."
            )
            if not self.selected_agg_cols and agg_options:
                self.selected_agg_cols = [agg_options[0]]
            self.selected_agg_col = self.selected_agg_cols[0] if self.selected_agg_cols else None


        with c1:
            self.max_sample_count = st.number_input(
                "**Num of Scenarios**",
                value=20,
                min_value=1,
                max_value=50000,
                step=50,
                key="max_sample_count_mnl",
                help="전체 조합에서 최대 몇 개를 샘플링해서 분석할지 설정합니다."
            )
        # with c2:
            self.loc_count = st.number_input("**Num of locations**", min_value=1, max_value=8, value=2, step=1, key="loc_count",
            help="몇개의 Location 으로 설정할지")
        self.selected_metrics = {"97.5%", "Total"}
    
    def _render_custom_group(self):
        # Create Custom Group column with default value "Others"
        if "Custom Group" not in self.df_filtered.columns:
            self.df_filtered["Custom Group"] = "Others"
        if "Custom Group" not in self.df_orig.columns:
            self.df_orig["Custom Group"] = "Others"

        with st.expander("Edit Custom Group", expanded=False):
            # Base column selection (전역 BASE_COL_OPTIONS 사용)
            base_options = [c for c in BASE_COL_OPTIONS if self.df_filtered is not None and c in self.df_filtered.columns] or BASE_COL_OPTIONS
            base_col = st.selectbox(
                "**Base Column**",
                options=base_options,
                key="custom_group_base_col",
                help="Select the base column to create groups from"
            )
            
            # Group list input
            group_list_str = st.text_input(
                "**Group List**",
                value="LCC, Skyteam, FSC, Others",
                key="custom_group_list",
                placeholder="e.g., LCC, FSC, Others",
                help="Enter group names separated by commas"
            )
            
            # Parse group list
            group_list = [g.strip() for g in group_list_str.split(",") if g.strip()] if group_list_str else []
            
            # Get unique values from base column
            available_items = sorted(self.df_filtered[base_col].unique()) if base_col in self.df_filtered.columns else []
            
            if len(available_items) > 0 and len(group_list) > 0:
                # Create matrix DataFrame based on current Custom Group values
                matrix_data = {"Item": available_items}
                for group in group_list:
                    # Check if item is assigned to this group in current Custom Group column
                    group_values = []
                    for item in available_items:
                        item_df = self.df_filtered[self.df_filtered[base_col] == item]
                        if len(item_df) > 0:
                            current_group = item_df["Custom Group"].iloc[0]
                            group_values.append(current_group == group)
                        else:
                            group_values.append(False)
                    matrix_data[group] = group_values
                
                matrix_df = pd.DataFrame(matrix_data)
                
                edited_df = st.data_editor(
                    matrix_df,
                    column_config={
                        "Item": st.column_config.TextColumn(
                            "Item",
                            disabled=True,
                            width="medium"
                        ),
                        **{group: st.column_config.CheckboxColumn(
                            group,
                            default=False,
                            width="small"
                        ) for group in group_list}
                    },
                    use_container_width=True,
                    hide_index=True
                )
                st.caption("→ Unselected values default to 'Others'")
                
                import numpy as np
                # Create probability table: item -> list of checked groups
                probability_mappings = {}
                
                # edited_df의 체크박스 값이 제대로 읽히도록 fillna(False) 적용
                edited_df_filled = edited_df.copy()
                for group in group_list:
                    if group in edited_df_filled.columns:
                        edited_df_filled[group] = edited_df_filled[group].fillna(False).replace('', False).astype(bool)
                
                # iloc를 사용하여 인덱스 기반으로 안정적으로 처리
                for idx in range(len(edited_df_filled)):
                    item = edited_df_filled.iloc[idx]['Item']
                    checked_groups = []
                    for group in group_list:
                        if group in edited_df_filled.columns:
                            checkbox_value = edited_df_filled.iloc[idx][group]
                            if pd.notna(checkbox_value) and bool(checkbox_value):
                                checked_groups.append(group)
                    
                    if len(checked_groups) > 0:
                        probability_mappings[item] = checked_groups
                    else:
                        probability_mappings[item] = ["Others"]
                
                # Apply probability-based assignment to df_filtered
                if len(probability_mappings) > 0:
                    np.random.seed(RANDOM_SEED)
                    
                    def assign_group_by_probability(value):
                        if value in probability_mappings:
                            checked_groups = probability_mappings[value]
                            return np.random.choice(checked_groups)
                        return "Others"
                    self.df_filtered["Custom Group"] = self.df_filtered[base_col].apply(assign_group_by_probability)
                    
                    # Show summary
                    summary = {}
                    for item, groups in probability_mappings.items():
                        if groups != ["Others"]:
                            prob = 100.0 / len(groups)
                            summary[item] = f"{', '.join(groups)} ({prob:.1f}% each)"
                    
                    if summary:
                        st.success(f"✅ Custom Group updated with probability-based assignment!")
                    else:
                        st.success(f"✅ Custom Group updated!")
                else:
                    st.warning("Please assign at least one item to a group")
            elif len(available_items) == 0:
                st.info("No data available for the selected base column")
            elif len(group_list) == 0:
                st.info("Please enter group names in Group List")
    

    def render_assign_units_tab(self):
        """Assign Units 탭 렌더링"""
        all_unit = sorted(self.df_filtered[self.relocation_unit].unique())
        
        # 1) Names in columns
        name_cols = st.columns(self.loc_count)
        default_names = [f"T{i+1}" for i in range(self.loc_count)]
        input_names = []
        for i, c in enumerate(name_cols):
            with c:
                nm = st.text_input(f"Name #{i+1}", value=default_names[i], key=f"loc_name_{i}")
                nm = (nm or default_names[i]).strip()
                if nm == "":
                    nm = default_names[i]
                input_names.append(nm)
        
        # ensure uniqueness
        seen = set()
        self.loc_names = []
        for nm in input_names:
            base = nm
            uniq = nm
            suffix = 1
            while uniq in seen:
                suffix += 1
                uniq = f"{base}_{suffix}"
            seen.add(uniq)
            self.loc_names.append(uniq)
        
        # 2) Fixed selection in columns
        fixed_cols = st.columns(self.loc_count)
        self.fixed_per_loc = {}
        used_fixed = set()
        for i, c in enumerate(fixed_cols):
            name = self.loc_names[i]
            available = [u for u in all_unit if u not in used_fixed]
            with c:
                fixed = st.multiselect(f"{name} fixed units", options=available, default=[], key=f"fixed_{name}")
            self.fixed_per_loc[name] = sorted(fixed)
            used_fixed.update(fixed)
        
        remaining = [u for u in all_unit if u not in used_fixed]
        self.moving_unit = st.multiselect("Allocate Units", options=remaining, default=remaining[:min(4, len(remaining))], key="moving_unit")
        
        # Per-unit allowed targets (checkbox matrix)
        self.allowed_targets_map = {}
        self.unassigned_units_map = {}  # 미배치 허용 유닛 맵
        if len(self.moving_unit) > 0 and len(self.loc_names) > 0:
            with st.expander("Per-unit allowed locations"):
                # Build checkbox matrix dataframe
                matrix_data = {"Unit": sorted(self.moving_unit)}
                for loc in self.loc_names:
                    matrix_data[loc] = [True for _ in range(len(matrix_data["Unit"]))]
                matrix_data["미배치"] = [False for _ in range(len(matrix_data["Unit"]))]  # 미배치 컬럼 추가
                matrix_df = pd.DataFrame(matrix_data)
                edited_df = st.data_editor(
                    matrix_df,
                    column_config={
                        "Unit": st.column_config.TextColumn("Unit", disabled=True, width="large"),
                        **{loc: st.column_config.CheckboxColumn(loc, default=True, width="small") for loc in self.loc_names},
                        "미배치": st.column_config.CheckboxColumn("미배치", default=False, width="small")
                    },
                    use_container_width=True,
                    hide_index=True
                )
                # None 값을 0(False)으로 채우기
                edited_df = edited_df.fillna(0)
                for _, row in edited_df.iterrows():
                    unit = row["Unit"]
                    allowed_locs = [loc for loc in self.loc_names if bool(row[loc])]
                    self.allowed_targets_map[unit] = allowed_locs
                    # 미배치 허용 여부 저장
                    self.unassigned_units_map[unit] = bool(row.get("미배치", False))
        
        # Inject JavaScript to style allocate Units multiselect
        st.markdown("""
        <script>
        (function() {
            function styleallocateSelect() {
                const selects = document.querySelectorAll('div[data-testid="stMultiSelect"]');
                selects.forEach(select => {
                    const label = select.querySelector('label');
                    if (label && label.textContent.includes('allocate')) {
                        const inputDiv = select.querySelector('div[data-baseweb="select"]');
                        if (inputDiv) {
                            inputDiv.style.border = '2px solid #ff1493';
                            inputDiv.style.boxShadow = '0 0 0 3px rgba(255, 20, 147, 0.15)';
                            inputDiv.style.borderRadius = '8px';
                            inputDiv.style.backgroundColor = '#fff0f5';
                        }
                        const tags = select.querySelectorAll('div[data-baseweb="tag"]');
                        tags.forEach(tag => {
                            tag.style.backgroundColor = '#ffb6c1';
                            tag.style.color = '#8b008b';
                            tag.style.borderColor = '#ff1493';
                        });
                        const input = select.querySelector('input');
                        if (input) input.style.caretColor = '#ff1493';
                        const svgs = select.querySelectorAll('svg');
                        svgs.forEach(svg => svg.style.color = '#ff1493');
                    }
                });
            }
            styleallocateSelect();
            setTimeout(styleallocateSelect, 100);
            setTimeout(styleallocateSelect, 500);
        })();
        </script>
        """, unsafe_allow_html=True)
        
        # Schematic preview
        if any(len(self.fixed_per_loc[name]) > 0 for name in self.fixed_per_loc) or len(self.moving_unit) > 0:
            st.markdown("#### 🗺️ Assignment schematic (preview)")
            fig = render_relocation_schematic(self.loc_names, self.fixed_per_loc, self.moving_unit)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)


class RunAnalysis:
    """Relocation 분석을 수행하고 결과를 관리하는 클래스"""
    
    @classmethod
    def from_saved(cls, data: dict):
        """저장된 JSON에서 로드. 재분석 없이 결과 복원."""
        self = object.__new__(cls)
        self.df_filtered = data.get("df_filtered")
        self.results_df = data.get("results_df")
        if self.df_filtered is None:
            self.df_filtered = pd.DataFrame()
        if self.results_df is None:
            self.results_df = pd.DataFrame()
        self.relocation_unit = data.get("relocation_unit", "unit")
        self.fixed_per_loc = data.get("fixed_per_loc") or {}
        self.moving_unit = data.get("moving_unit") or {}
        self.allowed_targets_map = data.get("allowed_targets_map") or {}
        self.loc_names = data.get("loc_names") or []
        self.unit_min = data.get("unit_min")
        _sd, _ed = data.get("start_date"), data.get("end_date")
        self.start_date = pd.to_datetime(_sd) if _sd is not None else None
        self.end_date = pd.to_datetime(_ed) if _ed is not None else None
        self.selected_metrics = data.get("selected_metrics") or []
        sc = data.get("selected_agg_cols") or [data.get("selected_agg_col", "total_seat_count")]
        self.selected_agg_cols = list(sc) if isinstance(sc, (list, tuple)) else [sc]
        self.selected_agg_col = self.selected_agg_cols[0] if self.selected_agg_cols else "total_seat_count"
        self.max_sample_count = data.get("max_sample_count")
        self.unassigned_units_map = data.get("unassigned_units_map") or {}
        self.analysis_count = data.get("analysis_count")
        self.total_combinations = data.get("total_combinations")
        self.is_sampled = data.get("is_sampled", False)
        self.category_dict = CATEGORY_DICT
        self.dep_category = f"departure {self.selected_agg_col}"
        self.arr_category = f"arrival {self.selected_agg_col}"
        self.dep_info = self.category_dict.get(self.dep_category, {})
        self.arr_info = self.category_dict.get(self.arr_category, {})
        # session_state 동기화
        st.session_state["relocation_results_df"] = self.results_df
        st.session_state["relocation_df_filtered"] = self.df_filtered
        st.session_state["relocation_start_date"] = self.start_date
        st.session_state["relocation_end_date"] = self.end_date
        st.session_state["relocation_unit_min"] = self.unit_min
        st.session_state["relocation_unit"] = self.relocation_unit
        st.session_state["relocation_selected_agg_col"] = self.selected_agg_col
        st.session_state["relocation_selected_agg_cols"] = self.selected_agg_cols
        st.session_state["relocation_loc_names"] = self.loc_names
        return self
    
    def __init__(self, df_filtered, relocation_unit, fixed_per_loc, moving_unit, 
                 allowed_targets_map, loc_names, unit_min, start_date, end_date,
                 selected_metrics, selected_agg_col, max_sample_count, unassigned_units_map=None,
                 selected_agg_cols=None):
        """분석 실행 및 결과 저장. selected_agg_cols가 있으면 여러 agg에 대해 분석 후 결과 병합."""
        import time
        
        # 입력 파라미터 저장
        self.df_filtered = df_filtered
        self.relocation_unit = relocation_unit
        self.fixed_per_loc = fixed_per_loc
        self.moving_unit = moving_unit
        self.allowed_targets_map = allowed_targets_map
        self.loc_names = loc_names
        self.unit_min = unit_min
        self.start_date = start_date
        self.end_date = end_date
        self.selected_metrics = selected_metrics
        self.selected_agg_cols = selected_agg_cols if isinstance(selected_agg_cols, (list, tuple)) and len(selected_agg_cols) > 0 else [selected_agg_col]
        self.selected_agg_col = self.selected_agg_cols[0]
        
        # Category dict 설정 (원본 컬럼명 직접 사용)
        self.category_dict = CATEGORY_DICT
        self.dep_category = f"departure {self.selected_agg_col}"
        self.arr_category = f"arrival {self.selected_agg_col}"
        self.dep_info = self.category_dict.get(self.dep_category, {})
        self.arr_info = self.category_dict.get(self.arr_category, {})
        
        # 분석 실행
        # LOC 기반 모든 조합 생성 (L^M)
        all_assignments = build_loc_assignments(
            fixed_per_loc,
            moving_unit,
            allowed_targets_map=allowed_targets_map if allowed_targets_map and len(allowed_targets_map) > 0 else None,
            unassigned_units_map=unassigned_units_map if unassigned_units_map and len(unassigned_units_map) > 0 else None
        )
        total_combinations = len(all_assignments)
        
        # 샘플링 적용
        if total_combinations > max_sample_count:
            import random
            random.seed(RANDOM_SEED)
            all_assignments = random.sample(all_assignments, max_sample_count)
            self.analysis_count = max_sample_count
            self.total_combinations = total_combinations
            self.is_sampled = True
        else:
            self.analysis_count = total_combinations
            self.total_combinations = total_combinations
            self.is_sampled = False
        
        # Progress bar 생성
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 시작 시간 기록
        start_time = time.time()
        
        # 각 조합별 분석
        results = []
        for idx, loc_map in enumerate(all_assignments, 1):
            # 진행률 업데이트
            progress = idx / self.analysis_count
            progress_bar.progress(progress)
            
            # 상태 텍스트 업데이트
            elapsed_time = time.time() - start_time
            if idx > 1:
                avg_time_per_scenario = elapsed_time / (idx - 1)
                remaining_scenarios = self.analysis_count - idx + 1
                estimated_remaining_time = avg_time_per_scenario * remaining_scenarios
                status_text.text(f"📊 Progress {idx:,}/{self.analysis_count:,} ({progress*100:.1f}%) | "
                                f"⏱️ Elapsed: {elapsed_time:.1f}s | "
                                f"🕐 ETA: {estimated_remaining_time:.1f}s")
            else:
                status_text.text(f"📊 Progress {idx:,}/{self.analysis_count:,} ({progress*100:.1f}%)")
            
            # LOC 기반 분석
            per_loc_stats = self.run_scenario(loc_map)
            scenario_id = scenario_fingerprint(loc_map, idx)
            # LOC 별 리스트 컬럼 구성 (리스트 그대로 보이게 유지)
            loc_list_cols = {loc_name: sorted(units) for loc_name, units in sorted(loc_map.items(), key=lambda x: x[0])}
            results.append({
                'Scenario_ID': scenario_id,
                **loc_list_cols,
                **per_loc_stats
            })
        
        # 완료 메시지
        total_elapsed_time = time.time() - start_time
        progress_bar.progress(1.0)
        status_text.text(f"✅ Done. {self.analysis_count:,} scenarios analyzed | Elapsed: {total_elapsed_time:.1f}s")
        
        # 결과 DataFrame 생성 및 후처리
        self.results_df = pd.DataFrame(results)
        self._process_results()
    
    def process_passenger_data(self, df_filtered, category="departure od-passenger count", keep_cols=None):
        """show_profile과 동일한 방식으로 승객 데이터 처리"""
        
        # 전역 category_dict 사용
        category_dict = CATEGORY_DICT
        
        # category가 category_dict에 없으면 빈 DataFrame 반환
        if category not in category_dict:
            return pd.DataFrame()
        
        agg_col = category_dict[category]["agg_col"]
        flight_io = category_dict[category]["flight_io"]
        mean = category_dict[category]["mean"]
        sigma = category_dict[category]["sigma"]
        min_max_clip = category_dict[category]["min_max_clip"]
        
        # flight_io 필터링
        df_filtered = df_filtered[df_filtered["flight_io"] == flight_io]
        
        # 안전 장치: 필요한 집계 컬럼이 없으면 total_seat_count로 대체, 그것도 없으면 빈 DF 반환
        if agg_col not in df_filtered.columns:
            if "total_seat_count" in df_filtered.columns:
                agg_col = "total_seat_count"
            else:
                return pd.DataFrame()
        
        # 필요한 컬럼만 선택 (+ 추가로 유지할 컬럼들)
        base_cols = ["scheduled_gate_local", agg_col] + BASE_COL_OPTIONS
        if keep_cols is None:
            keep_cols = []
        # 존재하는 컬럼만 유지하고 중복은 제거
        extra_cols = [c for c in keep_cols if c in df_filtered.columns and c not in base_cols]
        df_filtered = df_filtered[base_cols + extra_cols]
        # 음수 또는 0인 승객 수 제거
        df_filtered = df_filtered[df_filtered[agg_col] > 0]
        
        if len(df_filtered) == 0:
            return pd.DataFrame()  # 빈 DataFrame 반환
        
        # 승객 수만큼 행 복제 (show_profile과 동일한 방식)
        df_filtered = df_filtered.loc[df_filtered.index.repeat(df_filtered[agg_col])].reset_index(drop=True)
        # Show-up time 생성 (항공사별로 독립적인 분포)
        df_filtered[f"scheduled_gate_local(min)"] = 0
        df_filtered = create_normal_dist_col_by_airline(
            df=df_filtered,
            ref_col=f"scheduled_gate_local(min)",
            new_col="SHOW(min)",
            mean=mean,
            sigma=sigma,
            min_max_clip=min_max_clip,
            relocation_unit=keep_cols[0] if keep_cols and len(keep_cols) > 0 else "terminal_carrier",
            unit="m",
            iteration=1,
            datetime=False
        )
        df_filtered["SHOW"] = df_filtered["scheduled_gate_local"] + pd.to_timedelta(df_filtered["SHOW(min)"], unit="m")
        df_filtered = df_filtered.drop(columns=[f"scheduled_gate_local(min)", "SHOW(min)"])
        return df_filtered
    
    def _run_scenario_for_agg(self, loc_map: dict, agg: str) -> dict:
        """Run scenario stats for a single aggregate base (used when multiple aggs selected)."""
        results = {}
        for loc_name, units in sorted(loc_map.items(), key=lambda x: x[0]):
            mask = self.df_filtered[self.relocation_unit].isin(sorted(units))
            if mask.sum() == 0:
                for io, label in [("departure", "Dep"), ("arrival", "Arr")]:
                    prefix = f"{loc_name}_{label}"
                    if "97.5%" in self.selected_metrics:
                        results[f"{prefix}_97.5%"] = 0
                    if "95%" in self.selected_metrics:
                        results[f"{prefix}_95%"] = 0
                    if "90%" in self.selected_metrics:
                        results[f"{prefix}_90%"] = 0
                    if "Mean" in self.selected_metrics:
                        results[f"{prefix}_Mean"] = 0
                    if "Total" in self.selected_metrics:
                        results[f"{prefix}_Total"] = 0
                continue
            for io, label in [("departure", "Dep"), ("arrival", "Arr")]:
                category = f"{io} {agg}"
                df_proc = self.process_passenger_data(self.df_filtered[mask], category, keep_cols=[self.relocation_unit])
                if len(df_proc) > 0:
                    count_df, _ = make_count_df(df_proc, self.start_date, self.end_date, 'SHOW', self.relocation_unit, buffer_day=False, freq_min=self.unit_min)
                    if len(count_df) > 0:
                        total_df = count_df.groupby("Time")["index"].sum()
                        stats = {
                            '97.5%': total_df.quantile(0.975),
                            '95%': total_df.quantile(0.95),
                            '90%': total_df.quantile(0.90),
                            'Mean': total_df.mean(),
                            'Total': count_df['index'].sum(),
                            'Time_Slots': len(total_df)
                        }
                    else:
                        stats = {'97.5%': 0, '95%': 0, '90%': 0, 'Mean': 0, 'Total': 0, 'Time_Slots': 0}
                else:
                    stats = {'97.5%': 0, '95%': 0, '90%': 0, 'Mean': 0, 'Total': 0, 'Time_Slots': 0}
                prefix = f"{loc_name}_{label}"
                if "97.5%" in self.selected_metrics:
                    results[f"{prefix}_97.5%"] = int(stats['97.5%'])
                if "95%" in self.selected_metrics:
                    results[f"{prefix}_95%"] = int(stats['95%'])
                if "90%" in self.selected_metrics:
                    results[f"{prefix}_90%"] = int(stats['90%'])
                if "Mean" in self.selected_metrics:
                    results[f"{prefix}_Mean"] = int(stats['Mean'])
                if "Total" in self.selected_metrics:
                    results[f"{prefix}_Total"] = int(stats['Total']) if isinstance(stats['Total'], (int, np.integer, np.int64, np.int32)) else int(stats['Total'])
        return results

    def run_scenario(self, loc_map: dict) -> dict:
        """Compute per-LOC stats for dep/arr seat/od/tr using existing generation. Single agg: delegate; multi agg: merge."""
        if len(self.selected_agg_cols) == 1:
            return self._run_scenario_for_agg(loc_map, self.selected_agg_cols[0])
        per_loc_stats = {}
        for agg in self.selected_agg_cols:
            suffix = f"_{agg}"
            stats = self._run_scenario_for_agg(loc_map, agg)
            for k, v in stats.items():
                per_loc_stats[k + suffix] = v
        return per_loc_stats

    def _process_results(self):
        """결과 DataFrame 후처리 (Distance, Rank, Final_Score 계산). 다중 agg 시 suffix별로 처리."""
        # agg별 suffix: 단일이면 [""], 다중이면 ["", "_movement", ...] (첫 agg는 빈 suffix)
        suffixes = [""] if len(self.selected_agg_cols) == 1 else [f"_{a}" for a in self.selected_agg_cols]

        def calculate_distance_and_rank(df, metric_name, suffix):
            dep_cols = [col for col in df.columns if col.endswith(f'_Dep_{metric_name}{suffix}')]
            arr_cols = [col for col in df.columns if col.endswith(f'_Arr_{metric_name}{suffix}')]
            if len(dep_cols) > 0:
                distance_dep_col = f'Dist_Dep_{metric_name}{suffix}'
                rank_dep_col = f'Rank_Dep_{metric_name}{suffix}'
                df[distance_dep_col] = df[dep_cols].apply(lambda row: np.sqrt((row ** 2).sum()), axis=1)
                df[rank_dep_col] = df[distance_dep_col].rank(method='min', ascending=True).astype(int)
            if len(arr_cols) > 0:
                distance_arr_col = f'Dist_Arr_{metric_name}{suffix}'
                rank_arr_col = f'Rank_Arr_{metric_name}{suffix}'
                df[distance_arr_col] = df[arr_cols].apply(lambda row: np.sqrt((row ** 2).sum()), axis=1)
                df[rank_arr_col] = df[distance_arr_col].rank(method='min', ascending=True).astype(int)
            return df

        def calculate_terminal_ranks(df, metric_name, loc_names, suffix):
            for loc_name in loc_names:
                dep_col = f"{loc_name}_Dep_{metric_name}{suffix}"
                if dep_col in df.columns:
                    df[f"Rank_{dep_col}"] = df[dep_col].rank(method='min', ascending=True).astype(int)
                arr_col = f"{loc_name}_Arr_{metric_name}{suffix}"
                if arr_col in df.columns:
                    df[f"Rank_{arr_col}"] = df[arr_col].rank(method='min', ascending=True).astype(int)
            return df

        metric_order = ['97.5%', '95%', '90%', 'Mean']
        for metric in metric_order:
            if metric in self.selected_metrics:
                for suf in suffixes:
                    self.results_df = calculate_distance_and_rank(self.results_df, metric, suf)
        for metric in metric_order:
            if metric in self.selected_metrics:
                for suf in suffixes:
                    self.results_df = calculate_terminal_ranks(self.results_df, metric, self.loc_names, suf)
        
        # Rank가 들어간 모든 numeric 컬럼을 사용하여 Final_Score 계산
        numeric_cols_all = self.results_df.select_dtypes(include=[np.number]).columns.tolist()
        rank_cols = [c for c in numeric_cols_all if 'Rank' in c]
        if len(rank_cols) > 0:
            self.results_df['TotalRank'] = self.results_df[rank_cols].sum(axis=1)
            self.results_df['StdRank'] = self.results_df[rank_cols].std(axis=1)
            self.results_df['Final_Score'] = self.results_df['TotalRank'] + 0.5 * self.results_df['StdRank']
        
        # 컬럼 순서 재정렬 (Location별로 그룹화)
        ordered_columns = []
        
        # 1. Scenario_ID
        if 'Scenario_ID' in self.results_df.columns:
            ordered_columns.append('Scenario_ID')
        
        # 1.5 요청된 지표: TotalRank, StdRank, Final_Score (Scenario_ID 바로 옆)
        for col in ['TotalRank', 'StdRank', 'Final_Score']:
            if col in self.results_df.columns:
                ordered_columns.append(col)
        
        # 2. Distance와 Rank 컬럼들 (suffix별)
        for suf in suffixes:
            for metric in metric_order:
                if metric in self.selected_metrics:
                    for process in ['Dep', 'Arr']:
                        for col_suffix in ['Rank', 'Dist']:
                            col_name = f'{col_suffix}_{process}_{metric}{suf}'
                            if col_name in self.results_df.columns:
                                ordered_columns.append(col_name)
        # 3. Location 컬럼들
        for loc_name in self.loc_names:
            if loc_name in self.results_df.columns:
                ordered_columns.append(loc_name)
        # 4. Location별 통계 컬럼들 (suffix별)
        for suf in suffixes:
            for metric in metric_order:
                if metric in self.selected_metrics:
                    for loc_name in self.loc_names:
                        dep_col = f"{loc_name}_Dep_{metric}{suf}"
                        if dep_col in self.results_df.columns:
                            ordered_columns.append(dep_col)
                    for loc_name in self.loc_names:
                        arr_col = f"{loc_name}_Arr_{metric}{suf}"
                        if arr_col in self.results_df.columns:
                            ordered_columns.append(arr_col)
        # 4.5. 터미널 단위 랭킹 컬럼들
        for suf in suffixes:
            for metric in metric_order:
                if metric in self.selected_metrics:
                    for loc_name in self.loc_names:
                        for da in ['Dep', 'Arr']:
                            rank_col = f"Rank_{loc_name}_{da}_{metric}{suf}"
                            if rank_col in self.results_df.columns:
                                ordered_columns.append(rank_col)
        
        # 5. 나머지 컬럼들 (Total 등)
        remaining_cols = [col for col in self.results_df.columns if col not in ordered_columns]
        ordered_columns.extend(sorted(remaining_cols))
        
        # 컬럼 순서대로 재정렬
        self.results_df = self.results_df[ordered_columns]
        
        # session_state에도 저장 (하위 호환성)
        st.session_state['relocation_results_df'] = self.results_df
        st.session_state['relocation_df_filtered'] = self.df_filtered
        st.session_state['relocation_start_date'] = self.start_date
        st.session_state['relocation_end_date'] = self.end_date
        st.session_state['relocation_unit_min'] = self.unit_min
        st.session_state['relocation_unit'] = self.relocation_unit
        st.session_state['relocation_selected_agg_col'] = self.selected_agg_col
        st.session_state['relocation_selected_agg_cols'] = getattr(self, 'selected_agg_cols', [self.selected_agg_col])
        st.session_state['relocation_loc_names'] = self.loc_names
    
    def get_numeric_cols(self, chart_agg=None):
        """numeric_cols 계산. chart_agg가 있으면 해당 agg의 컬럼만 반환 (다중 agg 시)."""
        numeric_cols = self.results_df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_patterns = ['Scenario_ID', 'Rank']
        numeric_cols = [col for col in numeric_cols if not any(pattern in col for pattern in exclude_patterns)]
        numeric_cols = [col for col in numeric_cols if col not in self.loc_names]
        if chart_agg is not None and getattr(self, 'selected_agg_cols', None) and len(self.selected_agg_cols) > 1:
            suffix = f"_{chart_agg}"
            first_agg = self.selected_agg_cols[0]
            numeric_cols = [c for c in numeric_cols if (chart_agg == first_agg and not any(c.endswith(f"_{a}") for a in self.selected_agg_cols[1:])) or (chart_agg != first_agg and c.endswith(suffix))]
        return numeric_cols

    def get_results_df_for_agg(self, agg):
        """해당 agg에 해당하는 컬럼만 포함한 결과 DataFrame (suffix 제거). 차트용."""
        if not getattr(self, 'selected_agg_cols', None) or len(self.selected_agg_cols) <= 1:
            return self.results_df
        suffix = f"_{agg}"
        first_agg = self.selected_agg_cols[0]
        # 해당 agg 컬럼: 첫 agg면 suffix 없는 것, 그 외는 _agg suffix 있는 것
        keep = [c for c in self.results_df.columns if c in self.loc_names or c == 'Scenario_ID' or (agg == first_agg and not any(c.endswith(f"_{a}") for a in self.selected_agg_cols[1:])) or (agg != first_agg and c.endswith(suffix))]
        out = self.results_df[keep].copy()
        if agg != first_agg:
            out = out.rename(columns={c: c.replace(suffix, "") for c in out.columns if c.endswith(suffix)})
        return out

    @staticmethod
    def _compute_cumulative_series(count_df_dep, count_df_arr, fallback_index=None):
        """Relocation_Master에서 cumsum 1회만 계산. 결과를 JSON에 담고 Report는 그대로 사용.
        Returns: (all_index, dep_ts, arr_ts, cum_dep, cum_arr, net_cum)"""
        dep_ts = count_df_dep.groupby("Time")["index"].sum().sort_index() if len(count_df_dep) > 0 else pd.Series(dtype=float)
        arr_ts = count_df_arr.groupby("Time")["index"].sum().sort_index() if len(count_df_arr) > 0 else pd.Series(dtype=float)
        if len(dep_ts) == 0 and len(arr_ts) == 0 and fallback_index is not None:
            all_index = fallback_index
        else:
            all_index = dep_ts.index.union(arr_ts.index) if len(dep_ts) > 0 or len(arr_ts) > 0 else (fallback_index if fallback_index is not None else pd.Index([]))
        dep_ts = dep_ts.reindex(all_index, fill_value=0)
        arr_ts = arr_ts.reindex(all_index, fill_value=0)
        cum_dep = dep_ts.cumsum()
        cum_arr = arr_ts.cumsum()
        net_cum = cum_arr - cum_dep
        return all_index, dep_ts, arr_ts, cum_dep, cum_arr, net_cum

    def build_scenario_export(self, scenario_idx, terminal, dep_arr, capacity, agg_col_override=None, puffing_factor=1.0):
        """Headless build of one scenario export dict for report (no Streamlit). agg_col_override: use this agg instead of selected_agg_col. puffing_factor: 1.0=원본, 1.2=20% 증가(랜덤셔플 후 행 뻥튀기). Returns dict or None."""
        def _to_json_serializable(obj):
            if hasattr(obj, "tolist"):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, pd.Series):
                return {str(k): _to_json_serializable(v) for k, v in obj.items()}
            if isinstance(obj, pd.DataFrame):
                return obj.astype(object).where(pd.notnull(obj), None).to_dict(orient="split")
            if isinstance(obj, dict):
                return {k: _to_json_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_to_json_serializable(x) for x in obj]
            if isinstance(obj, (date, datetime)):
                return str(obj)
            if isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            if isinstance(obj, pd.Index):
                return [str(x) for x in obj]
            if pd.isna(obj):
                return None
            return obj

        try:
            row = self.results_df.loc[scenario_idx]
        except Exception:
            return None
        terminal_assignments = {}
        for loc in self.loc_names:
            if loc not in row.index:
                continue
            u = row[loc]
            if isinstance(u, (list, tuple)):
                terminal_assignments[loc] = list(u)
            elif hasattr(u, "__iter__") and not isinstance(u, (str, bytes)):
                try:
                    terminal_assignments[loc] = list(u)
                except Exception:
                    terminal_assignments[loc] = []
            else:
                terminal_assignments[loc] = []
        selected_units = terminal_assignments.get(terminal, [])
        if not selected_units:
            return None
        mask = self.df_filtered[self.relocation_unit].isin(selected_units)
        filtered_data = self.df_filtered[mask].copy()
        if len(filtered_data) == 0:
            return None
        # Puffing: frac=1 랜덤 셔플 후 puffing_factor배로 행 뻥튀기 (예: 1.35 → 35% 증가)
        pf = float(puffing_factor) if puffing_factor is not None else 1.0
        if pf > 0 and pf != 1.0:
            filtered_data = filtered_data.sample(frac=1)
            if pf > 1.0:
                extra = int(len(filtered_data) * (pf - 1.0))
                if extra > 0:
                    extra_df = filtered_data.sample(n=extra, replace=True)
                    filtered_data = pd.concat([filtered_data, extra_df], ignore_index=True)
            elif pf < 1.0:
                n_keep = max(1, int(len(filtered_data) * pf))
                filtered_data = filtered_data.iloc[:n_keep].copy()
        agg_col = agg_col_override if (agg_col_override and str(agg_col_override).strip()) else getattr(self, "calc_selected_agg_col", self.selected_agg_col)
        agg_col = (agg_col or "").strip() if agg_col is not None else ""
        if not agg_col:
            return None
        dep_category = f"departure {agg_col}"
        arr_category = f"arrival {agg_col}"
        if dep_category not in self.category_dict or arr_category not in self.category_dict:
            return None
        summary_extra_cols = ["dep/arr_airport", "A/C Code", "International/Domestic"]
        keep_for_summary = [self.relocation_unit] + [c for c in summary_extra_cols if c in filtered_data.columns]
        if dep_arr == "Both":
            df_proc_dep = self.process_passenger_data(filtered_data, dep_category, keep_cols=keep_for_summary)
            df_proc_arr = self.process_passenger_data(filtered_data, arr_category, keep_cols=keep_for_summary)
            if len(df_proc_dep) == 0 and len(df_proc_arr) == 0:
                return None
            count_df_dep, _ = make_count_df(df_proc_dep, self.start_date, self.end_date, "SHOW", self.relocation_unit, buffer_day=False, freq_min=self.unit_min) if len(df_proc_dep) > 0 else (pd.DataFrame(), None)
            count_df_arr, _ = make_count_df(df_proc_arr, self.start_date, self.end_date, "SHOW", self.relocation_unit, buffer_day=False, freq_min=self.unit_min) if len(df_proc_arr) > 0 else (pd.DataFrame(), None)
            if len(count_df_dep) > 0 and len(count_df_arr) > 0:
                count_df_combined = pd.concat([count_df_dep, count_df_arr], ignore_index=True)
                count_df = count_df_combined.groupby(["Time", self.relocation_unit])["index"].sum().reset_index()
            elif len(count_df_dep) > 0:
                count_df = count_df_dep.copy()
            elif len(count_df_arr) > 0:
                count_df = count_df_arr.copy()
            else:
                return None
        else:
            io_label = "departure" if dep_arr == "Dep" else "arrival"
            category = f"{io_label} {agg_col}"
            df_proc = self.process_passenger_data(filtered_data, category, keep_cols=keep_for_summary)
            if len(df_proc) == 0:
                return None
            count_df, _ = make_count_df(df_proc, self.start_date, self.end_date, "SHOW", self.relocation_unit, buffer_day=False, freq_min=self.unit_min)
            if len(count_df) == 0:
                return None
        time_series = count_df.groupby("Time")["index"].sum().sort_index()
        total_sum = float(time_series.sum())
        _avg_seat_batch = _compute_avg_seat(filtered_data, dep_arr)
        excess_over_capacity = (time_series - capacity).clip(lower=0)
        excess_sum = float(excess_over_capacity.sum())
        excess_share_pct = (excess_sum / total_sum * 100) if total_sum > 0 else 0.0
        total_delay_hours = compute_capacity_constraint_delay_hours(
            time_series.values, capacity, slot_duration_minutes=self.unit_min
        )
        vals = time_series.values
        sorted_vals = np.sort(vals)[::-1] if hasattr(vals, "__len__") and len(vals) > 0 else np.array([])
        sorted_values_block = {
            "statistics": {},
            "sorted_value_count": len(sorted_vals),
            "graph_data": {
                "x_index": list(range(len(sorted_vals))),
                "y_sorted_values": _to_json_serializable(sorted_vals.tolist() if hasattr(sorted_vals, "tolist") else list(sorted_vals)),
            },
        }
        if len(sorted_vals) > 0:
            _, v97_5 = _percentile_rank_value(sorted_vals, 97.5)
            _, v95 = _percentile_rank_value(sorted_vals, 95)
            _, v90 = _percentile_rank_value(sorted_vals, 90)
            _, v50 = _percentile_rank_value(sorted_vals, 50)
            sorted_values_block["statistics"] = {
                "max": float(np.max(sorted_vals)),
                "97.5%": v97_5,
                "95%": v95,
                "90%": v90,
                "median": v50,
                "mean": float(np.mean(sorted_vals)),
            }
        vals_ts = time_series.values
        has_vals = hasattr(vals_ts, "__len__") and len(vals_ts) > 0
        sorted_ts = np.sort(vals_ts)[::-1] if has_vals else np.array([])
        p95 = int(_percentile_rank_value(sorted_ts, 95)[1]) if has_vals else 0
        p97_5 = int(_percentile_rank_value(sorted_ts, 97.5)[1]) if has_vals else 0
        row_dict = dict(row.to_dict())
        if dep_arr == "Dep" or dep_arr == "Both":
            row_dict[f"{terminal}_Dep_95%"] = p95
            row_dict[f"{terminal}_Dep_97.5%"] = p97_5
        if dep_arr == "Arr" or dep_arr == "Both":
            row_dict[f"{terminal}_Arr_95%"] = p95
            row_dict[f"{terminal}_Arr_97.5%"] = p97_5
        scenario_id_str = format_scenario_id(scenario_idx) if hasattr(self.results_df, "columns") and "Scenario_ID" in self.results_df.columns else str(scenario_idx)
        _agg_df = self.get_results_df_for_agg(agg_col)
        scenario_export = {
            "_usage": "Scenario detail export for comparison reports.",
            "meta": {
                "scenario_id": scenario_id_str,
                "terminal": terminal,
                "dep_arr": dep_arr,
                "aggregate_base": agg_col,
                "date_range": {"start": str(self.start_date), "end": str(self.end_date)},
                "time_unit_minutes": self.unit_min,
                "terminal_assignments": {loc: list(units) for loc, units in terminal_assignments.items()},
                "terminal_names": list(self.loc_names),
                "default_assignments": getattr(self, "fixed_per_loc", None) and {loc: list(units) for loc, units in self.fixed_per_loc.items()} or None,
                "selected_units_at_terminal": list(selected_units),
                "puffing_factor": pf if pf != 1.0 else None,
            },
            "scenario_result_table_row": _to_json_serializable(row_dict),
            "statistics_table": _to_json_serializable(
                _agg_df.reset_index().astype(object).where(pd.notnull(_agg_df.reset_index()), None).to_dict(orient="records")
            ),
            "description": f"Scenario detail: Terminal={terminal}, Dep/Arr={dep_arr}, Aggregate={agg_col}.",
            # time_series: 시나리오 상세 Cumulative profile과 동일. 리포트 Cumulative comparison(Both)에서 time_index·values 그대로 사용
            "time_series": {
                "time_index": [str(t) for t in time_series.index],
                "values": _to_json_serializable(time_series.values),
                "capacity": int(capacity),
                "total_sum": total_sum,
                "excess_over_capacity_sum": excess_sum,
                "excess_as_percent_of_total": round(excess_share_pct, 2),
                "total_delay_hours_capacity_constraint": int(total_delay_hours) if total_delay_hours == int(total_delay_hours) else round(total_delay_hours, 2),
                **({"avg_seat": _avg_seat_batch} if _avg_seat_batch is not None else {}),
            },
            "sorted_values": sorted_values_block,
        }
        # Cumulative: Relocation_Master에서 cumsum 1회 계산 → JSON에 담음. Report는 그대로 사용 (두번계산 없음)
        if dep_arr == "Both":
            all_index, _dep_ts, _arr_ts, cum_dep, cum_arr, net_cum = self._compute_cumulative_series(count_df_dep, count_df_arr, fallback_index=time_series.index)
            time_index_cum = [str(t) for t in all_index]
            scenario_export["cumulative"] = {
                "time_index": time_index_cum,
                "cumulative_departure": _to_json_serializable(cum_dep.values.tolist()),
                "cumulative_arrival": _to_json_serializable(cum_arr.values.tolist()),
                "net_cumulative": _to_json_serializable(net_cum.values.tolist()),
            }
        else:
            cum_vals = time_series.values
            if hasattr(cum_vals, "__len__") and len(cum_vals) > 0:
                cum_arr = np.cumsum(np.asarray(cum_vals, dtype=float))
                cum_list = _to_json_serializable(cum_arr.tolist())
                time_index_cum = [str(t) for t in time_series.index]
                if dep_arr == "Dep":
                    scenario_export["cumulative"] = {
                        "time_index": time_index_cum,
                        "cumulative_departure": cum_list,
                        "cumulative_arrival": [],
                        "net_cumulative": cum_list,
                    }
                elif dep_arr == "Arr":
                    scenario_export["cumulative"] = {
                        "time_index": time_index_cum,
                        "cumulative_departure": [],
                        "cumulative_arrival": cum_list,
                        "net_cumulative": cum_list,
                    }
                else:
                    scenario_export["cumulative"] = {
                        "time_index": time_index_cum,
                        "cumulative_departure": [],
                        "cumulative_arrival": [],
                        "net_cumulative": cum_list,
                    }
        # Time Series pivot 데이터: 만들기 시 JSON에 포함되어 리포트 Time Series 섹션에서 사용
        try:
            count_df = count_df.copy()
            count_df["date"] = pd.to_datetime(count_df["Time"]).dt.date
            count_df["time_of_day"] = pd.to_datetime(count_df["Time"]).dt.strftime("%H:%M")
            count_df["day_of_week"] = pd.to_datetime(count_df["Time"]).dt.day_name()
            daily_hourly_table = count_df.groupby(["date", "day_of_week", "time_of_day"])[["index"]].sum().reset_index()
            daily_hourly_table["time_sort"] = pd.to_datetime(daily_hourly_table["time_of_day"], format="%H:%M").dt.time
            daily_hourly_pivot = daily_hourly_table.pivot_table(
                index="time_of_day",
                columns=["date", "day_of_week"],
                values="index",
                aggfunc="sum",
                fill_value=0,
            )
            time_index_sorted = sorted(daily_hourly_pivot.index, key=lambda x: pd.to_datetime(x, format="%H:%M").time())
            daily_hourly_pivot = daily_hourly_pivot.reindex(time_index_sorted)
            daily_hourly_pivot.columns = [f"{date} ({day})" for date, day in daily_hourly_pivot.columns]
            scenario_export["time_series"]["daily_hourly_pivot"] = _to_json_serializable(
                daily_hourly_pivot.astype(object).where(pd.notnull(daily_hourly_pivot), None).to_dict(orient="split")
            )
            group_col = self.relocation_unit
            if group_col in count_df.columns:
                stacked_pivot = count_df.pivot_table(
                    index="Time",
                    columns=group_col,
                    values="index",
                    aggfunc="sum",
                    fill_value=0,
                )
                scenario_export["time_series"]["stacked_pivot_group_col"] = group_col
                scenario_export["time_series"]["stacked_pivot"] = _to_json_serializable(
                    stacked_pivot.astype(object).where(pd.notnull(stacked_pivot), None).to_dict(orient="split")
                )
                group_sums = stacked_pivot.sum()
                total_sum_gr = group_sums.sum()
                group_percentages = (group_sums / total_sum_gr * 100) if total_sum_gr > 0 else group_sums * 0
                summary_df = pd.DataFrame({"Sum": group_sums, "Percentage (%)": group_percentages.round(2)})
                summary_df = summary_df.sort_values("Sum", ascending=False)
                # df_proc for Summary by Group 탭: Both면 concat(df_proc_dep, df_proc_arr), Dep/Arr면 df_proc
                if dep_arr == "Both":
                    df_proc_sbg = pd.concat([df_proc_dep, df_proc_arr], ignore_index=True) if len(df_proc_dep) > 0 and len(df_proc_arr) > 0 else (df_proc_dep if len(df_proc_dep) > 0 else df_proc_arr)
                else:
                    df_proc_sbg = df_proc
                summary_by_group_tabs = []
                if len(df_proc_sbg) > 0 and hasattr(df_proc_sbg, "columns"):
                    for tab_label, grp_col in SUMMARY_BY_GROUP_TABS:
                        if grp_col not in df_proc_sbg.columns:
                            continue
                        try:
                            _cnt, _ = make_count_df(df_proc_sbg, self.start_date, self.end_date, "SHOW", grp_col, buffer_day=False, freq_min=self.unit_min)
                            if len(_cnt) == 0:
                                continue
                            _piv = _cnt.pivot_table(index="Time", columns=grp_col, values="index", aggfunc="sum", fill_value=0)
                            _gs = _piv.sum()
                            _tot = _gs.sum()
                            _pct = (_gs / _tot * 100) if _tot > 0 else _gs * 0
                            _sf = pd.DataFrame({"Sum": _gs, "Percentage (%)": _pct.round(2)}).sort_values("Sum", ascending=False)
                            summary_by_group_tabs.append({"label": tab_label, "data": _sf.astype(object).where(pd.notnull(_sf), None).to_dict(orient="split")})
                        except Exception:
                            pass
                if summary_by_group_tabs:
                    scenario_export["time_series"]["summary_by_group"] = {"tabs": _to_json_serializable(summary_by_group_tabs)}
                else:
                    scenario_export["time_series"]["summary_by_group"] = _to_json_serializable(
                        summary_df.astype(object).where(pd.notnull(summary_df), None).to_dict(orient="split")
                    )
        except Exception:
            pass
        return scenario_export

    def render_analysis_summary(self):
        """분석 요약 5줄을 st.info로 표시"""
        _sd = str(self.start_date)[:10] if self.start_date else "—"
        _ed = str(self.end_date)[:10] if self.end_date else "—"
        def _list(d, loc):
            if isinstance(d, dict):
                v = d.get(loc, [])
                return list(v) if isinstance(v, (list, tuple)) else []
            return []
        def _moving_list(m):
            if isinstance(m, list):
                return m
            if isinstance(m, dict):
                return [u for units in m.values() for u in (units if isinstance(units, (list, tuple)) else [])]
            return []
        fixed, moving = self.fixed_per_loc or {}, self.moving_unit or {}
        loc_names = self.loc_names or []
        fixed_parts = [f"{loc} fixed: {_list(fixed, loc)}" for loc in loc_names]
        allocate_units = _moving_list(moving)
        allocate_str = str(allocate_units) if allocate_units else "[]"
        loc_summary = "  \n".join(fixed_parts) + f"  \nAllocate Units: {allocate_str}"
        agg_str = ", ".join(self.selected_agg_cols) if getattr(self, "selected_agg_cols", None) else (self.selected_agg_col or "—")
        metrics_str = ", ".join(self.selected_metrics) if self.selected_metrics else "—"
        sampled_str = f" (sampled: {self.analysis_count}/{self.total_combinations})" if getattr(self, "is_sampled", False) else ""
        ac = getattr(self, "analysis_count", None)
        ac_str = f"{ac:,}" if isinstance(ac, (int, np.integer)) else "—"
        n_rows = len(self.df_filtered) if self.df_filtered is not None else 0
        loaded_name = st.session_state.get("loaded_result_filename")
        loaded_saved_at = st.session_state.get("loaded_result_saved_at")
        lines = [
            f"○ Load File: {loaded_name}" if loaded_name else None,
            f"○ Created: {loaded_saved_at}" if loaded_name and loaded_saved_at else None,
            f"○ Relocation unit: {self.relocation_unit} | Period: {_sd} ~ {_ed} | Slot: {self.unit_min}min",
            f"○ Assignment by Location:\n{loc_summary}",
            f"○ Aggregate (Agg): {agg_str} | Metrics: {metrics_str}",
            f"○ Scenarios analyzed: {ac_str}{sampled_str}",
            f"○ Source data: {n_rows:,} rows",
        ]
        st.code("\n".join(l for l in lines if l is not None), language=None)

    def render_statistics_table(self):
        """Statistics table 렌더링"""
        df = self.results_df
        st.markdown("#### 📋 Statistics table")
        st.dataframe(df.set_index('Scenario_ID') if 'Scenario_ID' in df.columns else df, use_container_width=True, height=640)
    
    def render_1d_visualization(self):
        """Render 1D Visualization (Bar Chart)"""
        df = getattr(self, '_chart_df', self.results_df)
        numeric_cols = self.get_numeric_cols(getattr(self, '_chart_agg', None))
        if len(numeric_cols) == 0:
            st.info("No numeric columns available for visualization")
            return
        st.markdown("#### 📊 1D Visualization")
        default_bar = st.session_state.get('distance_viz_bar_chart_select', numeric_cols[:min(3, len(numeric_cols))])
        selected_bar_cols = st.multiselect(
            "Select for Bar Chart",
            options=numeric_cols,
            default=default_bar[0] if isinstance(default_bar, list) else [default_bar] if default_bar in numeric_cols else numeric_cols[:min(3, len(numeric_cols))],
            key="distance_viz_bar_chart_select"
        )
        if len(selected_bar_cols) > 0:
            sort_col = selected_bar_cols[0]
            sorted_df = df.sort_values(by=sort_col, ascending=True).reset_index(drop=True)
            if 'Scenario_ID' in sorted_df.columns:
                scenario_labels = sorted_df['Scenario_ID'].tolist()
            else:
                scenario_labels = [f'S{i}' for i in sorted_df.index]
            fig_bar = go.Figure()
            for col in selected_bar_cols:
                fig_bar.add_trace(go.Bar(
                    x=scenario_labels,
                    y=sorted_df[col],
                    name=col
                ))
            fig_bar.update_layout(
                title='Bar Chart',
                xaxis=dict(title='Scenario'),
                yaxis=dict(title='Value'),
                barmode='group',
                height=800
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Please select at least one metric for bar chart")
    
    def render_2d_visualization(self):
        """Render 2D Visualization (Scatter Plot)"""
        df = getattr(self, '_chart_df', self.results_df)
        numeric_cols = self.get_numeric_cols(getattr(self, '_chart_agg', None))
        if len(numeric_cols) == 0:
            st.info("No numeric columns available for visualization")
            return
        st.markdown("#### 📊 2D Visualization")
        default_x_idx = 1
        default_y_idx = min(2, len(numeric_cols)-1)
        c1, c2 = st.columns(2)
        with c1:
            x_axis = st.selectbox(
                "X-axis for Scatter",
                options=numeric_cols,
                index=default_x_idx,
                key="distance_viz_scatter_x"
            )
        with c2:
            y_axis = st.selectbox(
                "Y-axis for Scatter",
                options=numeric_cols,
                index=default_y_idx,
                key="distance_viz_scatter_y"
            )
        fig_scatter = go.Figure()
        if 'Scenario_ID' in df.columns:
            scenario_texts = df['Scenario_ID'].tolist()
        else:
            scenario_texts = [f'Scenario {i}' for i in df.index]
        fig_scatter.add_trace(go.Scatter(
            x=df[x_axis],
            y=df[y_axis],
            mode='markers',
            marker=dict(
                size=8,
                color=df.index,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Scenario Index")
            ),
            text=scenario_texts,
            hovertemplate=f'{x_axis}: %{{x}}<br>{y_axis}: %{{y}}<br>%{{text}}<extra></extra>'
        ))
        fig_scatter.update_layout(
            title='Scatter Plot',
            xaxis=dict(title=x_axis),
            yaxis=dict(title=y_axis),
            height=800
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    def render_3d_visualization(self):
        """Render 3D Visualization (3D Scatter Plot)"""
        df = getattr(self, '_chart_df', self.results_df)
        numeric_cols = self.get_numeric_cols(getattr(self, '_chart_agg', None))
        if len(numeric_cols) == 0:
            st.info("No numeric columns available for visualization")
            return
        st.markdown("#### 📊 3D Visualization")
        default_x_idx = 1
        default_y_idx = min(2, len(numeric_cols)-1)
        default_z_idx = min(3, len(numeric_cols)-1)
        c1, c2, c3 = st.columns(3)
        with c1:
            x_3d = st.selectbox("X-axis for 3D", options=numeric_cols, index=default_x_idx, key="distance_viz_3d_x")
        with c2:
            y_3d = st.selectbox("Y-axis for 3D", options=numeric_cols, index=default_y_idx, key="distance_viz_3d_y")
        with c3:
            z_3d = st.selectbox("Z-axis for 3D", options=numeric_cols, index=default_z_idx, key="distance_viz_3d_z")
        fig_3d = go.Figure()
        if 'Scenario_ID' in df.columns:
            scenario_texts_3d = df['Scenario_ID'].tolist()
        else:
            scenario_texts_3d = [f'Scenario {i}' for i in df.index]
        fig_3d.add_trace(go.Scatter3d(
            x=df[x_3d],
            y=df[y_3d],
            z=df[z_3d],
            mode='markers',
            marker=dict(
                size=6,
                color=df.index,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Scenario Index")
            ),
            text=scenario_texts_3d,
            hovertemplate=f'{x_3d}: %{{x}}<br>{y_3d}: %{{y}}<br>{z_3d}: %{{z}}<br>%{{text}}<extra></extra>',
            name='Scenarios'
        ))
        
        fig_3d.update_layout(
            title='3D Scatter Plot',
            scene=dict(
                xaxis_title=x_3d,
                yaxis_title=y_3d,
                zaxis_title=z_3d
            ),
            height=800
        )
        st.plotly_chart(fig_3d, use_container_width=True)
    
    def render_distance_visualization(self):
        """Render distance visualization with 3 graphs (Bar, Scatter, 3D Scatter). 다중 agg 시 선택한 agg만 시각화."""
        agg_options = getattr(self, 'selected_agg_cols', None) or [self.selected_agg_col]
        if len(agg_options) > 1:
            chart_agg = st.selectbox(
                "**Aggregate base (for chart)**",
                options=agg_options,
                index=0,
                key="distance_viz_agg_select",
                help="차트에 표시할 집계 기준을 선택하세요.",
            )
            self._chart_agg = chart_agg
            self._chart_df = self.get_results_df_for_agg(chart_agg)
        else:
            self._chart_agg = None
            self._chart_df = self.results_df
        col1_select, col2_select, col3_select = st.columns(3)
        with col1_select:
            self.render_1d_visualization()
        with col2_select:
            self.render_2d_visualization()
        with col3_select:
            self.render_3d_visualization()
    
    def render_radial_chart(self):
        """Render Radial Chart Visualization"""
        numeric_cols = self.get_numeric_cols()
        
        if len(numeric_cols) == 0:
            st.info("No numeric columns available for visualization")
            return
        
        st.markdown("#### 📊 Radial Chart Visualization")
        
        # 시나리오 선택 (Multiselect, 여러 개 선택 가능)
        scenario_options = self.results_df.index.tolist()
        
        # 시나리오 표시 형식 준비
        if 'Scenario_ID' in self.results_df.columns:
            scenario_display_map = {format_scenario_id(opt): opt for opt in scenario_options}
            scenario_display_options = sorted(scenario_display_map.keys())
        else:
            scenario_display_map = None
            scenario_display_options = None
        
        # Radial Chart용 시나리오 선택
        if 'Scenario_ID' in self.results_df.columns:
            selected_scenarios_display = st.multiselect(
                "Select Scenarios for Radial Chart",
                options=scenario_display_options,
                default=scenario_display_options[:min(5, len(scenario_display_options))] if len(scenario_display_options) > 0 else [],
                key="radial_chart_scenario_select"
            )
            
            # 선택한 표시 형식을 원래 시나리오 ID로 변환
            selected_scenarios = [scenario_display_map[disp] for disp in selected_scenarios_display]
        else:
            selected_scenarios = st.multiselect(
                "Select Scenarios for Radial Chart",
                options=scenario_options,
                default=scenario_options[:min(5, len(scenario_options))] if len(scenario_options) > 0 else [],
                key="radial_chart_scenario_select"
            )
        
        # Numeric 컬럼 선택 (Multiselect)
        selected_radial_cols = st.multiselect(
            "Select Numeric Columns",
            options=numeric_cols,
            default=numeric_cols[:min(7, len(numeric_cols))] if len(numeric_cols) > 0 else [],
            key="radial_chart_column_select"
        )
        
        # 방사형 차트 생성
        if len(selected_scenarios) > 0 and len(selected_radial_cols) > 0:
            # 선택한 시나리오의 데이터 필터링
            filtered_df = self.results_df.loc[selected_scenarios]
            
            # 전체 시나리오에서 각 컬럼의 Min과 Max 계산
            col_min_max = {}
            for col in selected_radial_cols:
                col_min_max[col] = {
                    'min': self.results_df[col].min(),
                    'max': self.results_df[col].max()
                }
            
            # 좌/우 컬럼 구성: 좌=Actual, 우=Normalized
            col_left, col_right = st.columns(2)
            # 공통 팔레트
            colors = (
                px.colors.qualitative.Dark24
                + px.colors.qualitative.Set1
                + px.colors.qualitative.Bold
                + px.colors.qualitative.D3
            )
            # 실제값 플롯
            with col_left:
                fig_actual = go.Figure()
                # 전체 축 범위 (모든 선택 컬럼 전역 Min/Max)
                global_min = min([col_min_max[col]['min'] for col in selected_radial_cols])
                global_max = max([col_min_max[col]['max'] for col in selected_radial_cols])
                if global_max == global_min:
                    global_max = global_min + 1
                for idx, scenario in enumerate(selected_scenarios):
                    scenario_data = filtered_df.loc[scenario]
                    actual_values = [scenario_data[col] for col in selected_radial_cols]
                    display_name = format_scenario_id(scenario) if 'Scenario_ID' in self.results_df.columns else f"Scenario_{int(scenario)+1:03d}" if str(scenario).isdigit() else f"{scenario}"
                    fig_actual.add_trace(go.Scatterpolar(
                        r=actual_values + [actual_values[0]],
                        theta=selected_radial_cols + [selected_radial_cols[0]],
                        fill=None,
                        name=display_name,
                        line=dict(color=colors[idx % len(colors)], width=3),
                        hovertemplate='<b>%{theta}</b><br>Value: %{r}<extra></extra>'
                    ))
                fig_actual.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[global_min, global_max]
                        )
                    ),
                    showlegend=True,
                    title='Radial Chart (Actual)',
                    height=800
                )
                st.plotly_chart(fig_actual, use_container_width=True)
                
            # 정규화 플롯
            with col_right:
                fig_norm = go.Figure()
                for idx, scenario in enumerate(selected_scenarios):
                    scenario_data = filtered_df.loc[scenario]
                    norm_values = []
                    for col in selected_radial_cols:
                        val = scenario_data[col]
                        col_min = col_min_max[col]['min']
                        col_max = col_min_max[col]['max']
                        if col_max == col_min:
                            normalized_val = 0.5
                        else:
                            normalized_val = (val - col_min) / (col_max - col_min)
                        norm_values.append(normalized_val)
                    display_name = format_scenario_id(scenario) if 'Scenario_ID' in self.results_df.columns else f"Scenario_{int(scenario)+1:03d}" if str(scenario).isdigit() else f"{scenario}"
                    fig_norm.add_trace(go.Scatterpolar(
                        r=norm_values + [norm_values[0]],
                        theta=selected_radial_cols + [selected_radial_cols[0]],
                        fill=None,
                        name=display_name,
                        line=dict(color=colors[idx % len(colors)], width=3),
                        hovertemplate='<b>%{theta}</b><br>Normalized: %{r:.2f}<extra></extra>'
                    ))
                fig_norm.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )
                    ),
                    showlegend=True,
                    title='Radial Chart (Normalized 0-1)',
                    height=800
                )
                st.plotly_chart(fig_norm, use_container_width=True)
            st.caption("Left: Actual values, Right: Normalized (0-1 by column across all scenarios).")
        elif len(selected_scenarios) == 0:
            st.info("Please select at least one scenario for radial chart")
        elif len(selected_radial_cols) == 0:
            st.info("Please select at least one numeric column for radial chart")
    
    def render_calculation_process(self):
        """Render Calculation Process section with scenario selection and visualization."""
        
        C1, C2, C3, C4 = st.columns(4)
        with C1:
            # 시나리오 선택 (results_df는 Scenario_ID를 index로 사용)
            scenario_options = self.results_df.index.tolist()
            if len(scenario_options) == 0:
                st.warning("No scenarios available")
                return
            
            # 시나리오 ID를 "Scenario_XXX" 형식으로 변환하는 함수
            # selectbox에서 표시할 형식으로 변환 (이미 "Scenario_001" 형식이므로 문자열 정렬로 충분)
            scenario_display_map = {format_scenario_id(opt): opt for opt in scenario_options}
            scenario_display_options = sorted(scenario_display_map.keys())  # 문자열 정렬로 충분
            
            selected_scenario_display = st.selectbox(
                "Select Scenario",
                options=scenario_display_options,
                key="calc_profile_scenario_select"
            )
            
            # 선택한 표시 형식을 원래 시나리오 ID로 변환
            selected_scenario = scenario_display_map[selected_scenario_display]
            
            # 선택한 시나리오의 터미널별 배치안 가져오기
            scenario_row = self.results_df.loc[selected_scenario]

        with C2:

            # Calculation Process: Relocation Settings에서 선택된 agg만 사용
            agg_options = list(getattr(self, "selected_agg_cols", None) or [self.selected_agg_col] if self.selected_agg_col else [])
            if len(agg_options) == 0:
                agg_options = [self.selected_agg_col] if self.selected_agg_col is not None else []
            calc_default = getattr(self, "calc_selected_agg_col", self.selected_agg_col)
            default_idx = agg_options.index(calc_default) if calc_default in agg_options else 0
            self.calc_selected_agg_col = st.selectbox(
                "**Aggregate base (for Calculation Process)**",
                options=agg_options,
                index=default_idx,
                key="calc_profile_agg_base",
                help="This does not change existing results; it only controls the profile charts.",
            )
            
            # 터미널별 배치안 추출
            terminal_assignments = {}
            for loc_name in self.loc_names:
                if loc_name in scenario_row.index:
                    units = scenario_row[loc_name]
                    if isinstance(units, list):
                        terminal_assignments[loc_name] = units
                    elif pd.notna(units) and units != '':
                        # 문자열로 저장된 리스트인 경우 처리
                        if isinstance(units, str) and units.startswith('[') and units.endswith(']'):
                            try:
                                import ast
                                terminal_assignments[loc_name] = ast.literal_eval(units)
                            except:
                                terminal_assignments[loc_name] = [units]
                        else:
                            terminal_assignments[loc_name] = [units] if not isinstance(units, list) else units
                    else:
                        terminal_assignments[loc_name] = []
                else:
                    terminal_assignments[loc_name] = []
        with C3:
            # 터미널 선택
            available_terminals = [loc for loc in self.loc_names if len(terminal_assignments.get(loc, [])) > 0]
            if len(available_terminals) == 0:
                st.warning("No terminals with assigned units in selected scenario")
                return
            
            selected_terminal = st.selectbox(
                "Select Terminal",
                options=available_terminals,
                key="calc_profile_terminal_select"
            )
        
        with C4:
            # Dep/Arr 선택
            selected_io = st.selectbox(
                "Select Departure or Arrival",
                options=["Dep", "Arr", "Both"],
                key="calc_profile_io_select"
            )
        
        # Field descriptions for LLM: explain what each key means (included in JSON so GPT/Gemini can interpret the export)
        FIELD_DESCRIPTIONS = {
            "overview": (
                "This JSON is a scenario detail export from a relocation/capacity analysis. "
                "Each scenario is one assignment of 'units' (e.g. airlines or carriers) to terminals. "
                "The data describes time-based demand (e.g. passengers or seats per time slot) for the selected terminal and Dep/Arr, "
                "and is intended for comparison across scenarios (e.g. via GPT or Gemini)."
            ),
            "meta": {
                "scenario_id": "Unique scenario identifier (e.g. Scenario_001).",
                "terminal": "Terminal/location name for which the following time series and profiles are computed.",
                "dep_arr": "Departure only (Dep), Arrival only (Arr), or Both (combined).",
                "aggregate_base": "Metric used for counts: e.g. total_seat_count, total_pax, movement.",
                "date_range": "Analysis period start and end dates.",
                "time_unit_minutes": "Length of each time slot in minutes (e.g. 60 = hourly).",
                "terminal_assignments": "For this scenario, which units are assigned to which terminal (location -> list of unit IDs).",
                "default_assignments": "Optional. Fixed/default units per terminal before relocation (location -> list of unit IDs). Same as Relocation Settings 'fixed units'.",
                "selected_units_at_terminal": "Units at the selected terminal; the data below refers to these units only.",
            },
            "scenario_result_table_row": "One row from the Scenario Result Table (Basic Info > 시나리오 결과 테이블): all metrics and terminal assignments for this scenario (e.g. Dist_Dep_95%, Dist_Arr_90%, rank columns, per-terminal unit lists). Same structure as the table visible in Basic Info.",
            "statistics_table": "Full Statistics table (Basic Info > 시나리오 결과 테이블): all scenarios as rows with all columns (Scenario_ID, Dist_Dep_97.5%, per-terminal metrics, ranks, etc.). Use for cross-scenario comparison.",
            "time_series": {
                "time_index": "Ordered list of time slot labels (e.g. timestamps). Each position corresponds to one slot.",
                "values": "Demand value (e.g. passengers or seats) per time slot, same order as time_index.",
                "capacity": "User-defined capacity threshold. Slots with value >= capacity are considered overloaded.",
                "total_sum": "Sum of all values across time slots (total volume in the period).",
                "excess_over_capacity_sum": "Sum of (value - capacity) for every slot where value > capacity. Total 'overflow' volume.",
                "excess_as_percent_of_total": "Percentage of total volume that exceeds capacity (excess_over_capacity_sum / total_sum * 100).",
                "total_delay_hours_capacity_constraint": "Total delay hours from capacity constraint. Cascading overflow: overflow from each slot carries to next; each overflowed unit contributes 1 slot duration (e.g. 1 hour) to total delay.",
                "daily_hourly_pivot": "Table: rows = time-of-day (e.g. 00:00–23:59), columns = dates; cell = demand in that hour on that date.",
                "stacked_pivot": "Table: rows = time slots, columns = groups (e.g. by unit); cell = demand by that group in that slot. Used for stacked bar chart.",
                "summary_by_group": "Per-group totals and percentage of total volume (from stacked_pivot).",
            },
            "sorted_values": {
                "statistics": "Percentiles and summary: max, 97.5%, 95%, 90%, 85%, 80%, median, mean of the time series values.",
                "graph_data": "Data for the 'Sorted Values' chart: x_index = rank (0=highest), y_sorted_values = demand values sorted descending.",
                "design_criteria": "Selected design basis: Max or a percentile (e.g. 95%) used as the design demand.",
                "design_criteria_value": "Numeric value of the selected design criteria (e.g. 95th percentile value).",
            },
            "cumulative": {
                "min_baseline": "Optional vertical shift so cumulative curve does not go below this value (display only).",
                "time_index": "Time slot labels for cumulative series (same length as arrays below).",
                "departure_per_slot": "Departure demand per time slot (same order as time_index).",
                "arrival_per_slot": "Arrival demand per time slot.",
                "cumulative_departure": "Cumulative sum of departures over time.",
                "cumulative_arrival": "Cumulative sum of arrivals over time.",
                "net_cumulative": "Cumulative arrivals minus cumulative departures (net accumulation).",
            },
        }
        
        # 선택한 터미널의 배치안에 해당하는 데이터 필터링
        selected_units = terminal_assignments[selected_terminal]
        if len(selected_units) == 0:
            st.warning(f"No units assigned to {selected_terminal} in selected scenario")
            return
        
        mask = self.df_filtered[self.relocation_unit].isin(selected_units)
        filtered_data = self.df_filtered[mask].copy()
        
        if len(filtered_data) == 0:
            st.warning(f"No data found for selected units in {selected_terminal}")
            return
        
        # Dep, Arr, 또는 Both에 따라 처리
        if selected_io == "Both":
            # Dep와 Arr 모두 처리하여 합계
            dep_category = f"departure {getattr(self, 'calc_selected_agg_col', self.selected_agg_col)}"
            arr_category = f"arrival {getattr(self, 'calc_selected_agg_col', self.selected_agg_col)}"
            summary_extra_cols = ["dep/arr_airport", "A/C Code", "International/Domestic"]
            keep_for_summary = [self.relocation_unit] + [c for c in summary_extra_cols if c in filtered_data.columns]
            
            # Departure 데이터 처리
            df_proc_dep = self.process_passenger_data(filtered_data, dep_category, keep_cols=keep_for_summary)
            # Arrival 데이터 처리
            df_proc_arr = self.process_passenger_data(filtered_data, arr_category, keep_cols=keep_for_summary)
            
            if len(df_proc_dep) == 0 and len(df_proc_arr) == 0:
                st.warning(f"No processed data available for {selected_terminal} - Both")
                return
            
            # 시간별 집계 (Dep와 Arr 각각)
            count_df_dep, _ = make_count_df(df_proc_dep, self.start_date, self.end_date, 'SHOW', self.relocation_unit, buffer_day=False, freq_min=self.unit_min) if len(df_proc_dep) > 0 else (pd.DataFrame(), None)
            count_df_arr, _ = make_count_df(df_proc_arr, self.start_date, self.end_date, 'SHOW', self.relocation_unit, buffer_day=False, freq_min=self.unit_min) if len(df_proc_arr) > 0 else (pd.DataFrame(), None)
            
            # Dep와 Arr 데이터 합치기
            if len(count_df_dep) > 0 and len(count_df_arr) > 0:
                # 두 DataFrame을 합치고 시간별로 합계
                count_df_combined = pd.concat([count_df_dep, count_df_arr], ignore_index=True)
                count_df = count_df_combined.groupby(['Time', self.relocation_unit])['index'].sum().reset_index()
            elif len(count_df_dep) > 0:
                count_df = count_df_dep.copy()
            elif len(count_df_arr) > 0:
                count_df = count_df_arr.copy()
            else:
                st.warning(f"No data available for {selected_terminal} - Both")
                return
            
            # df_proc도 합치기 (stacked bar chart용)
            if len(df_proc_dep) > 0 and len(df_proc_arr) > 0:
                df_proc = pd.concat([df_proc_dep, df_proc_arr], ignore_index=True)
            elif len(df_proc_dep) > 0:
                df_proc = df_proc_dep.copy()
            else:
                df_proc = df_proc_arr.copy()
        else:
            # Dep 또는 Arr만 처리
            io_label = "departure" if selected_io == "Dep" else "arrival"
            agg_key = getattr(self, "calc_selected_agg_col", self.selected_agg_col)
            category = f"{io_label} {agg_key}"
            summary_extra_cols = ["dep/arr_airport", "A/C Code", "International/Domestic"]
            keep_for_summary = [self.relocation_unit] + [c for c in summary_extra_cols if c in filtered_data.columns]
            
            # 승객 데이터 처리
            df_proc = self.process_passenger_data(filtered_data, category, keep_cols=keep_for_summary)
            
            if len(df_proc) == 0:
                st.warning(f"No processed data available for {selected_terminal} - {selected_io}")
                return
            
            # 시간별 집계
            count_df, _ = make_count_df(df_proc, self.start_date, self.end_date, 'SHOW', self.relocation_unit, buffer_day=False, freq_min=self.unit_min)
            
            if len(count_df) == 0:
                st.warning(f"No data available for {selected_terminal} - {selected_io}")
                return
        
        # 시간별 합계 계산
        time_series = count_df.groupby("Time")["index"].sum().sort_index()

        # JSON export payload for scenario detail (for GPT/Gemini comparison reports)
        def _to_json_serializable(obj):
            if hasattr(obj, "tolist"):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, pd.Series):
                return {str(k): _to_json_serializable(v) for k, v in obj.items()}
            if isinstance(obj, pd.DataFrame):
                return obj.astype(object).where(pd.notnull(obj), None).to_dict(orient="split")
            if isinstance(obj, dict):
                return {k: _to_json_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_to_json_serializable(x) for x in obj]
            if isinstance(obj, (date, datetime)):
                return str(obj)
            if isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            if pd.isna(obj):
                return None
            return obj

        # One row from the Scenario Result Table (기본정보 > 시나리오 결과 테이블) for this scenario — all metrics and terminal assignments
        scenario_result_row = self.results_df.loc[selected_scenario]
        scenario_export = {
            "_usage": "Scenario detail export for comparison reports. Upload this JSON with other scenario JSONs to GPT or Gemini to generate a comparison report.",
            "field_descriptions": FIELD_DESCRIPTIONS,
            "meta": {
                "scenario_id": format_scenario_id(selected_scenario) if hasattr(self.results_df, "columns") and "Scenario_ID" in self.results_df.columns else str(selected_scenario),
                "terminal": selected_terminal,
                "dep_arr": selected_io,
                "aggregate_base": getattr(self, "calc_selected_agg_col", self.selected_agg_col),
                "date_range": {"start": str(self.start_date), "end": str(self.end_date)},
                "time_unit_minutes": self.unit_min,
                "terminal_assignments": {loc: list(units) for loc, units in terminal_assignments.items()},
                "default_assignments": getattr(self, "fixed_per_loc", None) and {loc: list(units) for loc, units in self.fixed_per_loc.items()} or None,
                "selected_units_at_terminal": list(selected_units),
            },
            "scenario_result_table_row": _to_json_serializable(scenario_result_row.to_dict()),
            "statistics_table": _to_json_serializable(
                self.results_df.reset_index().astype(object).where(pd.notnull(self.results_df.reset_index()), None).to_dict(orient="records")
            ),
            "description": (
                f"Scenario detail: Terminal={selected_terminal}, Dep/Arr={selected_io}, "
                f"Aggregate={getattr(self, 'calc_selected_agg_col', self.selected_agg_col)}. "
                f"Time series and cumulative profiles for units assigned to this terminal in this scenario."
            ),
            "time_series": {
                "time_index": [str(t) for t in time_series.index],
                "values": _to_json_serializable(time_series.values),
            },
        }

        time_series_tab, sorted_values_tab, cumulative_tab = st.tabs(["Time Series", "Sorted Values", "Cumulative"])



        with time_series_tab:
            # Capacity 입력창
            capacity = st.number_input(
                "**Capacity**",
                min_value=0,
                value=100,
                step=1,
                key="capacity_input",
                help="이 값 이상인 셀은 글씨색이 빨간색으로 표시됩니다."
            )
            
            # Capacity 초과분 합계 및 비중 계산 (time_series: 시간별 합계)
            total_sum = float(time_series.sum())
            excess_over_capacity = (time_series - capacity).clip(lower=0)
            excess_sum = float(excess_over_capacity.sum())
            excess_share_pct = (excess_sum / total_sum * 100) if total_sum > 0 else 0.0

            # 용량 제약 지연시간 계산 (캐스케이딩 오버플로우)
            total_delay_hours, overflow_per_slot = compute_capacity_constraint_delay_hours(
                time_series.values,
                capacity,
                slot_duration_minutes=self.unit_min,
                return_overflow_per_slot=True,
            )
            # 평균/최대/95% 지연시간
            throughput = total_sum
            avg_delay_hours = (total_delay_hours / throughput) if throughput > 0 else 0.0
            max_delay_hours, p95_delay_hours = _delay_stats_from_overflow(
                overflow_per_slot, capacity, self.unit_min
            )
            avg_seat_val = _compute_avg_seat(filtered_data, selected_io)

            # KPI 카드 표시
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            with kpi1:
                st.metric(
                    label="Throughput",
                    value=f"{total_sum:,.0f}",
                    help="단순합계: 전체 기간 동안 모든 시간대 수요값의 합계."
                )
            with kpi2:
                st.metric(
                    label="Total Excess over Capacity",
                    value=f"{excess_sum:,.0f}",
                    help="Sum of (value − Capacity) for all time slots where value > Capacity."
                )
            with kpi3:
                st.metric(
                    label="Excess as % of Total",
                    value=f"{excess_share_pct:.2f}%",
                    help="Share of total volume that exceeds Capacity (excess total / total volume × 100)."
                )
            with kpi4:
                st.metric(
                    label="Total Delay Hours (Capacity)",
                    value=f"{int(total_delay_hours):,}" if total_delay_hours == int(total_delay_hours) else f"{total_delay_hours:,.2f}",
                    help="총 지연시간(시간). 용량 초과로 다음 시간대로 넘어가는 분이 누적되어 지연된 시간의 합계. "
                         "예: 05시 2편 overflow + 06시 5편 overflow → 7 delay-hours (1h slot 기준)."
                )

            # 평균/최대/95% 지연시간 카드
            kpi5, kpi6, kpi7, kpi8 = st.columns(4)
            with kpi5:
                st.metric(
                    label="Avg Delay (throughput당)",
                    value=_fmt_delay_hours_min(avg_delay_hours),
                    help="Total Delay Hours ÷ Throughput. 단위당 평균 지연시간."
                )
            with kpi6:
                st.metric(
                    label="Max Delay",
                    value=_fmt_delay_hours_min(max_delay_hours),
                    help="FIFO 가정 하에 가장 오래 대기한 단위의 지연시간."
                )
            with kpi7:
                st.metric(
                    label="95th Percentile Delay",
                    value=_fmt_delay_hours_min(p95_delay_hours),
                    help="지연 단위 중 95%가 이 값 이하로 대기한 시간."
                )
            # Avg seat KPI (aggregate base 상관없이 전체좌석수/전체항공편)
            avg_seat_val = _compute_avg_seat(filtered_data, selected_io)
            if avg_seat_val is not None:
                with kpi8:
                    st.metric(
                        label="Avg Seat",
                        value=f"{avg_seat_val:,.1f}",
                        help="전체좌석수합 ÷ 전체항공편합. aggregate base와 무관하게 항상 계산."
                    )
            
            # Export: Time Series section
            scenario_export["time_series"]["capacity"] = int(capacity)
            scenario_export["time_series"]["total_sum"] = total_sum
            scenario_export["time_series"]["excess_over_capacity_sum"] = excess_sum
            scenario_export["time_series"]["excess_as_percent_of_total"] = round(excess_share_pct, 2)
            _dh = total_delay_hours
            scenario_export["time_series"]["total_delay_hours_capacity_constraint"] = int(_dh) if _dh == int(_dh) else round(_dh, 2)
            if avg_seat_val is not None:
                scenario_export["time_series"]["avg_seat"] = avg_seat_val
            
            # 날짜와 시간 추출
            count_df['date'] = pd.to_datetime(count_df['Time']).dt.date
            count_df['time_of_day'] = pd.to_datetime(count_df['Time']).dt.strftime('%H:%M')
            count_df['day_of_week'] = pd.to_datetime(count_df['Time']).dt.day_name()
            
            # 요일별 시간대별 집계 (단위별 합계)
            daily_hourly_table = count_df.groupby(['date', 'day_of_week', 'time_of_day'])[['index']].sum().reset_index()
            
            # 시간대를 정렬하기 위한 변환 (HH:MM 문자열을 시간으로 변환하여 정렬)
            daily_hourly_table['time_sort'] = pd.to_datetime(daily_hourly_table['time_of_day'], format='%H:%M').dt.time
            
            # 피벗 테이블 생성 (시간대를 행으로, 날짜를 열로)
            daily_hourly_pivot = daily_hourly_table.pivot_table(
                index='time_of_day',
                columns=['date', 'day_of_week'],
                values='index',
                aggfunc='sum',
                fill_value=0
            )
            
            # 시간대 인덱스를 시간 순서대로 정렬
            time_index_sorted = sorted(daily_hourly_pivot.index, key=lambda x: pd.to_datetime(x, format='%H:%M').time())
            daily_hourly_pivot = daily_hourly_pivot.reindex(time_index_sorted)
            
            # 컬럼명을 날짜와 요일로 합치기
            daily_hourly_pivot.columns = [f"{date} ({day})" for date, day in daily_hourly_pivot.columns]
            
            # 스타일링: Capacity 이상인 셀은 글씨색만 빨간색으로 표시 (배경은 채우지 않음)
            def highlight_capacity(val):
                """Capacity 이상인 값에 글씨색 빨간색 적용"""
                if pd.notna(val) and val >= capacity:
                    return 'color: #e53935'
                return ''
            
            styled_df = daily_hourly_pivot.style.applymap(highlight_capacity).format("{:.0f}")
            
            # 두 개의 컬럼으로 나란히 배치
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### 📋 요일별 시간대별 데이터 테이블")
                # 전체 시간대가 보이도록 높이를 충분히 크게 설정
                num_rows = len(daily_hourly_pivot)
                table_height = max(400, min(num_rows * 35 + 150, 1000))
                st.dataframe(styled_df, use_container_width=True, height=table_height)
            
            with col2:
                st.markdown("##### 📊 요일별 시간대별 막대 그래프 (00:00~23:59)")
                
                # 날짜별로 그룹화하여 막대 그래프 생성 (stack 되지 않게)
                fig2 = go.Figure()
                
                # 각 날짜별로 막대 추가
                unique_dates = sorted(daily_hourly_table['date'].unique())
                colors = px.colors.qualitative.Set3  # 색상 팔레트
                
                for idx, date_val in enumerate(unique_dates):
                    date_data = daily_hourly_table[daily_hourly_table['date'] == date_val].copy()
                    date_data = date_data.sort_values('time_sort')
                    
                    day_name = date_data['day_of_week'].iloc[0]
                    date_str = f"{date_val} ({day_name})"
                    
                    fig2.add_trace(go.Bar(
                        x=date_data['time_of_day'],
                        y=date_data['index'],
                        name=date_str,
                        marker=dict(color=colors[idx % len(colors)])
                    ))
                
                # Capacity 값을 빨간색 점선으로 추가
                fig2.add_hline(
                    y=capacity,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Capacity: {capacity}",
                    annotation_position="right",
                    annotation=dict(
                        font_size=12,
                        font_color="red",
                        bgcolor="white",
                        bordercolor="red",
                        borderwidth=1
                    )
                )
                
                # 모든 고유한 시간대를 가져와서 정렬
                all_times = sorted(daily_hourly_table['time_of_day'].unique(), 
                                 key=lambda x: pd.to_datetime(x, format='%H:%M').time())
                
                # x축에 표시할 시간대 설정 (모든 시간대 또는 적절한 간격으로)
                if len(all_times) > 24:
                    tick_interval = max(1, len(all_times) // 12)
                    tickvals = all_times[::tick_interval]
                else:
                    tickvals = all_times
                
                fig2.update_layout(
                    title=f'Daily Time Series by Date: {selected_terminal} - {selected_io}',
                    xaxis_title='Time of Day (00:00~23:59)',
                    yaxis_title=f'{self.selected_agg_col.replace("_", " ").title()}',
                    height=750,
                    barmode='group',  # stack 되지 않게 그룹 모드
                    hovermode='x unified',
                    hoverlabel=dict(align='right', bgcolor='white', bordercolor='black', font_size=12),
                    legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
                    xaxis=dict(
                        type='category',  # 카테고리 타입으로 설정
                        tickmode='array',
                        tickvals=tickvals,  # 표시할 시간대 지정
                        tickangle=45,
                        tickfont=dict(size=10)  # 폰트 크기 조정
                    )
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # 아래에 stacked bar 그래프 추가
            st.markdown("#### 📊 시간대별 Group별 Stacked Bar Chart")
            
            # df_proc에서 사용 가능한 그룹 컬럼 확인 (전역 BASE_COL_OPTIONS 기반)
            base_group_cols = BASE_COL_OPTIONS + ["Custom Group", "International/Domestic", "A/C Code"]
            available_group_cols = [c for c in base_group_cols if c in df_proc.columns]
            
            # 기본값으로 relocation_unit 사용
            default_group_col = self.relocation_unit if self.relocation_unit in available_group_cols else (available_group_cols[0] if available_group_cols else None)
            
            if default_group_col is None:
                st.warning("No available group columns found for stacked bar chart")
                return
            
            # 그룹 컬럼 선택
            selected_group_col = st.selectbox(
                "**Group by**",
                options=available_group_cols,
                index=available_group_cols.index(default_group_col) if default_group_col in available_group_cols else 0,
                key="calc_profile_stacked_bar_group_select"
            )
            
            # 선택한 그룹 컬럼으로 count_df 재집계
            if selected_group_col != self.relocation_unit:
                # 선택한 컬럼이 df_proc에 있는지 확인하고, count_df를 재집계
                if selected_group_col in df_proc.columns:
                    count_df_grouped, _ = make_count_df(df_proc, self.start_date, self.end_date, 'SHOW', selected_group_col, buffer_day=False, freq_min=self.unit_min)
                else:
                    st.warning(f"Selected group column '{selected_group_col}' not found in processed data")
                    return
            else:
                count_df_grouped = count_df.copy()
            
            # count_df를 시간대별, 선택한 그룹별로 pivot
            stacked_pivot = count_df_grouped.pivot_table(
                index='Time',
                columns=selected_group_col,
                values='index',
                aggfunc='sum',
                fill_value=0
            )
            
            # Stacked bar chart 생성
            fig3 = go.Figure()
            
            # 각 그룹별로 막대 추가
            for group_val in stacked_pivot.columns:
                fig3.add_trace(go.Bar(
                    x=stacked_pivot.index,
                    y=stacked_pivot[group_val],
                    name=str(group_val),
                    hovertemplate=f'<b>{group_val}</b><br>Time: %{{x}}<br>Value: %{{y}}<extra></extra>'
                ))
            
            # Capacity 값을 빨간색 점선으로 추가
            fig3.add_hline(
                y=capacity,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Capacity: {capacity}",
                annotation_position="right",
                annotation=dict(
                    font_size=12,
                    font_color="red",
                    bgcolor="white",
                    bordercolor="red",
                    borderwidth=1
                )
            )
            
            fig3.update_layout(
                title=f'Time Series Stacked by {selected_group_col}: {selected_terminal} - {selected_io}',
                xaxis_title='Time',
                yaxis_title=f'{self.selected_agg_col.replace("_", " ").title()}',
                height=750,
                barmode='stack',  # stacked bar 모드
                hovermode='x unified',
                hoverlabel=dict(align='right', bgcolor='white', bordercolor='black', font_size=12),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig3, use_container_width=True)
            
            # 각 그룹별 합계와 비중 계산
            group_sums = stacked_pivot.sum()
            total_sum = group_sums.sum()
            group_percentages = (group_sums / total_sum * 100) if total_sum > 0 else group_sums * 0
            
            # 합계와 비중을 DataFrame으로 표시 (선택한 그룹 기준)
            summary_df = pd.DataFrame({
                'Sum': group_sums,
                'Percentage (%)': group_percentages.round(2)
            })
            summary_df = summary_df.sort_values('Sum', ascending=False)
            
            # Summary by Group 탭용: 여러 그룹 기준별 summary 수집
            summary_by_group_tabs = []
            for tab_label, group_col in SUMMARY_BY_GROUP_TABS:
                if group_col not in df_proc.columns:
                    continue
                try:
                    _cnt, _ = make_count_df(df_proc, self.start_date, self.end_date, 'SHOW', group_col, buffer_day=False, freq_min=self.unit_min)
                    if len(_cnt) == 0:
                        continue
                    _piv = _cnt.pivot_table(index='Time', columns=group_col, values='index', aggfunc='sum', fill_value=0)
                    _gs = _piv.sum()
                    _tot = _gs.sum()
                    _pct = (_gs / _tot * 100) if _tot > 0 else _gs * 0
                    _sf = pd.DataFrame({'Sum': _gs, 'Percentage (%)': _pct.round(2)}).sort_values('Sum', ascending=False)
                    summary_by_group_tabs.append({"label": tab_label, "data": _sf.astype(object).where(pd.notnull(_sf), None).to_dict(orient="split")})
                except Exception:
                    pass
            
            # Export: daily hourly pivot and stacked summary (pass through _to_json_serializable so index/columns with pd.Timestamp become strings)
            try:
                scenario_export["time_series"]["daily_hourly_pivot"] = _to_json_serializable(daily_hourly_pivot.astype(object).where(pd.notnull(daily_hourly_pivot), None).to_dict(orient="split"))
                scenario_export["time_series"]["stacked_pivot_group_col"] = selected_group_col
                scenario_export["time_series"]["stacked_pivot"] = _to_json_serializable(stacked_pivot.astype(object).where(pd.notnull(stacked_pivot), None).to_dict(orient="split"))
                if summary_by_group_tabs:
                    scenario_export["time_series"]["summary_by_group"] = {"tabs": _to_json_serializable(summary_by_group_tabs)}
                else:
                    scenario_export["time_series"]["summary_by_group"] = _to_json_serializable(summary_df.astype(object).where(pd.notnull(summary_df), None).to_dict(orient="split"))
            except Exception as _e:
                pass
            
            st.markdown("#### 📊 Summary by Group")
            if summary_by_group_tabs:
                _sbg_tabs = st.tabs([t["label"] for t in summary_by_group_tabs])
                for tab_widget, tab_data in zip(_sbg_tabs, summary_by_group_tabs):
                    with tab_widget:
                        d = tab_data["data"]
                        _sf = pd.DataFrame(d.get("data", []), columns=d.get("columns", []), index=d.get("index", []))
                        st.dataframe(_sf, use_container_width=True)
            else:
                st.dataframe(summary_df, use_container_width=True)
            
            st.markdown("#### 📊 Time Series Data")
            st.write(stacked_pivot)

        
        with sorted_values_tab:
            # 두 번째 그래프: 내림차순 정렬 그래프 (Max가 가장 왼쪽에)
            st.markdown(f"#### 📊 Sorted Values: {selected_terminal} - {selected_io}")
            sorted_values = time_series.values.copy()
            # 내림차순으로 정렬하여 Max가 가장 왼쪽(인덱스 0)에 오도록
            sorted_values = np.sort(sorted_values)[::-1]
            
            # 통계값 계산: 해당 퍼센타일 순위의 실제 값 사용 (보간 없음)
            max_val = float(sorted_values[0]) if len(sorted_values) > 0 else 0.0
            _, p975 = _percentile_rank_value(sorted_values, 97.5)
            _, p95 = _percentile_rank_value(sorted_values, 95)
            _, p90 = _percentile_rank_value(sorted_values, 90)
            _, p85 = _percentile_rank_value(sorted_values, 85)
            _, p80 = _percentile_rank_value(sorted_values, 80)
            _, median_val = _percentile_rank_value(sorted_values, 50)
            mean_val = float(np.mean(sorted_values)) if len(sorted_values) > 0 else 0.0
            
            # Export: Sorted Values statistics + full graph data (blue curve: x = index, y = sorted value)
            scenario_export["sorted_values"] = {
                "statistics": {
                    "max": float(max_val),
                    "97.5%": float(p975),
                    "95%": float(p95),
                    "90%": float(p90),
                    "85%": float(p85),
                    "80%": float(p80),
                    "median": float(median_val),
                    "mean": float(mean_val),
                },
                "sorted_value_count": len(sorted_values),
                "graph_data": {
                    "x_index": list(range(len(sorted_values))),
                    "y_sorted_values": _to_json_serializable(sorted_values),
                },
            }
            
            # 레이아웃: 그래프 70%, 선택 박스 30%
            col_graph, col_settings = st.columns([0.7, 0.3])
            
            with col_graph:
                fig2 = go.Figure()
                # x축을 0부터 시작하여 왼쪽에서 오른쪽으로 그려지도록
                x_values = np.arange(len(sorted_values))
                
                fig2.add_trace(go.Scatter(
                    x=x_values,
                    y=sorted_values,
                    mode='lines+markers',
                    name='Sorted Values',
                    line=dict(width=2, color='blue'),
                    marker=dict(size=3)
                ))
                
                # 통계값 표시: 해당 순위의 인덱스·값 사용 (보간 없음)
                stats_labels = ['Max', '97.5%', '95%', '90%', 'Median', 'Mean']
                stats_colors = ['red', 'orange', 'yellow', 'green', 'purple', 'cyan']
                idx_max = 0
                idx_975, _ = _percentile_rank_value(sorted_values, 97.5)
                idx_95, _ = _percentile_rank_value(sorted_values, 95)
                idx_90, _ = _percentile_rank_value(sorted_values, 90)
                idx_50, _ = _percentile_rank_value(sorted_values, 50)
                stats_indexed = [
                    (idx_max, max_val),
                    (idx_975, p975),
                    (idx_95, p95),
                    (idx_90, p90),
                    (idx_50, median_val),
                    (np.argmin(np.abs(sorted_values - mean_val)) if len(sorted_values) > 0 else 0, mean_val),
                ]
                for (stat_idx, stat_val), stat_label, stat_color in zip(stats_indexed, stats_labels, stats_colors):
                    # 라벨과 값을 함께 표시
                    fig2.add_trace(go.Scatter(
                        x=[stat_idx],
                        y=[stat_val],
                        mode='markers+text',
                        marker=dict(size=12, color=stat_color, symbol='diamond'),
                        text=[f'{stat_label}: {stat_val:.1f}'],
                        textposition='top center',
                        name=stat_label,
                        showlegend=True
                    ))
                
                # 설계기준값 선택 및 빨간 점선 표시
                design_criteria_map = {
                    'Max': max_val,
                    '97.5%': p975,
                    '95%': p95,
                    '90%': p90,
                    '85%': p85,
                    '80%': p80
                }
                
                # 설계기준값을 우측 컬럼에서 선택하도록 함 (아래에서 처리)
                # 여기서는 선택된 값을 사용하여 빨간 점선 추가
                selected_design_criteria = st.session_state.get('design_criteria_select', '95%')
                design_value = design_criteria_map.get(selected_design_criteria, p95)
                
                # 빨간 점선으로 설계기준값 표시
                fig2.add_trace(go.Scatter(
                    x=x_values,
                    y=[design_value] * len(x_values),
                    mode='lines',
                    name=f'Design Criteria ({selected_design_criteria})',
                    line=dict(width=2, color='red', dash='dash'),
                    showlegend=True
                ))
                
                fig2.update_layout(
                    title=f'Sorted Values with Statistics: {selected_terminal} - {selected_io}',
                    xaxis_title='Index (Sorted, Descending)',
                    yaxis_title=f'{self.selected_agg_col.replace("_", " ").title()}',
                    height=750,
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hoverlabel=dict(align='right', bgcolor='white', bordercolor='black', font_size=12)
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            with col_settings:
                st.markdown("#### ⚙️ Design Settings")
                
                # 설계기준값 선택
                design_criteria_options = ['Max', '97.5%', '95%', '90%', '85%', '80%']
                selected_design_criteria = st.selectbox(
                    "Select Design Criteria",
                    options=design_criteria_options,
                    index=design_criteria_options.index('95%'),
                    key='design_criteria_select'
                )
                
                design_criteria_map = {
                    'Max': max_val,
                    '97.5%': p975,
                    '95%': p95,
                    '90%': p90,
                    '85%': p85,
                    '80%': p80
                }
                design_value = design_criteria_map.get(selected_design_criteria, p95)

                # 대당 수용능력 입력
                capacity_per_unit = st.number_input(
                    "Capacity per Unit (unit / hour)",
                    min_value=0.1,
                    value=20.0,
                    step=1.0,
                    key='capacity_per_unit_input',
                    help="Enter the capacity per unit per hour"
                )
                
                # 필요한 시설 개수 계산
                if capacity_per_unit > 0:
                    required_facilities = design_value / capacity_per_unit
                    st.markdown("#### 📊 Calculation Result")
                    st.markdown(f"**Design Criteria Value:** {design_value:.1f}")
                    st.markdown(f"**Capacity per Unit:** {capacity_per_unit:.1f} unit / hour")
                    st.markdown(f"**Required Facilities:** {required_facilities:.1f} facilities")
                    st.info(f"**{design_value:.1f} / {capacity_per_unit:.1f} = {required_facilities:.1f} facilities required**")
                    # Export: design criteria only (facilities required 정보는 JSON에 포함하지 않음)
                    if "sorted_values" in scenario_export:
                        scenario_export["sorted_values"]["design_criteria"] = selected_design_criteria
                        scenario_export["sorted_values"]["design_criteria_value"] = float(design_value)


        with cumulative_tab:
            st.markdown("#### 📈 Cumulative profile")

            # User-configurable minimum baseline for cumulative profile (visual shift only)
            min_baseline = st.number_input(
                "**Minimum baseline (cumulative)**",
                value=0.0,
                step=1.0,
                key="cumulative_min_baseline",
                help="If the cumulative curve goes below this value, it will be shifted upward so that its minimum equals this baseline.",
            )

            # 항상 Dep/Arr 둘 다 계산해서 사용
            dep_category = f"departure {getattr(self, 'calc_selected_agg_col', self.selected_agg_col)}"
            arr_category = f"arrival {getattr(self, 'calc_selected_agg_col', self.selected_agg_col)}"
            df_dep = self.process_passenger_data(filtered_data, dep_category, keep_cols=[self.relocation_unit])
            df_arr = self.process_passenger_data(filtered_data, arr_category, keep_cols=[self.relocation_unit])

            if len(df_dep) == 0 and len(df_arr) == 0:
                st.info("No data for cumulative Dep/Arr.")
            else:
                dep_df, _ = make_count_df(
                    df_dep, self.start_date, self.end_date, "SHOW",
                    self.relocation_unit, buffer_day=False, freq_min=self.unit_min
                ) if len(df_dep) > 0 else (pd.DataFrame(), None)
                arr_df, _ = make_count_df(
                    df_arr, self.start_date, self.end_date, "SHOW",
                    self.relocation_unit, buffer_day=False, freq_min=self.unit_min
                ) if len(df_arr) > 0 else (pd.DataFrame(), None)

                # Relocation_Master에서 cumsum 1회만 계산 → JSON에 담고 Report는 그대로 사용
                all_index, dep_ts, arr_ts, cum_dep, cum_arr, net_cum = self._compute_cumulative_series(dep_df, arr_df)

                # 1) 상단 누적 그래프: Dep / Arr / Both에 따라 다른 시리즈 사용 (막대 그래프)
                fig_cum = go.Figure()

                # baseline shift 계산 (선택된 IO에 따라 한 개의 기준 시리즈만 사용)
                if selected_io == "Dep":
                    base_series = cum_dep
                elif selected_io == "Arr":
                    base_series = cum_arr
                else:
                    base_series = net_cum

                offset = 0.0
                if len(base_series) > 0:
                    cur_min = float(base_series.min())
                    if cur_min < min_baseline:
                        offset = float(min_baseline - cur_min)

                # Export: Cumulative section
                scenario_export["cumulative"] = {
                    "min_baseline": float(min_baseline),
                    "time_index": [str(t) for t in all_index],
                    "departure_per_slot": _to_json_serializable(dep_ts.reindex(all_index, fill_value=0).values),
                    "arrival_per_slot": _to_json_serializable(arr_ts.reindex(all_index, fill_value=0).values),
                    "cumulative_departure": _to_json_serializable((cum_dep + offset).values),
                    "cumulative_arrival": _to_json_serializable((cum_arr + offset).values),
                    "net_cumulative": _to_json_serializable((net_cum + offset).values),
                }

                if selected_io == "Dep":
                    fig_cum.add_trace(go.Bar(
                        x=all_index,
                        y=(cum_dep + offset).values,
                        name="Cumulative Departure",
                        marker_color="#EF553B",
                    ))
                    y_title = f"Cumulative {self.selected_agg_col.replace('_', ' ').title()} (Departure)"
                elif selected_io == "Arr":
                    fig_cum.add_trace(go.Bar(
                        x=all_index,
                        y=(cum_arr + offset).values,
                        name="Cumulative Arrival",
                        marker_color="#00CC96",
                    ))
                    y_title = f"Cumulative {self.selected_agg_col.replace('_', ' ').title()} (Arrival)"
                else:  # Both: 누적 Arr - 누적 Dep
                    fig_cum.add_trace(go.Bar(
                        x=all_index,
                        y=(net_cum + offset).values,
                        name="Cumulative Arrivals - Departures",
                        marker_color="#2ca02c",
                    ))
                    y_title = f"Net cumulative {self.selected_agg_col.replace('_', ' ').title()} (Arr - Dep)"

                # y축 라벨에 baseline shift 여부를 명시적으로 표시
                if offset != 0.0:
                    y_title = y_title + f" (shifted ≥ {min_baseline:.0f})"

                fig_cum.update_layout(
                    xaxis_title="Time",
                    yaxis_title=y_title,
                    height=600,
                    hovermode="x unified",
                    barmode="relative",
                    hoverlabel=dict(align='right', bgcolor='white', bordercolor='black', font_size=12),
                )
                st.plotly_chart(fig_cum, use_container_width=True)

                # 2) 슬롯별 Dep/Arr을 하나의 그래프에 (+ / -)로 표현
                st.markdown("#### 📊 Departure / Arrival counts per time slot (+ / -)")
                fig_da = go.Figure()
                fig_da.add_trace(go.Bar(
                    x=all_index,
                    y=dep_ts.values,
                    name="Departure per slot",
                    marker_color="#EF553B",
                ))
                fig_da.add_trace(go.Bar(
                    x=all_index,
                    y=-arr_ts.values,
                    name="Arrival per slot (negative)",
                    marker_color="#00CC96",
                ))
                fig_da.update_layout(
                    xaxis_title="Time",
                    yaxis_title=f"{self.selected_agg_col.replace('_', ' ').title()} per slot (+Dep / -Arr)",
                    height=525,
                    barmode="relative",  # 위/아래로 보이도록
                    hovermode="x unified",
                    hoverlabel=dict(align='right', bgcolor='white', bordercolor='black', font_size=12),
                    legend=dict(
                        x=0.99,
                        y=0.99,
                        xanchor="right",
                        yanchor="top",
                        bgcolor="rgba(255,255,255,0.6)",
                        bordercolor="rgba(0,0,0,0.2)",
                        borderwidth=1,
                    ),
                )
                st.plotly_chart(fig_da, use_container_width=True)

        # Download scenario detail as JSON (for GPT/Gemini comparison reports)
        st.markdown("---")
        st.markdown("#### 📥 Export scenario detail")
        export_desc = (
            "Choose which sections to include, then download. Use the file with GPT or Gemini to generate comparison reports across scenarios."
        )
        st.caption(export_desc)
        
        # Detailed data guide for LLM (shown in expander and included in JSON)
        DATA_GUIDE_FOR_LLM = """
# Data guide for JSON export (for LLM interpretation)

This JSON is a **scenario detail export** from a relocation/capacity analysis. Each **scenario** is one possible assignment of "units" (e.g. airlines, carriers, or flight groups) to terminals/locations. The file contains time-based demand metrics and profiles for **one scenario** so you can compare multiple scenario files.

---

## Top-level keys

- **meta**: Summary of which scenario, terminal, and options were selected. Use this to label the scenario in comparisons.
- **scenario_result_table_row**: The full row for this scenario from the "Scenario Result Table" (Basic Info tab). Contains all summary metrics (e.g. peak-load distances, ranks) and per-terminal unit lists. Compare these rows across scenario JSONs to see how metrics change by assignment.
- **time_series** (optional): Demand per time slot and related tables for the selected terminal and Dep/Arr. See below.
- **sorted_values** (optional): Percentiles and design-capacity calculation. See below.
- **cumulative** (optional): Cumulative departure/arrival series. See below.
- **field_descriptions**: Short definitions of each key. Use for reference when a key is unclear.
- **data_guide_for_llm**: This guide. Read it first when interpreting the JSON.

---

## meta (object)

- **scenario_id**: Unique ID for this scenario (e.g. "Scenario_001"). Use as the scenario name in reports.
- **terminal**: The terminal/location name for which time_series, sorted_values, and cumulative data are computed. Only data for units assigned to this terminal in this scenario is included.
- **dep_arr**: "Dep" = departure only, "Arr" = arrival only, "Both" = departure and arrival combined. All demand values in this file follow this choice.
- **aggregate_base**: The metric being counted (e.g. total_seat_count, total_pax, movement). All numeric values in time_series and cumulative are in this unit (e.g. passengers or seats).
- **date_range**: Start and end date of the analysis period. All time slots fall within this range.
- **time_unit_minutes**: Length of each time slot in minutes (e.g. 60 = hourly). Time series and cumulative arrays are in this resolution.
- **terminal_assignments**: For this scenario, which units are assigned to which terminal. Format: object with terminal name as key and list of unit IDs as value. Describes the full scenario assignment.
- **selected_units_at_terminal**: List of unit IDs at the selected terminal. The time_series, sorted_values, and cumulative data refer only to these units.

---

## scenario_result_table_row (object)

One row from the Scenario Result Table (Basic Info > 시나리오 결과 테이블). Keys and values match the table columns. Typical contents:

- **Scenario_ID**: Same as meta.scenario_id.
- **Dist_Dep_XX%**, **Dist_Arr_XX%**: Euclidean distance of peak-load metrics (e.g. 95th percentile) from origin; lower often means better spread of load.
- **Rank_*****: Rank of this scenario for that metric (e.g. 1 = best). Lower rank = better for peak-load metrics.
- **Per-terminal columns**: Each location name as key; value is the list of unit IDs assigned to that terminal in this scenario.
- Other numeric columns: Various peak percentiles (e.g. 95%, 90%) and totals by terminal. Units match aggregate_base (e.g. passengers or seats).

Use this object to compare scenarios by peak loads, ranks, and assignments without opening the time-series arrays.

---

## time_series (object, optional)

Demand over time for the selected terminal and Dep/Arr (and selected units only).

- **time_index**: List of time-slot labels (e.g. datetime strings). Same length as **values**. Position i corresponds to the i-th slot.
- **values**: Demand in each time slot (same order as time_index). Unit is aggregate_base (e.g. passengers or seats per slot).
- **capacity**: User-set capacity threshold. Slots where value ≥ capacity are considered overloaded.
- **total_sum**: Sum of all values (total volume in the period).
- **excess_over_capacity_sum**: Sum of (value − capacity) for every slot where value > capacity. Total "overflow" volume.
- **excess_as_percent_of_total**: (excess_over_capacity_sum / total_sum) × 100. Share of volume that exceeds capacity.
- **daily_hourly_pivot** (optional): Table: rows = time-of-day (e.g. "00:00"–"23:59"), columns = dates; cell = demand in that hour on that date. "split" format: keys "index", "columns", "data".
- **stacked_pivot** (optional): Table: rows = time slots, columns = groups (e.g. by unit); cell = demand by that group. "split" format. Used for stacked bar chart.
- **summary_by_group** (optional): Per-group totals and percentage of total volume. "split" format.

Interpretation: Higher total_sum or higher excess_as_percent_of_total at a given capacity means more congestion; compare across scenarios.

---

## sorted_values (object, optional)

Statistics and design-capacity calculation based on the same time series (sorted descending).

- **statistics**: Summary of the demand distribution: **max**, **97.5%**, **95%**, **90%**, **85%**, **80%** (percentiles), **median**, **mean**. Unit is aggregate_base. Use these to compare peak and average load across scenarios.
- **graph_data**: Data for the "Sorted Values" chart. **x_index**: rank (0 = highest demand slot). **y_sorted_values**: demand values sorted from highest to lowest. Same length as x_index.
- **design_criteria** (if present): Selected design basis, e.g. "95%" or "Max".
- **design_criteria_value** (if present): Numeric value of that basis (e.g. 95th percentile). Used as design demand.

Interpretation: Lower design_criteria_value usually means a better scenario for peak demand.

---

## cumulative (object, optional)

Cumulative departure and arrival over time (same terminal and units).

- **min_baseline**: Optional vertical shift applied so the cumulative curve does not go below this value (display only; does not change totals).
- **time_index**: Time-slot labels for the series below. Same length as the arrays.
- **departure_per_slot**: Demand in each time slot for departures only. Same order as time_index.
- **arrival_per_slot**: Demand in each time slot for arrivals only.
- **cumulative_departure**: Cumulative sum of departures over time (after optional shift).
- **cumulative_arrival**: Cumulative sum of arrivals over time (after optional shift).
- **net_cumulative**: Cumulative arrivals minus cumulative departures (net accumulation over time).

Interpretation: Steep slopes indicate high activity; compare slopes and net_cumulative across scenarios to discuss peak periods and net accumulation.

---

## How to use this file for comparison

1. Read **meta** and **scenario_result_table_row** to identify the scenario and its summary metrics and assignments.
2. Compare **scenario_result_table_row** across multiple JSONs (ranks, Dist_*, per-terminal lists).
3. If present, use **time_series** (total_sum, excess_*, capacity) and **sorted_values** (statistics, design_criteria_value) to compare demand level.
4. If present, use **cumulative** to compare time patterns (peaks, net accumulation) across scenarios.
5. Refer to **field_descriptions** or this **data_guide_for_llm** when a key or unit is unclear.
"""
        
        with st.expander("ℹ️ Data guide for export (for LLM)"):
            st.markdown(DATA_GUIDE_FOR_LLM)
        
        try:
            # Build export payload: 모든 섹션 기본 포함 (Time Series, Sorted Values, Cumulative, Analysis & Comparison)
            export_copy = {
                "_usage": scenario_export["_usage"],
                "data_guide_for_llm": DATA_GUIDE_FOR_LLM.strip(),
                "field_descriptions": {"overview": scenario_export["field_descriptions"]["overview"], "meta": scenario_export["field_descriptions"]["meta"]},
                "meta": scenario_export["meta"],
                "description": scenario_export["description"],
                "scenario_result_table_row": scenario_export["scenario_result_table_row"],
                "statistics_table": scenario_export.get("statistics_table"),
            }
            if scenario_export.get("statistics_table") is not None and "statistics_table" in scenario_export.get("field_descriptions", {}):
                export_copy["field_descriptions"]["statistics_table"] = scenario_export["field_descriptions"]["statistics_table"]
            if "time_series" in scenario_export:
                export_copy["time_series"] = scenario_export["time_series"]
                export_copy["field_descriptions"]["time_series"] = scenario_export["field_descriptions"]["time_series"]
            if "sorted_values" in scenario_export:
                export_copy["sorted_values"] = scenario_export["sorted_values"]
                export_copy["field_descriptions"]["sorted_values"] = scenario_export["field_descriptions"]["sorted_values"]
            if "cumulative" in scenario_export:
                export_copy["cumulative"] = scenario_export["cumulative"]
                export_copy["field_descriptions"]["cumulative"] = scenario_export["field_descriptions"]["cumulative"]
            if "scenario_result_table_row" in scenario_export["field_descriptions"]:
                export_copy["field_descriptions"]["scenario_result_table_row"] = scenario_export["field_descriptions"]["scenario_result_table_row"]
            # Analysis settings & Comparison (기본 포함)
            try:
                numeric_cols = self.get_numeric_cols()
                scenario_index_list = self.results_df.index.tolist()
                scenario_id_display = [format_scenario_id(ix) if "Scenario_ID" in self.results_df.columns else str(ix) for ix in scenario_index_list]
                export_copy["analysis_settings"] = _to_json_serializable({
                    "date_range": {"start": str(self.start_date), "end": str(self.end_date)},
                    "aggregate_base": getattr(self, "calc_selected_agg_col", self.selected_agg_col) or self.selected_agg_col,
                    "time_unit_minutes": self.unit_min,
                    "locations": list(self.loc_names),
                    "fixed_per_loc": dict(self.fixed_per_loc),
                    "moving_unit": list(self.moving_unit),
                    "dep_info": self.dep_info,
                    "arr_info": self.arr_info,
                    "is_sampled": getattr(self, "is_sampled", False),
                    "analysis_count": getattr(self, "analysis_count", None),
                    "total_combinations": getattr(self, "total_combinations", None),
                })
                col_stats_rows = []
                for ix in scenario_index_list:
                    row = {"scenario_id": format_scenario_id(ix) if "Scenario_ID" in self.results_df.columns else str(ix)}
                    for col in numeric_cols:
                        if col in self.results_df.columns:
                            val = self.results_df.loc[ix, col]
                            row[col] = float(val) if pd.notna(val) and np.issubdtype(type(val), np.number) else val
                    col_stats_rows.append(_to_json_serializable(row))
                col_min_max = {}
                for col in numeric_cols:
                    if col in self.results_df.columns:
                        ser = self.results_df[col]
                        try:
                            col_min_max[col] = {"min": float(ser.min()), "max": float(ser.max())}
                        except Exception:
                            col_min_max[col] = {"min": 0, "max": 0}
                radial_rows = []
                for ix in scenario_index_list:
                    vals = []
                    for col in numeric_cols:
                        if col in self.results_df.columns:
                            v = self.results_df.loc[ix, col]
                            vals.append(float(v) if pd.notna(v) and np.issubdtype(type(v), np.number) else 0)
                    radial_rows.append({
                        "scenario_id": format_scenario_id(ix) if "Scenario_ID" in self.results_df.columns else str(ix),
                        "values": _to_json_serializable(vals),
                    })
                export_copy["comparison"] = _to_json_serializable({
                    "column_statistics": {
                        "columns": numeric_cols,
                        "rows": col_stats_rows,
                    },
                    "radial_chart": {
                        "columns": numeric_cols,
                        "col_min_max": col_min_max,
                        "rows": radial_rows,
                    },
                })
            except Exception as _e:
                pass

            payload = json.dumps(export_copy, ensure_ascii=False, indent=2)
            meta = scenario_export.get("meta", {})
            scenario_id = (meta.get("scenario_id") or "scenario").replace(" ", "_")
            aggregate_base = (meta.get("aggregate_base") or "agg").replace(" ", "_")
            terminal = (meta.get("terminal") or "terminal").replace(" ", "_")
            dep_arr = (meta.get("dep_arr") or "Both").replace(" ", "_")
            capacity = scenario_export.get("time_series", {}).get("capacity", "na")
            if not isinstance(capacity, (int, float)):
                capacity = "na"
            file_name = f"{scenario_id}_{aggregate_base}_{terminal}_{dep_arr}_{capacity}.json"
            st.download_button(
                label="Download scenario detail (JSON)",
                data=payload,
                file_name=file_name,
                mime="application/json",
                key="scenario_detail_json_download",
            )
        except Exception as e:
            st.warning(f"Export not available: {e}")


def format_scenario_id(x):
    """시나리오 ID를 "Scenario_XXX" 형식으로 변환 (001부터 시작)"""
    x_str = str(x)
    if x_str.startswith('Scenario_'):
        # Scenario_XXX 형식에서 숫자 추출
        parts = x_str.split('_')
        if len(parts) > 1 and parts[1].isdigit():
            num = int(parts[1]) + 1  # 001부터 시작
            return f"Scenario_{num:03d}"
        return x_str
    elif '_' in x_str:
        parts = x_str.split('_')
        if len(parts) > 1 and parts[1].isdigit():
            num = int(parts[1]) + 1  # 001부터 시작
            return f"Scenario_{num:03d}"
    try:
        num = int(x_str) + 1  # start from 001 instead of 000
        return f"Scenario_{num:03d}"
    except:
        return x_str

def build_loc_assignments(
    loc_fixed: dict,
    moving_list: list,
    allowed_targets: list | None = None,
    allowed_targets_map: dict | None = None,
    unassigned_units_map: dict | None = None
) -> list:
    """Generate all assignments mappings for moving units across allowed LOCs.
    - If allowed_targets_map is provided, each unit can have its own allowed location list.
    - Else, allowed_targets (global) is used for all units.
    - If unassigned_units_map is provided, units can be left unassigned (미배치).
    Returns list of {loc_name: [units...]}.
    """
    loc_names = [name for name in sorted(loc_fixed.keys())]
    base = {name: sorted(units) for name, units in ((k, loc_fixed[k]) for k in loc_names)}
    M = len(moving_list)
    L = len(loc_names)
    # Determine allowed indices per unit
    sorted_units = sorted(moving_list)
    UNASSIGNED = -1  # 미배치를 나타내는 특수 값
    
    if allowed_targets_map:
        allowed_idx_lists = []
        for unit in sorted_units:
            targets = allowed_targets_map.get(unit, [])
            if targets is None or len(targets) == 0:
                idxs = list(range(L))
            else:
                target_set = set(targets)
                idxs = [i for i, nm in enumerate(loc_names) if nm in target_set]
                if len(idxs) == 0:
                    idxs = list(range(L))
            
            # 미배치 옵션 추가 (미배치가 허용된 경우)
            if unassigned_units_map and unassigned_units_map.get(unit, False):
                idxs.append(UNASSIGNED)
            
            allowed_idx_lists.append(idxs)
    else:
        if allowed_targets is None or len(allowed_targets) == 0:
            idxs = list(range(L))
        else:
            allowed_set = set(allowed_targets)
            idxs = [i for i, nm in enumerate(loc_names) if nm in allowed_set]
            if len(idxs) == 0:
                idxs = list(range(L))
        
        # 미배치 옵션 추가 (전역적으로 미배치가 허용된 경우)
        if unassigned_units_map:
            for unit in sorted_units:
                unit_idxs = list(idxs)
                if unassigned_units_map.get(unit, False):
                    unit_idxs.append(UNASSIGNED)
                allowed_idx_lists.append(unit_idxs)
        else:
            allowed_idx_lists = [idxs for _ in range(M)]
    
    if M == 0:
        return [base]
    assignments = []
    for choice in product(*allowed_idx_lists):
        mapping = {name: list(base[name]) for name in loc_names}
        for idx, unit in enumerate(sorted_units):
            if choice[idx] == UNASSIGNED:
                # 미배치: 해당 unit을 어떤 location에도 추가하지 않음
                continue
            mapping[loc_names[choice[idx]]].append(unit)
        # sort units per loc for stability
        mapping = {k: sorted(v) for k, v in mapping.items()}
        assignments.append(mapping)
    return assignments


def processing_data():
    df_orig = pd.read_parquet("mnl_1208_1214_gd.parquet")
    df_orig["total_seat_count"]=(df_orig["total_pax"]*1.16).astype(int)
    


    # ##############################################
    # # operating_carrier_iata가 5J이면서 domestic인 경우 변경
    # # 1. 5J domestic이면서 ILO인 경우 → 5J*dom*20%
    # mask_ilo = (df_orig["operating_carrier_iata"] == "5J") & (df_orig["International/Domestic"] == "domestic") & (df_orig["dep/arr_airport"].isin(
    #     ["DGT","GES","TUG","CYZ","OZC","PAG","RXS","DPL","KLO","LAO","SJI","VRC"]
    #     ))
    # df_orig.loc[mask_ilo, "operating_carrier_iata"] = "5Jdom20%"
    
    # # 2. 5J domestic이지만 ILO가 아닌 경우 → 5J*dom*80%
    # mask_dom_not_ilo = (df_orig["operating_carrier_iata"] == "5J") & (df_orig["International/Domestic"] == "domestic") & (df_orig["operating_carrier_iata"] != "5Jdom20%")
    # df_orig.loc[mask_dom_not_ilo, "operating_carrier_iata"] = "5Jdom80%"
    # ##############################################


    ##############################################
    group_1 = ["CEB"]                          # G1
    group_2 = ["DVO"]                          # G2
    group_3 = ["ILO"]                          # G3 (High-Upper)
    group_4 = ["CGY", "MPH"]                   # G4 (High-Lower)
    group_5 = ["BCD", "PPS"]                   # G5
    group_6 = ["DRP", "BXU"]                   # G6
    group_7 = ["TAC", "ZAM", "TAG"]             # G7
    group_8 = ["DGT", "GES"]                   # G8
    group_9 = ["TUG","PAG","OZC","RXS","CYZ"]   # G9 (Very Low-Upper)
    group_10 = ["DPL","KLO","LAO","VRC","SJI"]  # G10 (Very Low-Lower)

    route_groups = {
        "5J_G1": group_1,
        "5J_G2": group_2,
        "5J_G3": group_3,
        "5J_G4": group_4,
        "5J_G5": group_5,
        "5J_G6": group_6,
        "5J_G7": group_7,
        "5J_G8": group_8,
        "5J_G9": group_9,
        "5J_G10": group_10,
    }


    base_mask = (
        (df_orig["operating_carrier_iata"] == "5J") &
        (df_orig["International/Domestic"] == "domestic")
    )

    for group_name, airports in route_groups.items():
        mask = base_mask & df_orig["dep/arr_airport"].isin(airports)
        df_orig.loc[mask, "operating_carrier_iata"] = group_name
    ##############################################


    df_orig = df_orig[df_orig["total_seat_count"]>0]
    df_orig["scheduled_gate_local"]=pd.to_datetime(df_orig["scheduled_gate_local"])
    # terminal_carrier 컬럼 생성
    df_orig["terminal_carrier"] = "[" + df_orig["terminal"] + "] " + df_orig["operating_carrier_name"]
    df_orig["terminal_iata"] = "[" + df_orig["terminal"] + "] " + df_orig["operating_carrier_iata"]

    df_orig["terminal_carrier_int/dom"] = "[" + df_orig["terminal"] + "] " + df_orig["operating_carrier_name"] + " (" + df_orig["International/Domestic"].str[:3] + ")"
    df_orig["terminal_iata_int/dom"] = "[" + df_orig["terminal"] + "] " + df_orig["operating_carrier_iata"] + " (" + df_orig["International/Domestic"].str[:3] + ")"

    df_orig["terminal_iata_int/dom_dest"] = "[" + df_orig["terminal"] + "] " + df_orig["operating_carrier_iata"] + " (" + df_orig["International/Domestic"].str[:3] + ")" + " | "  + df_orig["dep/arr_airport"]

    try : 
        df_orig["Carrier_Destination"] = "[" + df_orig["terminal"] + "] " + "[" + df_orig["operating_carrier_name"] + "|" + df_orig["operating_carrier_iata"] +"] - "  +  df_orig["country_name"] + " | "  + df_orig["dep/arr_airport"]+ " | " + df_orig["flight_distance"].astype(str) + "km"
    except :
        df_orig["Carrier_Destination"] = "[" + df_orig["terminal"] + "] " + "[" + df_orig["operating_carrier_name"] + "] - "+  df_orig["country_name"] + " | "  + df_orig["dep/arr_airport"]
        
    return df_orig

def _render_puff_editor(self):
    """Helper to render puff factor editor and return puffed df based on current df_filtered."""
    df = (self.df_filtered if self.df_filtered is not None else self.df_orig).copy()

    # 기준이 될 그룹 컬럼 선택 (stacked bar 색상도 동일 컬럼 사용)
    candidate_group_cols = [
        c
        for c in [
            "terminal_carrier_int/dom",
            "terminal_iata_int/dom_dest",
            "terminal_carrier",
            "terminal",
            "operating_carrier_iata",
            "International/Domestic",
        ]
        if c in df.columns
    ]
    if not candidate_group_cols:
        st.info("No suitable group column found for puffing.")
        return df, None

    group_col = st.selectbox(
        "Group column (for puff & color)",
        options=candidate_group_cols,
        index=0,
        help="Rows will be puffed per unique value of this column, and stacked bar color will use the same groups.",
        key="puff_group_col",
    )

    groups = sorted(df[group_col].dropna().unique().tolist())
    if len(groups) == 0:
        st.info("No groups available to puff for the selected column.")
        return df, None

    puff_df = pd.DataFrame({"group": groups, "puff_factor": [1 for _ in groups]})

    st.caption("Edit `puff_factor` to virtually multiply each group's rows (1 = no change).")
    edited = st.data_editor(
        puff_df,
        num_rows="fixed",
        column_config={
            "group": st.column_config.TextColumn("Group", disabled=True, width="large"),
            "puff_factor": st.column_config.NumberColumn(
                "Puff factor",
                min_value=0,
                max_value=100,
                step=1,
                help="0 = remove group, 1 = original, 2 = double, ...",
            ),
        },
        hide_index=True,
        key="puff_factor_editor",
    )

    factor_map = (
        edited.set_index("group")["puff_factor"].fillna(1).astype(int).to_dict()
        if len(edited) > 0
        else {}
    )

    if len(factor_map) == 0:
        df_puffed = df
    else:
        df["__puff_factor__"] = df[group_col].map(factor_map).fillna(1).astype(int)
        # factor == 0 인 그룹은 제거
        df = df[df["__puff_factor__"] > 0]
        df_puffed = df.loc[df.index.repeat(df["__puff_factor__"])].copy()
        df_puffed.drop(columns=["__puff_factor__"], inplace=True, errors="ignore")

    return df_puffed, group_col



_SCENARIOS_SAVE_DIR = os.path.join(_PROJECT_ROOT, "Scenarios")


def _make_scenario_zip_bytes(scenario_list):
    """Build zip file in memory from list of scenario export dicts. Returns bytes or None."""
    if not scenario_list:
        return None
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for exp in scenario_list:
            meta = exp.get("meta") or {}
            sid = meta.get("scenario_id", "S")
            term = meta.get("terminal", "T1")
            da = meta.get("dep_arr", "Dep")
            cap = (exp.get("time_series") or {}).get("capacity", 0)
            agg_name = (meta.get("aggregate_base") or "agg").replace(" ", "_")
            safe_sid = "".join(c if c.isalnum() or c in "_" else "_" for c in str(sid))
            fname = f"scenario_{safe_sid}_{agg_name}_{term}_{da}_{cap}.json"
            zf.writestr(fname, json.dumps(exp, ensure_ascii=False, indent=2))
    buf.seek(0)
    return buf.getvalue()


@st.fragment
def _render_scenario_comparison_tab():
    """Scenario comparison: configure and generate HTML report."""

    if _build_scenario_report_html is None:
        st.warning("Report module is not available. Please check `utils/generate_scenario_report_rev2.py` exists.")
        return

    scenarios_to_report = []

    def _default_display_for_slot(sidx, slot_index, scenario_options, scenario_display, count_so_far):
        """When the same scenario is used multiple times, append (1), (2), ... suffixes."""
        base = scenario_display[scenario_options.index(sidx)] if sidx in scenario_options else str(sidx)
        if count_so_far == 0:
            return base
        return f"{base} ({count_so_far + 1})"

    def _render_auto_report():
        analysis = st.session_state.get("current_analysis")
        if analysis is None or not hasattr(analysis, "results_df") or analysis.results_df is None or len(analysis.results_df) == 0:
            st.info("To use auto report, first run **Run analysis** in the **Result** tab.")
        else:
            scenario_options = analysis.results_df.index.tolist()
            if hasattr(analysis.results_df, "columns") and "Scenario_ID" in analysis.results_df.columns:
                scenario_display = [format_scenario_id(opt) for opt in scenario_options]
            else:
                scenario_display = [f"S_{i}" for i in range(len(scenario_options))]

            # Scenario slots: per-slot scenario + display name, can be extended
            if "report_scenario_slots" not in st.session_state:
                default_slots = []
                for i, opt in enumerate(scenario_options[: min(2, len(scenario_options))]):
                    disp = scenario_display[i] if i < len(scenario_display) else str(opt)
                    default_slots.append((opt, disp))
                st.session_state["report_scenario_slots"] = default_slots

            slots = st.session_state["report_scenario_slots"]
            agg_options = list(getattr(analysis, "selected_agg_cols", None) or [getattr(analysis, "selected_agg_col", "total_seat_count")])
            if not agg_options:
                agg_options = [getattr(analysis, "selected_agg_col", "total_seat_count")]
            terminal_options = list(analysis.loc_names) if analysis.loc_names else ["T1"]
            default_aggs = st.session_state.get("report_auto_aggs", agg_options)
            default_terms = st.session_state.get("report_auto_terminals", terminal_options if terminal_options else [])

            with st.form("report_config_form", clear_on_submit=False):
                # Each row: [Scenario select] [Display name] [Delete]
                updated_slots = []
                delete_clicked_idx = None
                for i, (sidx, disp) in enumerate(list(slots)):
                    c1, c2, c3 = st.columns([2, 3, 1])
                    with c1:
                        try:
                            sel_index = scenario_options.index(sidx) if sidx in scenario_options else 0
                        except (ValueError, TypeError):
                            sel_index = 0
                        new_sidx = st.selectbox(
                            f"Slot {i + 1} – Scenario",
                            options=scenario_options,
                            index=min(sel_index, len(scenario_options) - 1) if scenario_options else 0,
                            format_func=lambda x, so=scenario_options, sd=scenario_display: sd[so.index(x)] if x in so else str(x),
                            key=f"report_slot_scenario_{i}",
                            label_visibility="collapsed",
                        )
                    with c2:
                        ph = scenario_display[scenario_options.index(new_sidx)] if new_sidx in scenario_options else str(new_sidx)
                        new_disp = st.text_input(
                            "Display name for report",
                            value=disp,
                            key=f"report_slot_name_{i}",
                            label_visibility="collapsed",
                            placeholder=ph,
                        )
                        base = scenario_display[scenario_options.index(new_sidx)] if new_sidx in scenario_options else str(new_sidx)
                        final_disp = (new_disp or "").strip() or base
                        updated_slots.append((new_sidx, final_disp))
                    with c3:
                        if len(slots) > 1 and st.form_submit_button(f"🗑 {i + 1}"):
                            delete_clicked_idx = i

                if updated_slots:
                    st.session_state["report_scenario_slots"] = updated_slots

                add_clicked = st.form_submit_button("➕ Add slot")
                st.caption("If display names are identical, (1), (2) will be appended automatically.")

                selected_aggs = st.multiselect(
                    "**Aggregate base** (multi-select allowed)",
                    options=agg_options,
                    default=default_aggs if default_aggs else agg_options,
                    key="report_auto_aggs",
                )
                if not selected_aggs and agg_options:
                    selected_aggs = [agg_options[0]]

                selected_terminals = st.multiselect(
                    "**Terminal** (multi-select, targets to export)",
                    options=terminal_options,
                    default=default_terms if default_terms else terminal_options[:1],
                    key="report_auto_terminals",
                )
                if not selected_terminals:
                    selected_terminals = terminal_options[:1] if terminal_options else ["T1"]

                _ = st.form_submit_button("Apply")

            # form 제출 후 처리: 추가/삭제 반영
            if delete_clicked_idx is not None:
                slots_after = st.session_state["report_scenario_slots"]
                new_list = slots_after[:delete_clicked_idx] + slots_after[delete_clicked_idx + 1 :]
                st.session_state["report_scenario_slots"] = new_list
                st.rerun()
            if add_clicked:
                slots_after = st.session_state["report_scenario_slots"]
                last_sidx = slots_after[-1][0] if slots_after else (scenario_options[0] if scenario_options else None)
                count_same = sum(1 for s, _ in slots_after if s == last_sidx)
                new_disp = _default_display_for_slot(last_sidx, len(slots_after), scenario_options, scenario_display, count_same)
                st.session_state["report_scenario_slots"] = slots_after + [(last_sidx, new_disp)]
                st.rerun()

            selected_scenario_slots = st.session_state.get("report_scenario_slots", [])

            # Build combination list
            combos = []
            for (sidx, display_name) in selected_scenario_slots:
                for agg in selected_aggs:
                    for term in selected_terminals:
                        combos.append((sidx, display_name, agg, term))

            st.markdown("**Dep / Arr / Both and capacity per combination**")
            edited_combo_df = None
            if not combos:
                st.caption("Select at least one Scenario, Agg, and Terminal to display the combination table.")
            else:
                def _default_caps(a):
                    if (a or "").strip().lower() == "movement":
                        return 8, 8, 12
                    return 2000, 1800, 2200
                rows = []
                for (sidx, display_name, agg, term) in combos:
                    dep_cap, arr_cap, both_cap = _default_caps(agg)
                    rows.append({
                        "Scenario": display_name,
                        "Agg": agg,
                        "Terminal": term,
                        "Puffing_factor": 1.0,
                        "Dep": True,
                        "Dep_cap": dep_cap,
                        "Arr": True,
                        "Arr_cap": arr_cap,
                        "Both": True,
                        "Both_cap": both_cap,
                    })
                combo_df = pd.DataFrame(rows)
                column_config = {
                    "Scenario": st.column_config.TextColumn("Scenario (display name in report)", disabled=True),
                    "Agg": st.column_config.TextColumn("Agg", disabled=True),
                    "Terminal": st.column_config.TextColumn("Terminal", disabled=True),
                    "Puffing_factor": st.column_config.NumberColumn(
                        "Puffing factor",
                        min_value=0.1,
                        max_value=10.0,
                        step=0.05,
                        default=1.0,
                        help="1.0 = original, 1.35 = +35% (row duplication after random shuffle)",
                    ),
                    "Dep": st.column_config.CheckboxColumn("Dep"),
                    "Dep_cap": st.column_config.NumberColumn("Dep cap", min_value=0, step=100),
                    "Arr": st.column_config.CheckboxColumn("Arr"),
                    "Arr_cap": st.column_config.NumberColumn("Arr cap", min_value=0, step=100),
                    "Both": st.column_config.CheckboxColumn("Both"),
                    "Both_cap": st.column_config.NumberColumn("Both cap", min_value=0, step=100),
                }
                edited_combo_df = st.data_editor(
                    combo_df,
                    column_config=column_config,
                    key="report_combo_editor",
                    use_container_width=True,
                    num_rows="fixed",
                )
                st.caption(
                    f"Total {len(combos)} combinations. "
                    "Set Puffing factor (1.0 = original, 1.35 = +35%), Dep/Arr/Both, and capacity, "
                    "then click the button below to generate the HTML report."
                )

            # Only generate JSON/HTML when the 'Build!' button is clicked
            build_clicked = st.button("🔄 Build! (generate HTML report)", key="report_auto_make")

            if build_clicked:
                # Basic validation: Scenario / Agg / Terminal selections
                if not selected_scenario_slots:
                    st.warning("Please select at least one scenario.")
                elif not selected_aggs:
                    st.warning("Please select at least one Agg.")
                elif not selected_terminals:
                    st.warning("Please select at least one terminal.")
                elif not combos or edited_combo_df is None or len(edited_combo_df) != len(combos):
                    st.warning("Please review the combination table before trying again.")
                else:
                    # From here: JSON + HTML generation (with progress)
                    def _default_caps(a):
                        if (a or "").strip().lower() == "movement":
                            return 8, 8, 12
                        return 2000, 1800, 2200

                    combos_for_build = combos  # (sidx, display_name, agg, term)
                    # 중복 display_name → (1), (2) 자동 부여 (슬롯 단위로만! 같은 슬롯의 여러 조합은 동일 이름 유지)
                    n_per_slot = len(selected_aggs) * len(selected_terminals)
                    display_name_counts: dict[str, int] = {}
                    slot_report_names = []
                    for (sidx, display_name) in selected_scenario_slots:
                        cnt = display_name_counts.get(display_name, 0)
                        display_name_counts[display_name] = cnt + 1
                        rn = display_name if cnt == 0 else f"{display_name} ({cnt})"
                        slot_report_names.append(rn)
                    combos_with_unique_names = [
                        (sidx, slot_report_names[i // n_per_slot], agg, term)
                        for i, (sidx, display_name, agg, term) in enumerate(combos_for_build)
                    ]

                    # Count total tasks (only Dep/Arr/Both actually checked)
                    total_tasks = 0
                    for i in range(len(combos_for_build)):
                        _, _, agg, _ = combos_for_build[i]
                        row = edited_combo_df.iloc[i]
                        def_cap_dep, def_cap_arr, def_cap_both = _default_caps(agg)
                        use_dep = bool(row.get("Dep", True) if hasattr(row, "get") else True)
                        use_arr = bool(row.get("Arr", True) if hasattr(row, "get") else True)
                        use_both = bool(row.get("Both", False) if hasattr(row, "get") else False)
                        for use_it in (use_dep, use_arr, use_both):
                            if use_it:
                                total_tasks += 1

                    if total_tasks == 0:
                        st.warning(
                            "There is no Dep/Arr/Both setting enabled for the selected combinations. "
                            "Please check that at least one of Dep/Arr/Both is checked in each row."
                        )
                    else:
                        built = []
                        done = 0
                        from math import ceil

                        with st.status("Generating report...", expanded=True) as status:
                            progress = st.progress(0.0)

                            for i in range(len(combos_for_build)):
                                sidx, _, agg, term = combos_for_build[i]
                                report_display_name = combos_with_unique_names[i][1]
                                row = edited_combo_df.iloc[i]
                                def_cap_dep, def_cap_arr, def_cap_both = _default_caps(agg)
                                use_dep = bool(row.get("Dep", True) if hasattr(row, "get") else True)
                                cap_dep = int(row.get("Dep_cap", def_cap_dep) or def_cap_dep) if hasattr(row, "get") else def_cap_dep
                                use_arr = bool(row.get("Arr", True) if hasattr(row, "get") else True)
                                cap_arr = int(row.get("Arr_cap", def_cap_arr) or def_cap_arr) if hasattr(row, "get") else def_cap_arr
                                use_both = bool(row.get("Both", False) if hasattr(row, "get") else False)
                                cap_both = int(row.get("Both_cap", def_cap_both) or def_cap_both) if hasattr(row, "get") else def_cap_both
                                puff = float(row.get("Puffing_factor", 1.0) or 1.0) if hasattr(row, "get") else 1.0

                                for dep_arr_label, use_it, cap in [
                                    ("Dep", use_dep, cap_dep),
                                    ("Arr", use_arr, cap_arr),
                                    ("Both", use_both, cap_both),
                                ]:
                                    if not use_it:
                                        continue
                                    exp = analysis.build_scenario_export(
                                        sidx, term, dep_arr_label, cap, agg_col_override=agg, puffing_factor=puff
                                    )
                                    done += 1
                                    progress.progress(min(done / total_tasks, 1.0))
                                    if exp is None:
                                        continue
                                    meta = exp.get("meta") or {}
                                    if not (meta.get("aggregate_base") or "").strip():
                                        continue
                                    meta["report_display_name"] = report_display_name
                                    meta["scenario_id"] = report_display_name
                                    exp["meta"] = meta
                                    built.append(exp)

                            if built:
                                try:
                                    status.update(label="Generating HTML report...", state="running")
                                    html = _build_scenario_report_html(built)
                                    status.update(
                                        label="Done. Use the button below to download the HTML report.",
                                        state="complete",
                                    )
                                    st.download_button(
                                        label="📥 Download Scenario_Comparison_Report.html",
                                        data=html,
                                        file_name="Scenario_Comparison_Report.html",
                                        mime="text/html",
                                        key="report_download_btn",
                                    )
                                    st.success(f"Generated HTML report based on **{len(built)}** scenarios.")
                                except Exception as e:
                                    status.update(label="Error occurred while generating the report.", state="error")
                                    st.error(f"Error while generating report: {e}")
                            else:
                                status.update(label="No scenario JSON was generated.", state="error")
                                st.warning(
                                    "No scenario JSON was generated for the selected combinations. "
                                    "Please ensure at least one of Dep/Arr/Both is checked in each row "
                                    "and that the terminal has assigned units."
                                )

    _render_auto_report()


@st.fragment
def _render_scenario_detail_tab(analysis):
    """Scenario Detail tab: show per-scenario calculation process."""
    with st.container():
        analysis.render_calculation_process()


@st.fragment
def show_result(analysis=None):
    # session_state에서 최신 analysis 객체 가져오기 (항상 최신 값 사용)
    if analysis is None or 'current_analysis' in st.session_state:
        analysis = st.session_state.get('current_analysis', analysis)
    
    if analysis is None:
        st.info("Please run analysis first")
        return
    
    main_tab1, main_tab2, main_tab3 = st.tabs(["📋 Basic Info", "🔍 Scenario Detail", "⚖️ Scenario Comparison"])
    
    with main_tab1:
        analysis.render_analysis_summary()
        with st.container():
            analysis.render_statistics_table()
        with st.container():
            analysis.render_distance_visualization()
    
    with main_tab2:
        _render_scenario_detail_tab(analysis)
    
    with main_tab3:
        _render_scenario_comparison_tab()



def _render_result_section(settings, result_mode):
    """Result 섹션 공통 UI: Run/Load, Save, show_result"""
    if result_mode == "New":
        if st.button("🚀 Run analysis"):
            selected_aggs = getattr(settings, "selected_agg_cols", None) or ([settings.selected_agg_col] if settings.selected_agg_col else [])
            if not selected_aggs:
                selected_aggs = ["total_seat_count"]
            analysis = RunAnalysis(
                df_filtered=settings.df_filtered,
                relocation_unit=settings.relocation_unit,
                fixed_per_loc=settings.fixed_per_loc,
                moving_unit=settings.moving_unit,
                allowed_targets_map=settings.allowed_targets_map if len(settings.allowed_targets_map) > 0 else None,
                loc_names=settings.loc_names,
                unit_min=settings.unit_min,
                start_date=settings.start_date,
                end_date=settings.end_date,
                selected_metrics=settings.selected_metrics,
                selected_agg_col=selected_aggs[0],
                selected_agg_cols=selected_aggs,
                max_sample_count=settings.max_sample_count,
                unassigned_units_map=settings.unassigned_units_map if len(settings.unassigned_units_map) > 0 else None
            )
            st.session_state['current_analysis'] = analysis
            st.session_state.pop('loaded_result_filename', None)
            st.session_state.pop('loaded_result_saved_at', None)
            st.rerun()
    else:
        if save_result and list_saved_results:
            saved_list = list_saved_results()
            if saved_list:
                for i, f in enumerate(saved_list):
                    name, saved_at = f.get("name", "—"), (f.get("saved_at", "")[:19] or "—")
                    st.caption("---")
                    rcol1, rcol2 = st.columns([1, 3])
                    with rcol1:
                        st.caption(f"**{name}**")
                        st.caption(saved_at)
                    with rcol2:
                        if st.button("📥 Load File", key=f"load_result_btn_{i}"):
                            data = load_result(f["filepath"])
                            loaded = RunAnalysis.from_saved(data)
                            st.session_state['current_analysis'] = loaded
                            st.session_state['loaded_result_filename'] = f.get("name", "—")
                            _sat = f.get("saved_at", "")
                            st.session_state['loaded_result_saved_at'] = str(_sat)[:19] if _sat else "—"
                            st.rerun()

                        if delete_result and st.button("🗑️ Remove .", key=f"delete_result_btn_{i}", help="Delete Scenario"):
                            if delete_result(f["filepath"]):
                                st.success("삭제됨")
                                st.rerun()
                            else:
                                st.error("삭제 실패")

            else:
                st.info("저장된 결과가 없습니다. **New Analysis**로 분석을 실행한 뒤 저장해주세요.")
    
    analysis = st.session_state.get('current_analysis')
    if analysis and save_result and hasattr(analysis, 'results_df') and analysis.results_df is not None and len(analysis.results_df) > 0:
        st.divider()
        save_col1, save_col2 = st.columns([2, 1])
        with save_col1:
            save_name = st.text_input("💾 Save name", placeholder="e.g. Manila_Major_Airlines_Relocation", key="save_result_name")
        with save_col2:
            st.write("")
            st.write("")
            if st.button("💾 Save", key="save_result_btn", disabled=not (save_name and save_name.strip())):
                try:
                    path = save_result(analysis, save_name.strip())
                    st.success(f"Saved: `{os.path.basename(path)}`")
                except Exception as e:
                    st.error(f"Save failed: {e}")
    
    if analysis:
        show_result(analysis)


@st.fragment
def main(df_orig):
    apply_css()
    apply_button_css()
    # Mode 선택을 Relocation Settings 앞의 탭으로
    # Load일 때는 Relocation Settings, Assign Units 탭 자체를 표시하지 않음
    load_mode = st.toggle("Load", value=True, key="result_mode_toggle", help="ON: Load saved results / OFF: New analysis")
    result_mode = "Load" if load_mode else "New"

    # RelocationSetting 클래스 인스턴스 생성
    settings = RelocationSetting(df_orig)

    if result_mode == "New":
        # New Analysis: Mode(선택됨) 뒤에 Relocation Settings, Assign Units, Result 탭
        relocation_settings_tab, assign_units_tab, result_tab = st.tabs(["Relocation Settings", "Assign Units", "Result"])
        with relocation_settings_tab:
            settings.render_relocation_settings_tab()
        with assign_units_tab:
            settings.render_assign_units_tab()
        with result_tab:
            _render_result_section(settings, result_mode)
        for i in range(100):
            st.write("")
    else:
        # Load: Relocation Settings, Assign Units 없음. Result만 표시
        _render_result_section(settings, result_mode)

def _run_relocation_page():
    st.write("")
    st.write("""<h1>Relocation Master</h1>""", unsafe_allow_html=True)
    ms = MasterplanInput()
    df_orig = processing_data()
    main(df_orig)

if __name__ in ("__main__", "__page__"):
    _run_relocation_page()