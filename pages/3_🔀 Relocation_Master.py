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
from utils.masterplan import MasterplanInput

# region agent log helper
def _agent_debug_log(hypothesis_id, location, message, data, run_id="initial"):
    """Lightweight NDJSON logger for debug mode (do not remove until debugging is complete)."""
    try:
        import time as _time
        payload = {
            "id": f"log_{int(_time.time()*1000)}",
            "timestamp": int(_time.time()*1000),
            "location": location,
            "message": message,
            "data": data,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
        }
        debug_log_path = "c:\\Users\\qkrru\\Desktop\\바탕 화면\\creative_code\\DMK_레포지토리\\simulation_module\\yeah_construction - 20250118\\.cursor\\debug.log"
        with open(debug_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        # Logging failures must never break the app
        pass
# endregion


st.set_page_config(
    page_title="Relocation Master",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# apply_css 함수 정의 (의존성 제거)
def apply_css():
    css = """
    <style>
    @font-face {
        font-family: 'Pretendard-Black';
        src: url('https://fastly.jsdelivr.net/gh/Project-Noonnu/noonfonts_2107@1.1/Pretendard-Black.woff') format('woff');
    }
    @font-face {
        font-family: 'Pretendard-Regular';
        src: url('https://fastly.jsdelivr.net/gh/Project-Noonnu/noonfonts_2107@1.1/Pretendard-Regular.woff') format('woff');
    }
    h1, h2, .css-18e3th9 {
        font-family: 'Pretendard-Black', sans-serif;
    }
    h3 {
        font-family: 'Pretendard-Black', sans-serif;
        padding-top: 30px;
    }
    h4, h5, h6, .css-1d391kg, .css-1vbd788 {
        font-family: 'Pretendard-Regular', sans-serif;
    }
    body, p, .css-1cpxqw2, .css-1d391kg, .css-18e3th9, .css-1vbd788, .css-hxt7ib, .css-1bpgk57, .css-1aumxhk, .css-13n8z4g, .css-1kyxreq, .css-1avcm0n, .css-1ia65gd, textarea {
        font-family: 'Pretendard-Regular', sans-serif;
        font-size: 20px;
    }

    /* Hot pink styling for the "allocate Units" multiselect - will be applied via JavaScript */

    /* Card container */
    .card {
        border: 1px solid #e6e8eb;
        background: #ffffff;
        border-radius: 12px;
        padding: 16px 18px;
        margin: 12px 0 16px 0;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }
    .card h3 { margin: 0 0 8px 0; }
    </style>
    """
    return st.markdown(css, unsafe_allow_html=True)

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

# 그룹화할 공통 컬럼 리스트 (공유 변수)
BASE_COL_OPTIONS = [
    "terminal_carrier", 
    "operating_carrier_name", 
    "Carrier_Destination", 
    "flight_number", 
    "terminal_carrier_int/dom", 
    "terminal_iata",
    "terminal_iata_int/dom",
    "terminal_iata_int/dom_dest"
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
                     8, "#ff5ca8", "#ffd6e7", 3)
    fig.add_annotation(x=allocate_x, y=allocate_y+allocate_box_height/2-15, text="Allocate", showarrow=False, font=dict(size=18, color="#9c1850", weight="bold"))
    # Draw each item as a small box with rounded corners
    start_y = allocate_y + allocate_box_height/2 - 45
    for idx, item in enumerate(allocate_items):
        item_y = start_y - idx * box_height_per_item
        item_width = 160
        add_rounded_rect(allocate_x-item_width/2, item_y-12, allocate_x+item_width/2, item_y+12,
                        6, "#ff5ca8", "#ffffff", 1.5)
        fig.add_annotation(x=allocate_x, y=item_y, text=item, showarrow=False, font=dict(size=14), align="center")

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
                         8, "#3c6df0", "#e6ecff", 2)
        # title
        fig.add_annotation(x=x, y=y+loc_box_height/2-15, text=name, showarrow=False, font=dict(size=18, color="#1f3fa5", weight="bold"))
        # Draw each item as a small box with rounded corners
        start_y = y + loc_box_height/2 - 35
        for idx, item in enumerate(items):
            item_y = start_y - idx * box_height_per_item
            item_width = 140
            add_rounded_rect(x-item_width/2, item_y-12, x+item_width/2, item_y+12,
                             6, "#3c6df0", "#ffffff", 1.5)
            fig.add_annotation(x=x, y=item_y, text=item, showarrow=False, font=dict(size=14), align="center")

        # arrow from allocate to each location
        fig.add_annotation(x=x, y=y+loc_box_height/2, ax=allocate_x, ay=allocate_y-allocate_box_height/2,
                           xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=2.5, arrowcolor="#666")

    fig.update_xaxes(visible=False, range=[0, width])
    fig.update_yaxes(visible=False, range=[0, height])
    # Remove height limit to show all items
    fig.update_layout(height=height+60, width=None, margin=dict(l=10, r=10, t=20, b=20))
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

    # region agent log
    try:
        sample_minutes = {}
        for m in [0, 20, 40]:
            mask_m = count_df_complete["Time"].dt.minute == m
            sample_minutes[str(m)] = int(count_df_complete.loc[mask_m, "index"].sum())
        _agent_debug_log(
            hypothesis_id="H3",
            location="make_count_df",
            message="Aggregated counts by Time minute",
            data={
                "freq_min": freq_min,
                "sample_minutes_sum": sample_minutes,
            },
        )
    except Exception:
        pass
    # endregion

    return count_df_complete, ranking_order

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
        height=500
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
        filter_col, puff_group_col, setting_col = st.columns([0.2,0.6,0.2])
        
        with filter_col:
            self._render_date_method()

        with puff_group_col:
            _render_puffing(self)
            self._render_custom_group()

        with setting_col:
            self._render_relocation_settings()
    
    def _render_date_method(self):
        """Date & Method 섹션 렌더링"""
        st.markdown("<div class='card'><h4>🧩Date & Method</h4>", unsafe_allow_html=True)
        
        # Select Date Range
        min_date = self.df_orig["scheduled_gate_local"].dt.date.min() + pd.Timedelta(days=1)
        max_date = self.df_orig["scheduled_gate_local"].dt.date.max()
        date_range = st.date_input(
            "**Select Date Range**",
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
        with st.expander("🔍 Additional Filters", expanded=False):
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

        # region agent log
        try:
            minute_counts_orig = self.df_orig["scheduled_gate_local"].dt.minute.value_counts().to_dict()
            minute_counts_filt = self.df_filtered["scheduled_gate_local"].dt.minute.value_counts().to_dict()
            _agent_debug_log(
                hypothesis_id="H1",
                location="RelocationSetting._render_date_method",
                message="scheduled_gate_local minute distribution (orig vs filtered)",
                data={
                    "unit_min": self.unit_min,
                    "minute_counts_orig": {k: minute_counts_orig.get(k, 0) for k in [0, 20, 40]},
                    "minute_counts_filtered": {k: minute_counts_filt.get(k, 0) for k in [0, 20, 40]},
                },
            )
        except Exception:
            pass
        # endregion

        # Create Custom Group column with default value "Others"
        if "Custom Group" not in self.df_filtered.columns:
            self.df_filtered["Custom Group"] = "Others"
        if "Custom Group" not in self.df_orig.columns:
            self.df_orig["Custom Group"] = "Others"
        
        # Setting Unit minutes
        self.unit_min = st.number_input(
            "**Time Unit (minutes)**", 
            value=60, 
            min_value=5, 
            max_value=60, 
            step=5,
            key="unit_min_mnl"
        )
        
        # Select Aggregate base (원본 컬럼명 직접 사용)
        agg_options = []
        if "total_seat_count" in self.df_filtered.columns:
            agg_options.append("total_seat_count")
        if "total_pax" in self.df_filtered.columns:
            agg_options.append("total_pax")
        if "movement" in self.df_filtered.columns:
            agg_options.append("movement")
        if "od_pax" in self.df_filtered.columns:
            agg_options.append("od_pax")
        if "tr_pax" in self.df_filtered.columns:
            agg_options.append("tr_pax")
        
        # 원본 컬럼명을 직접 사용 (변환 없음)
        self.selected_agg_col = st.selectbox("**Aggregate base**", options=agg_options, index=0, key="agg_base")
        st.caption(f"**→Original / Filtered:** {self.df_orig.shape} / {self.df_filtered.shape}")
    
    def _render_custom_group(self):
        """Custom Group 섹션 렌더링"""
        st.markdown("<div class='card'><h4>🧩Custom Group</h4>", unsafe_allow_html=True)
        
        # Base column selection
        base_col = st.selectbox(
            "**Base Column**",
            options=BASE_COL_OPTIONS,
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
            with st.expander("📊 Group Settings"):
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
    
    def _render_relocation_settings(self):
        """Relocation Settings 섹션 렌더링"""
        st.markdown("<div class='card'><h4>🧩Relocation Settings</h4>", unsafe_allow_html=True)
        
        self.max_sample_count = st.number_input(
            "**Max Sample Count**", 
            value=5, 
            min_value=1, 
            max_value=50000, 
            step=50,
            key="max_sample_count_mnl",
            help="전체 조합에서 최대 몇 개를 샘플링해서 분석할지 설정합니다."
        )
        
        # Build relocation unit options including Custom Group
        self.relocation_unit = st.selectbox(
            "**Relocation Unit**",
            options=BASE_COL_OPTIONS + ["Custom Group"],
            key="relocation_unit_mnl",
            help="Relocation 분석을 위한 단위를 설정합니다."
        )
        
        selected_metrics_labels = st.multiselect(
            "Metrics to include",
            options=["97.5%", "95%", "90%", "Mean", "Total"],
            default=["97.5%", "Total"],
            key="metrics_select"
        )
        self.selected_metrics = set("Total" if m == "Total" else m for m in selected_metrics_labels)
        
        self.loc_count = int(st.number_input("Number of locations", min_value=1, max_value=8, value=1, step=1, key="loc_count"))
    
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
    
    def __init__(self, df_filtered, relocation_unit, fixed_per_loc, moving_unit, 
                 allowed_targets_map, loc_names, unit_min, start_date, end_date,
                 selected_metrics, selected_agg_col, max_sample_count, unassigned_units_map=None):
        """분석 실행 및 결과 저장"""
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
        self.selected_agg_col = selected_agg_col
        
        # Category dict 설정 (원본 컬럼명 직접 사용)
        self.category_dict = CATEGORY_DICT
        self.dep_category = f"departure {selected_agg_col}"
        self.arr_category = f"arrival {selected_agg_col}"
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

        # region agent log
        try:
            show_minute_counts = df_filtered["SHOW"].dt.minute.value_counts().to_dict()
            _agent_debug_log(
                hypothesis_id="H2",
                location="RunAnalysis.process_passenger_data",
                message="SHOW minute distribution after processing",
                data={
                    "category": category,
                    "unit_min": getattr(self, "unit_min", None),
                    "show_minute_counts": {k: show_minute_counts.get(k, 0) for k in [0, 20, 40]},
                },
            )
        except Exception:
            pass
        # endregion

        return df_filtered
    
    def run_scenario(self, loc_map: dict) -> dict:
        """Compute per-LOC stats for dep/arr seat/od/tr using existing generation."""
        results = {}
        for loc_name, units in sorted(loc_map.items(), key=lambda x: x[0]):
            mask = self.df_filtered[self.relocation_unit].isin(sorted(units))
            if mask.sum() == 0:
                # fill zeros for all categories
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
                    # Time_Slots는 요청에 포함되지 않았으므로 기본 제외
                continue

            for io, label in [("departure", "Dep"), ("arrival", "Arr")]:
                category = f"{io} {self.selected_agg_col}"
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
                # Truncate percentile/mean values to integers (e.g., 44.2224 -> 44)
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
    
    def _process_results(self):
        """결과 DataFrame 후처리 (Distance, Rank, Final_Score 계산)"""
        # Distance 컬럼 계산 함수
        def calculate_distance_and_rank(df, metric_name):
            """Distance와 Rank 컬럼을 계산하는 함수 (유클리드 거리: 제곱합의 제곱근, Dep와 Arr 각각 계산)"""
            dep_cols = [col for col in df.columns if col.endswith(f'_Dep_{metric_name}')]
            arr_cols = [col for col in df.columns if col.endswith(f'_Arr_{metric_name}')]
            
            # Dep Distance 계산
            if len(dep_cols) > 0:
                distance_dep_col = f'Dist_Dep_{metric_name}'
                rank_dep_col = f'Rank_Dep_{metric_name}'
                df[distance_dep_col] = df[dep_cols].apply(
                    lambda row: np.sqrt((row ** 2).sum()), axis=1
                )
                df[rank_dep_col] = df[distance_dep_col].rank(method='min', ascending=True).astype(int)
            
            # Arr Distance 계산
            if len(arr_cols) > 0:
                distance_arr_col = f'Dist_Arr_{metric_name}'
                rank_arr_col = f'Rank_Arr_{metric_name}'
                df[distance_arr_col] = df[arr_cols].apply(
                    lambda row: np.sqrt((row ** 2).sum()), axis=1
                )
                df[rank_arr_col] = df[distance_arr_col].rank(method='min', ascending=True).astype(int)
            
            return df
        
        # 모든 metric에 대해 Distance와 Rank 계산
        metric_order = ['97.5%', '95%', '90%', 'Mean']
        for metric in metric_order:
            if metric in self.selected_metrics:
                self.results_df = calculate_distance_and_rank(self.results_df, metric)
        
        # 터미널 단위 랭킹 계산 함수
        def calculate_terminal_ranks(df, metric_name, loc_names):
            """터미널 단위 컬럼에 대해 랭킹 계산"""
            for loc_name in loc_names:
                # Dep 컬럼 랭킹
                dep_col = f"{loc_name}_Dep_{metric_name}"
                if dep_col in df.columns:
                    rank_col = f"Rank_{dep_col}"
                    df[rank_col] = df[dep_col].rank(method='min', ascending=True).astype(int)
                
                # Arr 컬럼 랭킹
                arr_col = f"{loc_name}_Arr_{metric_name}"
                if arr_col in df.columns:
                    rank_col = f"Rank_{arr_col}"
                    df[rank_col] = df[arr_col].rank(method='min', ascending=True).astype(int)
            
            return df
        
        # 모든 metric에 대해 터미널 단위 랭킹 계산
        for metric in metric_order:
            if metric in self.selected_metrics:
                self.results_df = calculate_terminal_ranks(self.results_df, metric, self.loc_names)
        
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
        
        # 2. Distance와 Rank 컬럼들 (Scenario_ID 바로 다음, 97.5% → 95% → 90% → Mean 순서, Dep → Arr 순서)
        for metric in metric_order:
            if metric in self.selected_metrics:
                for process in ['Dep', 'Arr']:
                    for col_suffix in ['Rank']:
                        col_name = f'{col_suffix}_{process}_{metric}'
                        if col_name in self.results_df.columns:
                            ordered_columns.append(col_name)
        
        for metric in metric_order:
            if metric in self.selected_metrics:
                for process in ['Dep', 'Arr']:
                    for col_suffix in ["Dist"]:
                        col_name = f'{col_suffix}_{process}_{metric}'
                        if col_name in self.results_df.columns:
                            ordered_columns.append(col_name)
        
        # 3. Location 컬럼들
        for loc_name in self.loc_names:
            if loc_name in self.results_df.columns:
                ordered_columns.append(loc_name)
        
        # 4. Location별 통계 컬럼들 (Metric → Dep/Arr → Location 순서)
        for metric in metric_order:
            if metric in self.selected_metrics:
                # Dep 먼저 (모든 Location의 Dep)
                for loc_name in self.loc_names:
                    dep_col = f"{loc_name}_Dep_{metric}"
                    if dep_col in self.results_df.columns:
                        ordered_columns.append(dep_col)
                # Arr 나중 (모든 Location의 Arr)
                for loc_name in self.loc_names:
                    arr_col = f"{loc_name}_Arr_{metric}"
                    if arr_col in self.results_df.columns:
                        ordered_columns.append(arr_col)
        
        # 4.5. 터미널 단위 랭킹 컬럼들 (Metric → Dep/Arr → Location 순서)
        for metric in metric_order:
            if metric in self.selected_metrics:
                # Dep 먼저 (모든 Location의 Dep Rank)
                for loc_name in self.loc_names:
                    rank_dep_col = f"Rank_{loc_name}_Dep_{metric}"
                    if rank_dep_col in self.results_df.columns:
                        ordered_columns.append(rank_dep_col)
                # Arr 나중 (모든 Location의 Arr Rank)
                for loc_name in self.loc_names:
                    rank_arr_col = f"Rank_{loc_name}_Arr_{metric}"
                    if rank_arr_col in self.results_df.columns:
                        ordered_columns.append(rank_arr_col)
        
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
        st.session_state['relocation_loc_names'] = self.loc_names
    
    def get_numeric_cols(self):
        """numeric_cols 계산하는 공통 메서드"""
        numeric_cols = self.results_df.select_dtypes(include=[np.number]).columns.tolist()
        # Scenario_ID, Location 컬럼, Rank 컬럼 제외
        exclude_patterns = ['Scenario_ID', 'Rank']
        numeric_cols = [col for col in numeric_cols if not any(pattern in col for pattern in exclude_patterns)]
        # Location 이름으로 된 컬럼 제외 (리스트가 들어있는 컬럼)
        numeric_cols = [col for col in numeric_cols if col not in self.loc_names]
        return numeric_cols
    
    def render_statistics_table(self):
        """Statistics table 렌더링"""
        st.markdown("#### 📋 Statistics table")
        st.dataframe(self.results_df.set_index('Scenario_ID'), use_container_width=True, height=640)
    
    def render_distance_visualization(self):
        """Render distance visualization with 3 graphs (Bar, Scatter, 3D Scatter)"""
        numeric_cols = self.get_numeric_cols()
        
        if len(numeric_cols) == 0:
            st.info("No numeric columns available for visualization")
            return
        
        # 선택 옵션들
        col1_select, col2_select, col3_select = st.columns(3)
        
        # session_state에서 이전 값 가져오기 (없으면 기본값)
        default_bar = st.session_state.get('distance_viz_bar_chart_select', numeric_cols[:min(3, len(numeric_cols))])
        default_x_idx = 1
        default_y_idx = min(2, len(numeric_cols)-1)
        default_z_idx = min(3, len(numeric_cols)-1)
        
        with col1_select:
            st.markdown("#### 📊 1D Visualization")
            selected_bar_cols = st.multiselect(
                "Select for Bar Chart",
                options=numeric_cols,
                default=default_bar[0] if isinstance(default_bar, list) else [default_bar] if default_bar in numeric_cols else numeric_cols[:min(3, len(numeric_cols))],
                key="distance_viz_bar_chart_select"
            )
            
            if len(selected_bar_cols) > 0:
                # 첫 번째 선택된 컬럼의 값으로 정렬
                sort_col = selected_bar_cols[0]
                sorted_df = self.results_df.sort_values(by=sort_col, ascending=True).reset_index(drop=True)
                
                # Scenario_ID 사용 (없으면 인덱스 사용)
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
        
        with col2_select:
            st.markdown("#### 📊 2D Visualization")
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
            
            # Scenario_ID 사용 (없으면 인덱스 사용)
            if 'Scenario_ID' in self.results_df.columns:
                scenario_texts = self.results_df['Scenario_ID'].tolist()
            else:
                scenario_texts = [f'Scenario {i}' for i in self.results_df.index]
            
            # 데이터 포인트 추가
            fig_scatter.add_trace(go.Scatter(
                x=self.results_df[x_axis],
                y=self.results_df[y_axis],
                mode='markers',
                marker=dict(
                    size=8,
                    color=self.results_df.index,
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
        
        with col3_select:
            st.markdown("#### 📊 3D Visualization")
            c1, c2, c3 = st.columns(3)
            with c1:
                x_3d = st.selectbox(
                    "X-axis for 3D",
                    options=numeric_cols,
                    index=default_x_idx,
                    key="distance_viz_3d_x"
                )
            with c2:
                y_3d = st.selectbox(
                    "Y-axis for 3D",
                    options=numeric_cols,
                    index=default_y_idx,
                    key="distance_viz_3d_y"
                )
            with c3:
                z_3d = st.selectbox(
                    "Z-axis for 3D",
                    options=numeric_cols,
                    index=default_z_idx,
                    key="distance_viz_3d_z"
                )
            
            fig_3d = go.Figure()
            
            # Scenario_ID 사용 (없으면 인덱스 사용)
            if 'Scenario_ID' in self.results_df.columns:
                scenario_texts_3d = self.results_df['Scenario_ID'].tolist()
            else:
                scenario_texts_3d = [f'Scenario {i}' for i in self.results_df.index]
            
            # 데이터 포인트 추가
            fig_3d.add_trace(go.Scatter3d(
                x=self.results_df[x_3d],
                y=self.results_df[y_3d],
                z=self.results_df[z_3d],
                mode='markers',
                marker=dict(
                    size=6,
                    color=self.results_df.index,
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
    
    def render_scenario_compare(self):
        """Render Scenario Comparison with Column Statistics"""
        numeric_cols = self.get_numeric_cols()
        
        if len(numeric_cols) == 0:
            st.info("No numeric columns available for visualization")
            return
        
        st.markdown("#### 📊 Column Statistics (Scenario Comparison)")
        
        # 시나리오 선택 (Multiselect, 여러 개 선택 가능)
        scenario_options = self.results_df.index.tolist()
        
        # 시나리오 표시 형식 준비
        if 'Scenario_ID' in self.results_df.columns:
            scenario_display_map = {format_scenario_id(opt): opt for opt in scenario_options}
            scenario_display_options = sorted(scenario_display_map.keys())
        else:
            scenario_display_map = None
            scenario_display_options = None
        
        # Column Statistics용 시나리오 선택 (2개만)
        if scenario_display_map is not None and scenario_display_options is not None:
            stats_scenarios_display = st.multiselect(
                "Select 2 Scenarios for Comparison",
                options=scenario_display_options,
                default=scenario_display_options[:min(2, len(scenario_display_options))] if len(scenario_display_options) > 0 else [],
                max_selections=2,
                key="scenario_compare_stats_scenario_select"
            )
            
            # 2개 초과 선택 시 경고
            if len(stats_scenarios_display) > 2:
                st.warning("Please select maximum 2 scenarios for comparison")
                stats_scenarios_display = stats_scenarios_display[:2]
            
            # 선택한 표시 형식을 원래 시나리오 ID로 변환
            stats_selected_scenarios = [scenario_display_map[disp] for disp in stats_scenarios_display]
        else:
            stats_selected_scenarios = st.multiselect(
                "Select 2 Scenarios for Comparison",
                options=scenario_options,
                default=scenario_options[:min(2, len(scenario_options))] if len(scenario_options) > 0 else [],
                max_selections=2,
                key="scenario_compare_stats_scenario_select"
            )
            
            # 2개 초과 선택 시 경고
            if len(stats_selected_scenarios) > 2:
                st.warning("Please select maximum 2 scenarios for comparison")
                stats_selected_scenarios = stats_selected_scenarios[:2]
        
        # Column Statistics용 Numeric 컬럼 선택 (Multiselect)
        selected_stats_cols = st.multiselect(
            "Select Numeric Columns for Comparison",
            options=numeric_cols,
            default=numeric_cols[:min(7, len(numeric_cols))] if len(numeric_cols) > 0 else [],
            key="scenario_compare_stats_column_select"
        )
        
        # Column Statistics 표시
        if len(stats_selected_scenarios) == 2 and len(selected_stats_cols) > 0:
            stats_filtered_df = self.results_df.loc[stats_selected_scenarios]
            
            # Aggregate base에 따른 단위 매핑 함수
            def get_unit_from_agg_key(agg_col):
                """Aggregate base에 따라 단위 반환 (원본 컬럼명 기반)"""
                agg_col_lower = agg_col.lower()
                if 'seat' in agg_col_lower or 'total_seat_count' in agg_col_lower:
                    return 'seat'
                elif 'od_pax' in agg_col_lower:
                    return 'pax'
                elif 'tr_pax' in agg_col_lower:
                    return 'pax'
                elif 'total_pax' in agg_col_lower:
                    return 'pax'
                elif 'movement' in agg_col_lower:
                    return 'move'
                else:
                    return ''  # 기본값
            
            unit_label = get_unit_from_agg_key(self.selected_agg_col)
            
            # Description 생성 함수 (더 직관적으로 개선)
            def get_column_description(col_name, unit_min, loc_names):
                """컬럼 이름에 따라 Description 생성"""
                col_lower = col_name.lower()
                
                # Dist_Dep_XX% 계열
                if 'dist_dep' in col_lower and '%' in col_name:
                    percentile = col_name.split('%')[0].split('_')[-1] if '%' in col_name else ""
                    return f"Peak load: {percentile}th busiest {unit_min}-min slot for departure facilities (check-in, security, immigration) across all terminals"
                
                # Dist_Arr_XX% 계열
                elif 'dist_arr' in col_lower and '%' in col_name:
                    percentile = col_name.split('%')[0].split('_')[-1] if '%' in col_name else ""
                    return f"Peak load: {percentile}th busiest {unit_min}-min slot for arrival facilities (immigration, carousel, customs) across all terminals"
                
                # {Location}_Dep_XX% 계열
                elif '_dep_' in col_lower and '%' in col_name:
                    percentile = col_name.split('%')[0].split('_')[-1] if '%' in col_name else ""
                    for loc in loc_names:
                        if col_name.startswith(loc):
                            return f"Peak load: {percentile}th busiest {unit_min}-min slot for departure facilities at {loc}"
                    return f"Peak load: {percentile}th busiest {unit_min}-min slot for departure facilities"
                
                # {Location}_Arr_XX% 계열
                elif '_arr_' in col_lower and '%' in col_name:
                    percentile = col_name.split('%')[0].split('_')[-1] if '%' in col_name else ""
                    for loc in loc_names:
                        if col_name.startswith(loc):
                            return f"Peak load: {percentile}th busiest {unit_min}-min slot for arrival facilities at {loc}"
                    return f"Peak load: {percentile}th busiest {unit_min}-min slot for arrival facilities"
                
                # {Location}_Dep_Total 계열
                elif '_dep_total' in col_lower or (col_lower.endswith('_total') and '_dep' in col_lower):
                    for loc in loc_names:
                        if col_name.startswith(loc):
                            return f"Total departure volume: Sum of all departure passengers/flights at {loc} during the period"
                    return f"Total departure volume: Sum of all departure passengers/flights during the period"
                
                # {Location}_Arr_Total 계열
                elif '_arr_total' in col_lower or (col_lower.endswith('_total') and '_arr' in col_lower):
                    for loc in loc_names:
                        if col_name.startswith(loc):
                            return f"Total arrival volume: Sum of all arrival passengers/flights at {loc} during the period"
                    return f"Total arrival volume: Sum of all arrival passengers/flights during the period"
                
                return ""  # 기본값
            
            stats_data = []
            scenario1_name = format_scenario_id(stats_selected_scenarios[0]) if 'Scenario_ID' in self.results_df.columns else f"Scenario_{int(stats_selected_scenarios[0])+1:03d}" if str(stats_selected_scenarios[0]).isdigit() else str(stats_selected_scenarios[0])
            scenario2_name = format_scenario_id(stats_selected_scenarios[1]) if 'Scenario_ID' in self.results_df.columns else f"Scenario_{int(stats_selected_scenarios[1])+1:03d}" if str(stats_selected_scenarios[1]).isdigit() else str(stats_selected_scenarios[1])
            
            for col in selected_stats_cols:
                scenario1_val = stats_filtered_df.loc[stats_selected_scenarios[0], col]
                scenario2_val = stats_filtered_df.loc[stats_selected_scenarios[1], col]
                diff = scenario2_val - scenario1_val
                
                # Total 계열은 improvement 계산하지 않음
                col_lower = col.lower()
                is_total = '_total' in col_lower or col_lower.endswith('_total')
                
                # 개선율 계산: (차이 / 시나리오1 값) * 100 (Total 계열 제외)
                if is_total:
                    improvement_pct = "N/A"
                else:
                    if scenario1_val != 0:
                        improvement_pct = (diff / scenario1_val) * 100
                        improvement_pct = f"{improvement_pct:.2f}%"
                    else:
                        improvement_pct = "N/A" if diff != 0 else "0.00%"
                
                description = get_column_description(col, self.unit_min, self.loc_names)
                
                stats_data.append({
                    'Column': col,
                    scenario1_name: f"{scenario1_val:.2f}",
                    scenario2_name: f"{scenario2_val:.2f}",
                    'Difference': f"{diff:.2f}",
                    'Improvement %': improvement_pct,
                    'Description': description
                })
            
            stats_df = pd.DataFrame(stats_data)
            
            # 카드 형태로 표시 (직관적인 비교)
            st.markdown("##### 📊 Card View")
            # 2열 그리드로 카드 표시
            num_cols = len(selected_stats_cols)
            cols_per_row = 2
            num_rows = (num_cols + cols_per_row - 1) // cols_per_row
            
            for row_idx in range(num_rows):
                card_cols = st.columns(cols_per_row)
                for col_idx in range(cols_per_row):
                    card_idx = row_idx * cols_per_row + col_idx
                    if card_idx < num_cols:
                        col = selected_stats_cols[card_idx]
                        with card_cols[col_idx]:
                            # 카드 스타일 적용
                            scenario1_val = stats_filtered_df.loc[stats_selected_scenarios[0], col]
                            scenario2_val = stats_filtered_df.loc[stats_selected_scenarios[1], col]
                            diff = scenario2_val - scenario1_val
                            
                            col_lower = col.lower()
                            is_total = '_total' in col_lower or col_lower.endswith('_total')
                            
                            if is_total:
                                improvement_pct = "N/A"
                            else:
                                if scenario1_val != 0:
                                    improvement_pct_val = (diff / scenario1_val) * 100
                                    improvement_pct = f"{improvement_pct_val:.2f}%"
                                else:
                                    improvement_pct = "N/A" if diff != 0 else "0.00%"
                            
                            description = get_column_description(col, self.unit_min, self.loc_names)
                            
                            # Improvement 효과 메시지 생성
                            effect_html = ""
                            
                            if not is_total and improvement_pct != "N/A":
                                try:
                                    improvement_val = float(improvement_pct.replace('%', ''))
                                    if improvement_val > 0:
                                        # 개선된 경우
                                        effect_icon = "✅"
                                        effect_color = "#16a34a"
                                        effect_text1 = f"Alleviating infrastructure load by {improvement_pct} can reduce potential infrastructure investment costs"
                                        effect_text2 = "Passenger waiting time can be significantly reduced"
                                        effect_html = f'<div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #e5e7eb;"><div style="font-size: 12px; color: {effect_color}; line-height: 1.8;"><div style="margin-bottom: 6px;"><span style="font-size: 16px; margin-right: 6px;">{effect_icon}</span>{effect_text1}</div><div><span style="font-size: 16px; margin-right: 6px;">{effect_icon}</span>{effect_text2}</div></div></div>'
                                    elif improvement_val < 0:
                                        # 악화된 경우
                                        effect_icon = "🚨"
                                        effect_color = "#dc2626"
                                        abs_improvement = abs(improvement_val)
                                        effect_text1 = f"Increasing infrastructure load by {abs_improvement:.2f}% may raise potential infrastructure investment costs"
                                        effect_text2 = "Passenger waiting time may significantly increase"
                                        effect_html = f'<div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #e5e7eb;"><div style="font-size: 12px; color: {effect_color}; line-height: 1.8;"><div style="margin-bottom: 6px;"><span style="font-size: 16px; margin-right: 6px;">{effect_icon}</span>{effect_text1}</div><div><span style="font-size: 16px; margin-right: 6px;">{effect_icon}</span>{effect_text2}</div></div></div>'
                                except:
                                    pass
                            
                            # 카드 배경색 결정 (개선/악화)
                            if not is_total and improvement_pct != "N/A":
                                try:
                                    improvement_val = float(improvement_pct.replace('%', ''))
                                    if improvement_val < 0:
                                        card_color = "#fee2e2"  # 빨간색 (악화)
                                        delta_color = "inverse"
                                    elif improvement_val > 0:
                                        card_color = "#d1fae5"  # 초록색 (개선)
                                        delta_color = "normal"
                                    else:
                                        card_color = "#f3f4f6"  # 회색 (변화 없음)
                                        delta_color = "off"
                                except:
                                    card_color = "#f3f4f6"
                                    delta_color = "off"
                            else:
                                card_color = "#f3f4f6"
                                delta_color = "off"
                            
                            # 카드 HTML 생성
                            card_html = f"""
                            <div style="
                                background-color: {card_color};
                                border: 1px solid #e5e7eb;
                                border-radius: 8px;
                                padding: 16px;
                                margin-bottom: 16px;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                            ">
                                <h4 style="margin-top: 0; color: #1f2937; font-size: 14px; font-weight: bold;">{col}</h4>
                                <div style="font-size: 13px; color: #4b5563; line-height: 1.5; margin-top: 6px; margin-bottom: 12px; padding-bottom: 12px; border-bottom: 1px solid #e5e7eb;">{description}</div>
                                <div style="display: flex; justify-content: space-between; margin: 12px 0;">
                                    <div style="flex: 1;">
                                        <div style="font-size: 11px; color: #6b7280; margin-bottom: 4px;">{scenario1_name}</div>
                                        <div style="font-size: 18px; font-weight: bold; color: #1f2937;">{scenario1_val:.0f} {unit_label}</div>
                                    </div>
                                    <div style="flex: 1; text-align: right;">
                                        <div style="font-size: 11px; color: #6b7280; margin-bottom: 4px;">{scenario2_name}</div>
                                        <div style="font-size: 18px; font-weight: bold; color: #1f2937;">{scenario2_val:.0f} {unit_label}</div>
                                    </div>
                                </div>
                                <div style="border-top: 1px solid #e5e7eb; padding-top: 8px; margin-top: 8px;">
                                    <div style="display: flex; justify-content: space-between; font-size: 12px;">
                                        <span style="color: #6b7280;">Difference:</span>
                                        <span style="font-weight: bold; color: {'#dc2626' if diff < 0 else '#16a34a' if diff > 0 else '#6b7280'};">{diff:+.0f} {unit_label}</span>
                                    </div>
                                    <div style="display: flex; justify-content: space-between; font-size: 12px; margin-top: 4px;">
                                        <span style="color: #6b7280;">Improvement:</span>
                                        <span style="font-weight: bold; color: {'#dc2626' if improvement_pct != 'N/A' and str(improvement_pct).startswith('-') else '#16a34a' if improvement_pct != 'N/A' and not str(improvement_pct).startswith('-') and improvement_pct != '0.00%' else '#6b7280'};">{improvement_pct}</span>
                                    </div>
                                    {effect_html}
                                </div>
                            </div>
                            """
                            st.markdown(card_html, unsafe_allow_html=True)
        
        elif len(stats_selected_scenarios) < 2:
            st.info("Please select 2 scenarios for comparison")
        elif len(selected_stats_cols) == 0:
            st.info("Please select at least one numeric column for comparison")
    
    def render_analysis_info(self):
        """Render analysis configuration and information"""
        st.info(f"""
        **Analysis configuration:**
        - **Date range:** {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}
        - **Category (aggregate):** {self.selected_agg_col}
        - **Time unit:** {self.unit_min} minutes
        - **Locations:** {', '.join(self.loc_names)}
        - **Fixed per location:** {json.dumps({k: v for k, v in self.fixed_per_loc.items()}, ensure_ascii=False)}
        - **allocate:** {', '.join(sorted(self.moving_unit))}
        
        **Distribution parameters:**
        - **Departure:** mean={self.dep_info.get('mean', 'N/A')}, sigma={self.dep_info.get('sigma', 'N/A')}, clip=[{self.dep_info.get('min_max_clip', ['N/A', 'N/A'])[0]}, {self.dep_info.get('min_max_clip', ['N/A', 'N/A'])[1]}]
        - **Arrival:** mean={self.arr_info.get('mean', 'N/A')}, sigma={self.arr_info.get('sigma', 'N/A')}, clip=[{self.arr_info.get('min_max_clip', ['N/A', 'N/A'])[0]}, {self.arr_info.get('min_max_clip', ['N/A', 'N/A'])[1]}]

        **Explanation of calculated values:**
        - **Terminal A_Dep_95%**: Passenger (or seat) volume at the 5th busiest departure time slot  
        (e.g., if the range has 100 one-hour slots, this is the 5th busiest slot).
        - **Terminal B_Arr_90%**: Passenger (or seat) volume at the 100th busiest arrival time slot  
        (e.g., if the range has 1000 one-hour slots, this is the 100th busiest slot).
        - **Dist_Dep_95%** represents the distance from the baseline (0,0,0) to the (Dep_95% of T1, T2, T3) point in a 3-dimensional space.
        - **Rank_Arr_90%** ranks scenarios by their Dist_Arr_90% score. Lower peak-load distance = better rank.
        """)
        
        if self.is_sampled:
            st.warning(f"⚠️ Sampling **{self.analysis_count:,}** out of {self.total_combinations:,} assignments.")
        else:
            st.info(f"✅ Analyzing all {self.total_combinations:,} assignments.")
    
    def render_calculation_process(self):
        """Render Calculation Process section with scenario selection and visualization."""
        st.markdown("---")
        st.markdown("### 📈 Calculation Process")
        
        C1, C2, C3 = st.columns(3)
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
        with C2:
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
        
        with C3:
            # Dep/Arr 선택
            selected_io = st.selectbox(
                "Select Departure or Arrival",
                options=["Dep", "Arr", "Both"],
                key="calc_profile_io_select"
            )
            
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
            dep_category = f"departure {self.selected_agg_col}"
            arr_category = f"arrival {self.selected_agg_col}"
            
            # Departure 데이터 처리
            df_proc_dep = self.process_passenger_data(filtered_data, dep_category, keep_cols=[self.relocation_unit])
            # Arrival 데이터 처리
            df_proc_arr = self.process_passenger_data(filtered_data, arr_category, keep_cols=[self.relocation_unit])
            
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
            category = f"{io_label} {self.selected_agg_col}"
            
            # 승객 데이터 처리
            df_proc = self.process_passenger_data(filtered_data, category, keep_cols=[self.relocation_unit])
            
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

        time_series_tab, sorted_values_tab = st.tabs(["Time Series", "Sorted Values"])



        with time_series_tab:
            # Capacity 입력창
            capacity = st.number_input(
                "**Capacity**",
                min_value=0,
                value=100,
                step=1,
                key="capacity_input",
                help="이 값 이상인 셀은 연한 빨간색으로 표시됩니다."
            )
            
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
            
            # 스타일링: Capacity 이상인 셀을 연한 빨간색으로 표시
            def highlight_capacity(val):
                """Capacity 이상인 값에 연한 빨간색 배경 적용"""
                if val >= capacity:
                    return 'background-color: #ffcccc'  # 연한 빨간색
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
                    height=500,
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
            
            # df_proc에서 사용 가능한 그룹 컬럼 확인
            available_group_cols = []
            if self.relocation_unit in df_proc.columns:
                available_group_cols.append(self.relocation_unit)
            if "A/C Code" in df_proc.columns:
                available_group_cols.append("A/C Code")
            if "terminal_carrier" in df_proc.columns:
                available_group_cols.append("terminal_carrier")
            if "operating_carrier_name" in df_proc.columns:
                available_group_cols.append("operating_carrier_name")
            if "terminal_iata_int/dom" in df_proc.columns:
                available_group_cols.append("terminal_iata_int/dom")

            if "International/Domestic" in df_proc.columns:
                available_group_cols.append("International/Domestic")
            if "Custom Group" in df_proc.columns:
                available_group_cols.append("Custom Group")
            
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
                height=500,
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
            
            # 합계와 비중을 DataFrame으로 표시
            summary_df = pd.DataFrame({
                'Sum': group_sums,
                'Percentage (%)': group_percentages.round(2)
            })
            summary_df = summary_df.sort_values('Sum', ascending=False)
            
            st.markdown("#### 📊 Summary by Group")
            st.dataframe(summary_df, use_container_width=True)
            
            st.markdown("#### 📊 Time Series Data")
            st.write(stacked_pivot)

        
        with sorted_values_tab:
            # 두 번째 그래프: 내림차순 정렬 그래프 (Max가 가장 왼쪽에)
            st.markdown(f"#### 📊 Sorted Values: {selected_terminal} - {selected_io}")
            sorted_values = time_series.values.copy()
            # 내림차순으로 정렬하여 Max가 가장 왼쪽(인덱스 0)에 오도록
            sorted_values = np.sort(sorted_values)[::-1]
            
            # 통계값 계산 (원본 데이터 기준)
            original_values = time_series.values.copy()
            max_val = original_values.max()
            p975 = np.percentile(original_values, 97.5)
            p95 = np.percentile(original_values, 95)
            p90 = np.percentile(original_values, 90)
            median_val = np.median(original_values)
            mean_val = original_values.mean()
            
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
            
            # 통계값 표시 (Max는 항상 인덱스 0에 있음)
            stats_y = [max_val, p975, p95, p90, median_val, mean_val]
            stats_labels = ['Max', '97.5%', '95%', '90%', 'Median', 'Mean']
            stats_colors = ['red', 'orange', 'yellow', 'green', 'purple', 'cyan']
            
            for stat_val, stat_label, stat_color in zip(stats_y, stats_labels, stats_colors):
                # Max는 항상 인덱스 0에 있음
                if stat_label == 'Max':
                    stat_idx = 0
                else:
                    # 내림차순 정렬된 배열에서 해당 값의 인덱스 찾기
                    stat_idx = np.argmin(np.abs(sorted_values - stat_val))
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
            
            fig2.update_layout(
                title=f'Sorted Values with Statistics: {selected_terminal} - {selected_io}',
                xaxis_title='Index (Sorted, Descending)',
                yaxis_title=f'{self.selected_agg_col.replace("_", " ").title()}',
                height=500,
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hoverlabel=dict(align='right', bgcolor='white', bordercolor='black', font_size=12)
            )
            st.plotly_chart(fig2, use_container_width=True)


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
        return df

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
        return df

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


def _render_puffing(self):
    """Puffing 탭 UI 및 1시간 단위 Stacked Bar 시각화."""
    st.markdown("<div class='card'><h4>🎈 Puffing</h4>", unsafe_allow_html=True)

    if self.df_filtered is None or len(self.df_filtered) == 0:
        st.info("No filtered data available. Please configure Date & Method first.")
        return

    with st.expander("Edit puff groups & preview chart", expanded=False):
        df_puffed, group_col = _render_puff_editor(self)

        # 시간 축 선택 및 집계 (항상 1시간 단위)
        time_candidates = [c for c in ["SHOW", "scheduled_gate_local"] if c in df_puffed.columns]
        if len(time_candidates) > 0:
            time_col = time_candidates[0]
            df_puffed[time_col] = pd.to_datetime(df_puffed[time_col])
            # 1시간 단위로 절단해서 집계
            df_puffed["Time_Hour"] = df_puffed[time_col].dt.floor("H")
            agg_df = (
                df_puffed.groupby(["Time_Hour", group_col])
                .size()
                .reset_index(name="count")
                .sort_values("Time_Hour")
            )

            st.markdown("#### 📊 Puff 결과 – Stacked Bar (1-hour bins)")
            fig = px.bar(
                agg_df,
                x="Time_Hour",
                y="count",
                color=group_col,
                barmode="stack",
            )
            fig.update_layout(
                xaxis_title="Time (hourly)",
                yaxis_title="Count",
                hovermode="x unified",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Time column (`SHOW` or `scheduled_gate_local`) not found – stacked bar is skipped.")

        # Puff 후 df_filtered에 반영하여 이후 분석에 사용되도록 함
        self.df_filtered = df_puffed

@st.fragment
def show_result(analysis=None):
    # session_state에서 최신 analysis 객체 가져오기 (항상 최신 값 사용)
    if analysis is None or 'current_analysis' in st.session_state:
        analysis = st.session_state.get('current_analysis', analysis)
    
    if analysis is None:
        st.info("Please run analysis first")
        return
    
    init_tab, result_table_tab, radial_chart_tab, scenario_compare_tab, scenario_profile_tab = st.tabs(["init","Result Table","Radial Chart","Scenario Compare","Scenario Profile"])
    
    with init_tab:
        analysis.render_analysis_info()

    with result_table_tab:
        analysis.render_statistics_table()
        analysis.render_distance_visualization()
    
    with radial_chart_tab:
        analysis.render_radial_chart()
        
    with scenario_compare_tab:
        analysis.render_scenario_compare()
        
    with scenario_profile_tab:
        analysis.render_calculation_process()



@st.fragment
def main(df_orig):
    apply_css()
    relocation_settings_tab, assign_units_tab, result_tab = st.tabs(["Relocation Settings","Assign Units","Result"])

    # RelocationSetting 클래스 인스턴스 생성
    settings = RelocationSetting(df_orig)
    
    with relocation_settings_tab:
        settings.render_relocation_settings_tab()
    
    with assign_units_tab:
        settings.render_assign_units_tab()
    
    # 분석 실행
    with result_tab:
        if st.button("🚀 Run analysis"):
            # RunAnalysis 클래스 인스턴스 생성 (분석 실행)
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
                selected_agg_col=settings.selected_agg_col,
                max_sample_count=settings.max_sample_count,
                unassigned_units_map=settings.unassigned_units_map if len(settings.unassigned_units_map) > 0 else None
            )
            # session_state에 analysis 객체 저장 (최신 값 유지)
            st.session_state['current_analysis'] = analysis
            show_result(analysis)

if __name__ == "__main__":
    st.title("🔀 Relocation Master")
    ms=MasterplanInput()
    # ms.select_airport_block()
    df_orig = processing_data()
    st.write(df_orig.head(100))



    
    main(df_orig)