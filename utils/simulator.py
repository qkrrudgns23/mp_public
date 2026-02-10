import streamlit as st
from utils.cirium import *
import pandas as pd
import numpy as np
import graphviz
import plotly.express as px
import datetime
import plotly.graph_objects as go
from datetime import datetime, time, date
from functools import reduce
import heapq
import random

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
    </style>
    """
    return st.markdown(css, unsafe_allow_html=True)


def select_target_day(df_orig, key="default"):
    c1, c2, c3, c4, c5, c6 = st.columns([0.15,0.25,0.15,0.15,0.15,0.15])

    # NOTE : Select Departure or Arrival Flight
    with c1:
        flight_io_list = st.multiselect(
            "**Flight Dep/Arr**",
            ["d", "a"],
            default=["d"],
            key=f"select Flight I/O_oag{key}",
        )
    with c2:
        target_year_list = st.multiselect(
            "**Reference year**",
            [2019, 2024, 2025],
            default=[2019, 2024, 2025],
            key=f"select Reference year{key}",
        )

        df_orig = df_orig[df_orig["scheduled_gate_local"].notna()]
        df_orig = df_orig[df_orig["flight_io"].isin(flight_io_list)]

        df_orig["scheduled_gate_local"] = pd.to_datetime(
            df_orig["scheduled_gate_local"]
        )
        df_orig["day_of_week"] = df_orig["scheduled_gate_local"].dt.day_name()
        df_orig["week_number"] = df_orig["scheduled_gate_local"].dt.isocalendar().week
        df_orig["year"] = df_orig["scheduled_gate_local"].dt.year
        df_orig = df_orig[df_orig["year"].isin(target_year_list)]

    with c3:
        target_terminal_list = st.multiselect(
            "**Terminal**",
            df_orig["terminal"].unique(),
            default=df_orig["terminal"].unique(),
            key=f"select Reference terminal{key}",
        )
        df_orig = df_orig[df_orig["terminal"].isin(target_terminal_list)]




    with c4:
        method = st.selectbox(
            "**method**", ["Annual","Winter season", "Summer season", "First Half", "Second Half"], key=f"method"
        )
        method_dict = {"Annual":{"top date":30, "peak start":15, "peak end":30},
                        "Winter season":{"top date":13, "peak start":7, "peak end":13},
                        "Summer season":{"top date":18, "peak start":9, "peak end":18},
                        "First Half":{"top date":15, "peak start":8, "peak end":15},
                        "Second Half":{"top date":15, "peak start":8, "peak end":15}}

    with c5:
        top_date = st.number_input(
            "**Top date**", value=method_dict[method]["top date"], min_value=1, key=f"top_date{key}"
        )

    with c6:

        future_year = st.selectbox(
            "**Target year**", range(2000, 2050), index=25, key=f"target_year{key}"
        )


    # 각 행의 주차 계산
    def first_sunday(year):
        first_day = pd.Timestamp(f"{year}-01-01")
        first_sunday = first_day + pd.DateOffset(days=(6 - first_day.weekday()))
        return first_sunday

    # 각 행의 첫 번째 일요일로부터의 일수 계산
    first_sundays = {year: first_sunday(year) for year in df_orig["year"].unique()}

    df_orig["first_sunday"] = df_orig["year"].map(first_sundays)
    df_orig["days_from_first_sunday"] = (
        df_orig["scheduled_gate_local"] - df_orig["first_sunday"]
    ).dt.days
    df_grouped = df_orig.groupby(["days_from_first_sunday", "year"])["total_seat_count"].agg("sum").unstack()
    df_orig["dates"]=df_orig["scheduled_gate_local"].dt.date.astype(str)
    df_grouped["Average"] = df_grouped.mean(axis=1)
    df_grouped = df_grouped.stack().reset_index(name="count")

    def method_filter(df, method):
        if method=="Summer season":
            start_mask = df["days_from_first_sunday"]>=84
            end_mask = df["days_from_first_sunday"]<=293
            df = df[start_mask & end_mask]
        elif method =="Winter season":
            start_mask = df["days_from_first_sunday"]<84
            end_mask = df["days_from_first_sunday"]>293
            df = df[start_mask | end_mask]
        elif method =="First Half":
            start_mask = df["days_from_first_sunday"]>=0
            end_mask = df["days_from_first_sunday"]<=181
            df = df[start_mask & end_mask]
        elif method =="Second Half":
            start_mask = df["days_from_first_sunday"]>=182
            df = df[start_mask]

        return df
    df_grouped = method_filter(df_grouped, method)
    df_grouped_max = (
        df_grouped[df_grouped["year"] == "Average"]
        .sort_values(by="count", ascending=False)
        .head(top_date)
    )
    df_grouped_max["year"] = "max"




    fig = go.Figure()
    for year in df_grouped["year"].unique():
        df_year = df_grouped[df_grouped["year"] == year]
        # 호버 텍스트 생성
        if year != "Average":
            df_year[f"{year}_date"] = pd.to_datetime(
                first_sundays[year]
            ) + pd.to_timedelta(df_year["days_from_first_sunday"], unit="d")
            hover_text = [
                f"▶Week: {week}<br>▶day: {day_name}<br>Date: {date}<br>Seat Count: {count}"
                for count, date, week, day_name in zip(
                    df_year["count"],
                    df_year[f"{year}_date"].dt.date,
                    df_year[f"{year}_date"].dt.isocalendar().week,
                    df_year[f"{year}_date"].dt.day_name(),
                )
            ]
            fig.add_trace(
                go.Scatter(
                    x=df_year["days_from_first_sunday"],
                    y=df_year["count"],
                    mode="lines",
                    name=str(year),
                    hoverinfo="text",  # 'text'만 표시하도록 설정
                    hovertext=hover_text,
                    line=dict(width=0.3, color=None),
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=df_year["days_from_first_sunday"],
                    y=df_year["count"],
                    mode="lines",
                    name=str(year),
                    line=dict(width=2, color="#007aff"),
                )
            )

    fig.add_trace(
        go.Scatter(
            x=df_grouped_max["days_from_first_sunday"],
            y=df_grouped_max["count"],
            mode="markers",
            name="Peak Periods",
            marker=dict(color="red", size=15),
        )
    )
    fig.update_layout(
        xaxis_title="Days from [First Week] & [First Sunday]", yaxis_title="Total Seat Count"
    )


    df_grouped_max["date"] = df_grouped_max["days_from_first_sunday"].apply(
        lambda x: first_sunday(future_year) + pd.DateOffset(days=x)
    )
    df_grouped_max["day_of_week"] = df_grouped_max["date"].dt.day_name()
    df_grouped_max["week_number"] = df_grouped_max["date"].dt.isocalendar().week
    df_grouped_max["date"] = df_grouped_max["date"].dt.date
    seleced_target_day = df_grouped_max.drop(
        ["year", "count", "days_from_first_sunday"], axis=1
    ).reset_index(drop=True)
    seleced_target_day.index = seleced_target_day.index + 1
    
    st.caption(f"""✅ Based on the :blue[**{method}**] characteristics from :blue[**{(target_year_list)}**], select :blue[**{top_date} predicted Peak Period**] for the year :blue[**[{future_year}]**]""")
    st.caption(
        "✅ :blue[**Annual:**] From :blue[**Jan 1st to Dec 31st**]"
        if method == "Annual"
        else "✅ :blue[**Summer Season:**] From :blue[**Last Sunday of March to Last Saturday of October**]"
        if method == "Summer season"
        else "✅ :blue[**Winter Season:**] From :blue[**Last Sunday of October to Last Saturday of March**]"
        if method == "Winter season"
        else "✅ :blue[**First Half:**] From :blue[**1st Sunday to 26th Saturday**]"
        if method == "First Half"
        else "✅ :blue[**Second Half:**] From :blue[**26th Sunday to Last Day of Year**]"
    )

    st.caption(f"""✅ :blue[**Measurement Day:**] Select date from the :blue[**{method_dict[method]["peak start"]}th to {method_dict[method]["peak end"]}th**] Peak Period""")
    st.plotly_chart(fig)



    # def highlight_rows(row):
    #     if row.name in range(method_dict[method]["peak start"], method_dict[method]["peak end"]+1):
    #         return ['background-color: #A0E7A0']*len(row)
    #     return ['']*len(row)
    
    def highlight_rows(row):
        if row.name in range(method_dict[method]["peak start"], method_dict[method]["peak end"]+1):
            return ['background-color: rgba(160, 231, 160, 0.5)'] * len(row)  # 0.5는 투명도 50%
        return [''] * len(row)


    styled_df = seleced_target_day.style.apply(highlight_rows, axis=1)
    st.subheader("**Peak Period**")
    st.write(styled_df.to_html(escape=False), unsafe_allow_html=True)



    df_grouped_total = df_grouped.sort_values(by="count", ascending=False)
    df_grouped_total["year"] = df_grouped_total["year"].astype(str)
    df_grouped_total = df_grouped_total.set_index(["year","days_from_first_sunday"]).unstack().T.reset_index().drop(["level_0"],axis=1)
    df_grouped_total=df_grouped_total//1
    df_grouped_total=df_grouped_total
    st.table(df_grouped_total)


def show_df_schedule(df, group, key, time_col, y_label, title, unit_min=30):
    """
    Visualize schedule data with a bar chart.
    :param df: DataFrame containing the schedule data
    :param key: A unique key to avoid duplication when the function is used multiple times
    :param time_col: time column name(str)
    :param unit_min: Time unit for rounding schedule times (default is 15 minutes)

    :streamlit_param group_col: Column name for grouping data, selected via radio widget
    """
    df = df.copy()
    df[time_col] = df[time_col].dt.floor(f"{unit_min}T")
    df_grouped = (
        df.groupby([time_col, group]).size().unstack().fillna(0).stack().reset_index()
    )
    df_grouped.columns = [time_col, group, y_label]
    # Create a bar chart
    fig = px.bar(df_grouped, x=time_col, y=y_label, color=group, title=title)
    start_date = df_grouped[time_col].min()
    end_date = df_grouped[time_col].max()
    fig.update_xaxes(
        range=[
            datetime.combine(start_date.date(), time.min),
            datetime.combine(end_date.date() + pd.Timedelta(days=1), time.min),
        ],
        tickformat="%Y-%m-%d %H:%M",  # 년-월-일 시:분 형식으로 표시
    )

    # Display the bar chart
    st.plotly_chart(fig)


def show_distribution(df, group_col, time_col):
    """
    Draw a graph with the x-axis as "time_col" and the y-axis as the "count of the dataframe", with the legend and grouping determined by "group_col".
    :param df: DataFrame to plot
    :param group_col: Column name for grouping (str)
    :param time_col: Column name for time axis (str)
    :return: The resulting figure (fig)
    """
    # Create a copy of the dataframe to avoid modifying the original
    df = df.copy()

    # Ensure the time column is in integer format
    df["minute difference"] = df[time_col].astype(int)

    # Group the dataframe by time and group column, then unstack to get counts
    df_grouped = df.groupby(["minute difference", group_col]).size().unstack()

    # Create a complete time index
    min_ck = df_grouped.index.min()
    max_ck = df_grouped.index.max()
    time_index = pd.DataFrame({"minute difference": range(min_ck, max_ck + 1)})

    # Merge the grouped dataframe with the complete time index
    df_grouped = pd.merge(time_index, df_grouped, on="minute difference", how="left")
    df_grouped = df_grouped.set_index("minute difference")

    # Normalize the data
    normalized = df_grouped / df_grouped.sum(axis=0)
    normalized = normalized.fillna(0)
    normalized = normalized.rolling(window=3, min_periods=1, center=True).mean()
    normalized = normalized.reset_index()

    # Draw and display the figure
    fig = px.line(
        normalized, x="minute difference", y=normalized.columns[1:], markers=True
    )
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[0],  # y 좌표는 0으로 설정하거나, 필요에 따라 조정할 수 있습니다
            mode="markers",
            marker=dict(size=19, color="#007aff", symbol="circle"),
            name=f"Flight",
            showlegend=True,
        )
    )

    # 레이아웃 업데이트 (선택사항)
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig)


def create_normal_dist_col(
    df,
    ref_col,
    new_col,
    mean,
    sigma,
    min_max_clip,
    unit="m",
    iteration=6,
    datetime=False,
):
    """
    Adds a new column to the DataFrame by creating normally distributed random values
    and adding them to an existing column.
    :param df: DataFrame to modify
    :param ref_col: Reference column to which the normal distribution values will be added
    :param new_col: Name of the new column to be created
    :param mean: Mean of the normal distribution
    :param sigma: Standard deviation of the normal distribution
    :param min_max_clip: Tuple (min, max) to clip the normal distribution values
    :param unit: Unit of time for timedelta if datetime is True (default is "m" for minutes)
    :param iteration: Number of iterations for resampling out-of-range values (default is 6)
    :param datetime: Boolean flag to handle datetime operations (default is False)
    :return: DataFrame with the new column added
    """
    # Generate normally distributed random values
    random_arr = np.random.normal(mean, sigma, size=len(df))
    # Check if the mean is within the clipping range
    assert (
        min_max_clip[0] <= mean <= min_max_clip[1]
    ), "mean 값이 clipping 범위를 넘어서고 있습니다 >> min~max clipping 범위 내로 mean값을 재설정해주세요"
    # Resample values outside the clipping range up to "iteration" times
    for _ in range(iteration):
        out_of_range_indices = np.where(
            (random_arr < min_max_clip[0]) | (random_arr > min_max_clip[1])
        )
        random_arr[out_of_range_indices] = np.random.normal(
            mean, sigma, size=len(out_of_range_indices[0])
        )
    # Add the generated values to the reference column and create the new column
    if datetime == False:
        df[new_col] = df[ref_col] + random_arr
    elif datetime == True:
        # Convert the random values to timedelta if handling datetime
        timedelta_arr = pd.to_timedelta(random_arr, unit=unit)
        timedelta_arr = timedelta_arr.round("S")
        df[new_col] = df[ref_col] + timedelta_arr
    return df


@st.fragment
def select_airport(return_dict):
    """
    Select and filter flight schedule based on user input.
    :param return_dict: Dictionary to store the filtered DataFrame
    :return: Updated return_dict with the filtered DataFrame

    :streamlit_param airport_code: Airport code selected via selectbox widget
    :streamlit_param selected_date: Date selected via selectbox widget
    :streamlit_param terminal: Terminal selected via selectbox widget
    """
    container_select_airport = st.container(border=True)
    with container_select_airport:
        (
            select_airport_tab,
            show_up_tab,
            pax_data_tab,
            profile_airport_tab,
        ) = st.tabs(
            [
                "**SELECT AIRPORT**",
                "**SHOW-UP PATTERN**",
                "**GENERATE PASSENGERS**",
                "**🔎 SURVEY DATE**",
            ]
        )

        # =======================================================================
        # NOTE: SELECT AIRPORT
        with select_airport_tab:
            st.title(":blue[SELECT AIRPORT]")

            uploaded_file = st.file_uploader("Upload Schedule File")
            c1, c2, c3, c4, c5 = st.columns(5, gap="small")
            # NOTE : 공항선택
            with c1:
                # NOTE : 시리움 연결하기
                # ["ICN", "IST", "NRT", "BLR","MNL", "SGN", "BKK","DXB","CGK","PER","DAC","GRU","MVD"]
                if uploaded_file is None:
                    airport_options = {
                        "ICN": "Inchoen Airport",
                        "IST": "Istanbul Airport",
                        "NRT": "Narita International Airport",
                        "BLR": "Bangalore International Airport",
                        "MNL": "Manila International Airport",
                        "SGN": "Tan Son Nhat International Airport",
                        "BKK": "Suvarnabhumi International Airport",
                        "DXB": "Dubai International Airport",
                        "CGK": "Soekarno-Hatta International Airport",
                        "PER": "Perth International Airport",
                        "DAC": "Hazrat Shahjalal International Airport",
                        "GRU": "Sao Paulo International Airport",
                        "MVD": "Carrasco International Airport",
                        "BAH": "Bahrain International Airport",
                        "UGC": "Urgench International Airport",
                        "HAN": "Noi Bai International Airport",
                        "BGI": "Grantley Adams International Airport",
                    }

                    code_str = [
                        st.selectbox(
                            "**Airport**",
                            options=list(airport_options.keys()),
                            format_func=lambda x: f"{x} - {airport_options[x]}",
                            # default=[list(airport_options.keys())[0]],  # 첫 번째 항목을 기본 선택
                            index=1,
                            key="select_airport__",
                        )
                    ]
                    if len(code_str) == 0:
                        st.write("기본공항 : UGC")
                        code_str = ["UGC"]

                    airport_code = code_str[0]
                    return_dict["airport_code"] = airport_code
                    st.write(f"{airport_code}_schedule_ready.parquet")
                    df_orig = pd.read_parquet("cirium/" + f"{airport_code}_schedule_ready_.parquet"
                    )

                    # df_orig = pd.read_parpassenger_queuest(
                    #     "../../oag_database/" + f"{airport_code}_OAG_SC_Processed.parpassenger_queuest"
                    # )
                    df_orig["flight_number"]= df_orig["operating_carrier_iata"] + df_orig["flight_number"].astype(str)
                    st.write(df_orig.groupby(["year"])["total_seat_count"].sum())

                    uploded_file_is = False
                # NOTE : 직접 업로드
                else:
                    df_orig = pd.read_parquet(uploaded_file)
                    uploded_file_is = True

            # NOTE : 날짜선택
            with c2:
                """
                Cirium 데이터에는 2가지 종류가 있다.
                1) Flight Status : 과거실적데이터
                2) Schedule : 미래(today~ +1year)
                selected_date 가, today ~ 미래일 경우 >> Schedule에서 받아와야 함
                selected_date 가  today 미만일 경우 >> Flight_status 에서 가져와야 함
                """
                if uploded_file_is:
                    crowded_date = (
                        df_orig["scheduled_gate_local"].dt.date.value_counts().index[0]
                    )
                    selected_date = st.date_input(
                        "select date",
                        value=crowded_date,
                        min_value=df_orig["scheduled_gate_local"].min().date(),
                        max_value=df_orig["scheduled_gate_local"].max().date(),
                    )
                else:
                    selected_date = st.date_input(
                        "select date", value=date(2024, 10, 5)
                    )
                return_dict["selected_date"] = selected_date

            # NOTE:출발편 / 출발편 "d", 도착편 "a"
            with c3:
                flight_io = st.selectbox(
                    "**Flight Dep/Arr**",
                    ["d", "a"],
                    key="select Flight I/O",
                )
                return_dict["flight_io"] = flight_io

                # NOTE: 중요!!!
                # Scheduled_gate_local >> 미래 날자를 선택했을 경우, 미래는 Actual이 없다 >> Scheduled_gate_local 을 쓴다
                # 여객들이 공항에 언제 도착할지를 선택하는 시간이 보딩패스에 예상 출발시간이므로, 예상 출발시간을 써야 한다.
                # 예를들어 VIncent 가 14:30 출발 비행기를 탄다 >> 14:30분은 예정일까 실제일까?? >> 이건 예정시간이다 >> scheduled_gate_local

                df = df_orig[
                    (df_orig["flight_io"] == flight_io)
                    & (df_orig["primary_usage"] == "Passenger")
                    & (df_orig["scheduled_gate_local"].dt.date == selected_date)
                ]
                flight_io_dt = "scheduled_gate_local"


            # NOTE : 터미널 선택
            with c4:
                # 실렉트가 멀티인지, 싱글인지 이것도 중요
                terminal_list = df[f"terminal"].value_counts().index
                terminal = st.multiselect(
                    "**Terminal**",
                    terminal_list,
                    default=terminal_list[0],
                    key="terminal_list",
                )
                return_dict["terminal"] = terminal
                
                df = df[df[f"terminal"].isin(terminal)].reset_index(
                    drop=True
                )  # Filter by selected terminal


                # 실렉트가 멀티인지, 싱글인지 이것도 중요
                airline_lists = df[f"operating_carrier_iata"].value_counts().index.tolist()
                selected_airline_lists = st.multiselect(
                    "**Airline**",
                    airline_lists,
                    default=airline_lists,  # 모든 항목을 기본 선택
                    key="airline_lists",
                )
                
                df = df[df[f"operating_carrier_iata"].isin(selected_airline_lists)].reset_index(
                    drop=True
                )  # Filter by selected terminal



            # NOTE : 그래프 컬러(그룹핑) 기준
            with c5:
                group = st.selectbox(
                    "**grouping**",
                    [
                        "operating_carrier_iata",
                        "flight_number",
                        "International/Domestic",
                        "dep/arr_airport",
                    ],
                    key=f"Select Group Column",
                )
            st.success(f"✅ Total : {len(df)} Flight 🛫")

            # NOTE : 바차트 그림 그리는 함수
            show_df_schedule(
                df=df,
                group=group,
                key="generation",
                time_col=flight_io_dt,
                y_label="Movement (Dep)",
                title="",
                unit_min=60,
            )
            with st.expander("Raw Flight Data"):
                st.dataframe(df, hide_index=True)
                return_dict["df"] = df.copy()


        # =======================================================================

        # =======================================================================
        # NOTE: SHOW-UP PATTERN
        with show_up_tab:
            st.title(":blue[SHOW-UP PATTERN]")
            group_columns = st.multiselect(
                "Category",
                [
                    "International/Domestic",
                    "country_name",
                    "region_name",
                    "alliance",
                ],
                ["International/Domestic"],
            )
            df["new_col"] = df[group_columns].astype(str).agg(" & ".join, axis=1)
            col_values = df["new_col"].unique().tolist()

            # df_pax의 최종모습 (360편) >> 한편당 100명 >> 36000명
            df_pax_list = []

            # Iterate over each unique group value
            for value in col_values:
                container = st.container(border=True)
                with container:
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.subheader(value)
                    with c2:
                        load_factor = st.number_input(
                            f"**Load Factor**?",
                            step=0.01,
                            key=f"load_factor{value}",
                            value=0.85,
                        )
                    with c3:
                        mean = st.number_input(
                            f"**mean time**?",
                            step=1,
                            format="%d",
                            key=f"mean_value{value}",
                            value=- 120,
                        )
                        sigma = abs(mean/3)
                        mean_max_clip = [-400,-30]

                    value_list = value.split(" & ")

                    # NOTE : 예시 값들
                    # group_columns = ["Int/Dom", "carirers"]
                    # value_list = ["Int", "Korean Air"]
                    # df[group_columns] == value_list
                    filter_condition = (df[group_columns] == value_list).all(axis=1)
                    filtered_df = df[filter_condition]
                    filtered_df["Passenger (Total)"] = round(
                        filtered_df["total_seat_count"] * load_factor
                    )

                    # Passenger (Total)가 260명일 경우, 260개의 동일한 로우로 뻥튀기
                    filtered_df = filtered_df.loc[
                        filtered_df.index.repeat(filtered_df["Passenger (Total)"])
                    ].reset_index(drop=True)

                    # flight_io_dt = scheduled_gate_local >> 0

                    filtered_df[f"{flight_io_dt}(min)"] = 0
                    filtered_df = create_normal_dist_col(
                        df=filtered_df,
                        ref_col=f"{flight_io_dt}(min)",
                        new_col="SHOW(min)",
                        mean=mean,
                        sigma=sigma,
                        min_max_clip=mean_max_clip,
                        unit="m",
                        iteration=6,
                        datetime=False,
                    )
                    df_pax_list.append(filtered_df)
                    df_pax = pd.concat(df_pax_list)
                    df_pax["SHOW"] = df_pax[flight_io_dt] + pd.to_timedelta(
                        df_pax["SHOW(min)"], unit="m"
                    )
                    df_pax["All Passengers"] = "All Passengers"
                    df_pax["One Source"] = "One Source"
                    return_dict["df_pax_orig"] = df_pax.copy()

            proportion_series = df_pax["SHOW(min)"].round().value_counts().idxmax()
            if return_dict["flight_io"] == "departure":
                st.success(
                    f"✅ Passengers arrived🚶‍♂️ most frepassenger_queuesntly :blue[**{proportion_series*-1} minutes Before**] aircraft departure 🛫"
                )
                with st.popover("💡Info"):
                    st.image("data/raw/simulation_explain/arrival_distribution.png")
            elif return_dict["flight_io"] == "arrival":
                st.success(
                    f"✅ Passengers arrived🚶‍♂️ most frepassenger_queuesntly :blue[**{proportion_series} minutes After**] aircraft Arrival 🛬"
                )

            show_distribution(df_pax, group_col="new_col", time_col="SHOW(min)")
        # =======================================================================

        # =======================================================================
        # NOTE: GENERATE PASSENGERS
        with pax_data_tab:
            st.title(":blue[GENERATE PASSENGERS]")

            # NOTE: Select Bar-chart Group column
            group = st.selectbox(
                "**grouping**",
                [
                    "operating_carrier_iata",
                    "flight_number",
                    "International/Domestic",
                    "dep/arr_airport",
                    "gate",
                    "country_code",
                ],
                key=f"Select Group Column_",
            )
            average_seat = int(df["total_seat_count"].mean())
            total_pax = len(df_pax)
            flight = len(df)
            load_factor = total_pax / df["total_seat_count"].sum()
            st.success(
                f"✅ Total : {total_pax} pax🚶‍♂️  = Flight({flight}) x avg_seat({average_seat}) x Load_factor({int(load_factor*1000)/10}%)"
            )

            # NOTE: Show Passenger Bar-Graph
            show_df_schedule(
                df=df_pax,
                group=group,
                key="pax_generation",
                time_col="SHOW",
                y_label="Departure Passenger",
                title="Passenger Traffic",
                unit_min=15,
            )

            with st.expander("Raw Passenger Data"):
                st.dataframe(df_pax, hide_index=True)
        # =======================================================================

        # =======================================================================
        # NOTE: Historical Data (Movements)
        # SOURCE : df_orig >> 예를들어 ICN 선택했다면 ICN의 데이터가 들어온다
        with profile_airport_tab:

            st.write(df_orig.groupby(["terminal","year"])["total_seat_count"].sum())
            st.title(":blue[SURVEY DATE]")
            select_target_day(df_orig, key="basic")
        # =======================================================================

        # st.dataframe(df)

    return return_dict

    # return_dict['df_pax']


# =======================================================================
# NOTE : df_pax에 새로운 컬럼을 할당하고, value를 채워주는 함수
@st.fragment
def add_properties(return_dict, num_of_properties=1):
    """
    Add new property columns to the DataFrame based on user inputs.
    """
    return_dict["add_properties"] = {}

    st.title(f":blue[ADD PROPERTIES]")

    st.info(
        """
    💡 Used when passenger ratio information is known and needs to be reflected in surveys / simulations.\n
    📌 CASE 1) When the ratio of PRM passengers is known \n
    📌 CASE 2) When the ratio of passenger nationality is known \n
    📌 ETC..
    """
    )



    # 사용자로부터 새 속성의 수를 입력받음
    st.subheader(f"**How many New Properties you want**?")

    # NOTE: 몇개의 새로운 컬럼을 생성하고 싶은가요?
    num_of_properties = st.number_input(
        f"**How many New Properties you want**?",
        min_value=1,
        step=1,
        format="%d",
        key=f"how many",
        value=num_of_properties,
    )
    st.divider()

    # NOTE : 같은 값으로 들어가지 못하게 Alert를 하는 작업 필요
    for idx in range(num_of_properties):
        st.subheader(f"Property No {idx + 1}")
        col1, col2, col3 = st.columns(3)

        # NOTE : SET COLUMN NAME
        with col1:
            # 새 속성 열 이름 입력
            column_name = st.text_input(
                "**Column Name**",
                value="PRM Status",
                key=f"new_property_column_name{idx}",
            )

        # NOTE : SET VERTICAL_AXIS 즉
        with col2:
            # 세로축 속성선택
            vertical_property = st.selectbox(
                "**Vertical Column**",
                [
                    "operating_carrier_iata",
                    "flight_number",
                    "tail_number",
                    "International/Domestic",
                    "dep/arr_airport" "gate",
                    "country_code",
                ],
                key=f"exist_property{idx}",
            )
            vertical_index = return_dict["df"][vertical_property].unique()
        with col3:
            # 쉼표로 구분된 새 속성 값 입력
            horizontal_str = st.text_input(
                "**Horizontal Column (separated by commas)**",
                value="With children, Elderly, Disabilities, Non-PRM",
                key=f"new_property{idx}",
            )
            horizontal_columns = [item.strip() for item in horizontal_str.split(",")]

        # 선택된 속성 값으로 속성 행렬 DataFrame 생성
        property_matrix = pd.DataFrame(index=vertical_index, columns=horizontal_columns)
        property_matrix = property_matrix.fillna(1 / len(property_matrix.columns))
        # 사용자에게 속성 행렬 편집 허용
        property_matrix = st.data_editor(
            property_matrix, use_container_width=True, key=f"property{idx}"
        )
        property_matrix = property_matrix.div(property_matrix.sum(axis=1), axis=0)
        # 속성 행렬을 반환 사전에 저장
        return_dict["add_properties"][column_name] = property_matrix

        ######################
        def sample_node(row, edited_df):
            """
            Sample a node based on the probabilities from the choice matrix.
            :param row: The row of the DataFrame to sample from
            :param edited_df: The choice matrix DataFrame with probabilities
            :return: The sampled node
            """
            probabilities = edited_df.loc[row]
            return np.random.choice(probabilities.index, p=probabilities.values)

        # Copy the passenger DataFrame from return_dict
        df_pax = return_dict["df_pax_orig"].copy()
        # Iterate over each process transition in the choice
        df_pax[column_name] = df_pax[vertical_property].apply(
            sample_node, args=(property_matrix.fillna(0),)
        )
        with st.expander("Show Passenger Data"):
            st.dataframe(df_pax)
        return_dict["df_pax_orig"] = df_pax.copy()
        #####################

    return return_dict


# =======================================================================


def set_process(return_dict):
    """
    Set up the process based on user input.
    :param return_dict: Dictionary to store the process configuration
    :return: Updated return_dict with the process configuration

    :streamlit_param component_list: List of processes entered via text_input widget(checkin, departure, security...)
    :streamlit_param node_list: List of node names for each process entered via text_input widget
    :streamlit_param process_category: Category type for each process, selected via selectbox widget(node or condition)
    :streamlit_param property_col: Property column for conditions, selected via selectbox widget
    """

    st.title(f":blue[SET TOUCH-POINT]")

    st.info(
        """
    💡 You can create a flow for any Touch-Point in your airport.\n
    💡 This allows you to customize the flow according to your airport"s specific operations and needs.\n
    """
    )

    # Input for process list
    component_list = st.text_input(
        "**Enter Touch-Point, separated by commas**",
        value="Check in, Boarding Pass Control, Security, Passport",
        key=f"set_process",
    )
    return_dict["component_list"] = [item.strip() for item in component_list.split(",")]
    node_value_list = []
    default_destination = return_dict["component_list"][1:] + [None] * 10000
    default_process_category_values = [
        "A",
        "BP",
        "SC",
        "PC",
        "A0,B0,C0",
        "A1,B1,C1",
        "A2,B2,C2",
        "A3,B3,C3",
        "A4,B4,C4",
        "A5,B5,C5",
        "A6,B6,C6",
        "A7,B7,C7",
        "A8,B8,C8",
    ]
    for process in return_dict["component_list"]:
        return_dict[process] = {}
        return_dict[process]["source"] = []
        return_dict[process]["dst"] = []
        return_dict[process]["node_list"] = None

    for idx, process in enumerate(return_dict["component_list"]):
        container = st.container(border=True)
        with container:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader(f"**:blue[{process}]**")
            with col2:
                # Input for node list
                node_list = st.text_input(
                    "**Enter Service-Point, separated by commas**",
                    default_process_category_values[idx],
                    key=f"how many category{process}{idx}",
                )

                # Select destination for each process
                if idx == len(return_dict["component_list"]) - 1:
                    dst_col = []
                else:
                    dst_col = [default_destination[idx]]
                dst = list(return_dict["component_list"])
                dst.remove(process)

                if dst_col:
                    return_dict[process]["dst"] = dst_col

            with col3:
                if idx == 0:
                    add_property_list = list(return_dict["add_properties"].keys())
                    property_list = st.multiselect(
                        "**What kind of sources?**",
                        [
                            "operating_carrier_name",
                            "operating_carrier_iata",
                            "flight_number",
                            "tail_number",
                            "International/Domestic",
                            "All Passengers",
                            "dep/arr_airport",
                            "gate",
                            "country_code",
                            "One Source",
                        ]+add_property_list,
                        ["operating_carrier_name"],
                        key=f"source_{process}",
                    )

                    property_col = " & ".join(property_list)
                    return_dict["df_pax_orig"][property_col] = (
                        return_dict["df_pax_orig"][property_list]
                        .astype(str)
                        .agg(" & ".join, axis=1)
                        .copy()
                    )
                    return_dict[process]["source"] = [property_col]
                    return_dict[process]["root"] = 1
                    node_list = [item.strip() for item in node_list.split(",")]
                    return_dict[process]["node_list"] = node_list
                    node_value_list += node_list

                else:
                    return_dict[process]["root"] = 0
                    node_list = [item.strip() for item in node_list.split(",")]
                    return_dict[process]["node_list"] = node_list
                    node_value_list += node_list

    # Warning for duplicate node values
    component_value_series = pd.Series(return_dict["component_list"]).value_counts()
    node_value_series = pd.Series(node_value_list).value_counts()
    exceed_component = component_value_series[component_value_series > 1].index.tolist()
    exceed_nodes = node_value_series[node_value_series > 1].index.tolist()

    if len(exceed_nodes) > 0:
        for exceed_value in exceed_nodes:
            st.error(
                f"value name [{exceed_value}] Duplicated!!  please use unique node name",
                icon="🚨",
            )
    else:
        st.write("")
    if len(exceed_component) > 0:
        for exceed_value in exceed_component:
            st.error(
                f"value name [{exceed_value}] Duplicated!!  please use unique component name",
                icon="🚨",
            )
    else:
        st.write("")

    return return_dict


def node_chunk(process, return_dict):
    """
    Configure facilities for each process node based on user input.
    :param process: The process name
    :param return_dict: Dictionary to store the facility configuration
    :return: Updated return_dict with the facility configuration

    :streamlit_param node_list: List of nodes for the process, taken from return_dict
    :streamlit_param facility_num: Number of facilities for each process node, input via number_input widget
    :streamlit_param max_capacity_length: Maximum passenger_queuesue length for each facility, input via number_input widget
    :streamlit_param transaction_time: Time taken by one counter machine to process each passenger, input via number_input widget
    :streamlit_param transaction_df: DataFrame for facility transaction times, edited via data_editor widget
    :streamlit_param capa_df: DataFrame for hourly transaction capacity, edited via data_editor widget
    """
    node_list = return_dict[process]["node_list"]
    st.title(f":blue[{process.upper()}]")
    return_dict[process]["facility_detail_list"] = []
    return_dict[process]["facility_type"] = [] ## 2024.04.24 수정 ##
    return_dict[process]["facility_nums"] = []
    return_dict[process]["max_capacity"] = []
    for idx, col in enumerate(st.tabs(node_list)):
        if "capacity_datas" in return_dict:
            if node_list[idx] in return_dict["capacity_datas"]:
                default_value = False
                designed_capa = return_dict["capacity_datas"][node_list[idx]]
                designed_capa = designed_capa.set_index("Unnamed: 0")
                designed_capa.index.name = "index"
            else:
                default_value = True
        else:
            default_value = True

        with col:
            facility_type = st.selectbox(
                "**Facility Type**",
                [
                    "normal_facility","infinite_facility"
                ],
                index=0,
                key=f"{process} facility_type{idx}{process}",
            )  ## 2024.04.24 수정 ##
            return_dict[process]["facility_type"].append(facility_type)  ## 2024.04.24 수정 ##


            col1, col2, col3 = st.columns(3)
            with col1:
                ###################  ## 2024.04.24 수정 ## #######################
                st.subheader(f"1. How many Desks/Facilities?")
                # Input for the number of facilities
                if facility_type == "infinite_facility":  
                    facility_num=1
                    st.write("보이지 않게 하지만, 자동으로 facility_num = 1로 설정")

                elif facility_type == "normal_facility":   
                    if default_value == True:
                        default_facility_num = 2
                    elif default_value == False:
                        default_facility_num = len(designed_capa.columns)

                    facility_num = st.number_input(
                        f"💡 :blue[**Facilities**] : Num of Desks within Node",
                        min_value=1,
                        value=default_facility_num,
                        step=1,
                        format="%d",
                        key=f"{process} facilities{idx}{process}",
                    )

                facility_list = [f"no{str(i+1)}" for i in range(facility_num)]
                return_dict[process]["facility_detail_list"].append(
                    [f"{node_list[idx]}_" + fac_num for fac_num in facility_list]
                )
                return_dict[process]["facility_nums"].append(facility_num)
                ###################  ## 2024.04.24 수정 ## #######################

            with col2:
                ###################  ## 2024.04.24 수정 ## #######################
                st.subheader(f"**2. Max Queue**")
                if facility_type == "infinite_facility":
                    max_queue_length=1
                    st.write("보이지 않게 하지만, 자동으로 max_queue_length = 1로 설정")
                else:
                    max_queue_length = st.number_input(
                        f"💡 :blue[**Max Queue**] : maximum pax who can queue at Node (space constraint)",
                        min_value=0,
                        value=50,
                        step=1,
                        format="%d",
                        key=f"max_queue{idx}{process}",
                    )
                return_dict[process]["max_capacity"].append(max_queue_length)
                ###################  ## 2024.04.24 수정 ## #######################
            with col3:
                st.subheader("3. Transaction Time")
                transaction_time = st.number_input(
                    "💡 :blue[**Transaction Time**] : needed seconds for each Facility to process one person",
                    min_value=1,
                    value=120,
                    step=1,
                    format="%d",
                    key=f"Transaction Time{idx}{process}",
                )
                transaction_df = pd.DataFrame(
                    index=["unit(sec)"], columns=facility_list
                )
                transaction_df = transaction_df.fillna(transaction_time)
                transaction_edited = st.data_editor(
                    transaction_df,
                    use_container_width=True,
                    key=f"Transaction Time_{idx}{process}",
                )

            st.subheader("4. Hourly Transaction Time")
            input_view, over_view = st.tabs(["Input View", "Over View"])
            with input_view:
                c1, c2 = st.columns([0.6, 0.4])
                with c1:
                    st.subheader("")
                    st.write("")
                    st.write("")
                    st.write("")
                    st.info(
                        """
                    📉 Red line: Capacity\n
                    🟦 Blue bar graph: Passengers by time period\n
                    🚨 Caution : When the blue bar graph is higher than the red line, "delays occur".\n
                    """
                    )
                    if default_value == True:
                        times = [
                            time(hour=hour, minute=minute)
                            for hour in range(24)
                            for minute in range(0, 60, 10)
                        ]
                        capa_df = pd.DataFrame(index=times, columns=facility_list)
                        capa_df.iloc[0, :] = transaction_edited.copy().iloc[0, :]
                        capa_df = capa_df.fillna(method="ffill")
                    elif default_value == False:
                        capa_df = designed_capa

                    capa_df = capa_df.astype(float)

                    capa_edited = st.data_editor(
                        capa_df,
                        column_config={
                            col: st.column_config.NumberColumn(format="%.2f")
                            for col in capa_df.select_dtypes(include=["float"]).columns
                        },
                        use_container_width=True,
                        key=f"capacity_{idx}{process}",
                        height=5300,
                    )
                    capa_edited = capa_edited.replace(0, 0.0000001)


                    st.write(capa_edited)
                    return_dict[process][node_list[idx]] = np.repeat(
                        capa_edited.values, 10, axis=0
                    )
                with c2:
                    st.subheader("")
                    unit_min = 10
                    total_capa = 1 / capa_edited.div(60)
                    total_capa["capacity"] = total_capa.fillna(0).sum(axis=1) * unit_min
                    total_capa = total_capa.reset_index()
                    total_capa["index"] = total_capa["index"].astype(str)
                    ##########################################################
                    df_pax = return_dict["df_pax"].copy()
                    df_pax_filtered = df_pax[
                        df_pax[process + "_component"] == node_list[idx]
                    ]
                    df_pax_filtered["SHOW"] = df_pax_filtered["SHOW"].dt.floor("10T")
                    # st.write(f"총 {node_list[idx]} 예상 이용여객 수 : {len(df_pax_filtered)}")

                    grouped = df_pax_filtered["SHOW"].value_counts().reset_index()
                    grouped["index"] = grouped["SHOW"].astype(str).str[-8:]

                    total_capa = pd.merge(total_capa, grouped, on="index", how="left")
                    total_capa["count"] = total_capa["count"].fillna(0)

                    fig = go.Figure()
                    fig.add_trace(
                        go.Bar(
                            y=total_capa["index"],
                            x=total_capa["count"],
                            name="Count",
                            orientation="h",
                            marker_color="#007aff",
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            y=total_capa["index"],
                            x=total_capa["capacity"],
                            mode="lines",
                            name="Capacity",
                            line=dict(color="red", width=5),
                        )
                    )
                    fig.update_layout(
                        yaxis_autorange="reversed",
                        height=5640,
                        legend=dict(x=0.8, y=0.9),  # 레전드를 그래프에 겹치게 설정
                        margin=dict(t=0),  # 위쪽 마진을 줄임
                    )

                    st.plotly_chart(fig)
                ##########################################################

                ##########################################################
            with over_view:
                c1, _ = st.columns([0.2, 0.8])
                with c1:
                    num = np.random.randint(0, 10000000) #20250626#
                    group = st.selectbox(
                        "**grouping**",
                        [
                            "operating_carrier_name",
                            "operating_carrier_iata",
                            "flight_number",
                            "tail_number",
                            "International/Domestic",
                            "dep/arr_airport",
                        ],
                        key=f"node_chunk{node_list[idx]}{num}",
                    )

                unit_min = 15
                df_pax = return_dict["df_pax"]
                selected_date = return_dict["selected_date"]
                start_time = "SHOW"

                df_filtered = df_pax[df_pax[f"{process}_component"] == node_list[idx]]
                st.info(
                    f"Service-Point  ({node_list[idx]}) : {len(df_filtered)} passengers"
                )

                # capacity
                capa_edited = pd.DataFrame(
                    return_dict[process][node_list[idx]][::unit_min]
                )
                total_capa = 1 / capa_edited.div(60)
                total_capa["capacity"] = total_capa.fillna(0).sum(axis=1) * unit_min
                time_range = pd.date_range(
                    start=selected_date,
                    end=selected_date + pd.to_timedelta(1, unit="d"),
                    freq=f"{unit_min}T",
                )[:-1]
                total_capa.index = time_range

                start_count_df, start_count_ranking_order = make_count_df(
                    df_filtered,
                    selected_date,
                    selected_date,
                    start_time,
                    group,
                    freq_min=unit_min,
                )

                max_y = max(
                    start_count_df.groupby("Time")["index"].sum().max() * 1.2,
                    total_capa["capacity"].max(),
                )
                show_bar(
                    start_count_df,
                    start_count_ranking_order,
                    group,
                    capa_df=total_capa,
                    max_y=max_y,
                )

    return return_dict


st.fragment


def choice_matrix(return_dict):
    """
    Create a choice matrix for processes based on user input.
    :param return_dict: Dictionary to store the choice matrix and related configurations
    :return: Updated return_dict with the choice matrix

    :streamlit_param component_list: List of processes taken from return_dict
    :streamlit_param matrix_type: Type of input for the choice matrix, selected via selectbox widget
    :streamlit_param group_col: Detail property for grouping, selected via selectbox widget
    :streamlit_param choice_matrix_df: DataFrame for the choice matrix, edited via data_editor widget
    """
    component_list = return_dict["component_list"]

    totla_node_list = []
    for component in component_list:
        totla_node_list += return_dict[component]["node_list"]
    return_dict["totla_node_list"] = totla_node_list
    return_dict["choice_matrix"] = {}

    st.title(f":blue[SET PASSENGER FLOW]")

    st.info(
        """
    💡 This section sets passenger flow between each Touch-Point and Service-Point. \n
    💡 :blue[**Vertical axis**] of the MATRIX represents the preceding service-point,  :blue[**Horizontal**] axis represents the subsepassenger_queuesnt service-point. \n
    💡 Values indicate the :blue[**probability(%)**] of passenger movement.\n
    """
    )
    sources, idx_list, cols, dsts, dst_lists = [], [], [], [], []
    for component in component_list:
        if return_dict[component]["root"]:
            source = return_dict[component]["source"][0]
            index = return_dict["df_pax_orig"][source].unique()
            columns = return_dict[component]["node_list"]
            dst = component
            sources.append(source)
            idx_list.append(index)
            cols.append(columns)
            dsts.append(dst)
            dst_lists.append([dst])
        if not return_dict[component]["dst"]:
            continue

        source = component
        index = return_dict[component]["node_list"]
        columns = []
        dstl = []
        for dst in return_dict[component]["dst"]:
            columns += return_dict[dst]["node_list"]
            dstl.append(dst)
        dst = ", ".join(return_dict[component]["dst"])
        sources.append(source)
        idx_list.append(index)
        cols.append(columns)
        dsts.append(dst)
        dst_lists.append(dstl)

    tab_list = [col + " → " + dst for col, dst in zip(sources, dsts)]
    show_node_list = [
        col + "_to_" + dst
        for col, dst in zip(sources, dsts)
        if not (return_dict.get(dst, {}).get("root", False))
    ]
    return_dict["show_node_list"] = show_node_list

    return_dict["priority_conditions"] = {}
    for idx, tab in zip(range(len(sources)), st.tabs(tab_list)):
        with tab:
            source, index, columns, dst, dstl = (
                sources[idx],
                idx_list[idx],
                cols[idx],
                dsts[idx],
                dst_lists[idx],
            )
            st.subheader(f":blue[{source}]  →  :blue[{dst}]")

            return_dict["priority_conditions"][
                dst
            ] = {}  # component의 other conditions 담는 공간
            if st.toggle("More conditions?", key=f"cond_toggle{idx}{source}"):
                toggle = "on"
                c1, c2 = st.columns([0.15, 0.85])
                with c1:
                    add_conditions = st.number_input(
                        "Add Priority Condition",
                        min_value=1,
                        max_value=10,
                        value=1,
                        key=f"add_conditions{idx}{source}",
                    )
                for add_idx in range(add_conditions):
                    return_dict["priority_conditions"][dst][
                        f"priority_{add_idx}"
                    ] = {}  # component의 other conditions >> priority #1 담는 공간
                    container = st.container(border=True)
                    with container:
                        dst_col = dst
                        dst_values = return_dict[dst]["node_list"]
                        df = return_dict["df_pax_orig"].copy()
                        if add_idx == 0:
                            st.subheader(f"**:blue[{add_idx+1}st] Priority Condition**")
                        elif add_idx == 1:
                            st.subheader(f"**:blue[{add_idx+1}nd] Priority Condition**")
                        elif add_idx == 2:
                            st.subheader(f"**:blue[{add_idx+1}rd] Priority Condition**")
                        else :
                            st.subheader(f"**:blue[{add_idx+1}th] Priority Condition**")


                        c1, c2, c3, c4, c5 = st.columns(5)
                        with c1:
                            num_conditions = st.number_input(
                                "CONDITIONS",
                                min_value=1,
                                max_value=5,
                                value=1,
                                key=f"num_conditions{idx}{source}{add_idx}",
                            )
                        with c2:
                            if return_dict[dst]["root"] == 1:
                                source_values = df[source].unique()
                            else:
                                source_values = return_dict[source]["node_list"]
                            source_idx = st.multiselect(
                                f"SOURCE",
                                source_values,
                                source_values[:2],
                                key=f"source_value{source}{add_idx}",
                            )

                        conditions = []
                        for c_idx in range(int(num_conditions)):
                            with c3:
                                cond_col = st.selectbox(
                                    f"FILTER",
                                    [
                                        "operating_carrier_id",
                                        "International/Domestic",
                                        "dep/arr_airport",
                                        "operating_carrier_name",
                                        "flight_number",
                                        "terminal",
                                        "PRM Status",
                                    ],
                                    key=f"col{c_idx}{source}{add_idx}",
                                )

                                unique_values = df[cond_col].unique()
                            with c4:
                                cond_value = st.multiselect(
                                    f"VALUES",
                                    unique_values,
                                    unique_values[:2],
                                    key=f"value{c_idx}{source}{add_idx}",
                                )
                                conditions.append((cond_col, cond_value))

                        with c5:
                            to_value = st.multiselect(
                                f"DESTINATION",
                                dst_values,
                                dst_values[:1],
                                key=f"dst{c_idx}{source}{add_idx}",
                            )
                        cond_sentences = " and ".join(
                            [
                                f"[{cond_col}]  is  {cond_value}"
                                for cond_col, cond_value in conditions
                            ]
                        )
                        sentence = f":blue[**If {cond_sentences}**], then {dst_col} becomes :blue[**{to_value}**]."
                        st.success(sentence)
                        return_dict["priority_conditions"][dst][f"priority_{add_idx}"][
                            "conditions"
                        ] = conditions
                        return_dict["priority_conditions"][dst][f"priority_{add_idx}"][
                            "to_value"
                        ] = to_value

                        choice_matrix_df = pd.DataFrame(
                            index=source_idx, columns=to_value, dtype=float
                        )

                        # Create columns for layout
                        c1, _ = st.columns([0.2, 0.8])
                        with c1:
                            # Select input type for the choice matrix
                            matrix_type = st.selectbox(
                                "",
                                [
                                    "Check-Box(filled)",
                                    "Probabilistic(%)",
                                    "Check-Box",
                                ],
                                key=f"input_type_{idx}{add_idx}",
                            )
                        # Create a DataFrame for the choice matrix with the appropriate index and columns
                        if matrix_type == "Check-Box":
                            # Initialize the choice matrix with zeros and convert to boolean
                            choice_matrix_df = choice_matrix_df.fillna(0)
                            edited_df = st.data_editor(
                                choice_matrix_df.astype(bool),
                                use_container_width=True,
                                key=f"ck_box_{idx}{add_idx}",
                            )
                        elif matrix_type == "Check-Box(filled)":
                            # Initialize the choice matrix with zeros and fill randomly selected columns with ones
                            choice_matrix_df = choice_matrix_df.fillna(0)
                            np.random.seed(42)
                            for i in choice_matrix_df.index:
                                random_col = np.random.choice(choice_matrix_df.columns)
                                choice_matrix_df.loc[i, random_col] = 1
                            edited_df = st.data_editor(
                                choice_matrix_df.astype(bool),
                                use_container_width=True,
                                key=f"ck_box_filled_{idx}{add_idx}",
                            )
                        elif matrix_type == "Probabilistic(%)":
                            # Initialize the choice matrix with equal probabilities for each column
                            choice_matrix_df = choice_matrix_df.fillna(
                                1 / len(choice_matrix_df.columns)
                            )
                            edited_df = st.data_editor(
                                choice_matrix_df,
                                use_container_width=True,
                                key=f"choice_mat_{idx}{add_idx}",
                            )
                        # Normalize the choice matrix so that each row sums to 1
                        edited_df = edited_df.div(edited_df.sum(axis=1), axis=0)
                        return_dict["priority_conditions"][dst][f"priority_{add_idx}"][
                            "edited_df_orig"
                        ] = edited_df.copy()

                        if return_dict[dst]["root"] == 1:
                            edited_df.columns = [
                                totla_node_list.index(v) for v in edited_df.columns
                            ]
                            return_dict["priority_conditions"][dst][
                                f"priority_{add_idx}"
                            ]["edited_df"] = edited_df
                        else:
                            edited_df.index = [
                                totla_node_list.index(v) for v in edited_df.index
                            ]
                            edited_df.columns = [
                                totla_node_list.index(v) for v in edited_df.columns
                            ]
                            return_dict["priority_conditions"][dst][
                                f"priority_{add_idx}"
                            ]["edited_df"] = edited_df

            else:
                toggle = "off"
            ####################################################
            ####################################################
            ####################################################

            container_default = st.container(border=True)
            with container_default:
                if toggle == "off":
                    pass
                elif toggle == "on":
                    st.subheader(f"**Default Condition**")

                # Create columns for layout
                c1, _ = st.columns([0.2, 0.8])
                with c1:
                    choice_matrix_df = pd.DataFrame(
                        index=index, columns=columns, dtype=float
                    )

                    # Select input type for the choice matrix
                    matrix_type = st.selectbox(
                        "**Input Type**",
                        [
                            "Check-Box(filled)",
                            "Probabilistic(%)",
                            "Check-Box",
                        ],
                        key=f"input_type_{idx}",
                    )
                # Create a DataFrame for the choice matrix with the appropriate index and columns
                if matrix_type == "Check-Box":
                    # Initialize the choice matrix with zeros and convert to boolean
                    choice_matrix_df = choice_matrix_df.fillna(0)
                    edited_df = st.data_editor(
                        choice_matrix_df.astype(bool),
                        use_container_width=True,
                        key=f"ck_box_{idx}",
                    )
                elif matrix_type == "Check-Box(filled)":
                    # Initialize the choice matrix with zeros and fill randomly selected columns with ones
                    choice_matrix_df = choice_matrix_df.fillna(0)
                    np.random.seed(42)
                    for i in choice_matrix_df.index:
                        random_col = np.random.choice(choice_matrix_df.columns)
                        choice_matrix_df.loc[i, random_col] = 1
                    edited_df = st.data_editor(
                        choice_matrix_df.astype(bool),
                        use_container_width=True,
                        key=f"ck_box_filled_{idx}",
                    )
                elif matrix_type == "Probabilistic(%)":
                    # Initialize the choice matrix with equal probabilities for each column
                    choice_matrix_df = choice_matrix_df.fillna(
                        1 / len(choice_matrix_df.columns)
                    )
                    edited_df = st.data_editor(
                        choice_matrix_df,
                        use_container_width=True,
                        key=f"choice_mat_{idx}",
                    )
                # Normalize the choice matrix so that each row sums to 1
                edited_df = edited_df.div(edited_df.sum(axis=1), axis=0)

                return_dict["choice_matrix"][source] = {}
                return_dict["choice_matrix"][source]["matrix_type"] = matrix_type
                return_dict["choice_matrix"][source]["matrix_df"] = edited_df.values
                return_dict["choice_matrix"][source]["matrix_df_orig"] = edited_df
                return_dict["choice_matrix"][source]["dst"] = dstl
                return_dict["choice_matrix"][source]["row_idx"] = index
                # st.write(index, dstl)
            st.write("")
            st.write("")
            ####################################################
            ####################################################
            ####################################################

    if st.button(
        "**FINISH PASSENGER FLOW**",
        type="secondary",
        use_container_width=True,
        key="Apply Choice Matrix",
    ):
        return_dict = add_columns(return_dict)
        with st.expander("show detail"):
            st.dataframe(return_dict["df_pax"])

        df = return_dict["df_pax"].copy()


    return return_dict


@st.fragment
def show_specific_node(return_dict):
    df = return_dict["df_pax"].copy()

    survey_dict = {}
    survey_sheets = []
    c1, c2, c3, c4 = st.columns(4)
    comp_list = [
        component + "_component" for component in return_dict["component_list"]
    ]
    for filter_col, comp_tab in zip(comp_list, st.tabs(comp_list)):
        with comp_tab:
            node_list = df[filter_col].unique().tolist()
            for node, node_tab in zip(node_list, st.tabs(node_list)):
                with node_tab:
                    c1, c2 = st.columns(2)
                    with c1:
                        group = st.selectbox(
                            "**grouping**",
                            [
                                "operating_carrier_name",
                                "operating_carrier_iata",
                                "flight_number",
                                "tail_number",
                                "International/Domestic",
                                "dep/arr_airport",
                            ],
                            key=f"Select Group Column{node}",
                        )

                    unit_min = 15
                    df_filtered = df[df[filter_col] == node]
                    st.info(
                        f"Total Passenger at [Zone {node}] : {len(df_filtered)} pax"
                    )
                    show_df_schedule(
                        df=df_filtered,
                        group=group,
                        key="pax_generation",
                        time_col="SHOW",
                        y_label="Passenger (Dep)",
                        title="",
                        unit_min=unit_min,
                    )

                    # 여객 상위 80% 시간대만 뽑아내기
                    hour_counts = df_filtered["SHOW"].dt.hour.value_counts()
                    total_count = hour_counts.sum()
                    cumulative_percent = hour_counts.cumsum() / total_count
                    top_80_percent = cumulative_percent[cumulative_percent <= 1] # 80% cover
                    result = top_80_percent.index.tolist()
                    df_survey = df_filtered[df_filtered["SHOW"].dt.hour.isin(result)][
                        [group, "SHOW", filter_col]
                    ].copy()

                    df_survey["SHOW"] = df_survey["SHOW"].dt.floor(f"{unit_min}T")
                    is_root = return_dict[filter_col.replace("_component", "")]["root"]
                    if is_root:
                        df_survey = (
                            df_survey.sort_values(by=[group, "SHOW"])
                            .drop_duplicates(subset=[group, "SHOW"])
                            .reset_index(drop=True)
                        )
                        df_survey["Survey No"] = "NO. " + (df_survey.index + 1).astype(
                            str
                        )
                        df_survey = df_survey[["Survey No", filter_col, group, "SHOW"]]
                    else:
                        df_survey = (
                            df_survey.sort_values(by=[filter_col, "SHOW"])
                            .drop_duplicates(subset=[filter_col, "SHOW"])
                            .reset_index(drop=True)
                        )
                        df_survey["Survey No"] = "NO. " + df_survey.index.astype(str)
                        df_survey = df_survey[["Survey No", filter_col, "SHOW"]]

                    df_survey["passenger_queuesue_start"] = ""
                    df_survey["passenger_queuesue_end"] = ""
                    df_survey["passenger_queuesue_pax(last line pax)"] = ""
                    df_survey["open_facilities"] = ""
                    df_survey["Open Detail"] = ""
                    df_survey["Comment"] = ""


                    df_survey["passenger_queuesue_start"].loc[0] = (
                        df_survey["SHOW"].loc[0]
                        + pd.to_timedelta(1, unit="m")
                        + pd.to_timedelta(32, unit="S")
                    )
                    df_survey["passenger_queuesue_end"].loc[0] = (
                        df_survey["SHOW"].loc[0]
                        + pd.to_timedelta(12, unit="m")
                        + pd.to_timedelta(21, unit="S")
                    )
                    df_survey["passenger_queuesue_pax(last line pax)"].loc[0] = "75 pax"
                    df_survey["open_facilities"].loc[0] = "7 desk"
                    df_survey["Open Detail"].loc[0] = "C11-C17"
                    df_survey["SHOW"] = df_survey["SHOW"].astype(str)
                    df_survey = df_survey.rename({filter_col: "node"}, axis=1)
                    survey_sheets += [df_survey]
                    st.subheader("Survey Sheet")
                    st.info(f"Survey Passenger at Service-Point ({node}) : {len(df_survey)} passengers")
                    with st.popover("💡Info"):
                        st.image("data/raw/simulation_explain/survey_infos.png")
                        st.image("data/raw/simulation_explain/why_last_pax.png")
                        st.image("data/raw/simulation_explain/approximate.png")
                        st.image("data/raw/simulation_explain/survey_methodology.png")
                        st.image("data/raw/simulation_explain/process_overview.png")
                        st.image("data/raw/simulation_explain/ranking_chart.png")
                        st.image("data/raw/simulation_explain/survey_to_capacity.png")
                        st.image("data/raw/simulation_explain/additional_value.png")

                    st.table(df_survey)
                    survey_dict[filter_col + "♪" + node] = {}
                    survey_dict[filter_col + "♪" + node]["sample_num"] = len(df_survey)
    survey_sheets = pd.concat(survey_sheets, axis=0)
    return survey_dict, survey_sheets


def show_survey_result(return_dict):
    df = return_dict["df_pax"].copy()

    survey_dict = {}
    survey_sheets = []
    c1, c2, c3, c4 = st.columns(4)
    comp_list = [
        component + "_component" for component in return_dict["component_list"]
    ]
    for filter_col, comp_tab in zip(comp_list, st.tabs(comp_list)):
        with comp_tab:
            node_list = df[filter_col].unique().tolist()
            for node, node_tab in zip(node_list, st.tabs(node_list)):
                with node_tab:
                    c1, c2 = st.columns(2)
                    with c1:
                        group = st.selectbox(
                            "**grouping**",
                            [
                                "operating_carrier_name",
                                "operating_carrier_iata",
                                "flight_number",
                                "tail_number",
                                "International/Domestic",
                                "dep/arr_airport",
                            ],
                            key=f"Select Group Column{node}_result",
                        )

                    unit_min = 15
                    df_filtered = df[df[filter_col] == node]
                    st.info(
                        f"Total Passenger at [Zone {node}] : {len(df_filtered)} pax"
                    )
                    show_df_schedule(
                        df=df_filtered,
                        group=group,
                        key="pax_generation",
                        time_col="SHOW",
                        y_label="Passenger (Dep)",
                        title="",
                        unit_min=unit_min,
                    )


def add_columns(return_dict):
    """
    Add columns to the DataFrame based on the choice matrices for each process transition.
    :param return_dict: Dictionary containing the choice matrices and the passenger DataFrame
    :return: Updated return_dict with added columns to the passenger DataFrame
    """

    def sample_node(row, edited_df):
        """
        Sample a node based on the probabilities from the choice matrix.
        :param row: The row of the DataFrame to sample from
        :param edited_df: The choice matrix DataFrame with probabilities
        :return: The sampled node
        """
        probabilities = edited_df.loc[row]
        return np.random.choice(probabilities.index, p=probabilities.values)

    # Copy the passenger DataFrame from return_dict
    df_pax = return_dict["df_pax_orig"].copy()
    df_pax = df_pax.reset_index(drop=True)  # 중복 인덱스 문제 해결

    return_dict["df_pax"] = None

    # Iterate over each process transition in the choice matrix
    st.write()
    for source in return_dict["choice_matrix"].keys():
        vertical_process = source
        horizontal_process = return_dict["choice_matrix"][source]["dst"][0]

        # Get the choice matrix for the current process transition
        edited_df = return_dict["choice_matrix"][source]["matrix_df_orig"]
        horizontal_process_category = return_dict[horizontal_process]["root"]
        # If the horizontal process category is "condition", use the property column to sample nodes
        if horizontal_process_category == 1:  # is root
            property_col = return_dict[horizontal_process]["source"][0]
            df_pax[f"{horizontal_process}_component"] = df_pax[property_col].apply(
                sample_node, args=(edited_df.fillna(0),)
            )
        elif horizontal_process_category == 0:
            df_pax[f"{horizontal_process}_component"] = df_pax[
                f"{vertical_process}_component"
            ].apply(sample_node, args=(edited_df.fillna(0),))

        process_cond_dict = return_dict["priority_conditions"][horizontal_process]
        priority_process_cond_dict = list(process_cond_dict.keys())[::-1]

        df_pax[f"{horizontal_process}_edited_df"] = None
        for priority in priority_process_cond_dict:
            conds = process_cond_dict[priority]
            condition = pd.Series([True] * len(df_pax))
            for cond_col, cond_values in conds["conditions"]:
                condition &= df_pax[cond_col].isin(cond_values)
            n = len(df_pax.loc[condition])  # condition에 해당하는 행의 개수

            edited_df_values = pd.Series(
                [conds["edited_df"]] * n,  # for 문 대신 곱셈 사용
                index=df_pax.loc[condition].index,
            )
            df_pax.loc[condition, f"{horizontal_process}_edited_df"] = edited_df_values

            # process_component는 시뮬레이션 전에 probability로만 가는 값이기 때문에, temp_condition으로 vertical_process 값이 미리 확정되어있음
            if horizontal_process_category == 1:  # is root
                property_col = return_dict[horizontal_process]["source"][0]
                temp_condition = condition & (
                    df_pax[property_col].isin(conds["edited_df_orig"].index)
                )
                df_pax.loc[temp_condition, f"{horizontal_process}_component"] = (
                    df_pax.loc[temp_condition, property_col].apply(
                        sample_node, args=(conds["edited_df_orig"].fillna(0),)
                    )
                )
            elif horizontal_process_category == 0:
                temp_condition = condition & (
                    df_pax[f"{vertical_process}_component"].isin(
                        conds["edited_df_orig"].index
                    )
                )
                df_pax.loc[temp_condition, f"{horizontal_process}_component"] = (
                    df_pax.loc[temp_condition, f"{vertical_process}_component"].apply(
                        sample_node, args=(conds["edited_df_orig"].fillna(0),)
                    )
                )

        return_dict["df_pax"] = df_pax.copy()
    return return_dict


@st.fragment
def show_choice_matrix(return_dict, key, pax_flow=False):
    """
    Visualize the process choice matrix as a graph.
    :param return_dict: Dictionary containing the choice matrices and the passenger DataFrame
    :param key: A unique key to avoid duplication when the function is used multiple times
    :param pax_flow: Boolean flag to indicate whether to display passenger flow (default is False)
    """
    # Get the passenger DataFrame
    df_pax = return_dict["df_pax"].copy()

    # Initialize a Graphviz Digraph
    graph = graphviz.Digraph()

    # Set graph attributes
    graph.attr(
        "node", shape="ellipse", color="darkgray", fontcolor="darkgray"
    )  # Node and text color set to white
    graph.attr(ranksep="1", nodesep="0.15")
    graph.attr(rankdir="LR")  # Left to Right graph direction
    graph.attr(bgcolor="transparent")  # Set background to transparent

    arrow_shape = "vee"  # Arrow shape: options include normal, vee, box, diamond, etc.
    edge_length = "2"  # Arrow length
    node_size = "1"

    # Visualize the process nodes and transitions
    for from_to_process in return_dict["show_node_list"]:
        [from_process, to_process] = from_to_process.split("_to_")

        # Get the choice matrix for the current process transition
        edited_df = return_dict["choice_matrix"][from_process]["matrix_df_orig"]
        from_nodes = return_dict[from_process]["node_list"]
        to_nodes = return_dict[to_process]["node_list"]
        from_col = from_process + "_component"
        to_col = to_process + "_component"

        # Add nodes and edges to the graph
        for from_node in from_nodes:
            graph.node(
                from_node,
                color="darkgray",  # Node color set to white
                fontcolor="darkgray",  # Text color set to white
                width=node_size,
                height=node_size,
                fixedsize="true",
            )
            for to_node in to_nodes:
                graph.node(
                    to_node,
                    color="darkgray",  # Node color set to white
                    fontcolor="darkgray",  # Text color set to white
                    width=node_size,
                    height=node_size,
                    fixedsize="true",
                )
                if pax_flow:
                    # Display passenger flow if pax_flow is True
                    pax_flow_length = len(
                        df_pax[
                            (df_pax[from_col] == from_node)
                            & (df_pax[to_col] == to_node)
                        ]
                    )
                    if pax_flow_length > 0:
                        graph.edge(
                            from_node,
                            to_node,
                            arrowhead=arrow_shape,
                            minlen=edge_length,
                            label=str(pax_flow_length),
                            color="darkgray",  # Edge color set to white
                            fontcolor="darkgray",  # Label text color set to white
                        )
                else:
                    # Display edges based on the choice matrix
                    if edited_df.loc[from_node, to_node] > 0:
                        graph.edge(
                            from_node,
                            to_node,
                            arrowhead=arrow_shape,
                            minlen=edge_length,
                            color="darkgray",  # Edge color set to white
                            fontcolor="darkgray",  # Label text color set to white
                        )
    # Display the Graphviz graph in Streamlit
    st.graphviz_chart(graph, use_container_width=True)


@st.fragment
def set_capacity(return_dict):

    st.title(":blue[CAPACITY]")
    pax_flow, capa_input, capa_template = st.tabs(
        ["**FLOW VIEW**", "**CAPACITY**", "**CAPCITY TEMPLATE**"]
    )
    with pax_flow:
        c1, c2, _, _ = st.columns(4)
        with c1:
            st.success(
                f"""
                ✅ Airport Code :blue[**{return_dict['airport_code']}**] \n
                ✅ Terminal :blue[**{return_dict['terminal']}**] \n
                ✅ Date :blue[**{return_dict['selected_date']}**] \n
                ✅ Flight IO :blue[**{return_dict['flight_io']}**] \n  
                """
            )
        with c2:
            st.success(
                f"""
                ✅ Flights :blue[**{len(return_dict['df'])}**] \n  
                ✅ Passengers :blue[**{len(return_dict['df_pax'])}**] \n
                ✅ Touch-Point :blue[**{len(return_dict['component_list'])}**] \n
                ✅ Service-Point :blue[**{len(return_dict["totla_node_list"])}**] \n
                """
            )
        create_sankey(
            return_dict["df_pax"], return_dict["component_list"], suffix="_component"
        )

    with capa_input:
        uploaded_file = st.file_uploader("Upload Capacity excel", type=["xlsx"])
        if uploaded_file is None:
            return_dict.pop("capacity_datas", None)  # 파일이 없으면 capacity_datas 삭제
        elif uploaded_file.name.endswith((".xlsx")):
            capacity_datas = pd.read_excel(
                uploaded_file, sheet_name=None, index_col=False
            )
            return_dict["capacity_datas"] = capacity_datas

        for tab, process in zip(
            st.tabs(return_dict["component_list"]), return_dict["component_list"]
        ):
            with tab:
                return_dict = node_chunk(process, return_dict)

    with capa_template:
        make_capacity_excel(return_dict)
    return return_dict


# def set_choice_matrix(return_dict):
#     with st.expander("**SET CHOICE MATRIX**"):
#         return_dict = choice_matrix(return_dict)
#     return return_dict

import openpyxl


def make_capacity_excel(return_dict):
    # 엑셀 파일 경로
    excel_path = "data/raw/excel_utility/simulator_capacity.xlsx"
    # 기존 파일이 있는 경우 모든 시트 삭제 (Sheet1 제외)
    book = openpyxl.load_workbook(excel_path)
    # Sheet1을 제외한 모든 시트 삭제
    for sheet_name in book.sheetnames:
        if sheet_name != "Sheet1":
            del book[sheet_name]
    book.save(excel_path)
    book.close()

    # 기존 파일이 있는 경우에는 append 모드로, 없는 경우에는 새로 생성
    with pd.ExcelWriter(
        excel_path, mode="a", engine="openpyxl", if_sheet_exists="replace"
    ) as writer:
        for component in return_dict["component_list"]:
            for idx, node in enumerate(return_dict[component]["node_list"]):
                values = return_dict[component][node]
                values = values[
                    ::10
                ]  # values는 현재 1분단위 증강 데이터 > 다시 10분단위변환

                facility_num = return_dict[component]["facility_nums"][idx]
                facility_columns = [
                    node + f"_no{str(i+1)}" for i in range(facility_num)
                ]

                times = [
                    time(hour=hour, minute=minute)
                    for hour in range(24)
                    for minute in range(0, 60, 10)
                ]

                df = pd.DataFrame(data=values, columns=facility_columns, index=times)
                df.to_excel(writer, sheet_name=str(node))

    # 파일을 읽어서 다운로드 버튼 생성
    with open("data/raw/excel_utility/simulator_capacity.xlsx", "rb") as f:
        st.download_button(
            label="Download Capacity Template",
            data=f.read(),
            file_name="simulator_capacity.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


@st.fragment
def run(return_dict):
    container_simulation = st.container(border=True)

    with container_simulation:
        (
            ADD_PROPERTIES,
            TOUCH_POINT,
            CHOICE_MATRIX,
            CAPACITY,
            RUN_SIMULATION,
            OPTIMIZATION,
        ) = st.tabs(
            [
                "**ADD PROPERTIES**",
                "**TOUCH-POINT**",
                "**PASSENGER FLOW**",
                "**CAPACITY**",
                "**RUN SIMULATION**",
                "**OPTIMIZATION**",
            ]
        )
        with ADD_PROPERTIES:
            return_dict = add_properties(return_dict)
        with TOUCH_POINT:
            return_dict = set_process(return_dict)
        with CHOICE_MATRIX:
            return_dict = choice_matrix(return_dict)
        if "df_pax" in return_dict:
            # with SURVEY:
            #     # st.write("제작중")
            #     show_survey(return_dict)
            with CAPACITY:
                _ = set_capacity(return_dict)
                with RUN_SIMULATION:
                    ###################  ## 2024.04.24 수정 ## #######################
                    optimize = st.toggle("Operation schedule recommendations", key=f"ops_sch_recom", value=False)
                    if optimize:
                        for process in return_dict["component_list"] : 
                            return_dict[process]["target_waiting_time"]=[]
                            target_waiting_time = 5
                            for facility_type in return_dict[process]["facility_type"] :
                                if facility_type == "normal_facility" :
                                    return_dict[process]["target_waiting_time"].append(target_waiting_time)
                                elif facility_type == "infinite_facility":
                                    return_dict[process]["target_waiting_time"].append(None)
                    else:
                        for process in return_dict["component_list"] : 
                            return_dict[process]["target_waiting_time"]=[]
                            for node in return_dict[process]["node_list"] :
                                return_dict[process]["target_waiting_time"].append(None)
                    ###################  ## 2024.04.24 수정 ## #######################

                    if st.button(
                        "Run Simulation", use_container_width=True, type="primary"
                    ):
                        _ = run_simulation(return_dict)
        else:
            # with SURVEY:
            #     st.write("*YOU HAVE TO FINISH CHOICE MATRIX!*")
            with CAPACITY:
                st.write("*YOU HAVE TO FINISH CHOICE MATRIX!*")
            with RUN_SIMULATION:
                st.write("*YOU HAVE TO FINISH CAPACITY!*")
            with OPTIMIZATION:
                st.write("*YOU HAVE TO FINISH SIMULATION!*")
        with RUN_SIMULATION:
            if "sim_df_pax" in return_dict:
                _ = view(return_dict)
                if st.toggle("AEMOS Survey?", key=f"AEMOS Survey"):
                    show_aemos_template(df=return_dict["sim_df_pax"],
                                        component_list=return_dict["component_list"]
                                        )
                    st.write("--")

        with OPTIMIZATION:
            _ = optimize_configuration(return_dict)

    return return_dict


# @st.fragment
def run_simulation(return_dict):
    pax = run_sim(return_dict)
    with st.expander("sim_df_pax"):
        st.write(pax)

    return_dict["sim_df_pax"] = pax.copy()
    return_dict["sim_df_pax"].to_parquet("sim_df_pax_.parquet")
    return return_dict









def show_aemos_template(df, component_list, time_interval_min=15):
    st.header("Survey Template")
    num_of_passengers=len(df)
    num_of_flights=len(df["flight_number"].unique())
    origin_airport_name=(df["origin_airport"].unique())
    terminal_name=(df["terminal"].unique())


    tab_list=component_list+["Total Template"]
    template=[]
    for tab, component in zip(st.tabs(tab_list), tab_list):
        with tab:
            if component !="Total Template":
                group_col_list = st.multiselect(
                    f"Queue groupby", ["operating_carrier_name", "International/Domestic","terminal","country_name","region_name","aircraft_class"], 
                    default=[],  # 디폴트 값 설정
                    key=f"method_{component}"
                )
                df["group_col"] = df[group_col_list].astype(str).agg("_".join, axis=1) if group_col_list else ""

                df["Measurement Time"] = df[f"{component}_on_pred"].dt.floor(f"{time_interval_min}T")
                
                touch_point_template=[]
                for node in sorted(df[f"{component}_pred"].unique()):
                    df_node = df[df[f"{component}_pred"]==node]
                    if group_col_list == []:       
                        service_point_template = df_node.groupby("Measurement Time").size().reset_index(name="Exp pax")
                        service_point_template["Service Point"] = node.replace(component +"_","")
                    else:
                        service_point_template = df_node.groupby(["group_col", "Measurement Time"]).size().reset_index(name="Exp pax")
                        service_point_template["Service Point"] = node.replace(component +"_","") + " (" +service_point_template["group_col"] +")"
                        service_point_template = service_point_template.drop({"group_col"},axis=1)

                    touch_point_template+=[service_point_template]
                touch_point_template = pd.concat(touch_point_template)
                touch_point_template["Touch Point"] = component
                touch_point_template=touch_point_template[["Touch Point","Service Point","Measurement Time","Exp pax"]]
                touch_point_template[["Queue Start","Sample Apperarnce","Queue Pax","Open Resources","Open Detail","Queue End","Comment"]]= ""
                template +=[touch_point_template]


                num_of_service_point = len(touch_point_template["Service Point"].unique())
                num_of_samples = len(touch_point_template)
                st.caption(f"Touch Point : {component}")
                st.caption(f"Service Point : {num_of_service_point}")
                st.caption(f"Samples : {num_of_samples}")
                st.caption(f"Sample Ratio : {int(num_of_samples/num_of_passengers*1000)/10}%  (=num_of_samples / num_of_passengers)")
                st.dataframe(touch_point_template, hide_index=True)

            if component =="Total Template":
                template = pd.concat(template)

                num_of_touch_point = len(component_list)
                num_of_service_point = len(template["Service Point"].unique())
                num_of_samples = len(template)
                st.caption(f"Origin Airport : {origin_airport_name}")
                st.caption(f"Terminal : {terminal_name}")

                st.caption(f"Flights : {num_of_flights}")
                st.caption(f"Passengers : {num_of_passengers}")
                st.caption(f"Touch Point : {num_of_touch_point}")
                st.caption(f"Service Point : {num_of_service_point}")
                st.caption(f"Samples : {num_of_samples}")
                st.caption(f"Sample Ratio : {int(num_of_samples/(num_of_passengers*num_of_touch_point)*1000)/10}%  (=num_of_samples / (num_of_passengers x num_of_touch_point))")
                st.dataframe(template, hide_index=True)




@st.fragment
def view(return_dict):

    component_list = return_dict["component_list"]
    sim_df_orig = return_dict["sim_df_pax"]
    def create_time_categories(time_diff):
        conditions = [
            time_diff <= pd.Timedelta(minutes=10),
            time_diff <= pd.Timedelta(minutes=20),
            time_diff <= pd.Timedelta(minutes=30),
            time_diff > pd.Timedelta(minutes=30),
        ]

        choices = ["0~10 min", "10~20 min", "20~30 min", "30~ min"]

        return np.select(conditions, choices, default="over 60min")

    # 각 component에 대해 적용
    for component in component_list:
        time_diff = (
            sim_df_orig[f"{component}_pt_pred"] - sim_df_orig[f"{component}_on_pred"]
        )
        sim_df_orig[f"{component}_delay ◎"] = create_time_categories(time_diff)

    selected_date = return_dict["selected_date"]
    freq_min = 10

    total_throughput = len(sim_df_orig)
    unfinished = sim_df_orig[f"{component_list[-1]}_done_pred"].isna().sum()

    st.info(
        f"""
    ✅ Total Throughput :blue[**{total_throughput} pax**] \n
    🚨 Un Finished :blue[**{unfinished} pax**] \n
    """
    )
    create_sankey(sim_df_orig, component_list)
    info_dfs = []

    for component, comp_tab in zip(
        component_list + ["TOTAL"], st.tabs(component_list + ["TOTAL"])
    ):
        with comp_tab:
            if component != "TOTAL":
                start_time = f"{component}_on_pred"
                end_time = f"{component}_pt_pred"
                process_time = f"{component}_pt"

                node_list = return_dict[component]["node_list"]
                facility_detail_list = return_dict[component]["facility_detail_list"]
                facility_nums = return_dict[component]["facility_nums"]
                limit_qaueus = return_dict[component]["max_capacity"]

                sim_df = sim_df_orig[sim_df_orig[end_time].notna()]
                for node, facility_detail, facility_num, limit_passenger_queuesue, tab in zip(
                    node_list,
                    facility_detail_list,
                    facility_nums,
                    limit_qaueus,
                    st.tabs(node_list),
                ):
                    filtered_sim_df = sim_df[
                        sim_df[f"{component}_pred"] == f"{component}_{node}"
                    ]

                    if len(filtered_sim_df) > 0:

                        with tab:
                            c1, c2 = st.columns(2)
                            diff_arr = (
                                filtered_sim_df[end_time] - filtered_sim_df[start_time]
                            ).dt.total_seconds()
                            throughput = int(len(filtered_sim_df))
                            max_queue_length = filtered_sim_df[f"{component}_passenger_queues"].max()

                            total_delay = int(diff_arr.sum() / 60)
                            max_delay = int(diff_arr.max() / 60)
                            average_delay = int((total_delay / throughput) * 100) / 100
                            average_transaction_time = (
                                int(filtered_sim_df[process_time].mean() * 10) / 10
                            )

                            with c1:
                                st.success(
                                    f"""
                                ✅ Throughput :blue[**{throughput} pax**] \n
                                ✅ Delay (Max) :blue[**{max_delay} min**] \n
                                ✅ Delay (Avg) :blue[**{average_delay} min**] \n
                                """
                                )
                                group = st.selectbox(
                                    "**group**",
                                    [
                                        "operating_carrier_iata",
                                        f"{component}_delay ◎",
                                        f"{component}_fac",

                                        "International/Domestic",
                                        "operating_carrier_name",
                                        "country_code",
                                        "flight_number",
                                        "tail_number",
                                    ]
                                    + [
                                        f"{component}_pred"
                                        for component in component_list
                                    ],
                                    key=f"view_group{node}{component}",
                                )
                                st.write(group)
                            with c2:
                                st.success(
                                    f"""
                                ✅ Transaction (Avg) :blue[**{average_transaction_time} sec**] \n
                                ✅ facility num :blue[**{facility_num} EA**] \n
                                ✅ limit passenger_queue :blue[**{limit_passenger_queuesue} pax**] \n
                                ✅ true max_queue_length :blue[**{max_queue_length} pax**] \n
                                """
                                )

                            info_df = pd.DataFrame(
                                {
                                    "component": [component],
                                    "node": [node],
                                    "Throughput(pax)": [throughput],
                                    "max_delay(min)": [max_delay],
                                    "average_delay(min)": [average_delay],
                                    "total_delay(min)": [total_delay],
                                    "average_transaction_time(sec)": [
                                        average_transaction_time
                                    ],
                                    "total_transaction_time(sec)": [
                                        average_transaction_time * throughput
                                    ],
                                    "facility_num(EA)": [facility_num],
                                    "limit_passenger_queuesue(pax)": [limit_passenger_queuesue],
                                }
                            )
                            info_dfs += [info_df]

                            ### 2025.05.12 ###
                            capa_edited = pd.DataFrame(
                                return_dict["processing_config_full_dict"][f"{component}_{node}"][::freq_min] ### 2025.05.07 ### [freq_min-1::freq_min]
                                )
                            ### 2025.05.12 ###
                            total_capa = 1 / capa_edited.div(60)
                            total_capa["capacity"] = (
                                total_capa.fillna(0).sum(axis=1) * freq_min
                            )





                            earliest_date = sim_df_orig["SHOW"].min().date()
                            st.write(earliest_date, selected_date)


                            time_range = pd.date_range(
                                start=selected_date - pd.to_timedelta(1, unit="d"),
                                end=selected_date + pd.to_timedelta(2, unit="d"),
                                freq=f"{freq_min}T",
                            )[:-1]



                            # 새로운 인덱스 설정
                            total_capa.index = time_range
                            capa_edited.index = time_range

                            passenger_queuesue_df, passenger_queuesue_ranking_order, waiting_time_df = (
                                make_passenger_queuesue_df(
                                    filtered_sim_df,
                                    selected_date,
                                    start_time,
                                    end_time,
                                    group,
                                    freq_min=1,
                                )
                            )

                            start_count_df, start_count_ranking_order = make_count_df(
                                filtered_sim_df,
                                selected_date,
                                selected_date,
                                start_time,
                                group,
                                freq_min=freq_min,
                            )
                            end_count_df, end_count_ranking_order = make_count_df(
                                filtered_sim_df,
                                selected_date,
                                selected_date,
                                end_time,
                                group,
                                freq_min=freq_min,
                            )
                            max_y = max(
                                start_count_df.groupby("Time")["index"].sum().max()
                                * 1.2,
                                total_capa["capacity"].max(),
                            )

                            valid_time_range = pd.date_range(
                                start=start_count_df[start_count_df[group].notna()][
                                    "Time"
                                ].min(),
                                end=end_count_df[end_count_df[group].notna()][
                                    "Time"
                                ].max(),
                                freq=f"{freq_min}T",
                            )
                            passenger_queuesue_df = passenger_queuesue_df[passenger_queuesue_df["Time"].isin(valid_time_range)]
                            waiting_time_df = waiting_time_df[
                                waiting_time_df["Time"].isin(valid_time_range)
                            ]

                            total_capa = total_capa[
                                total_capa.index.isin(valid_time_range)
                            ]

                            c1, c2 = st.columns(2)
                            with c1:
                                st.write(":blue[**Inflow**]  throughput")
                                show_bar(
                                    start_count_df,
                                    start_count_ranking_order,
                                    group,
                                    capa_df=total_capa,
                                    max_y=max_y,
                                )
                            with c2:
                                st.write(":blue[**Outflow**]  throughput")
                                show_bar(
                                    end_count_df,
                                    end_count_ranking_order,
                                    group,
                                    capa_df=total_capa,
                                    max_y=max_y,
                                )

                            c1, c2 = st.columns(2)
                            with c1:
                                st.write(":blue[**Queueing Passenger(number)**]")
                                show_bar(passenger_queuesue_df, passenger_queuesue_ranking_order, group)

                            with c2:
                                st.write(":blue[**Waiting Time(min)**]")
                                show_bar(waiting_time_df, passenger_queuesue_ranking_order, group)

                            # st.write("Recommanded Schedule") ### 2024.04.24 ###
                            # st.write(capa_edited) ### 2024.04.24 ###

            elif component == "TOTAL":
                info_dfs = pd.concat(info_dfs)

                summary = (
                    info_dfs.groupby("component")
                    .agg(
                        {
                            "Throughput(pax)": "sum",
                            "total_delay(min)": "sum",
                            "total_transaction_time(sec)": "sum",
                        }
                    )
                    .reset_index()
                )
                summary["average_delay(min)"] = (
                    summary["total_delay(min)"] / summary["Throughput(pax)"]
                )
                summary["average_transaction_time(sec)"] = (
                    summary["total_transaction_time(sec)"] / summary["Throughput(pax)"]
                )

                show_total(info_dfs, key=node)
                with st.expander("show detail"):
                    st.header("Total Summary")
                    st.dataframe(summary)

                    st.header("Node Summary")
                    st.dataframe(info_dfs)

                    st.header("simulation result pax df")
                    st.dataframe(sim_df_orig)


@st.fragment
def show_total(df, key):
    show_col = st.selectbox(
        "**group**",
        df.columns[2:],
        key=f"show_col{key}",
    )
    # 기본 바 차트
    fig = px.bar(
        df,
        x="node",
        y=show_col,
    )
    # 레이아웃 커스터마이징
    fig.update_layout(
        height=500,
    )
    fig.update_traces(marker_color="#007aff", opacity=0.8)  # 바 색상  # 투명도

    st.plotly_chart(fig)


def make_passenger_queuesue_df(df, selected_date, start_time, end_time, group, freq_min=1):
    df_copied = df.copy()

    time_range = pd.date_range(
        start=selected_date - pd.to_timedelta(1, unit="d"),
        end=selected_date + pd.to_timedelta(2, unit="d"),
        freq=f"{freq_min}T",
    )
    time_range_df = pd.DataFrame(time_range, columns=["Time"])
    df_expanded = df_copied[[start_time, end_time, group]].copy()
    df_expanded[start_time] = df_expanded[start_time].dt.round(f"{freq_min}T")
    df_expanded[end_time] = df_expanded[end_time].dt.round(f"{freq_min}T")

    df_expanded["start_int"] = (
        (
            (df_expanded[start_time] - df_expanded[start_time].min()).dt.total_seconds()
            // 60
        )
        .fillna(0)
        .astype(int)
    )
    df_expanded["end_int"] = (
        (
            (df_expanded[end_time] - df_expanded[start_time].min()).dt.total_seconds()
            // 60
        )
        .fillna(0)
        .astype(int)
    )
    df_expanded["Time"] = df_expanded.apply(
        lambda row: list(range(row["start_int"], row["end_int"], freq_min)), axis=1
    )
    df_expanded = df_expanded.explode("Time")
    df_expanded["Time"] = df_expanded[start_time] + pd.to_timedelta(
        df_expanded["Time"] - df_expanded["start_int"], unit="m"
    )
    df_expanded = df_expanded[
        df_expanded["Time"].isin(time_range)
    ]  # 하루를 넘어가거나, 하루 이전의 값을 사전 제거

    # count_df 만들기
    count_df = (
        df_expanded.groupby(["Time", group]).size().unstack(fill_value=0).reset_index()
    )
    count_df = count_df.melt(id_vars="Time", var_name=group, value_name="index")
    count_df = pd.merge(time_range_df, count_df, on="Time", how="left")
    count_df["index"] = count_df["index"].fillna(0)
    count_df[group] = count_df[group].fillna("")

    df_expanded["index"] = (df_expanded["end_int"] - df_expanded["start_int"]).fillna(0)
    waiting_time_df = df_expanded.groupby(["Time"])["index"].agg("mean").reset_index()
    waiting_time_df = pd.merge(time_range_df, waiting_time_df, on="Time", how="left")
    waiting_time_df[group] = "Total Flight"

    ranking_df = count_df.groupby(group)["index"].sum().sort_values(ascending=False)
    ranking_order = ranking_df.index.tolist()

    return count_df, ranking_order, waiting_time_df


def make_count_df(df, start_date, end_date, time_col, group, buffer_day=True, freq_min=1):
    df_copied = df.copy()
    if buffer_day==True:
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

    # 필요한 시간 열 설정(use loc_dict) 및 freq_min 단위로 반올림
    df_copied[time_col] = df_copied[time_col].dt.floor(f"{freq_min}T")

    # 그룹별로 카운트
    count_df = df_copied.groupby([time_col, group]).size().reset_index(name="index")
    count_df.columns = ["Time", group, "index"]
    count_df = pd.merge(time_range_df, count_df, on="Time", how="left")
    count_df["index"] = count_df["index"].fillna(0)
    count_df[group] = count_df[group].fillna("")
    # ranking 설정하기
    ranking_df = count_df.groupby(group)["index"].sum().sort_values(ascending=False)
    ranking_order = ranking_df.index.tolist()

    return count_df, ranking_order


def show_bar(df, ranking_order, group, capa_df=None, max_y=None):
    fig = px.bar(
        df,
        x="Time",
        y="index",
        color=group,
        category_orders={group: ranking_order},  # 범례의 순서 지정
    )
    if max_y is not None:
        fig.update_layout(yaxis_range=[0, max_y])  # y축 범위 설정: [최솟값, 최댓값]

    # if capa_df:
    if capa_df is not None:
        fig.add_scatter(
            x=capa_df.index,
            y=capa_df["capacity"],
            mode="lines",
            name="Capacity",
            line=dict(color="red", width=3),
        )

    fig.update_layout(
        xaxis_title="",
        yaxis_title="",
        barmode="stack",
        legend=dict(
            x=0.95,  # x좌표 (0~1 사이, 1에 가까울수록 오른쪽)
            y=1,  # y좌표 (0~1 사이, 1에 가까울수록 위쪽)
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(0,0,0,0)",  # 투명한 배경
        ),
    )
    st.plotly_chart(fig)


import plotly.graph_objects as go
import numpy as np




# def create_sankey(df, component_list, suffix="_pred"):
#     # 프로세스별 고유값과 인덱스 매핑
#     nodes = []
#     node_dict = {}
#     idx = 0

#     # 노드 인덱스 생성
#     for process in component_list:
#         col_name = f"{process}{suffix}"
#         for value in df[col_name].unique():
#             if value not in node_dict and pd.notna(value):
#                 node_dict[value] = idx
#                 # 각 value의 길이를 label로
#                 label_count = len(df[df[col_name] == value])
#                 nodes.append(f"{value} ({label_count})")
#                 idx += 1

#     # source, target, value 생성
#     sources, targets, values = [], [], []
#     for i in range(len(component_list) - 1):
#         source_col = f"{component_list[i]}{suffix}"
#         target_col = f"{component_list[i+1]}{suffix}"

#         flow = df.groupby([source_col, target_col]).size().reset_index()
#         for _, row in flow.iterrows():
#             if pd.notna(row[source_col]) and pd.notna(row[target_col]):
#                 sources.append(node_dict[row[source_col]])
#                 targets.append(node_dict[row[target_col]])
#                 values.append(row[0])

#     def get_plotly_colors(
#         n, palette_name="Pastel1", opacity=1
#     ):  # opacity 매개변수 추가
#         palettes = {
#             "Alphabet": px.colors.qualitative.Alphabet,
#             "Antique": px.colors.qualitative.Antique,
#             "D3": px.colors.qualitative.D3,
#             "Prism": px.colors.qualitative.Prism,
#             "T10": px.colors.qualitative.T10,
#         }

#         colors = palettes[palette_name][:n]
#         return [
#             f'{color.replace("rgb", "rgba").replace(")", f", {opacity})")}'
#             for color in colors
#         ]

#     # Sankey 다이어그램 생성
#     fig = go.Figure(
#         go.Sankey(
#             node=dict(
#                 label=nodes,
#                 # pad = 15,
#                 # thickness = 30,
#                 color="#007aff",  # 노드 색상
#                 hoverinfo="none",  # 링크 호버 비활성화
#             ),
#             link=dict(
#                 source=sources,
#                 target=targets,
#                 value=values,
#                 color=get_plotly_colors(len(sources), "Antique", opacity=0.5),
#                 hoverinfo="none",  # 링크 호버 비활성화
#             ),
#         )
#     )

#     fig.update_layout(title="Passenger Flow", font_size=14, height=900)

#     st.plotly_chart(fig)


def create_sankey(df, component_list, suffix="_pred"):
    # 노드 생성
    node_dict, labels, idx = {}, [], 0
    for proc in component_list:
        col = f"{proc}{suffix}"
        for v in df[col].unique():
            if pd.notna(v) and v not in node_dict:
                node_dict[v] = idx
                labels.append(f"{v} ({(df[col]==v).sum()})")
                idx += 1

    # 링크 생성
    sources, targets, values = [], [], []
    for i in range(len(component_list)-1):
        s, t = f"{component_list[i]}{suffix}", f"{component_list[i+1]}{suffix}"
        flow = df.groupby([s, t]).size().reset_index(name="cnt")
        for _, r in flow.iterrows():
            if pd.notna(r[s]) and pd.notna(r[t]):
                sources.append(node_dict[r[s]])
                targets.append(node_dict[r[t]])
                values.append(int(r["cnt"]))

    # 색상 생성
    #             "Alphabet": px.colors.qualitative.Alphabet,
    #             "Antique": px.colors.qualitative.Antique,
    #             "D3": px.colors.qualitative.D3,
    #             "Prism": px.colors.qualitative.Prism,
    #             "T10": px.colors.qualitative.T10,
    def get_colors(n, palette="D3", opacity=0.4):
        base = px.colors.qualitative.__dict__[palette]
        out = []
        for i in range(n):
            c = base[i % len(base)]
            if opacity<1: c = c.replace("rgb","rgba").replace(")",f", {opacity})")
            out.append(c)
        return out

    # source 노드별 색 매핑
    uniq = sorted(set(sources))
    pal = get_colors(len(uniq))
    cmap = {src: pal[i] for i, src in enumerate(uniq)}
    link_colors = [cmap[s] for s in sources]

    fig = go.Figure(go.Sankey(
        node=dict(label=labels, color="#007aff", hoverinfo="none"),
        link=dict(source=sources, target=targets, value=values, color=link_colors, hoverinfo="none")
    ))
    fig.update_layout(title="Passenger Flow", font_size=14, height=750)

    st.plotly_chart(fig)





def optimize_configuration(return_dict):
    with st.expander("**OPTIMIZE CONFIGURATION**"):
        st.write("Here we provide the optimal configuration setup for your simulation.")
    return return_dict


        

# Simulator Engine ########################################################
# Simulator Engine ########################################################
# Simulator Engine ########################################################
class DsSimulator:
    def __init__(self, graph, show_up_arr, dist_key, dist_map, df_pax, comp):
        """
        Delay Selection Simulator
        :param graph: DsGraph instance
        :param show_up_arr: array of integers - show up values (e.g., ck_on time clock)
        :param dist_key: distribution key
        :param dist_map: distribution map
        """
        self.graph = graph
        self.nodes = self.graph.nodes
        self.show_up_arr = show_up_arr
        self.N = len(show_up_arr)
        self.dist_key = dist_key
        self.dist_map = dist_map
        self.df_pax = df_pax
        self.idx = 0
        self.comp = comp

    def add_flow(self, t, greedy=False):
        """
        Add a flow to the node
        :param t: time step (seconds)
        :param greedy: boolean variable. whether greedily select the dst with minimum passenger_queuesue
        """
        source = self.comp[0]
        while self.idx < self.N and self.show_up_arr[self.idx] <= t:
            edited_df = self.df_pax.loc[self.idx][f"{source}_edited_df"]

            if edited_df is not None:  # edited_df 있는 경우
                dst = edited_df.columns
                prob = edited_df.loc[self.dist_key[self.idx]]
            else:
                dst, prob = self.dist_map[self.dist_key[self.idx]]

            ###################################### 2025.07.02
            ###################################### 2025.07.02
            ###################################### 2025.07.02
            # STEP 1: open_mask 기반으로 닫힌 노드 제외
            open_dst = []
            open_prob = []
            for i, d in enumerate(dst):
                node = self.nodes[d]
                config_now = node.processing_config_full[t // 60]  # ← second → t로 맞춤
                open_mask = any(ti is not None and ti > 0 for ti in config_now)
                if open_mask:
                    open_dst.append(d)
                    open_prob.append(prob[i])

            # STEP 2: 열린 노드가 있을 경우만 필터 반영
            if open_dst:
                dst = open_dst
                prob = np.array(open_prob)
                prob = prob / prob.sum()  # normalize
            ###################################### 2025.07.02
            ###################################### 2025.07.02
            ###################################### 2025.07.02



            if greedy:
                min_len = min(len(self.nodes[d].passenger_queues) for d in dst)
                candidates = [i for i, d in enumerate(dst) if len(self.nodes[d].passenger_queues) == min_len]
                node = self.nodes[dst[np.random.choice(candidates)]]

            else:
                # max_capacity 안 넘는 후보만 추림
                candidates = [i for i, d in enumerate(dst) if len(self.nodes[d].passenger_queues) < self.nodes[d].max_capacity]
                if candidates:
                    # 확률도 그에 맞게 정규화
                    norm_prob = prob[candidates] / np.sum(prob[candidates])
                    node = self.nodes[dst[np.random.choice(candidates, p=norm_prob)]]
                else:
                    # 모든 노드가 꽉 찼을 때는 기존 방식대로 뽑기
                    node = self.nodes[dst[np.random.choice(len(prob), p=prob)]]

            available_sum = (node.unoccupied_facilities > 0).sum()  # 1인 상태만 고려 합산
            if available_sum == 0:
                heapq.heappush(node.passenger_queues, (self.show_up_arr[self.idx], node.passenger_node_id))
                node.que_history[node.passenger_node_id] = len(node.passenger_queues)

            elif available_sum > 0:
                node.que_history[node.passenger_node_id] = len(node.passenger_queues)
                heapq.heappush(node.passenger_queues, (self.show_up_arr[self.idx], node.passenger_node_id))

            # node.passenger_queues = [
            #     (100, 0),  # 100초에 도착해서 수속중인 0번 승객
            #     (120, 1)   # 120초에 도착해서 수속중인 1번 승객
            # ]
            node.passenger_ids.append(self.idx)
            node.on_time[node.passenger_node_id] = t  # node별 time_stamp 저장장소
            node.passenger_node_id += 1
            self.idx += 1



    def run(self, start_time, end_time, add_flow_time, prod_time, unit=1): ### 2024.04.24 ###
        """
        :param st: starting a time clock (seconds) for Delay & Selection simulation.
        :param et: ending a time clock (seconds) for Delay & Selection simulation.
        :param unit: simulation unit (default = 1 second).
        """
        # 전체 진행률을 표시할 progress bar 생성
        progress_bar = st.progress(0)
        total_iterations = (end_time - start_time + 1)//1000
        import time as tm

        for t in range(start_time, end_time + 1):
            if t % unit:
                continue

            self.add_flow(t)
            simulation_finished = self.graph.prod(t, self.df_pax, self.comp)

            if simulation_finished=="simulation_finished":
                for node in self.nodes : # 시뮬레이션이 돌아가지 않는 시간대의 Capacity를 0으로 만들기 위한 조치
                    node.processing_config_full[:start_time//60+1,:]=None ### 2025.05.12 ###
                    node.processing_config_full[t//60:,:]=None


                break
            ### 2024.04.24 ###

            if t % 1000 ==0:
            # 진행률 계산 및 progress bar 업데이트
                progress = min(((t - start_time)//1000) / total_iterations, 1)
                progress_bar.progress(progress)

        progress_bar.progress(1.0)


class DsGraph:
    def __init__(
        self,
        N,
        components,
        num_nodes,
        comp_to_idx,
        idx,
        num_facilities,
        facility_type, ### 2024.04.24 ###
        node_labels,
        max_capacity,
        target_waiting_time,  ### 2024.04.24 ###
        fac_config,
        graph,
        source,
        logger,
    ):
        """
        Graph of nodes (components) for Delay & Selection simulation
        :param N: number of components.
        :param components: string list of components. each component contains a list of nodes. of length N
        :param num_nodes: integer list of number of nodes for each component. of length N
        :param comp_to_idx: dictionary mapping each component to a list of indices.
        :param idx: integer list of indices for each node in the component. of length # nodes.
        :param num_facilities: list of integers that is the amount of facility for each node. of length # nodes.
        :param node_labels: list of label tuple (comp, node) for each node. of length # nodes.
        :param max_capacity: maximum number of passenger_queuesues for each node in nodes[i]. of length # nodes.
        :param fac_config: list of facility config (passenger processing time) of each node. of length # nodes.
        :param graph: graph[i] contains a list of edges for node i. [dst, prob]. - [[j1, w1], ...] - of length # nodes.
        :param source: source data of the graph
        :param logger: logger object

        e.g.,
        N = 4
        components = ['check in', 'departure gate', 'security', 'passport']
        num_nodes = [8, 4, 4, 4]
        comp_to_idx = {'check in': [0, 1, 2, 3, 4, 5, 6, 7], 'departure gate': [8, 9, 10, 11],
                       'security': [12, 13, 14, 15], 'passport': [16, 17, 18, 19]}
        idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        num_facilities = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        node_labels = [('check in', 'A'), ('check in', 'B'), ... , ('passport', 'PC4')]
        max_capacity = [200, ..., 200]
        fac_config = [array([[180], ..., [180]], dtype=int64), ..., array([[180], ..., [180]], dtype=int64)]
        graph = [[[4, .5], [5, .5]], [[12, 1.0]], [[19, 1.0]], [[16, 1.0]], ..., [[19, 1.0]], [], [], [], []]
        """

        self.N = N
        self.components = components
        self.num_nodes = num_nodes
        self.comp_to_idx = comp_to_idx
        self.idx = idx
        self.num_facilities = num_facilities
        self.facility_type = facility_type  ### 2024.04.24 ###
        self.node_labels = node_labels
        self.max_capacity = max_capacity
        self.target_waiting_time=target_waiting_time  ### 2024.04.24 ###
        self.fac_config = fac_config
        self.graph = graph
        self.source = source
        self.logger = logger
        self.N0 = sum(self.num_nodes)
        self.nodes = []
        self.build()

    def build(self):
        """
        Initialize nodes with the given graph
        """
        
        def make_comp_lables(components, comp_to_idx, i):
            components = components + [None] * 100
            current_comp = [key for key, value in comp_to_idx.items() if i in value][0]
            current_idx = components.index(current_comp)
            components = components[current_idx:]
            return components

        for i in range(self.N0):
            node = DsNode(
                components=make_comp_lables(self.components, self.comp_to_idx, i),
                node_id=i,
                comp_to_idx=self.comp_to_idx,
                num_facilities=self.num_facilities[i],
                facility_type=self.facility_type[i],  ### 2024.04.24 ###
                facility_schedule=self.fac_config[i],
                max_capacity=self.max_capacity[i],
                target_waiting_time=self.target_waiting_time[i],  ### 2024.04.24 ###
                num_passengers=500000,
                processes=None,
                is_deterministic=True,
                # passenger_queues=None,
                destinations=(
                    [dst[0] for dst in self.graph[i]]
                    if len(self.graph[i]) > 0
                    else None
                ),  # dst_list = [4,5] >> 의미 4번, 5번 노드
                destination_choices=(
                    [dst[1] for dst in self.graph[i]]
                    if len(self.graph[i]) > 0
                    else None
                ),  # dst_choice = [0.9, 0.1] >> 의미 4번노드로 0.9확률, 5번노드로 0.1확률로 뿌리기
                
                node_label=f"{self.node_labels[i][0]}*^*{self.node_labels[i][1]}",  # node_label = check in_A >> 의미 component + _ + node
                bypass=False,
            )
            self.nodes.append(node)  # node 객체를 담은 것이 nodes이다.
        self.nodes = np.array(self.nodes, dtype=object)
        for node in self.nodes:
            if node.destinations is not None:
                node.destinations = [
                    self.nodes[i] for i in node.destinations
                ]  # node.dst_list에 목적지 노드의 객체 자체를 넣는 과정

    def prod(self, t, df_pax, comp):
        """
        Run simulation with Delay & Selection simulation.
        :param t: time step (second)
        :param t0: time step (minute)
        """
        for node in self.nodes:
            node.prod(t, self.nodes, df_pax)  

        if t%10000==0:
            cnt_sum=0    
            for node in self.nodes:
                cnt_sum+=node.passenger_node_id 
            if ((len(df_pax)*len(comp))==cnt_sum):
                return "simulation_finished"





from numpy import ndarray
from datetime import datetime, timedelta

class DsNode:
    def __init__(
        self,
        node_id,
        node_label: str,
        components,
        destination_choices,
        destinations,
        facility_schedule: ndarray,
        max_capacity,
        facility_type,
        target_waiting_time, 
        num_facilities,
        processes,
        comp_to_idx,
        num_passengers: int = 500_000,
        bypass: bool = False,
        is_deterministic: bool = False,
    ):
        self.node_id = node_id
        self.node_label = node_label
        self.components = components
        self.target_waiting_time = target_waiting_time 
        self.facility_type = facility_type 
        self.is_deterministic = is_deterministic
        self.max_capacity = max_capacity
        self.num_passengers = num_passengers
        self.destinations = destinations
        self.destination_choices = destination_choices
        self.bypass = bypass
        self.processes = processes
        self.comp_to_idx = comp_to_idx
        self.num_facilities = num_facilities

        self.occupied_facilities = []
        self.passenger_ids = []
        self.passenger_node_id = 0
        self.passenger_queues = []
        self.que_history = np.zeros(num_passengers, dtype=int) - 1
        self.facility_numbers = np.zeros(num_passengers, dtype=int) - 1 
        self.processing_time = np.zeros(num_passengers, dtype=int)
        self.on_time = np.zeros(num_passengers, dtype=int)
        self.done_time = np.zeros(num_passengers, dtype=int)
        self.move_time = np.zeros(num_passengers, dtype=int)
        # 2025.07.24 - 이동 대기 큐 추가
        self.moving_queues = []


        if (self.facility_type=="normal_facility") & (self.target_waiting_time is not None):
            self.unoccupied_facilities = np.array([1] + [-1] * (self.num_facilities - 1))
        else:
            self.unoccupied_facilities = np.ones(self.num_facilities, dtype=int)  
        self.processing_config_full = np.tile(facility_schedule, (3, 1))
        self.selection_config = 1 * (self.processing_config_full > 0)
        self.latest_opened = 0  



    def normalize(self, arr):
        return arr / sum(arr)


    def prod(self, second, nodes, passengers): ### 2024.04.24 ###
        if self.target_waiting_time is not None : 
            if (self.facility_type=="normal_facility") :
                if (second % (60) == 0)  :
                    self._adjust_facilities(target_delay=self.target_waiting_time*60, 
                                            second=second, 
                                            adjust_interval=1)
        self._counter_to_move(second)
        self._move_to_next_node(second, nodes, passengers) ### 2024.04.24 ###
        self._que_to_counter(second, nodes, passengers) ### 2024.04.24 ###



    # 2025.07.24 - 노드 간 이동 시간 처리 메서드 추가
    def _counter_to_move(self, second):
        """처리 완료된 승객들을 이동 상태로 전환"""
        while self.occupied_facilities and self.occupied_facilities[0][0] <= second:
            done_time, passenger_node_id, facility_number = heapq.heappop(
                self.occupied_facilities
            )
            
            self.done_time[passenger_node_id] = second
            self.processing_time[passenger_node_id] += second - done_time 
            self.unoccupied_facilities[facility_number] = 1
            
            # 이동 대기 큐에 추가 (600초 이동)
            move_time = second + 1200
            heapq.heappush(
                self.moving_queues,
                (move_time, passenger_node_id)
            )

    def _move_to_next_node(self, second, nodes, passengers):
        # 2025.07.24 - 이동 완료된 승객들 처리
        while self.moving_queues and self.moving_queues[0][0] <= second:
            move_time, passenger_node_id = heapq.heappop(
                self.moving_queues
            ) # 2025.07.24
            
            self.move_time[passenger_node_id] = second # 2025.07.24

            ##### STEP1 : SELECT DESTINATION ##########################
            if self.destinations is None:
                continue # 마지막 프로세스인 경우(ex : passport)
            passenger_id = self.passenger_ids[passenger_node_id]
            destination = self.select_destination(
                nodes, passengers, passenger_id, second
            )


            ##### STEP2 : SELECT TO→COUNTER OR TO→QUE ########################
            if destination:
                destination.passenger_ids.append(passenger_id)
                _passenger_node_id = destination.passenger_node_id
                destination.passenger_node_id += 1
                destination.on_time[_passenger_node_id] = max(
                    self.move_time[passenger_node_id],  # Security 완료 시각
                    second                              # 현재 시뮬레이터 시각
                ) # 2025.07.24

                destination_second = destination.on_time[_passenger_node_id] 
                destination_facility_number = destination.select_facility(
                    second=destination_second
                ) 
                destination.facility_numbers[_passenger_node_id] = (
                    destination_facility_number
                )

                ############################################################# 2025.07.03 add dod component
                destination_passenger_id = destination.passenger_ids[_passenger_node_id]
                dest_of_dest = destination.select_destination(
                    nodes, passengers, destination_passenger_id, second
                )
                ############################################################# 2025.07.03 add dod component






                if (destination_facility_number == 0) :
                    heapq.heappush(
                        destination.passenger_queues,
                        (destination.on_time[_passenger_node_id], _passenger_node_id),
                    ) # TO→QUE

                ############################################################# 2025.07.03 add dod component
                # 목적지의 목적지가 꽉 찼을 경우, 안보낸다
                elif (dest_of_dest is not None) and (dest_of_dest.max_capacity * np.random.normal(0.4, 0.15)  <= len(dest_of_dest.passenger_queues)):
                    destination.unoccupied_facilities[
                        destination_facility_number - 1
                    ] = 1
                    heapq.heappush(
                        destination.passenger_queues,
                        (destination.on_time[_passenger_node_id], _passenger_node_id),
                    )
                ############################################################# 2025.07.03 add dod component


                else:
                    adjusted_processing_time = destination.adjust_processing_time(
                        destination.processing_config_full[destination_second//60][
                            destination_facility_number - 1
                        ] 
                    )
                    heapq.heappush(
                        destination.occupied_facilities,
                        (
                            destination.on_time[_passenger_node_id]
                            + adjusted_processing_time,
                            _passenger_node_id,
                            destination_facility_number - 1,
                        ),
                    ) # TO→COUNTER
                    destination.processing_time[_passenger_node_id] = (
                        adjusted_processing_time
                    )
                destination.que_history[_passenger_node_id] = len(
                    destination.passenger_queues) 



    def select_destination(self, nodes, df_pax, pax_idx, second):
        start_component = self.components[0]
        destination_component = self.components[1]

        ### STEP1 : destination_nodes ##########################
        edited_df = None
        if destination_component is None:
            priority_destination_node_indices = None
        else:
            edited_df = df_pax.loc[pax_idx][f"{destination_component}_edited_df"]
            if edited_df is not None:  # edited_df 있는 경우
                priority_destination_node_indices = (
                    None if self.node_id not in edited_df.index else edited_df.columns
                )
            else:  # edited_df 없는 경우
                priority_destination_node_indices = None
        priority_destination_nodes = None if priority_destination_node_indices is None else nodes[priority_destination_node_indices]
        destination_nodes = priority_destination_nodes if priority_destination_nodes is not None else self.destinations


        ###################################### 2025.07.02
        ###################################### 2025.07.02
        ###################################### 2025.07.02
        if destination_nodes is not None:
            all_dst_choices=None
            if (edited_df is not None) and (self.node_id in edited_df.index):
                all_dst_choices = edited_df.loc[self.node_id].values
            else:
                all_dst_choices = self.destination_choices
            nonzero_indices = [i for i, p in enumerate(all_dst_choices) if p > 0]
            destination_nodes = [destination_nodes[i] for i in nonzero_indices]
            all_dst_choices = [all_dst_choices[i] for i in nonzero_indices]


            ### STEP2 : available_destination_nodes ##########################
            open_notmax_node_ids = []
            open_notmax_nodes = []
            open_node_ids =[]
            open_nodes=[]
            for i, node in enumerate(destination_nodes):
                notmax_mask=len(node.passenger_queues) < node.max_capacity
                config_now = node.processing_config_full[second // 60]
                open_mask = any(t is not None and t > 0 for t in config_now)
                if notmax_mask & open_mask:
                    open_notmax_node_ids.append(i) # [0,2,3]
                    open_notmax_nodes.append(node) # [object(0), object(2), object(3)]
                if open_mask :
                    open_node_ids.append(i) 
                    open_nodes.append(node) 

            ### STEP3 : dst_choices ##########################
            if len(open_notmax_node_ids) >= 1: # 맥스큐 안넘으면서 & 열려있는 경우>>
                open_notmax_choices = [all_dst_choices[i] for i in open_notmax_node_ids]
                return open_notmax_nodes[np.random.choice(len(open_notmax_choices), p=self.normalize(open_notmax_choices))]
            elif len(open_node_ids)>=1: # 맥스큐 넘으면서 & 열려있는 경우>>
                open_choices = [all_dst_choices[i] for i in open_node_ids] #하나라도 열린 node의 확률을 뽑는다.
                return open_nodes[np.random.choice(len(open_choices), p=self.normalize(open_choices))]
            else: # 닫혀있는 경우
                return destination_nodes[np.random.choice(len(all_dst_choices), p=self.normalize(all_dst_choices))]

            ###################################### 2025.07.02
            ###################################### 2025.07.02
            ###################################### 2025.07.02





    def _que_to_counter(self, second, nodes, passengers): 
        while self.passenger_queues and self.passenger_queues[0][0] <= second:
            counter_num = self.select_facility(second)
            if counter_num == 0:
                break  
            passenger_node_id = self.passenger_queues[0][1]
            passenger_id = self.passenger_ids[passenger_node_id]
            self.facility_numbers[passenger_node_id] = counter_num
            priority_destination_node_indices = None
            start_component = self.components[0]
            destination_component = self.components[1]
            adjusted_processing_time = self.adjust_processing_time(
                processing_time=self.processing_config_full[second//60][counter_num - 1]
            ) 

            ### STEP1 : destination_nodes ##########################
            edited_df=None
            if destination_component:
                edited_df = passengers.loc[passenger_id][f"{destination_component}_edited_df"]
                if edited_df is not None: # edited_df 있는 경우
                    priority_destination_node_indices = (
                        None if self.node_id not in edited_df.index else edited_df.columns
                    )
                else: # edited_df 없는 경우
                    priority_destination_node_indices = None
            priority_destination_nodes = None if priority_destination_node_indices is None else nodes[priority_destination_node_indices]
            destination_nodes = priority_destination_nodes if priority_destination_nodes is not None else self.destinations


            ################################################ 2025.07.02
            ## STEP2 : select prod or hold ############################
            if destination_nodes is not None:
                all_dst_choices=None
                if (edited_df is not None) and (self.node_id in edited_df.index):
                    all_dst_choices = edited_df.loc[self.node_id].values
                else:
                    all_dst_choices = self.destination_choices
                nonzero_indices = [i for i, p in enumerate(all_dst_choices) if p > 0]
                destination_nodes = [destination_nodes[i] for i in nonzero_indices]



                # 열린 노드만 필터링
                open_nodes = []
                for node in destination_nodes:
                    config_now = node.processing_config_full[second // 60]
                    open_mask = any(t is not None and t > 0 for t in config_now)
                    if open_mask:
                        open_nodes.append(node)


                def is_mostly_full(nodes):
                    num_of_nodes = len(nodes)

                    if num_of_nodes == 1:
                        threshold = np.random.normal(0.4, 0.15) 
                        return len(nodes[0].passenger_queues) >= nodes[0].max_capacity * threshold

                    elif num_of_nodes > 1:
                        full_threshold = np.random.normal(0.4, 0.15) 
                        full_nodes = [
                            n for n in nodes
                            if len(n.passenger_queues) >= n.max_capacity * full_threshold
                        ]
                        if len(full_nodes) / num_of_nodes >= 0.5:
                            return True # full_nodes 비율이 50% 초과면 break 
                        else:
                            return False
                    
                if open_nodes:
                    if is_mostly_full(open_nodes):
                        self.unoccupied_facilities[counter_num - 1] = 1
                        break
                else:
                    if is_mostly_full(destination_nodes):
                        self.unoccupied_facilities[counter_num - 1] = 1
                        break
            ################################################ 2025.07.02


            ### STEP3: prod ############################################
            heapq.heappop(self.passenger_queues)
            self.processing_time[passenger_node_id] = adjusted_processing_time
            heapq.heappush(
                self.occupied_facilities,
                (second + adjusted_processing_time, passenger_node_id, counter_num - 1),
            )






    def select_facility(self, second) -> int | None:
        if self.facility_type=="infinite_facility": # infinite_facility일 경우, 열고닫는것 없이 무조건 1번 카운터로 통과하는 것으로 계산
            return 1

        current_selection = self.selection_config[second//60] 
        available_facility_indices = np.where(
            np.multiply(current_selection, self.unoccupied_facilities) > 0
        )[0]

        if len(available_facility_indices) == 0:
            return 0

        if self.is_deterministic:
            index = 0
        else:
            probabilities = self.normalize(
                current_selection[available_facility_indices]
            )
            index = np.random.choice(len(available_facility_indices), p=probabilities)

        # NOTE: 선택된 시설(facility)의 가용 여부를 변경한다.
        self.unoccupied_facilities[available_facility_indices[index]] = 0

        facility_index = available_facility_indices[index] + 1
        return facility_index


    def adjust_processing_time(self, processing_time, sigma_multiple=0):
        if processing_time is None or processing_time == 0:
            return processing_time

        sigma = np.sqrt(processing_time)
        if sigma < 1:
            return processing_time

        adjusted_processing_time = int(
            round(np.random.normal(processing_time, sigma * sigma_multiple))
        )
        return max(adjusted_processing_time, processing_time // 2) 



    def check_condition(self, passenger, condition, component, check_time):
        criteria = condition.criteria

        if criteria == component:
            return True

        elif criteria == "Time":
            check_time = (datetime.min + timedelta(seconds=int(check_time))).time()
            operator = condition.operator
            condition_time = datetime.strptime(condition.value, "%H:%M")

            if operator == "start":
                return check_time >= condition_time.time()
            if operator == "end":
                return check_time <= condition_time.time()

        else:
            criteria_col = COL_FILTER_MAP.get(criteria, None)
            if not criteria_col:
                return False

            return passenger[criteria_col] in condition.value





    def _adjust_facilities(self, target_delay, second, adjust_interval=1, open_threshold=0.85, close_threshold=0.1):
        # 현재 시각의 분 단위 인덱스 계산
        idx_min = second // 60
        
        # 현재 설정 복사
        updated_config = self.processing_config_full[idx_min].copy()
        updated_config_orig = self.processing_config_full[idx_min].copy()

        # 닫힌 시설은 updated_config에서 None 처리
        updated_config[self.unoccupied_facilities == -1] = None

        # delay가 없으면 설정만 반영하고 종료
        latest_idx_min = np.argmax(self.done_time)
        if self.done_time[latest_idx_min] <= 0:
            self.processing_config_full[idx_min : idx_min+adjust_interval] = updated_config
            return

        # 최근 지연시간과 대기열 비율 계산
        recent_delay = self.done_time[latest_idx_min] - self.on_time[latest_idx_min]
        
        # 현재 열린 시설의 총 1분당 처리 용량 계산
        current_capacity = 0
        for i, status in enumerate(self.unoccupied_facilities):
            trans_time = updated_config_orig[i]
            if status >= 0 and trans_time is not None and trans_time > 0:
                current_capacity += 60 / trans_time  # 60초 기준 처리 가능 수
                
        # 최근 5분간 온 여객 수 계산 → 1분 평균 수요 추정
        time_diff = second - self.on_time
        demand_5min = np.sum((time_diff >= 0) & (time_diff < 300))
        demand_per_min = demand_5min / 5.0

        # 수요와 용량 차이 계산
        shortage = demand_per_min - current_capacity

        ###### 시설 열기 ######
        if (recent_delay > target_delay * open_threshold or len(self.passenger_queues) > 10):
            if shortage > 0:
                # 열 수 있는 후보 시설들의 평균 처리 시간 계산
                open_candidates = (self.unoccupied_facilities == -1) & (updated_config_orig > 0)
                if np.any(open_candidates):
                    avg_trans_time = np.mean(updated_config_orig[open_candidates])
                    # 부족분을 처리하기 위해 필요한 시설 수 계산 (10% 여유 추가)
                    iter_open = int(np.ceil(shortage * 1.1 * avg_trans_time / 60))

                    # iter_open 개수만큼 시설 열기
                    for _ in range(iter_open):
                        for fac in range(len(self.unoccupied_facilities)):
                            if self.unoccupied_facilities[fac] == -1:
                                self.unoccupied_facilities[fac] = 1
                                updated_config[fac] = self.processing_config_full[idx_min][fac]
                                self.latest_opened = second ### 2025.05.12 ###
                                break

        ###### 시설 닫기 (열린 후 1시간 지난 경우만) ######
        elif (recent_delay < target_delay * close_threshold) or (shortage < 0):
            if shortage < 0:
                # 닫을 수 있는 후보 시설들의 평균 처리 시간 계산
                closed_candidates = (self.unoccupied_facilities == 1) & (updated_config_orig > 0)
                if np.any(closed_candidates):
                    avg_trans_time = np.mean(updated_config_orig[closed_candidates])
                    # 10%의 여유를 두고 줄이도록 시설 수 계산
                    iter_closed = int(np.ceil(shortage * -1 * 0.9 * avg_trans_time / 60))

                    # 닫을 수 있는 시설 후보 선정 (1시간 이상 운영된 시설)
                    MIN_OPEN_LASTED_MIN = 40
                    candidates = [
                        i for i in reversed(range(len(self.unoccupied_facilities)))
                        if self.unoccupied_facilities[i] == 1 and (second - self.latest_opened) >= MIN_OPEN_LASTED_MIN*60 ### 2025.05.12 ###
                    ]

                    # 최소 1개 시설은 유지하면서 시설 닫기
                    MIN_OPEN_FACILITIES = 1
                    for i in range(min(len(candidates), iter_closed)):
                        if len([s for s in self.unoccupied_facilities if s >= 0]) > MIN_OPEN_FACILITIES and candidates:
                            fac_to_close = candidates[i]
                            self.unoccupied_facilities[fac_to_close] = -1
                            updated_config[fac_to_close] = None

        # 닫힌 시설은 config에서 None 처리
        updated_config[self.unoccupied_facilities == -1] = None

        # 변경된 설정 반영
        self.processing_config_full[idx_min : idx_min+adjust_interval] = updated_config









class DsInputPreprocessor:
    def __init__(self, df):
        """
        Preprocessor for DS input data. Assign input data to the roots of DS graph.
        :param df: Input data from streamlit user input
        """
        self.df = df

    def preprocess(
        self,
        component_key="component_list",
        node_list_key="node_list",
        fac_nums_key="facility_nums",
        facility_type_key = "facility_type", ### 2024.04.24 ###
        max_capacity_key="max_capacity",
        target_waiting_time_key="target_waiting_time", ### 2024.04.24 ###
        choice_mat_key="choice_matrix",
        mat_key="matrix_df",
        dst_key="dst",
        row_idx_key="row_idx",
    ):
        """
        preprocess DS input data from streamlit user input
        :param component_key: key of component in DS graph
        :param node_list_key: key of nodes in DS graph
        :param fac_nums_key: key of facilities in DS graph
        :param max_capacity_key: key of max passenger_queuesue in DS graph
        :param choice_mat_key: key of choice matrix in DS graph
        :param mat_key: key of matrix in DS graph
        :param dst_key: key of dst in DS graph
        :param row_idx_key: key of row index in DS graph
        :return: preprocessed DS input data for simulation
        """
        N = len(self.df[component_key])
        components = self.df[component_key]
        num_nodes = [len(self.df[comp][node_list_key]) for comp in components]

        raw_idx = []
        comp_to_idx = {}
        for i in range(len(components)):
            v = num_nodes[i]
            if not raw_idx:
                raw_idx.append([i for i in range(v)])
            else:
                raw_idx.append([i + 1 + raw_idx[-1][-1] for i in range(v)])
            comp_to_idx[components[i]] = raw_idx[-1]

        idx = list(reduce(lambda x, y: x + y, raw_idx, []))
        num_facilities = list(
            reduce(
                lambda x, y: x + y,
                [self.df[comp][fac_nums_key] for comp in components],
                [],
            )
        )
        node_labels = list(
            reduce(
                lambda x, y: x + y,
                [
                    [(comp, node) for node in self.df[comp][node_list_key]]
                    for comp in components
                ],
                [],
            )
        )

        max_capacity = []
        facility_type = [] ### 2024.04.24 ###
        target_waiting_time = [] ### 2024.04.24 ###
        for comp in components:
            max_capacity += self.df[comp][max_capacity_key]
            facility_type += self.df[comp][facility_type_key]  ### 2024.04.24 ###
            target_waiting_time += self.df[comp][target_waiting_time_key] ### 2024.04.24 ###

        fac_config = [
            self.df[comp][node] for (comp, node) in node_labels
        ]  # each array of length 1440 (1d)

        graph, dist_map = self.add_edges(
            idx, comp_to_idx, choice_mat_key, mat_key, dst_key, row_idx_key
        )
        return (
            N,
            components,
            num_nodes,
            comp_to_idx,
            idx,
            num_facilities,
            facility_type,  ### 2024.04.24 ###
            node_labels,
            max_capacity,
            target_waiting_time,  ### 2024.04.24 ###
            fac_config,
            graph,
            dist_map,
        )

    def add_edges(
        self, idx, comp_to_idx, choice_mat_key, mat_key, dst_key, row_idx_key
    ):
        """
        construct a graph and source
        :param comp_to_idx: mapping of a component to index
        :param choice_mat_key:
        :param mat_key:
        :param dst_key:
        :param row_idx_key:
        :return:
        """
        # st.write("idx")
        # st.write(idx)

        # st.write("comp_to_idx")
        # st.write(comp_to_idx)

        # st.write("choice_mat_key")
        # st.write(choice_mat_key)

        # st.write("mat_key")
        # st.write(mat_key)

        # st.write("dst_key")
        # st.write(dst_key)

        # st.write("row_idx_key")
        # st.write(row_idx_key)

        graph = [[] for _ in range(len(idx))]
        dist_map = {}
        choice_mat = self.df[choice_mat_key]
        for comp in choice_mat:
            mat = choice_mat[comp][mat_key]
            col_idx = []
            for dst in choice_mat[comp][dst_key]:
                col_idx += comp_to_idx[dst]
            col_idx = np.array(col_idx)
            mask = np.arange(len(mat[0]))
            row_idx = (
                np.array(comp_to_idx[comp])
                if comp in comp_to_idx
                else choice_mat[comp][row_idx_key]
            )
            for i in range(len(mat)):
                cols = mask[mat[i] > 0]
                if comp in comp_to_idx:
                    graph[row_idx[i]] = [[col_idx[col], mat[i][col]] for col in cols]
                else:
                    dist_map[row_idx[i]] = [
                        np.array([col_idx[col] for col in cols]),
                        np.array([mat[i][col] for col in cols]),
                    ]

        return graph, dist_map

    def filter_source(
        self, pax_key="df_pax", source_key="operating_carrier_name", ck_on_key="SHOW"
    ):
        """
        preprocess source data
        :param pax_key:
        :param source_key:
        :param ck_on_key:
        :return
        """
        data = self.df[pax_key]
        cols, arr = data.columns.values, data.values
        sorted_idx = np.argsort(arr[:, (cols == ck_on_key)].flatten())
        mask = (cols == ck_on_key) | (cols == source_key)
        filtered_data = arr[sorted_idx][:, mask]
        dist_key = filtered_data[:, 0].flatten()
        ck_on = filtered_data[:, -1].flatten()



        ### 2025.05.12 ###
        starting_time_stamp = ck_on[0]

        v0 = (
            starting_time_stamp.hour * 3600
            + starting_time_stamp.minute * 60
            + starting_time_stamp.second
        )


        ck_arr = np.round(
            [(td.total_seconds()) + v0 for td in (ck_on - np.array(ck_on[0]))]
        )

        v_end = ck_arr[-1]
        ### 2025.05.12 ###


        ### 2025.05.12 ###
        st.divider()
        if v_end <= 86400:
            v0 = v0 + 86400
            ck_arr = ck_arr + 86400
        ### 2025.05.12 ###


        return (
            cols,
            arr,
            sorted_idx,
            filtered_data,
            dist_key,
            ck_arr,
            starting_time_stamp,
            v0,
        )


class DsOutputWrapper:
    def __init__(self, passengers, components, nodes, starting_time):
        """
        Output Wrapper
        :param df_pax: original pax data
        :param nodes: list of nodes after simulation
        :param starting_time: starting time of simulation. [time stamp, integer value (total seconds)]
        """
        self.passengers = passengers
        self.components = components
        self.nodes = nodes
        self.starting_time = starting_time
        self.processing_config_full_dict={} ### 2024.04.24 ###

    def _add_column_dt(self, node, arr, col_label, method="normal"):
        if col_label not in self.passengers.columns:
            self.passengers.loc[:, col_label] = pd.NaT

        mask = arr > 0
        passenger_ids = node.passenger_ids[: len(arr[mask])]
        if mask.size > 0 and len(passenger_ids) > 0:
            if method == "normal":
                values_arr = (
                    pd.to_timedelta(arr[mask] - self.starting_time[1], unit="s")
                    + self.starting_time[0]
                )
            elif method == "delta":
                values_arr = arr[mask]

            self.passengers.iloc[passenger_ids, self.passengers.columns.get_loc(col_label)] = values_arr





    def _add_column_string(self, node, col_label):
        if col_label not in self.passengers.columns:
            self.passengers.loc[:, col_label] = ""

        passenger_ids = node.passenger_ids
        if len(passenger_ids) > 0:
            self.passengers.iloc[passenger_ids, self.passengers.columns.get_loc(col_label)] = str(node.node_label).replace("*^*", "_")

    def _add_column_info(self, node, arr, col_label):
        if col_label not in self.passengers.columns:
            self.passengers.loc[:, col_label] = pd.NaT

        mask = arr >= 0
        passenger_ids = node.passenger_ids[: len(arr[mask])]
        values_arr = arr[mask]

        self.passengers.iloc[passenger_ids, self.passengers.columns.get_loc(col_label)] = values_arr

    def write_pred(self):
        for node in self.nodes:
            node.passenger_ids = np.array(node.passenger_ids)

            # FIXME: 수정완료
            # ===> 만약 사용자가 Security에 SC_West, SC_East로 입력하면 이게 의도한 것과 다르게 된다.
            new_col_on = (node.node_label.split("*^*")[0]) + "_on_pred" # node_label = check in_A >> 의미 component + _ + node
            new_col_done = (node.node_label.split("*^*")[0]) + "_done_pred"
            new_pt = (node.node_label.split("*^*")[0]) + "_pt"
            new_passenger_queues = (node.node_label.split("*^*")[0]) + "_passenger_queues"
            new_fac = (node.node_label.split("*^*")[0]) + "_fac"
            new_col_fac = (node.node_label.split("*^*")[0]) + "_pred"

            self._add_column_dt(node, node.on_time, new_col_on, method="normal")
            self._add_column_dt(node, node.done_time, new_col_done, method="normal")
            self._add_column_dt(node, node.processing_time, new_pt, method="delta")
            self._add_column_info(node, node.que_history, new_passenger_queues)
            self._add_column_info(node, node.facility_numbers, new_fac)
            self._add_column_string(node, new_col_fac)
            self.processing_config_full_dict[str(node.node_label).replace("*^*", "_")] = node.processing_config_full ### 2024.04.24 ###

        for component in self.components:
            self.passengers[f"{component}_done_pred"] = pd.to_datetime(
                np.where(
                    self.passengers[f"{component}_done_pred"]< self.passengers[f"{component}_on_pred"],
                    None,
                    (self.passengers[f"{component}_done_pred"]),
                )
            )
            self.passengers[f"{component}_pt"]=(np.where(self.passengers[f"{component}_done_pred"].isna(), None, (self.passengers[f"{component}_pt"])))
            processing_time = pd.to_timedelta(
                self.passengers[f"{component}_pt"], unit="s"
            )

            # FIXME: 변수명 개선하기
            self.passengers[f"{component}_pt_pred"] = self.passengers[f"{component}_done_pred"] - processing_time


@st.cache_resource
def run_sim(return_dict, seed=0):
    import time as tm

    return_dict["df_pax"] = (
        return_dict["df_pax"]
        .sort_values(by="SHOW", ascending=True)
        .reset_index(drop=True)
    )

    np.random.seed(seed)
    ipp = DsInputPreprocessor(return_dict)
    cols, arr, sorted_idx, filtered_data, dist_key, td_arr, st_ts, st_int = (
        ipp.filter_source(
            pax_key="df_pax", source_key="operating_carrier_name", ck_on_key="SHOW"
        )
    )  



    (
        N,
        comp,
        num_nodes,
        comp_to_idx,
        idx,
        num_facilities,
        facility_type, 
        label,
        max_capacity,
        target_waiting_time, 
        fac_config,
        graph,
        dist_map,
    ) = ipp.preprocess()


    ################################################3
    graph = DsGraph(
        N=N,
        components=comp,
        num_nodes=num_nodes,
        comp_to_idx=comp_to_idx,
        idx=idx,
        num_facilities=num_facilities,
        facility_type = facility_type,### 2024.04.24 ###
        node_labels=label,
        max_capacity=max_capacity,
        target_waiting_time=target_waiting_time,### 2024.04.24 ###
        fac_config=fac_config,
        graph=graph,
        source=None,
        logger=None,
    )

    sim = DsSimulator(graph, td_arr, dist_key, dist_map, return_dict["df_pax"], comp)
    day_num=3




    stt=tm.time()
    add_flow_time=0
    prod_time=0
    sim.run(int(td_arr[0]), max(3600 * 24 * day_num - 1 , td_arr[-1] + 1), add_flow_time=add_flow_time, prod_time=prod_time) ### 2024.04.24 ### td_arr[0] & day_num - 1 
    ft=tm.time()
    taken_time=ft-stt
    st.write(f"sim.run : {taken_time}")

    ow = DsOutputWrapper(return_dict["df_pax"], comp, graph.nodes, [st_ts, st_int])



    ow.write_pred()

    ################################################3
    return_dict["processing_config_full_dict"] = ow.processing_config_full_dict ### atomated scheduler ###

    return ow.passengers
