import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
from datetime import datetime, time, timedelta
import base64


st.set_page_config(layout="wide")



# GENERAL INPUT
image_path = r"C:\Users\qkrru\OneDrive\바탕 화면\terminal_default_layout.png"
path=r"C:\Users\qkrru\OneDrive\바탕 화면\sim_pax_test_not_wait.csv"
df_orig=pd.read_csv(path)
df_orig

touch_point_list = [col.replace("_on_pred","") for col in df_orig.columns if '_on_pred' in col]
df_orig["scheduled_gate_departure_local"] = pd.to_datetime(df_orig["scheduled_gate_departure_local"])
selected_date = df_orig["scheduled_gate_departure_local"].dt.date.value_counts().index[0]
selected_terminal = df_orig["departure_terminal"].unique()[0]



# COLUMN INPUT
touch_point_dict={}
location_mapping_dict={}
location_mapping_dict[selected_terminal]={}

touch_point_length = len(touch_point_list)
x_dist_unit = 400/touch_point_length
for x_idx, component in enumerate(touch_point_list):
    touch_point_dict[component]=[f"{component}_pred", f"{component}_on_pred", f"{component}_done_pred", None]
    df_orig[f"{component}_on_pred"] = pd.to_datetime(df_orig[f"{component}_on_pred"])
    df_orig[f"{component}_done_pred"] = pd.to_datetime(df_orig[f"{component}_done_pred"])


    service_point_list=df_orig[f"{component}_pred"].unique()

    
    service_point_length = len(service_point_list)
    y_dist_unit = 120/service_point_length
    for y_idx, service_point in enumerate(service_point_list):
        # x = [0~400], y=[0~120]
        location_mapping_dict[selected_terminal][(component, service_point)]={"angle":90, "offset":(x_dist_unit*x_idx+60,y_dist_unit*y_idx+15)}














def find_location_zigzag(
    df, h_diff, v_diff, points_per_row, angle, offset, direction="right"
):
    cos_theta = np.cos(np.radians(angle))
    sin_theta = np.sin(np.radians(angle))

    def rotate_coordinates(x, y):
        x_rot = round(x * cos_theta - y * sin_theta, 2) + offset[0]
        y_rot = round(x * sin_theta + y * cos_theta, 2) + offset[1]
        return (x_rot, y_rot)

    def calculate_coordinates(i):
        row = i // points_per_row
        col = i % points_per_row
        if direction == "right":
            x = col if row % 2 == 0 else points_per_row - 1 - col
        elif direction == "left":
            x = points_per_row - 1 - col if row % 2 == 0 else col
        y = row * v_diff
        return x * h_diff, y

    coordinates = [
        rotate_coordinates(*calculate_coordinates(i)) for i in range(len(df))
    ]
    df["x"], df["y"] = zip(*coordinates)
    return df



def generate_data_for_time_range(touch_point_df_dict, target_time, duration_time, duration_unit=1):

    # time_range 생성
    time_range = pd.date_range(start=target_time, 
                            end=target_time + pd.Timedelta(minutes=duration_time), 
                            freq=f'{duration_unit}T')

    time_range_data = []
    for time in time_range:
        dfs = []
        for key, temp in touch_point_df_dict.items():
            temp2 = temp[
                (temp["QueueStart"] <= time)
                & (temp["QueueEnd"] >= time)
            ].copy()
            total = temp2.groupby(['Detail','QueueStart','QueueEnd','color_col'])["Pax"].sum().reset_index()
            total.insert(0, "Location", key)
            dfs+=[total]
        new_df = pd.concat(dfs, ignore_index=True)
        new_df.insert(0, "TargetTime", time)
        time_range_data+=[new_df]
    final_df = pd.concat(time_range_data, ignore_index=True)
    return final_df



def add_dummy_df2(df, color_col, time_col, frame_freq, x_col, y_col):
    start_time = df[time_col].min()
    end_time = df[time_col].max()
    time_range = pd.date_range(start=start_time, end=end_time, freq=f"{frame_freq}S")
    unique_colors = df[color_col].unique()
    dummy_expanded = pd.DataFrame(
        np.repeat(unique_colors, len(time_range)), columns=[color_col]
    )
    dummy_expanded[time_col] = np.tile(time_range, len(unique_colors))
    dummy_expanded[x_col]=None
    dummy_expanded[y_col]=None

    df = pd.concat([df,dummy_expanded])
    df=df.sort_values(by=[time_col,color_col])
    return df

def decide_pax_position(group, terminal):
    _, location, detail = group.name
    group = group.sort_values(by="QueueStart")
    # Pax 값만큼 줄확대시키기
    group["Pax"] = group["Pax"].apply(lambda x: [1] * x)
    group = group.explode("Pax").reset_index(drop=True)
    group.drop("Pax", axis=1, inplace=True)
    # display(group)
    angle = location_mapping_dict[terminal][(location, detail)]["angle"]
    offset = location_mapping_dict[terminal][(location, detail)]["offset"]
    points_per_row = 9
    group = find_location_zigzag(
        group,
        h_diff=1,
        v_diff=1,
        points_per_row=points_per_row,
        angle=angle,
        offset=offset,
    )
    return group

def make_queueline_scatter(df, terminal, duration_unit, image_path, location_mapping_dict):
    graph_df = (
        df.groupby(["TargetTime", "Location", "Detail"])
        .apply(decide_pax_position, terminal=terminal)
        .reset_index(drop=True)
    )
    # graph_df = add_dummy_df(graph_df, 'color_col')
    graph_df=add_dummy_df2(df=graph_df, 
                color_col='color_col', 
                time_col="TargetTime",
                frame_freq=duration_unit*60,
                x_col='x',
                y_col='y')

    neon_colors = ["#39FF14", "#FF073A", "#007aff", "#FE53BB", "#FFFF33", "#FF1493", "#BC13FE","#FFFFFF","#036F4B","#0FFEF9", "#FF7A40", "#D4B200","#9370DB","#000080","#8B0000"]
    category_order = graph_df['color_col'].value_counts().index.tolist()

    fig = px.scatter(
        graph_df,
        x="x",
        y="y",
        animation_frame="TargetTime",
        color='color_col',
        category_orders={'color_col': category_order},
        hover_data=["Location", "Detail"],  # 
        color_discrete_sequence=neon_colors  # 네온 계열의 쨍한 색상 시퀀스 설정
        )
    fig.update_traces(
        marker=dict(size=3.5),
    )
    fig.update_layout(
        width=1400,
        height=850,
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            title="",
            range=[0, 400],
            constrain="domain",
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            title="",
            range=[0, 140],
            constrain="domain",
            scaleratio=1,
        ),
        plot_bgcolor="rgba(0,0,0,0)",  # 그래프 배경색 투명
        # paper_bgcolor="rgba(0,0,0,0)",  # 종이 배경색 투명
    )
    encoded_image = base64.b64encode(
        open(image_path, "rb").read()
    )
    fig.add_layout_image(
        dict(
            source=f"data:image/png;base64,{encoded_image.decode()}",
            xref="x",
            yref="y",
            x=0,  # 이미지의 왼쪽 위치
            y=0,  # 이미지의 위쪽 위치
            sizex=400,  # 이미지의 너비
            sizey=150,  # 이미지의 높이
            sizing="contain",  # 이미지의 크기 조정 방식
            xanchor="left",  # 이미지의 x 앵커
            yanchor="bottom",  # 이미지의 y 앵커
            opacity=0.9,
            layer="below",
        )
    )
    fig.layout.pop("updatemenus")  # Play, Stop 버튼 없애기
    # fig.layout.updatemenus[0].buttons = []  # Play, Stop 버튼 없애기


    for (touch_point, service_point), value_dict in location_mapping_dict[selected_terminal].items():
        x_loc, y_loc = value_dict["offset"]

        fig.add_annotation(
            x=x_loc,
            y=125,
            text=touch_point,
            showarrow=False,
            font=dict(size=30, color="white"),
            opacity=0.75
        )
        fig.add_annotation(
            x=x_loc,
            y=y_loc-3,
            text=service_point,
            showarrow=False,
            font=dict(size=15, color="white"),
            bgcolor="rgba(0,0,0,0.5)",
            opacity=0.75
        )


    return fig


def show_queue_queueline_map(df_orig, touch_point_dict, selected_date, selected_terminal, image_path, hour=6, minute=0, time_range=180, duration_unit=1):
    with st.container(border=True):
        st.subheader(f":blue[Passenger Queue Overview]")
        c1,c2=st.columns(2)
        with c1:
            input_time = st.time_input('Select time', value=time(int(hour), int(minute)))
            target_time = datetime.combine(
                selected_date,input_time)
        with c2:
            color_col = st.selectbox(
                "Gruoping",
                ["flight_number","operating_carrier_name","operating_carrier_iata","International/Domestic"],
                key="queuline_map_fig_color_select",
            ) 

        touch_point_df_dict={}
        for key, [detail, queue_start, queue_end, temp] in touch_point_dict.items():
            df_touch_point=df_orig.groupby([detail, queue_start, queue_end, color_col]).size().reset_index(name="Pax")
            df_touch_point.columns=['Detail','QueueStart','QueueEnd','color_col','Pax']
            touch_point_df_dict[key]=df_touch_point



        new_df = generate_data_for_time_range(touch_point_df_dict, target_time, time_range, duration_unit) 
        if len(new_df)==0:
            st.write(f"✅ There are no passengers in queue between :blue[**{input_time.strftime('%H:%M')} and {(datetime.combine(datetime.today(), input_time) + timedelta(hours=time_range//60)).strftime('%H:%M')}**].")
            st.write("✅ Please select a different time period!")
        else:
            st.write(f"✅ Displaying queue only between :blue[**{input_time.strftime('%H:%M')} and {(datetime.combine(datetime.today(), input_time) + timedelta(hours=time_range//60)).strftime('%H:%M')}**].")
            fig = make_queueline_scatter(new_df, selected_terminal, duration_unit, image_path, location_mapping_dict) 
            st.plotly_chart(fig, use_container_width=True)




show_queue_queueline_map(
                        df_orig=df_orig,
                        touch_point_dict=touch_point_dict, 
                        selected_date=selected_date, 
                        selected_terminal=selected_terminal, 
                        image_path=image_path,
                        )
