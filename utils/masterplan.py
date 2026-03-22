from openpyxl import load_workbook
import pandas as pd
import numpy as np
import streamlit as st
import shutil
import plotly.express as px
import plotly.graph_objects as go
import openai
from SPARQLWrapper import SPARQLWrapper, JSON
from streamlit_folium import st_folium
import folium
from geopy.distance import geodesic

# Global random seed (create_normal_dist_coldragon)
RANDOM_SEED = 42


def select_target_day(df_orig, key="default"):
    c1, c2, c4, c5, c6 = st.columns([0.15, 0.25, 0.15, 0.15, 0.15])

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
            range(2018, 2040),
            default=[2018, 2019, 2024],
            key=f"select Reference year{key}",
        )

        df_orig = df_orig[df_orig["scheduled_gate_local"].notna()]
        df_orig = df_orig[df_orig["flight_io"].isin(flight_io_list)]

        df_orig["scheduled_gate_local"] = pd.to_datetime(df_orig["scheduled_gate_local"])
        df_orig["day_of_week"] = df_orig["scheduled_gate_local"].dt.day_name()
        df_orig["week_number"] = df_orig["scheduled_gate_local"].dt.isocalendar().week
        df_orig["year"] = df_orig["scheduled_gate_local"].dt.year
        df_orig = df_orig[df_orig["year"].isin(target_year_list)]

    with c4:
        method = st.selectbox(
            "**method**", ["Annual", "Winter season", "Summer season", "First Half", "Second Half"], key=f"method"
        )
        method_dict = {
            "Annual": {"top date": 30, "peak start": 15, "peak end": 30},
            "Winter season": {"top date": 13, "peak start": 7, "peak end": 13},
            "Summer season": {"top date": 18, "peak start": 9, "peak end": 18},
            "First Half": {"top date": 15, "peak start": 8, "peak end": 15},
            "Second Half": {"top date": 15, "peak start": 8, "peak end": 15},
        }

    with c5:
        top_date = st.number_input(
            "**Top date**", value=method_dict[method]["top date"], min_value=1, key=f"top_date{key}"
        )

    with c6:
        future_year = st.selectbox(
            "**Target year**", range(2000, 2050), index=26, key=f"target_year{key}"
        )

    def first_sunday(year):
        first_day = pd.Timestamp(f"{year}-01-01")
        first_sunday = first_day + pd.DateOffset(days=(6 - first_day.weekday()))
        return first_sunday

    first_sundays = {year: first_sunday(year) for year in df_orig["year"].unique()}

    df_orig["first_sunday"] = df_orig["year"].map(first_sundays)
    df_orig["days_from_first_sunday"] = (
        df_orig["scheduled_gate_local"] - df_orig["first_sunday"]
    ).dt.days
    df_grouped = df_orig.groupby(["days_from_first_sunday", "year"])["total_seat_count"].agg("sum").unstack()
    df_orig["dates"] = df_orig["scheduled_gate_local"].dt.date.astype(str)
    df_grouped["Average"] = df_grouped.mean(axis=1)
    df_grouped = df_grouped.stack().reset_index(name="count")

    def method_filter(df, method):
        if method == "Summer season":
            start_mask = df["days_from_first_sunday"] >= 84
            end_mask = df["days_from_first_sunday"] <= 293
            df = df[start_mask & end_mask]
        elif method == "Winter season":
            start_mask = df["days_from_first_sunday"] < 84
            end_mask = df["days_from_first_sunday"] > 293
            df = df[start_mask | end_mask]
        elif method == "First Half":
            start_mask = df["days_from_first_sunday"] >= 0
            end_mask = df["days_from_first_sunday"] <= 181
            df = df[start_mask & end_mask]
        elif method == "Second Half":
            start_mask = df["days_from_first_sunday"] >= 182
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
        if year != "Average":
            df_year[f"{year}_date"] = pd.to_datetime(first_sundays[year]) + pd.to_timedelta(
                df_year["days_from_first_sunday"], unit="d"
            )
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
                    hoverinfo="text",
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
    st.caption(f"""✅ :blue[**Design Day:**] Select date from the :blue[**{method_dict[method]["peak start"]}th to {method_dict[method]["peak end"]}th**] Peak Period""")
    st.plotly_chart(fig)

    def highlight_rows(row):
        if row.name in range(method_dict[method]["peak start"], method_dict[method]["peak end"] + 1):
            return ["background-color: rgba(160, 231, 160, 0.5)"] * len(row)
        return [""] * len(row)

    styled_df = seleced_target_day.style.apply(highlight_rows, axis=1)
    st.subheader("**Peak Period**")
    st.write(styled_df.to_html(escape=False), unsafe_allow_html=True)


def create_normal_dist_col(
    df,
    ref_col,
    new_col,
    mean,
    sigma,
    min_max_clip,
    unit="m",
    iteration=1,
    datetime=False,
):
    """Normally distributed random values ref_colIn addition to new_colcreate."""
    np.random.seed(RANDOM_SEED)
    random_arr = np.random.normal(mean, sigma, size=len(df))
    assert (
        min_max_clip[0] <= mean <= min_max_clip[1]
    ), "mean value clipping It's going out of bounds. >> min~max clipping within range meanPlease reset the value"
    for _ in range(iteration):
        out_of_range_indices = np.where(
            (random_arr < min_max_clip[0]) | (random_arr > min_max_clip[1])
        )
        random_arr[out_of_range_indices] = np.random.normal(
            mean, sigma, size=len(out_of_range_indices[0])
        )
    if datetime == False:
        df[new_col] = df[ref_col] + random_arr
    elif datetime == True:
        timedelta_arr = pd.to_timedelta(random_arr, unit=unit)
        timedelta_arr = timedelta_arr.round("S")
        df[new_col] = df[ref_col] + timedelta_arr
    return df


def make_count_df(df, start_date, end_date, time_col, group, buffer_day=True, freq_min=1):
    df_copied = df.copy()
    if buffer_day == True:
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
    df_copied[time_col] = df_copied[time_col].dt.floor(f"{freq_min}T")
    count_df = df_copied.groupby([time_col, group]).size().reset_index(name="index")
    count_df.columns = ["Time", group, "index"]
    count_df = pd.merge(time_range_df, count_df, on="Time", how="left")
    count_df["index"] = count_df["index"].fillna(0)
    count_df[group] = count_df[group].fillna("")
    ranking_df = count_df.groupby(group)["index"].sum().sort_values(ascending=False)
    ranking_order = ranking_df.index.tolist()
    return count_df, ranking_order


def show_bar(df, ranking_order, group, capa_df=None, max_y=None):
    fig = px.bar(
        df,
        x="Time",
        y="index",
        color=group,
        category_orders={group: ranking_order},
    )
    if max_y is not None:
        fig.update_layout(yaxis_range=[0, max_y])
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
            x=0.95,
            y=1,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    st.plotly_chart(fig)


class MasterplanInput:
    def __init__(
            self,
    ):
        self.data_source=None
        self.iata_code=None # str
        self.icao_code=None # str
        self.iata_code_list=None
        self.icao_code_list=None
        self.df_airport=pd.read_parquet("data/raw/airport/cirium_airport_ref.parquet")
        self.country_mapper={
                            "Russian Federation":"Russia",
                            "russia":"Russia",
                            "Republic of Korea":"South Korea",
                            "Korea":"South Korea",
                            "korea":"South Korea",
                            "Korea, Rep.":"South Korea",
                            "Korea, Dem. People's Rep.":"North Korea",
                            "Kyrgyzstan":"Kyrgyz Republic",
                            "Viet Nam":"Vietnam",
                            }
        self.country_code=None
        self.variable_mapper={"Domestic Passengers":"domestic_passenger",
                            "International Passengers":"international_passenger",
                            "Total Passengers":"total_pax",
                            "Total Cargo":"total_cargo",
                            "Total Aircraft Movements":"total_flight"}
        self.country=None
        self.df_orig=None # dataframe
        self.df=None # dataframe
        self.return_df=None # dataframe
        self.return_df_pax=None # dataframe
        self.return_df_cargo=None # dataframe
        self.airport_open_data=self.import_aiport_open_data()
        self.aci_watf=self.import_aci_watf()
        self.aci_airport=self.import_aci_airport()
        self.df_cities=self.import_df_cities()
        self.econ_df=self.import_econ_df()
        self.fleet_df=self.import_fleet_df()
        self.open_year=None,
        self.start_year=2018 # int
        self.end_year=2024 # int
        self.predict_end_year=2060 # int
        self.ref_year_table=None # dataframe
        self.filter_dict={}
        self.multiple_default_airport = ["ALA", "CIT"]
        self.single_default_airport="TGD"

    def import_aiport_open_data(self):
        return pd.read_csv('data/raw/airport/airports(open_data).csv')

    def import_aci_airport(self):
        return pd.read_csv('data/raw/airport/aci_airport.csv', encoding='cp949')

    def import_df_cities(self):
        return pd.read_csv('data/raw/forecast/worldcities.csv', encoding="cp949")

    def import_aci_watf(self, last_pred_year=2052):
        aci_watf = pd.read_excel("data/raw/forecast/watf.xlsx", sheet_name="watf_2023", header=1)
        year_list = [year for year in range(2015,last_pred_year+1)]
        aci_watf = aci_watf[["Metric","Region","Entity"]+year_list]
        aci_watf["Entity"] = aci_watf["Entity"].replace(self.country_mapper)
        aci_watf["Metric"] = aci_watf["Metric"].replace(self.variable_mapper)
        return aci_watf

    def import_fleet_df(self):
        fleet=pd.read_csv("data/raw/aircraft/capa_fleet.csv")
        fleet=fleet[["Operator Name","Operator IATA","Operator Country","Aircraft Manufacturer","Aircraft Production Group","Aircraft Class","Aircraft Type Series","Aircraft Type","Aircraft IATA Code","Aircraft ICAO Code","Aircraft Serial Number","Activity Name","Status","Operating Status","Aircraft Role","Aircraft Delivery Date","In Service Date"]]
        fleet = fleet[fleet["Status"].isin(["In Service","On Order","On Option"])]
        fleet = fleet[fleet["Aircraft Role"].isin(["Passenger","Cargo"])]
        fleet["Aircraft Delivery year"]=pd.to_datetime(fleet["Aircraft Delivery Date"]).dt.year
        return fleet

    def import_econ_df(self, imf_normal_last_pred_year=2029, imf_population_last_pred_year=2050):
        col_df = pd.DataFrame(columns=["LOCATION","Country","Variable","Scenario"] + [str(year) for year in range(1980,2101)])
        oecd_df = pd.read_csv("data/raw/forecast/oecd_data.csv")
        meta_data = pd.read_excel("data/raw/forecast/meta_data.xlsx", sheet_name="oecd_value_info")
        oecd_df = oecd_df[["LOCATION","Country","Variable","Scenario","TIME_PERIOD","OBS_VALUE"]]
        oecd_primary_index=meta_data["variable_long_name"][:6].tolist()
        oecd_df = oecd_df[oecd_df["Variable"].isin(oecd_primary_index)]
        oecd_variable_mapper = dict(zip(meta_data['variable_long_name'], meta_data['variable_short_name']))
        oecd_df["Variable"] = oecd_df["Variable"].map(oecd_variable_mapper)
        oecd_df["Country"] = oecd_df["Country"].replace(self.country_mapper)
        oecd_df.loc[oecd_df["Variable"] == "GDP local currency", "OBS_VALUE"] /= 1000000000

        oecd_df = oecd_df.pivot(
            index=["LOCATION","Country","Variable","Scenario"],
            columns='TIME_PERIOD',
            values='OBS_VALUE'
        ).reset_index()
        oecd_df=oecd_df[oecd_df["Scenario"]=="Baseline"]
        oecd_df["Scenario"]="OECD"
        oecd_df.columns = oecd_df.columns.astype(str)
        oecd_df=pd.concat([col_df, oecd_df])

        imf_df = pd.read_csv("data/raw/forecast/imf_data.csv")
        imf_df.columns = imf_df.columns.astype(str)
        meta_data = pd.read_excel("data/raw/forecast/meta_data.xlsx", sheet_name="imf_value_info")
        imf_variable_mapper = dict(zip(meta_data['IMF_WEO_Code'], meta_data['variable_short_name']))
        imf_primary_index=meta_data["IMF_WEO_Code"][:6].tolist()
        imf_df=imf_df[imf_df["WEO Subject Code"].isin(imf_primary_index)]
        imf_df["WEO Subject Code"] = imf_df["WEO Subject Code"].map(imf_variable_mapper)
        imf_df["Scenario"]="IMF"
        imf_df = imf_df.rename({"ISO":"LOCATION",
                        "WEO Subject Code":'Variable',
                        }, axis=1)
        imf_df = imf_df[imf_df["Variable"]!="Total Population"]

        imf_df["Country"] = imf_df["Country"].replace(self.country_mapper)
        imf_df = imf_df[["LOCATION","Variable","Country","Scenario"]+[str(year) for year in range(1980,imf_normal_last_pred_year+1)]]
        imf_df = pd.concat([col_df, imf_df])

        imf_pop = pd.read_csv("data/raw/forecast/imf_population.csv")
        imf_pop = imf_pop.rename({"Country Name":"Country",
                                "Country Code":"LOCATION",
                                "Series Name":"Variable",
                                "Series Code":"Scenario"
                                }, axis=1)
        imf_pop["Scenario"]="IMF"
        imf_pop.columns = ["Country","LOCATION","Variable","Scenario"] + [year_col[:4] for year_col in imf_pop.columns[4:]]
        imf_pop = imf_pop[imf_pop["Variable"].isin(["Population ages 15-64, total","Population, total"])]
        imf_pop["Variable"] = imf_pop["Variable"].map({"Population ages 15-64, total":"Working-age Population", "Population, total":"Total Population"})
        imf_pop["Country"] = imf_pop["Country"].replace(self.country_mapper)
        for year in range(imf_population_last_pred_year+1,imf_population_last_pred_year+11):
            imf_pop[str(year)]=(imf_pop[str(year-1)]/imf_pop[str(year-2)]) * imf_pop[str(year-1)]
        imf_pop = imf_pop[["LOCATION","Variable","Country","Scenario"]+[str(year) for year in range(1980,imf_population_last_pred_year+11)]]
        econ_df=pd.concat([oecd_df,imf_df,imf_pop])
        return econ_df


    @st.fragment
    def select_airport_block(self):
        mode = st.toggle("Existing Airport", value=True, label_visibility='visible')
        self.real_virtual = "real_airport" if mode else "virtual_airport"
        
        if self.real_virtual=="real_airport":
            self.iata_code , self.icao_code, self.data_source, self.open_year, self.df_orig = self.select_airport(key=self.real_virtual)
            if st.button(f"✅ Apply Airport Changes", type="primary", key=f"changes"):
                st.rerun();st.write("");st.write("")
        elif self.real_virtual =="virtual_airport":
            self.virtual_airport_block()





    def select_airport(self, 
                        key, 
                        multiple_airport=False):
        if multiple_airport==True:
            select_airport_tab, load_factor_tab, filter_tab = st.tabs(["**Select Airport**", "**Load-Factor Estimation**", "**Filter**"])
        elif multiple_airport==False:
            select_airport_tab, load_factor_tab = st.tabs(["**Select Airport**", "**Load-Factor Estimation**"])

        with select_airport_tab:
            source, code = st.columns(2)
            with source :
                data_source = st.selectbox(
                    "**Flight Data Source**",
                    ["Cirium_Status","Cirium_Schedule","OAG"], #
                    index=2,
                    key=f"Data Source{key}",    
                )
            with code :
                if data_source=="Cirium_Status":
                    import os
                    folder_path = r"C:\Users\qkrru\Desktop\desktop\creative_code\DMK_repository\cirium_database"
                    files = os.listdir(folder_path)
                    airport_list = [filename[:3] for filename in files if '_gd_table.' in filename]

                elif data_source=="Cirium_Schedule":
                    import os
                    folder_path = r"C:\Users\qkrru\Desktop\desktop\creative_code\DMK_repository\cirium_database"
                    files = os.listdir(folder_path)
                    airport_list = [filename[:3] for filename in files if '_schedule_ready_.parquet' in filename]

                elif data_source=="OAG":
                    import os
                    folder_path = "oag"
                    files = os.listdir(folder_path)
                    airport_list = [filename[:3] for filename in files if '_OAG_SC_Processed.' in filename]

                airport_label_map = {
                    row["airport_id"]: f'{row["airport_id"]} - {row["name"]} - {row["country_name"]}'
                    for _, row in self.df_airport.iterrows()
                }

                if multiple_airport:
                    source_number = key.split("*^*")[1]
                    try:
                        airport_index = airport_list.index(self.multiple_default_airport[int(source_number)])
                        st.write("airport_index")
                        st.write(airport_index)
                        st.write(key)

                    except (ValueError, IndexError):
                        airport_index = 0
                else:
                    airport_index = airport_list.index(self.single_default_airport) if self.single_default_airport in airport_list else 0
                iata_code = st.selectbox(
                    "**Airport**",
                    options=airport_list,
                    format_func=lambda x: airport_label_map.get(x, f"{x} - Unknown"),
                    index=airport_index,
                    key=f"select_airport__{key}"
                )
                icao_code=self.df_airport[self.df_airport["airport_id"]==iata_code]["airport_icao"].values[0]

                selected_airport=self.df_airport[self.df_airport["airport_id"]==iata_code]
                self.airport_lat = selected_airport.iloc[0]['lat']
                self.airport_long = selected_airport.iloc[0]['lon']
                self.country=selected_airport["country_name"].replace(self.country_mapper).values[0]
                self.country_code=selected_airport["country_code"].replace(self.country_mapper).values[0]
            

            st.info(
                f"""
                **Country : {self.country} ({self.country_code})** \n
                **Name : {selected_airport["name"].values[0]}({iata_code})**
                """
            )



        with load_factor_tab:
            for attempt in range(10):
                try:
                    open_source_pax_df = self.get_airport_passenger_data(iata_code)
                    break
                except Exception as e:
                    if attempt==9:
                        st.write(f"{attempt}tea failure")


            # Load airport passenger data from wiki_data
            open_source_pax_df = self.classify_passenger_data(open_source_pax_df)
            open_source_pax_df=open_source_pax_df.rename({"Year":"year","Month":"month","Passengers":"total_pax"},axis=1)
            pax_count_by_year = open_source_pax_df[open_source_pax_df["Type"]=="year"][["year","total_pax"]]
            pax_count_by_year = pax_count_by_year.set_index('year').reindex(
                range(pax_count_by_year['year'].min(), pax_count_by_year['year'].max() + 1)
            )
            pax_count_by_year['total_pax'] = pax_count_by_year['total_pax'].interpolate().astype(int)


            # Load schedule data
            if data_source=="Cirium_Status":
                df_orig=pd.read_parquet("../../cirium_database/" + f'{iata_code}_gd_table.parquet')
                df_orig=df_orig[df_orig["scheduled_gate_local"].dt.year>=2018]
            elif data_source=="Cirium_Schedule":
                df_orig=pd.read_parquet("../../cirium_database/" + f'{iata_code}_schedule_ready_.parquet')
                # df_orig["operating_carrier_name"] = df_orig["operating_carrier_iata"].copy()
                df_orig["movement"]=1
                df_orig["aircraft_class"]="Unknown"
                df_orig["aircraft_code_iata"]="Unknown"
                df_orig["aircraft_name"]="Unknown"
                df_orig["aircraft_manufacturer"]="Unknown"
                df_orig["aircraft_max_takeoff_weight"]="Unknown"
                df_orig["aircraft_max_takeoff_weight_lb"]="Unknown"
                df_orig["aircraft_max_takeoff_weight_kg"]="Unknown"
                df_orig["aircraft_max_takeoff_weight_ton"]="Unknown"
                df_orig["aircraft_max_takeoff_weight_ton"]="Unknown"

            elif data_source=="OAG":
                df_orig = pd.read_parquet("oag/" + f'{iata_code}_OAG_SC_Processed.parquet')
                df_orig=df_orig[df_orig["scheduled_gate_local"].dt.year<2026]
            df_orig["flight_number"]=df_orig["operating_carrier_iata"] + df_orig["flight_number"].astype(str).str.zfill(4)
            df_orig["year"] = df_orig["scheduled_gate_local"].dt.year
            df_orig['total_seat_count'] = df_orig['total_seat_count'].fillna(0)



            st.subheader("Load-Factor Estimation")

            seat_count_by_year = df_orig.groupby(["year"])["total_seat_count"].agg("sum")
            load_factor_table = pd.concat([pax_count_by_year, seat_count_by_year], axis=1)#
            load_factor_table["load_factor(%)"]=load_factor_table["total_pax"]/load_factor_table["total_seat_count"]
            load_factor_table = load_factor_table.reset_index().rename(columns={'index': 'year'})
            load_factor_table["transfer(%)"]=0.05
            load_factor_table = load_factor_table.sort_values(by="year", ascending=False)
            total_pax_series = load_factor_table["total_pax"]


            c1, c2 = st.columns([0.65,0.35])
            with c1:
                st.caption(f"load_factor(%) : total_pax / total_seat_count")
                st.caption("✅ indicates that the load_factor(%) value needs to be entered.")

                load_factor_table["year"] = load_factor_table.apply(
                    lambda row: f"{int(row['year'])}✅" if pd.isna(row["load_factor(%)"]) and pd.notna(row["total_seat_count"]) else int(row["year"]),
                    axis=1
                )

                load_factor_table = st.data_editor(
                    load_factor_table[["year","total_seat_count","load_factor(%)","transfer(%)"]],
                    use_container_width=True,
                    hide_index=True,
                    key=f"load_factor_table {key}",
                    column_config={
                        "total_seat_count": st.column_config.Column(disabled=True, help="✅ indicates that the load_factor(%) value needs to be entered."),
                        "year": st.column_config.Column(disabled=True, help="✅ indicates that the load_factor(%) value needs to be entered."),
                    }
                )
                load_factor_table["year"]=(load_factor_table["year"].astype(str).str.replace("✅", "").str.replace(".0", "")).astype(int)

                total_pax_series=load_factor_table["load_factor(%)"]*load_factor_table["total_seat_count"]//1

                load_factor_table['total_pax'] = np.where(
                    total_pax_series.isna(),
                    (load_factor_table['total_seat_count'] * load_factor_table['load_factor(%)'])//1,
                    total_pax_series
                )
                load_factor_table['transfer_pax']=load_factor_table['total_pax']*load_factor_table["transfer(%)"]//1

                date_num_2024 = len(df_orig[df_orig["year"]==2024]["scheduled_gate_local"].dt.date.unique())
                st.caption(f"2024 date num : {date_num_2024}")


                pax_count_by_year = load_factor_table.set_index("year")[["total_pax"]]
                # st.write(load_factor_table)
                # st.write(pax_count_by_year)

            with c2:
                st.caption("total_pax : Annual passenger data(from various sources)")
                st.caption(f"total_pax(E) = total_seat_count * load_factor(%)")
                st.dataframe(load_factor_table[["year","total_pax","transfer_pax"]], hide_index=True,use_container_width=True)



            c1,c2=st.columns([0.65,0.35])
            with c1:
                st.subheader("Passenger & Seat")
                fig = px.line(load_factor_table, x='year', y=['total_pax', 'total_seat_count'], markers=True)
                fig.update_layout(
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=0.95,
                        xanchor="right",
                        x=1
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.subheader("Load-Factor & Transfer(%)")
                fig = px.line(load_factor_table[load_factor_table["total_seat_count"].notna()], x='year', y=['load_factor(%)', 'transfer(%)'], markers=True)
                fig.update_layout(
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=0.95,
                        xanchor="right",
                        x=1
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
            df_orig = df_orig.merge(load_factor_table[['year', 'load_factor(%)',"transfer(%)"]], on='year', how='left')
            df_orig['total_pax'] = df_orig['total_seat_count'] * df_orig['load_factor(%)']
            df_orig['tr_pax']=df_orig['total_pax']*df_orig["transfer(%)"]
            df_orig['od_pax']=df_orig['total_pax']-df_orig['tr_pax']

            ####################################################################################3
            df_orig=df_orig.rename({"dep/arr_site":'dep/arr_airport', "airport":"origin_airport"}, axis=1)
            if "region_name" not in df_orig.columns:
                df_orig = pd.merge(
                    df_orig,
                    self.df_airport[["airport_id","region_name"]],
                    left_on="dep/arr_airport",
                    right_on="airport_id",
                    how="left"
                )
            else: 
                df_orig = pd.merge(
                    df_orig.drop("region_name", axis=1),
                    self.df_airport[["airport_id","region_name"]],
                    left_on="dep/arr_airport",
                    right_on="airport_id",
                    how="left"
                )
            ####################################################################################3


            aircraft_map={
            "Utility Jets":"Widebody Jets (E)",
            "Utility Jets (E)":"Widebody Jets (E)",
            "Regional Turboprops":"Regional Jets",
            "Business Turboprops":"Business Jets",
            "Regional Turboprops (E)":"Regional Jets",
            "Business Turboprops (E)":"Business Jets",
            }

            df_orig['aircraft_class'] = df_orig['aircraft_class'].replace(aircraft_map)
            business_conditions = [
                "Business - Air Taxi/Air Charter (E)",
                "Business - Air Taxi/Air Charter"
            ]
            df_orig.loc[df_orig['primary_usage'].isin(business_conditions), 'aircraft_class'] = 'Business Jets (E)'

            for col in ["primary_usage","terminal","aircraft_class", "aircraft_code_iata"]:
                df_orig[col]=df_orig[col].str.replace(" (E)","")
            df_orig = pd.merge(df_orig,
                                    self.df_airport[['airport_id', 'lat', 'lon', 'name', 'city',"category"]],
                                    left_on="dep/arr_airport", 
                                    right_on='airport_id', 
                                    how='left')
        if multiple_airport==True:
            with filter_tab:
                before_pax = df_orig["total_pax"].sum()
                df_orig, filter_dict = self.select_filter(df_orig, key=f"{key}_filter")
                after_pax = df_orig["total_pax"].sum()
                filtered_ratio = after_pax/before_pax
                st.caption(f"* ratio : {after_pax/before_pax}")
                pax_count_by_year["total_pax"] = (pax_count_by_year["total_pax"]*filtered_ratio) //1

            with select_airport_tab:
                with st.expander("Data Sample"):
                    st.dataframe(df_orig.head(10), hide_index=True)
            return iata_code, icao_code, data_source, pax_count_by_year, df_orig
        with select_airport_tab:
            with st.expander("Data Sample"):
                st.dataframe(df_orig.head(10), hide_index=True)
                st.write(df_orig["scheduled_gate_local"].max(), df_orig["scheduled_gate_local"].min())
        return iata_code, icao_code, data_source, pax_count_by_year, df_orig



    def show_cities_location(self, map_style, color, radius_distance):
        def haversine_vectorized(lat1, lon1, lat2_array, lon2_array):
            R = 6371  # Earth's radius in kilometers
            # Convert all latitudes/longitudes to radians
            lat1_rad = np.radians(lat1)
            lon1_rad = np.radians(lon1)
            lat2_rad = np.radians(lat2_array)
            lon2_rad = np.radians(lon2_array)
            # Compute differences
            dlat = lat2_rad - lat1_rad
            dlon = lon2_rad - lon1_rad
            # Haversine formula
            a = np.sin(dlat/2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2.0)**2
            c = 2 * np.arcsin(np.sqrt(a))
            distances = R * c
            return distances

        # Distance calculation and filtering
        distances = haversine_vectorized(self.airport_lat, self.airport_long, 
                                    self.df_cities['lat'].values, 
                                    self.df_cities['lng'].values)

        df_cities_filtered = self.df_cities[(distances <= radius_distance)&(self.df_cities["population"]>0)]
        df_cities_filtered["population"]=df_cities_filtered["population"].astype(int)
        # map creation

        def create_circle(lat, lon, radius_km, points=100):
            local_azimuths = np.linspace(0, 360, points)
            circle_coords = [
                geodesic(kilometers=radius_km).destination((lat, lon), az)
                for az in local_azimuths
            ]
            lats, lons = zip(*[(coord.latitude, coord.longitude) for coord in circle_coords])
            return lats, lons


        circle_lat, circle_lon = create_circle(self.airport_lat, self.airport_long, radius_distance)
        fig = px.scatter_mapbox(df_cities_filtered,
                                lat='lat',
                                lon='lng',
                                size="population",
                                color=color,
                                hover_name='city',
                                zoom=3)

        # add circle
        fig.add_trace(go.Scattermapbox(
            lat=circle_lat,
            lon=circle_lon,
            mode='lines',
            line=dict(width=0.5, color='red'),
            name=f"{radius_distance}km radius"
        ))

        # Show center point
        fig.add_trace(go.Scattermapbox(
            lat=[self.airport_lat],
            lon=[self.airport_long],
            mode='markers',
            marker=dict(size=10, color='red'),
            name="Center"
        ))

        # fig.update_layout(mapbox_style="open-street-map",
        #                 mapbox_center={"lat": self.airport_lat, "lon": self.airport_long},
        #                 margin={"l":0,"r":0,"t":0,"b":0})

        fig.update_layout(
            mapbox_style=map_style,
            mapbox=dict(
                center=dict(lat=self.airport_lat, lon=self.airport_long),
                zoom=4
            ),
            height=650,
            margin=dict(t=0, l=0, r=0, b=0),  # Set margin in all directions to 0
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                borderwidth=2
            ),
            )

        # Display map in streamlet
        graph, table=st.columns([0.5,0.5])
        with graph:
            st.plotly_chart(fig, use_container_width=True)
            st.write(f"● Bubble size : cities population")
            catchment_area_living = df_cities_filtered["population"].sum()
        with table:
            c1, c2 = st.columns(2)
            with c1:
                group_col = st.multiselect('**Category**', ["country","city"], default=["country"], key="groupby_category_select_pop")
            with c2:
                count_col = st.selectbox('**Population**', ["population"], index=0, key="count_select_pop")
            grouped_df=df_cities_filtered.groupby(group_col)[count_col].agg("sum")
            grouped_df.loc["※ Total ※"]=grouped_df.sum(axis=0)
            grouped_df=grouped_df.sort_values(ascending=False)
            st.write(f"Population within {radius_distance}km : {'{:,}'.format(catchment_area_living)} people")
            st.table(grouped_df)


    def show_earth_chart(self, df):
        st.subheader(f"**Location**")
        if self.iata_code!=None:
            dep_arr_list=df["dep/arr_airport"].unique()    
            selected_airport = self.df_airport[self.df_airport["airport_id"]==self.iata_code]
            dep_arr_airport = self.df_airport[self.df_airport["airport_id"].isin(dep_arr_list)]

            lat=selected_airport["lat"].values[0]
            lon=selected_airport["lon"].values[0]
        elif self.iata_code==None:
            dep_arr_list=df["dep/arr_airport"].unique()    
            selected_airport = self.df_airport[self.df_airport["airport_id"].isin(self.iata_code_list)]
            dep_arr_airport = self.df_airport[self.df_airport["airport_id"].isin(dep_arr_list)]
        fig = go.Figure()

        for lat, lon in zip(selected_airport["lat"], selected_airport["lon"]):
            fig.add_trace(go.Scattergeo(
                lon = [lon],
                lat = [lat],
                mode = 'markers',
                text = ['Center'],
                marker = dict(
                size = 7,
                color = '#FF0066'  # or 'rgb(255, 105, 180)'
                )
                ))

        # dst_airport (Add blue dots)
        fig.add_trace(go.Scattergeo(
            lon=dep_arr_airport["lon"],
            lat=dep_arr_airport["lat"],
            mode='markers',
            marker=dict(
                size=3,
                color='#00FF00'
            )
        ))

        # geo Setting up a globe using objects
        fig.update_geos(
            projection_type='orthographic',
            showland=True,
            showcountries=True,
            showocean=True,
            showcoastlines=True,
            bgcolor='rgba(0,0,0,0)',
            landcolor='rgb(42, 35, 35)',
            oceancolor = "#007aff",

            center=dict(
                lon=lon,
                lat=lat
            ),
            # Globe rotation settings
            projection_rotation=dict(
                lon=lon,
                lat=lat,
                roll=0
            )
        )
        fig.update_layout(
        height=1000,
        width=1000,
        paper_bgcolor='rgba(0,0,0,0)',  # Make the paper background transparent
        plot_bgcolor='rgba(0,0,0,0)' ,   # Make plot background transparent
        margin=dict(l=0, r=0, t=0, b=0),  # Set all margins to 0
        showlegend=False  # Remove legend
        )
        st.caption(f"{len(dep_arr_airport)} routes")
        st.plotly_chart(fig, use_container_width=True)




    def show_bubble_map(self, df, map_style):
        st.subheader(f"Routes")
        c1, _=st.columns([0.2,0.8])
        with c1:
            color = st.selectbox(
            "**Colored by**",
            [
                "airport_id",
                "country_code",
                "region_name",
                "city",
            ],
            index=2,
            key=f"bubble_color",
            )



        # Calculate counts by year
        site_counts = df.groupby(['year', 'dep/arr_airport']).size().reset_index(name='count')
        site_counts.columns = ['year', 'airport_id', 'count']
        # Combining counts and airport data
        df_with_counts = pd.merge(
            site_counts,
            self.df_airport[['airport_id', 'lat', 'lon', 'country_code',"region_name", 'city']],
            on='airport_id',
            how='left'
        )

        # Add column for selected airport
        if self.iata_code!=None:
            df_with_counts['is_selected'] = df_with_counts['airport_id'].isin([self.iata_code])
            max_val=df_with_counts["count"].max()*0.5
            years = df_with_counts['year'].unique()
            df_selected_extra = pd.concat([
                self.df_airport[self.df_airport['airport_id']==self.iata_code]
                .assign(year=year, count=max_val*0.5, is_selected=True)
                for year in years
            ], ignore_index=True)
            df_selected_extra[color]="selected airport"
        elif self.iata_code==None:
            df_with_counts['is_selected'] = df_with_counts['airport_id'].isin(self.iata_code_list)
            max_val=df_with_counts["count"].max()*0.5
            years = df_with_counts['year'].unique()
            df_selected_extra = pd.concat([
                self.df_airport[self.df_airport['airport_id'].isin(self.iata_code_list)]
                .assign(year=year, count=max_val*0.5, is_selected=True)
                for year in years
            ], ignore_index=True)
            df_selected_extra[color]="selected airport"
        # Display selected airports duplicated for all additional years



        # Full data integration
        # data integration
        df_final = pd.concat([df_with_counts, df_selected_extra], ignore_index=True)
        df_final['count'] = df_final['count'].clip(lower=max_val * 0.1)

        # Create and then merge all year-color combinations
        from itertools import product
        all_combinations = pd.DataFrame(product(df_final['year'].unique(), df_final[color].unique()),
                                        columns=['year', color])

        # Merge to find missing combinations dummy fill
        df_final = pd.merge(all_combinations, df_final, on=['year', color], how='left')
        df_final.fillna({
            'airport_id': 'selected airport',
            'lat': self.airport_lat,
            'lon': self.airport_long,
            'name': '',
            'city': '',
            'count': max_val * 0.2
        }, inplace=True)

        # draw a map
        fig = px.scatter_mapbox(df_final,
                                lat='lat',
                                lon='lon',
                                size='count',
                                color=color,
                                color_discrete_map={"selected airport": '#F9075D'},  # #F9075D
                                hover_name='airport_id',
                                hover_data=['name', 'city', 'count'],
                                animation_frame='year',
                                zoom=1.5)
        fig.update_layout(
            mapbox_style=map_style,
            mapbox_center=dict(lat=self.airport_lat, lon=self.airport_long),
            height=800,
            margin=dict(t=0, l=0, r=0, b=0),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, borderwidth=2),
            updatemenus=[],   # Remove play button
            )
        
        
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"※ Bubble size : Flight Movements")




    def heatmap(self, df):
        fig = go.Figure(data=go.Heatmap(
            z=df.T.values,                    # data value
            x=df.T.columns,  # xaxis label (hour)
            y=df.T.index,                     # yaxis label (date)
            colorscale=[
                        [0, 'black'],
                        [0.5, 'pink'],  # median is pink
                        [1, 'red']
                        ]
                        ))
        fig.update_layout(
            title='heatmap',
            height=600,
            )
        st.plotly_chart(fig)

    def issue_paper_block(self):
        st.subheader("**📄 Issue Paper Analysis**")
        
        if self.df_orig is None or len(self.df_orig) == 0:
            st.warning("⚠️ There is no schedule data. Please load the data first.")
            return
        
        # Data preprocessing
        df = self.df_orig.copy()
        if 'scheduled_gate_local' in df.columns:
            df['scheduled_gate_local'] = pd.to_datetime(df['scheduled_gate_local'])
            df['scheduled_gate_year'] = df['scheduled_gate_local'].dt.year
            df['scheduled_gate_month'] = df['scheduled_gate_local'].dt.month
            df['scheduled_gate_date'] = df['scheduled_gate_local'].dt.date
            df['scheduled_gate_hour'] = df['scheduled_gate_local'].dt.hour
            df['scheduled_gate_dayofweek'] = df['scheduled_gate_local'].dt.dayofweek
            df['scheduled_gate_dayname'] = df['scheduled_gate_local'].dt.day_name()
        
        # Tab Configuration
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📊 1. Executive Summary",
            "🚦 2. Operations Congestion",
            "📈 3. Airline & Route Trends",
            "🌐 4. Connectivity & Hub",
            "💰 5. Revenue Perspective",
            "🛡️ 6. Safety & Environment",
            "🎯 7. Strategic Recommendations"
        ])
        
        with tab1:
            self._executive_summary(df)
        
        with tab2:
            self._operations_congestion_analysis(df)
        
        with tab3:
            self._airline_route_trends(df)
        
        with tab4:
            self._connectivity_hub_analysis(df)
        
        with tab5:
            self._revenue_perspective(df)
        
        with tab6:
            self._safety_environment(df)
        
        with tab7:
            self._strategic_recommendations(df)
    
    def _executive_summary(self, df):
        st.markdown("### 📄 Executive Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔍 key discovery (Key Findings)")
            
            # 1. Congestion analysis by time zone
            if 'scheduled_gate_hour' in df.columns:
                hourly_movements = df.groupby('scheduled_gate_hour').size()
                peak_hour = hourly_movements.idxmax()
                peak_movements = hourly_movements.max()
                avg_movements = hourly_movements.mean()
                
                st.info(f"""
                **1. Identify peak times**
                - peak rush hour: **{peak_hour:02d}:00**
                - Number of peak hour operations: **{peak_movements:.0f}side**
                - Average number of flights per hour: **{avg_movements:.1f}side**
                - peak/average ratio: **{peak_movements/avg_movements:.2f}x**
                """)
            
            # 2. gate saturation
            if 'terminal' in df.columns and 'scheduled_gate_local' in df.columns:
                # Estimation of gate occupancy by time period (15minute by minute)
                df['time_slot'] = df['scheduled_gate_local'].dt.floor('15min')
                gate_occupancy = df.groupby(['time_slot', 'terminal']).size().reset_index(name='occupancy')
                max_occupancy = gate_occupancy['occupancy'].max()
                avg_occupancy = gate_occupancy['occupancy'].mean()
                
                st.info(f"""
                **2. gate saturation**
                - Maximum concurrently occupied gates: **{max_occupancy:.0f}dog**
                - Average concurrently occupied gates: **{avg_occupancy:.1f}dog**
                - saturation risk: **{'height' if max_occupancy > avg_occupancy * 2 else 'commonly'}**
                """)
            
            # 3. Airline market share changes
            if 'operating_carrier_name' in df.columns and 'scheduled_gate_year' in df.columns:
                carrier_trend = df.groupby(['scheduled_gate_year', 'operating_carrier_name']).size().reset_index(name='movements')
                if len(carrier_trend) > 0:
                    latest_year = carrier_trend['scheduled_gate_year'].max()
                    latest_data = carrier_trend[carrier_trend['scheduled_gate_year'] == latest_year]
                    top_carrier = latest_data.loc[latest_data['movements'].idxmax(), 'operating_carrier_name']
                    top_share = latest_data['movements'].max() / latest_data['movements'].sum() * 100
                    
                    st.info(f"""
                    **3. Major airline share**
                    - most occupied airline: **{top_carrier}**
                    - market share: **{top_share:.1f}%**
                    """)
        
        with col2:
            st.markdown("#### ⚠️ danger/opportunity signal (Risk/Opportunity Signals)")
            
            # Red flag 1: Structural delay zone
            if 'scheduled_gate_hour' in df.columns:
                hourly_movements = df.groupby('scheduled_gate_hour').size()
                congestion_zones = hourly_movements[hourly_movements > hourly_movements.quantile(0.75)]
                
                if len(congestion_zones) > 0:
                    st.warning(f"""
                    **⚠️ Red flag 1: Structural delay zone**
                    - rush hour: **{', '.join([f'{h:02d}:00' for h in congestion_zones.index[:3]])}**
                    - Risk of delays due to persistent congestion
                    """)
            
            # Sign of Opportunity 1: Night Operations
            if 'scheduled_gate_hour' in df.columns:
                night_flights = df[(df['scheduled_gate_hour'] >= 22) | (df['scheduled_gate_hour'] < 6)]
                night_ratio = len(night_flights) / len(df) * 100 if len(df) > 0 else 0
                
                if night_ratio < 10:
                    st.success(f"""
                    **💡 Opportunity Signal 1: Potential for expanded nighttime operations**
                    - Current night flight ratio: **{night_ratio:.1f}%**
                    - nighttime Slot There is room for expansion due to low utilization rate.
                    """)
        
        # Summary statistics table
        st.markdown("#### 📊 Summary Statistics")
        summary_stats = []
        
        if 'scheduled_gate_year' in df.columns:
            years = sorted(df['scheduled_gate_year'].unique())
            for year in years:
                year_df = df[df['scheduled_gate_year'] == year]
                summary_stats.append({
                    'Year': year,
                    'Total Movements': len(year_df),
                    'Total Passengers': year_df['total_pax'].sum() if 'total_pax' in year_df.columns else 0,
                    'Avg Daily Movements': len(year_df) / len(year_df['scheduled_gate_date'].unique()) if 'scheduled_gate_date' in year_df.columns else 0
                })
        
        if summary_stats:
            summary_df = pd.DataFrame(summary_stats)
            st.dataframe(summary_df, use_container_width=True)
    
    def _operations_congestion_analysis(self, df):
        st.markdown("### 🚦 airport congestion(Operations) analyze")
        
        # 1. Congestion index by time of day
        st.markdown("#### 1) Congestion index by time of day (Time-of-Day Congestion Index)")
        
        if 'scheduled_gate_hour' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # arrive/Peak times by departure
                if 'flight_io' in df.columns:
                    arrival_df = df[df['flight_io'] == 'a'] if 'a' in df['flight_io'].values else df
                    departure_df = df[df['flight_io'] == 'd'] if 'd' in df['flight_io'].values else df
                    
                    arrival_hourly = arrival_df.groupby('scheduled_gate_hour').size() if len(arrival_df) > 0 else pd.Series()
                    departure_hourly = departure_df.groupby('scheduled_gate_hour').size() if len(departure_df) > 0 else pd.Series()
                    
                    if len(arrival_hourly) > 0 or len(departure_hourly) > 0:
                        fig = go.Figure()
                        if len(arrival_hourly) > 0:
                            fig.add_trace(go.Bar(x=arrival_hourly.index, y=arrival_hourly.values, name='Arrival', marker_color='#1f77b4'))
                        if len(departure_hourly) > 0:
                            fig.add_trace(go.Bar(x=departure_hourly.index, y=departure_hourly.values, name='Departure', marker_color='#ff7f0e'))
                        fig.update_layout(
                            title='Arrival by time slot/number of departures',
                            xaxis_title='Hour',
                            yaxis_title='Number of Flights',
                            barmode='group',
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Congestion by overall time zone
                hourly_movements = df.groupby('scheduled_gate_hour').size()
                avg_movements = hourly_movements.mean()
                congestion_index = (hourly_movements / avg_movements).round(2)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=congestion_index.index,
                    y=congestion_index.values,
                    marker=dict(
                        color=congestion_index.values,
                        colorscale='Reds',
                        showscale=True
                    )
                ))
                fig.add_hline(y=1.5, line_dash="dash", line_color="red", annotation_text="High Congestion Threshold")
                fig.update_layout(
                    title='Congestion index by time of day (Congestion Index)',
                    xaxis_title='Hour',
                    yaxis_title='Congestion Index (vs Average)',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Structural Delay Zone discrimination
                delay_zones = congestion_index[congestion_index > 1.5]
                if len(delay_zones) > 0:
                    st.warning(f"**⚠️ Structural Delay Zones:** {', '.join([f'{h:02d}:00' for h in delay_zones.index])}")
        
        # 2. Gate saturation analysis
        st.markdown("#### 2) gate(Gate) Saturation analysis")
        
        if 'terminal' in df.columns and 'scheduled_gate_local' in df.columns:
            # Gate occupancy by time slot
            df['time_slot'] = df['scheduled_gate_local'].dt.floor('15min')
            gate_occupancy = df.groupby(['time_slot', 'terminal']).size().reset_index(name='occupancy')
            
            # Wide/Narrow-body ratio
            if 'aircraft_class' in df.columns:
                wide_narrow_ratio = df.groupby(['scheduled_gate_year', 'aircraft_class']).size().unstack(fill_value=0)
                if len(wide_narrow_ratio) > 0:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Gate share heatmap by time zone
                        occupancy_pivot = gate_occupancy.pivot_table(
                            index=gate_occupancy['time_slot'].dt.hour,
                            columns='terminal',
                            values='occupancy',
                            aggfunc='mean'
                        ).fillna(0)
                        
                        fig = px.imshow(
                            occupancy_pivot,
                            labels=dict(x="Terminal", y="Hour", color="Gate Occupancy"),
                            title="Gate share by terminal by time zone",
                            aspect="auto"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Wide/Narrow-body rate change
                        wide_narrow_pct = wide_narrow_ratio.div(wide_narrow_ratio.sum(axis=1), axis=0) * 100
                        fig = px.line(
                            wide_narrow_pct.reset_index(),
                            x='scheduled_gate_year',
                            y=wide_narrow_pct.columns.tolist(),
                            title="Wide/Narrow-body rate change",
                            labels={'value': 'Percentage (%)', 'scheduled_gate_year': 'Year'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            # Saturation risk time zone
            max_occupancy_by_hour = gate_occupancy.groupby(gate_occupancy['time_slot'].dt.hour)['occupancy'].max()
            avg_occupancy = gate_occupancy['occupancy'].mean()
            saturation_risk = max_occupancy_by_hour[max_occupancy_by_hour > avg_occupancy * 1.5]
            
            if len(saturation_risk) > 0:
                st.warning(f"**⚠️ Gate saturation risk period:** {', '.join([f'{h:02d}:00' for h in saturation_risk.index[:5]])}")
        
        # 3. Risk of increased taxi time
        st.markdown("#### 3) taxi time(Taxi Time) Identification of increased risk")
        
        if 'actual_taxi_time' in df.columns:
            # Estimating the number of simultaneous ground movement flights (15minute by minute)
            df['taxi_time_slot'] = df['scheduled_gate_local'].dt.floor('15min')
            simultaneous_taxi = df.groupby('taxi_time_slot').size()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=simultaneous_taxi.index,
                    y=simultaneous_taxi.values,
                    mode='lines',
                    name='Simultaneous Ground Movements',
                    line=dict(color='orange', width=2)
                ))
                fig.update_layout(
                    title='Number of simultaneous ground flights per hour',
                    xaxis_title='Time',
                    yaxis_title='Number of Flights',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Taxi time distribution
                if df['actual_taxi_time'].notna().sum() > 0:
                    fig = px.histogram(
                        df[df['actual_taxi_time'].notna()],
                        x='actual_taxi_time',
                        nbins=30,
                        title="Taxi time distribution",
                        labels={'actual_taxi_time': 'Taxi Time (minutes)', 'count': 'Frequency'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    def _airline_route_trends(self, df):
        st.markdown("### 📈 airline·Growth by route/shrinking trend")
        
        # 1. New route/Interruption Analysis
        st.markdown("#### 1) New route/Interruption Analysis (New & Suspended Routes Tracker)")
        
        if 'dep/arr_airport' in df.columns and 'scheduled_gate_year' in df.columns and 'scheduled_gate_month' in df.columns:
            # Route changes in the last 3 months
            latest_year = df['scheduled_gate_year'].max()
            latest_month = df[df['scheduled_gate_year'] == latest_year]['scheduled_gate_month'].max()
            
            # Routes for the last 3 months
            recent_df = df[
                (df['scheduled_gate_year'] == latest_year) & 
                (df['scheduled_gate_month'] >= latest_month - 2)
            ]
            
            # Previous 3 months route (same period previous year)
            if latest_year > df['scheduled_gate_year'].min():
                previous_df = df[
                    (df['scheduled_gate_year'] == latest_year - 1) & 
                    (df['scheduled_gate_month'] >= latest_month - 2)
                ]
                
                recent_routes = set(recent_df['dep/arr_airport'].unique())
                previous_routes = set(previous_df['dep/arr_airport'].unique())
                
                new_routes = recent_routes - previous_routes
                suspended_routes = previous_routes - recent_routes
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.success(f"**✅ New routes ({len(new_routes)}dog)**")
                    if new_routes:
                        for route in list(new_routes)[:10]:
                            st.write(f"- {route}")
                    else:
                        st.write("No new routes")
                
                with col2:
                    st.error(f"**❌ closed/suspended route ({len(suspended_routes)}dog)**")
                    if suspended_routes:
                        for route in list(suspended_routes)[:10]:
                            st.write(f"- {route}")
                    else:
                        st.write("closed/No interrupted routes")
        
        # 2. By airline Market Share change
        st.markdown("#### 2) By airline Market Share change")
        
        if 'operating_carrier_name' in df.columns and 'scheduled_gate_year' in df.columns:
            carrier_share = df.groupby(['scheduled_gate_year', 'operating_carrier_name']).size().reset_index(name='movements')
            carrier_share['share'] = carrier_share.groupby('scheduled_gate_year')['movements'].transform(
                lambda x: x / x.sum() * 100
            )
            
            # Show only top 10 airlines
            top_carriers = carrier_share.groupby('operating_carrier_name')['movements'].sum().nlargest(10).index
            carrier_share_filtered = carrier_share[carrier_share['operating_carrier_name'].isin(top_carriers)]
            
            fig = px.area(
                carrier_share_filtered,
                x='scheduled_gate_year',
                y='share',
                color='operating_carrier_name',
                title="Changes in market share by airline",
                labels={'share': 'Market Share (%)', 'scheduled_gate_year': 'Year'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # International/Domestic ratio
            if 'International/Domestic' in df.columns:
                intl_dom_ratio = df.groupby(['scheduled_gate_year', 'International/Domestic']).size().reset_index(name='count')
                intl_dom_ratio['percentage'] = intl_dom_ratio.groupby('scheduled_gate_year')['count'].transform(
                    lambda x: x / x.sum() * 100
                )
                
                fig = px.line(
                    intl_dom_ratio,
                    x='scheduled_gate_year',
                    y='percentage',
                    color='International/Domestic',
                    title="International/Domestic rate change",
                    labels={'percentage': 'Percentage (%)', 'scheduled_gate_year': 'Year'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # 3. LCC vs FSC change in specific gravity
        st.markdown("#### 3) LCC vs FSC change in specific gravity")
        
        if 'operating_carrier_name' in df.columns:
            # LCC inventory (general LCC Airlines actually need separate mapping)
            lcc_keywords = ['air', 'jet', 'express', 'asia', 'tiger', 'cebu', 'peach', 'vanilla']
            df['carrier_type'] = df['operating_carrier_name'].apply(
                lambda x: 'LCC' if any(keyword in str(x).lower() for keyword in lcc_keywords) else 'FSC'
            )
            
            if 'scheduled_gate_year' in df.columns:
                lcc_fsc_ratio = df.groupby(['scheduled_gate_year', 'carrier_type']).size().reset_index(name='count')
                lcc_fsc_ratio['percentage'] = lcc_fsc_ratio.groupby('scheduled_gate_year')['count'].transform(
                    lambda x: x / x.sum() * 100
                )
                
                fig = px.bar(
                    lcc_fsc_ratio,
                    x='scheduled_gate_year',
                    y='percentage',
                    color='carrier_type',
                    title="LCC vs FSC change in specific gravity",
                    labels={'percentage': 'Percentage (%)', 'scheduled_gate_year': 'Year'},
                    barmode='stack'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def _connectivity_hub_analysis(self, df):
        st.markdown("### 🌐 connectivity(Connectivity) & Hub Competitiveness Analysis")
        
        # 1. hub power (Hub Strength Index)
        st.markdown("#### 1) hub power (Hub Strength Index)")
        
        if 'dep/arr_airport' in df.columns and 'scheduled_gate_local' in df.columns:
            # Calculate the number of possible transfer combinations (simple estimation)
            # actually MCT(Minimum Connection Time) Based calculation required
            
            # Estimation of transfer possibility based on arrival-departure time difference (2-6time interval)
            arrival_df = df[df['flight_io'] == 'a'] if 'flight_io' in df.columns else df
            departure_df = df[df['flight_io'] == 'd'] if 'flight_io' in df.columns else df
            
            if len(arrival_df) > 0 and len(departure_df) > 0:
                # Simple transferable pair calculation
                connectivity_pairs = []
                for _, arr_row in arrival_df.head(1000).iterrows():  # Improve performance with sampling
                    arr_time = arr_row['scheduled_gate_local']
                    arr_origin = arr_row.get('origin_airport', '')
                    
                    potential_conn = departure_df[
                        (departure_df['scheduled_gate_local'] > arr_time + pd.Timedelta(hours=2)) &
                        (departure_df['scheduled_gate_local'] < arr_time + pd.Timedelta(hours=6)) &
                        (departure_df.get('dep/arr_airport', '') != arr_origin)
                    ]
                    
                    connectivity_pairs.append(len(potential_conn))
                
                avg_connectivity = np.mean(connectivity_pairs) if connectivity_pairs else 0
                
                st.info(f"""
                **Hub Connectivity Index (Hub Connectivity Index)**
                - Average number of transferable connections: **{avg_connectivity:.1f}dog**
                - herb strength: **{'strong' if avg_connectivity > 5 else 'commonly' if avg_connectivity > 2 else 'weakness'}**
                """)
            
            # Route network visualization
            route_counts = df.groupby('dep/arr_airport').size().reset_index(name='frequency')
            top_routes = route_counts.nlargest(20, 'frequency')
            
            fig = px.bar(
                top_routes,
                x='dep/arr_airport',
                y='frequency',
                title="Flight frequency by top route",
                labels={'frequency': 'Frequency', 'dep/arr_airport': 'Route'}
            )
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        # 2. Comparison with competing hubs (If you have data)
        st.markdown("#### 2) Comparison of transfer possibilities with competing hubs")
        st.info("💡 Competitive hub comparison requires multi-airport data. Currently only single airport analysis is available.")
    
    def _revenue_perspective(self, df):
        st.markdown("### 💰 Revenue Perspective Issue Paper")
        
        # 1. Night Operation Value Analysis
        st.markdown("#### 1) night operation(Night Operation) value analysis")
        
        if 'scheduled_gate_hour' in df.columns:
            # Night flight definition: 22:00-06:00
            df['is_night'] = (df['scheduled_gate_hour'] >= 22) | (df['scheduled_gate_hour'] < 6)
            night_flights = df[df['is_night']]
            
            night_ratio = len(night_flights) / len(df) * 100 if len(df) > 0 else 0
            night_pax = night_flights['total_pax'].sum() if 'total_pax' in night_flights.columns else 0
            total_pax = df['total_pax'].sum() if 'total_pax' in df.columns else 0
            night_pax_ratio = (night_pax / total_pax * 100) if total_pax > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Night flight ratio", f"{night_ratio:.1f}%")
            with col2:
                st.metric("Number of night passengers", f"{night_pax:,.0f}")
            with col3:
                st.metric("Night flight passenger ratio", f"{night_pax_ratio:.1f}%")
            
            # Nighttime operation status by time zone
            hourly_night = df[df['is_night']].groupby('scheduled_gate_hour').size()
            fig = px.bar(
                x=hourly_night.index,
                y=hourly_night.values,
                title="Number of flights by night time slot",
                labels={'x': 'Hour', 'y': 'Number of Flights'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            if night_ratio < 15:
                st.success("💡 **Opportunity to expand nighttime operations**: Currently, the rate of night flights is low. Slot Possible to improve utilization rate and increase profits")
        
        # 2. Non-Aviation Revenue Impact Model
        st.markdown("#### 2) Non-Aviation Revenue Impact Model (shopping/duty free time)")
        
        if 'scheduled_gate_local' in df.columns and 'total_pax' in df.columns:
            # Estimation of residence time (Arrival-departure interval)
            # actually OD vs Transfer Separation required
            st.info("💡 Analysis of residence time OD/Transfer Separated data is required. Currently only overall average length of stay is provided.")
            
            if 'flight_io' in df.columns:
                arrival_pax = df[df['flight_io'] == 'a']['total_pax'].sum() if 'a' in df['flight_io'].values else 0
                departure_pax = df[df['flight_io'] == 'd']['total_pax'].sum() if 'd' in df['flight_io'].values else 0
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("arrive passenger 수", f"{arrival_pax:,.0f}")
                with col2:
                    st.metric("number of departing passengers", f"{departure_pax:,.0f}")
    
    def _safety_environment(self, df):
        st.markdown("### 🛡️ safety·regulation·environmental issues")
        
        # 1. Safety margin based on runway congestion
        st.markdown("#### 1) Safety margin analysis based on runway congestion")
        
        if 'scheduled_gate_hour' in df.columns:
            # Frequency of takeoffs and landings by time of day
            hourly_movements = df.groupby('scheduled_gate_hour').size()
            
            # Safety margin calculation (Assumption: Risk for more than 60 flights per hour)
            safety_threshold = 60
            risk_hours = hourly_movements[hourly_movements > safety_threshold]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=hourly_movements.index,
                y=hourly_movements.values,
                marker=dict(
                    color=['red' if v > safety_threshold else 'green' for v in hourly_movements.values]
                )
            ))
            fig.add_hline(y=safety_threshold, line_dash="dash", line_color="red", annotation_text="Safety Threshold")
            fig.update_layout(
                title="Runway congestion and safety margin by time of day",
                xaxis_title="Hour",
                yaxis_title="Number of Movements",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            if len(risk_hours) > 0:
                st.warning(f"**⚠️ Time zone where safety margin is insufficient:** {', '.join([f'{h:02d}:00' for h in risk_hours.index])}")
        
        # 2. carbon emissions estimates
        st.markdown("#### 2) carbon emissions estimates (Carbon Emission Structure)")
        
        if 'aircraft_class' in df.columns:
            # Flight frequency by aircraft type
            aircraft_distribution = df.groupby('aircraft_class').size().reset_index(name='count')
            aircraft_distribution['percentage'] = aircraft_distribution['count'] / aircraft_distribution['count'].sum() * 100
            
            fig = px.pie(
                aircraft_distribution,
                values='percentage',
                names='aircraft_class',
                title="Distribution by aircraft type"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Wide-body rate change
            if 'scheduled_gate_year' in df.columns:
                wide_body_ratio = df.groupby('scheduled_gate_year').apply(
                    lambda x: (x['aircraft_class'].str.contains('Wide', case=False, na=False).sum() / len(x) * 100)
                ).reset_index(name='wide_body_pct')
                
                fig = px.line(
                    wide_body_ratio,
                    x='scheduled_gate_year',
                    y='wide_body_pct',
                    title="Wide-body aircraft rate change",
                    labels={'wide_body_pct': 'Wide-body Ratio (%)', 'scheduled_gate_year': 'Year'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("💡 Wide-body A decline in rates could mean a change in the structure of carbon emissions.")
    
    def _strategic_recommendations(self, df):
        st.markdown("### 🎯 Airport strategy proposal type issue paper")
        
        # 1. Airline Reallocation Scenario
        st.markdown("#### 1) Airline Reallocation Scenario")
        
        if 'terminal' in df.columns and 'scheduled_gate_hour' in df.columns:
            # Congestion by terminal and time slot
            terminal_hourly = df.groupby(['terminal', 'scheduled_gate_hour']).size().reset_index(name='movements')
            
            # Identify peak times by terminal
            terminal_peaks = terminal_hourly.loc[terminal_hourly.groupby('terminal')['movements'].idxmax()]
            
            st.dataframe(terminal_peaks[['terminal', 'scheduled_gate_hour', 'movements']], use_container_width=True)
            
            st.info("""
            **💡 Relocation implications:**
            - If peak times for each terminal overlap, congestion can be alleviated through relocation.
            - Load distribution by time zone is possible through terminal relocation for each airline.
            """)
        
        # 2. specific Zone Need for expansion
        st.markdown("#### 2) specific Zone Necessity of expansion issue paper")
        
        if 'scheduled_gate_hour' in df.columns:
            # Demand analysis by time zone
            hourly_demand = df.groupby('scheduled_gate_hour').size()
            peak_hours = hourly_demand.nlargest(5)
            
            st.markdown("**Expansion priorities by peak hour:**")
            for idx, (hour, count) in enumerate(peak_hours.items(), 1):
                st.write(f"{idx}. **{hour:02d}:00** - {count:.0f}side (Need for expansion: {'height' if count > hourly_demand.mean() * 1.5 else 'commonly'})")
            
            # Need for gate expansion
            if 'terminal' in df.columns:
                df['time_slot'] = df['scheduled_gate_local'].dt.floor('15min')
                gate_occupancy = df.groupby(['time_slot', 'terminal']).size().reset_index(name='occupancy')
                max_occupancy = gate_occupancy.groupby('terminal')['occupancy'].max()
                
                st.markdown("**Maximum simultaneous gate occupancy per terminal:**")
                for terminal, occupancy in max_occupancy.items():
                    st.write(f"- **{terminal}**: {occupancy:.0f}Simultaneous use of two gates")
        
        # 3. Comprehensive suggestions
        st.markdown("#### 3) Comprehensive strategy proposal")
        
        recommendations = []
        
        # Congestion-based suggestions
        if 'scheduled_gate_hour' in df.columns:
            hourly_movements = df.groupby('scheduled_gate_hour').size()
            if hourly_movements.max() > hourly_movements.mean() * 1.5:
                recommendations.append("**short term improvement:** Review schedule readjustment to relieve congestion during peak hours")
        
        # Nighttime operation basis suggestions
        if 'scheduled_gate_hour' in df.columns:
            night_ratio = ((df['scheduled_gate_hour'] >= 22) | (df['scheduled_gate_hour'] < 6)).sum() / len(df) * 100
            if night_ratio < 15:
                recommendations.append("**mid-term improvement:** nighttime Slot Opportunity to increase revenue through improved utilization")
        
        # Proposal based on gate expansion
        if 'terminal' in df.columns and 'scheduled_gate_local' in df.columns:
            df['time_slot'] = df['scheduled_gate_local'].dt.floor('15min')
            gate_occupancy = df.groupby(['time_slot', 'terminal']).size()
            if gate_occupancy.max() > 50:  # threshold
                recommendations.append("**long term improvement:** Gate expansion review (Risk of saturation at certain times)")
        
        if recommendations:
            for rec in recommendations:
                st.write(f"- {rec}")
        else:
            st.success("Currently operating in good condition. No further improvements.")





    def show_profile(self, df, key="defalut"):
        df["terminal_carrier"] = "[" + df["terminal"] + "] " + df["operating_carrier_name"] 

        df["scheduled_gate_local"]=pd.to_datetime(df["scheduled_gate_local"])

        filter_containder = st.container(border=True)

        filter_containder = st.container(border=True)
        with filter_containder:
            name_col, start_date_col, end_date_col = st.columns([1, 2, 2])
            with name_col:
                st.subheader("**Filter**")

            with start_date_col:
                start_date = st.date_input("start date", value=pd.to_datetime("2024-08-03"), key=f"{key}_start_date") # Manila is 2025-08-03
            with end_date_col:
                end_date = st.date_input("end date", value=pd.to_datetime("2024-08-09"), key=f"{key}_end_date")

            df["scheduled_gate_local"] = pd.to_datetime(df["scheduled_gate_local"])
            df_filtered=df[(df["scheduled_gate_local"].dt.date>=start_date)&(df["scheduled_gate_local"].dt.date<=end_date)]




            # Available filters
            available_filters = [
                "International/Domestic",
                "terminal",
                "region_name",
                "country_name",
                "terminal_carrier",
                "operating_carrier_name",
                "operating_carrier_iata",
                "dep/arr_airport",
                "primary_usage",
                "aircraft_class",
                "flight_number",
                "aircraft_type",
                "aircraft_code_iata",
            ]
            
            # Select multiple filters to apply
            selected_filters = st.multiselect(
                "**Select Filters to Apply**",
                available_filters,
                default=["terminal", "terminal_carrier"],
                key=f"filter_selection_{key}"
            )
            
            # Apply each selected filter
            if selected_filters:
                for filter_name in selected_filters:
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        st.write(f"**{filter_name}**")
                    with col2:
                        value_list = sorted(df_filtered[filter_name].value_counts().index.tolist())
                        selected_values = st.multiselect(
                            f"Select values for {filter_name}",
                            value_list,
                            default=value_list[:1],
                            key=f"filter_values_{filter_name}_{key}",
                        )
                        if selected_values:
                            df_filtered = df_filtered[df_filtered[filter_name].isin(selected_values)]
                            
                            # Save current filter information (For scenarios)
                            if f'current_filters_{key}' not in st.session_state:
                                st.session_state[f'current_filters_{key}'] = {}
                            st.session_state[f'current_filters_{key}'][filter_name] = selected_values

        group_container = st.container(border=True)
        with group_container:
            name_col, graph_method_col, category_col, group_col, unit_min_col = st.columns(5)
            with name_col:
                st.subheader("**Graph**")
            with graph_method_col:
                graph_method_dict = {
                    "*(normal) bar": "※ Bar graph with time on the horizontal axis",
                    "(normal) line": "※ Line graph with time on the horizontal axis",
                    "(compare) daily": "※ Graph comparing daily data from 00:00 to 11:59",

                    "(compare) yearly_average": "※ Graph showing yearly averages from 00:00 to 11:59. Displays average hourly values for each year (Jan 1 – Dec 31)",
                    "(compare) monthly_average": "※ Graph showing monthly averages from 00:00 to 11:59. Displays average hourly values for each month (1st–30th or 31st)",
                    "(compare) yearly_monthly_average" : "※ Graph showing yearly averages from 00:00 to 11:59. Displays average hourly values for each year-month",
                    "(compare) weekday_average": "※ Graph showing weekday averages from 00:00 to 11:59. Displays average hourly values for each weekday",
                    "*(compare) yearly_peak": "※ Graph showing yearly peak values from 00:00 to 11:59. Displays maximum hourly values for each year (Jan 1 – Dec 31)",
                    "(compare) monthly_peak": "※ Graph showing monthly peak values from 00:00 to 11:59. Displays maximum hourly values for each month (1st–30th or 31st)",
                    "*(compare) weekday_peak": "※ Graph showing weekday peak values from 00:00 to 11:59. Displays maximum hourly values for each weekday",

                    "(peak time) daily": "※ Graph with dates on the horizontal axis, showing daily peak passenger counts per unit time",
                    "(peak time) weekly": "※ Graph with weeks on the horizontal axis, showing weekly peak passenger counts per unit time",
                    "(peak time) monthly": "※ Graph with months on the horizontal axis, showing monthly peak passenger counts per unit time",
                }

                method = st.selectbox(
                "**Type**",
                list(graph_method_dict.keys()),
                index=0,
                key=f"graph_method_{key}",
                )

            with category_col:
                category = st.selectbox(
                "**Flight / Seat / Pax ?**",
                [
                    "flight count",
                    "departure flight count",
                    "arrival flight count",
                    "departure seat count",
                    "arrival seat count",
                    "departure od-passenger count",
                    "arrival od-passenger count",
                    "departure transfer-passenger count",
                    "arrival transfer-passenger count",
                ],
                index=3,
                key=f"Select category{key}",
                )

            if method=="*(normal) bar":
                with group_col:
                    group = st.selectbox(
                    "**Colored by**",
                    [
                        "International/Domestic",
                        "flight_number",
                        "terminal",
                        "flight_io",
                        "region_name",
                        "country_name",
                        "operating_carrier_name",
                        "operating_carrier_iata",
                        "terminal_carrier",
                        "aircraft_type",
                        "aircraft_code_iata",
                        "dep/arr_airport",
                        "primary_usage",
                        "total_seat_count",
                        "flight_distance_km",
                        "aircraft_class",
                        "origin_airport",
                        "operating_maximum_takeoff_weight_lb",
                    ],
                    index=0,
                    key=f"Select Group Column{key}",
                    )
            else:
                group="International/Domestic"

            with unit_min_col:
                unit_min = st.number_input("X-Axis unit(min)", value=15, key=f"x_axis_time_unit{key}")




        if category in["flight count","departure flight count","arrival flight count"]:
            # Create filter summary for flight count categories
            filter_summary = ""
            if selected_filters:
                filter_parts = []
                for filter_name in selected_filters:
                    values = sorted(df_filtered[filter_name].unique().tolist())
                    if values:
                        filter_parts.append(f"{filter_name} → {', '.join(map(str, values))}")
                filter_summary = " | ".join(filter_parts)
            else:
                filter_summary = "No filters applied"
                
            st.caption(
                f"""
                :blue[**[Date]**] {start_date} ~ {end_date}
                :blue[**[Filters]**] {filter_summary}\n
                :blue[**[X]**] Actual Runway Time , :blue[**[Y]**] {category} , :blue[**[Time-Unit]**] {unit_min}min \n
                {graph_method_dict[method]}
                """
                )

            time_column = "scheduled_gate_local"
            if category=="departure flight count":
                df_filtered=df_filtered[df_filtered["flight_io"]=="d"]
            elif category=="arrival flight count":
                df_filtered=df_filtered[df_filtered["flight_io"]=="a"]
            else:
                pass
        else:
            # Create filter summary string
            filter_summary = ""
            if selected_filters:
                filter_parts = []
                for filter_name in selected_filters:
                    values = sorted(df_filtered[filter_name].unique().tolist())
                    if values:
                        filter_parts.append(f"{filter_name} → {', '.join(map(str, values))}")
                filter_summary = " | ".join(filter_parts)
            else:
                filter_summary = "No filters applied"

            st.caption(
                f"""
                :blue[**[Date]**] {start_date} ~ {end_date}
                :blue[**[Filters]**] {filter_summary}\n
                :blue[**[X]**] Passenger Show-Up Time , :blue[**[Y]**] {category} , :blue[**[Time-Unit]**] {unit_min}min \n
                {graph_method_dict[method]}
                """
                )

            category_dict={
            "departure seat count":{"agg_col":"total_seat_count",
                                        "flight_io":"d",
                                        "mean":-180,
                                        "sigma":50,
                                        "min_max_clip":[-360,-37],
                                        },
            "arrival seat count":{"agg_col":"total_seat_count",
                                        "flight_io":"a",
                                        "mean":20,
                                        "sigma":6,
                                        "min_max_clip":[3,50],
                                        },                               
            "departure od-passenger count":{"agg_col":"od_pax",
                                        "flight_io":"d",
                                        "mean":-120,
                                        "sigma":26,
                                        "min_max_clip":[-360,-37],
                                        },      
            "arrival od-passenger count":{"agg_col":"od_pax",
                                        "flight_io":"a",
                                        "mean":20,
                                        "sigma":6,
                                        "min_max_clip":[3,50],
                                        },  
            "departure transfer-passenger count":{"agg_col":"tr_pax",
                                        "flight_io":"d",
                                        "mean":-50,
                                        "sigma":15,
                                        "min_max_clip":[-80,-37],
                                        },      
            "arrival transfer-passenger count":{"agg_col":"tr_pax",
                                        "flight_io":"a",
                                        "mean":20,
                                        "sigma":6,
                                        "min_max_clip":[3,50],
                                        },  
            }

            agg_col=category_dict[category]["agg_col"]
            flight_io=category_dict[category]["flight_io"]
            mean=category_dict[category]["mean"]
            sigma=category_dict[category]["sigma"]
            min_max_clip=category_dict[category]["min_max_clip"]


            df_filtered=df_filtered[df_filtered["flight_io"]==flight_io]
            if group==agg_col:
                group=agg_col
                df_filtered=df_filtered[["scheduled_gate_local",agg_col]]
            else:
                df_filtered=df_filtered[["scheduled_gate_local",group,agg_col]]

            df_filtered = df_filtered.loc[df_filtered.index.repeat(df_filtered[agg_col])].reset_index(drop=True)


            df_filtered[f"scheduled_gate_local(min)"] = 0
            df_filtered = create_normal_dist_col(
                df=df_filtered,
                ref_col=f"scheduled_gate_local(min)",
                new_col="SHOW(min)",
                mean=mean,
                sigma=sigma,
                min_max_clip=min_max_clip,
                unit="m",
                iteration=6,
                datetime=False,
            )
            df_filtered["SHOW"] = df_filtered["scheduled_gate_local"] + pd.to_timedelta(
                df_filtered["SHOW(min)"], unit="m"
            )
            time_column="SHOW"

        series_dict={"flight_distance_km":{"unit":2000, "suffix":' km'},
                    "total_seat_count":{"unit":100, "suffix":' seat'},
                    "standing_time":{"unit":100, "suffix":' min'},
                    "ground_time":{"unit":100, "suffix":' min'},
                    "operating_maximum_takeoff_weight_lb":{"unit":200000, "suffix":' lb'},
                    }

        if group in list(series_dict.keys()):
            unit=series_dict[group]["unit"]
            suffix=series_dict[group]["suffix"]
            selected_series=(df_filtered[group]//unit*unit)
            df_filtered[group] = (selected_series.astype(str) + '~' + (selected_series + unit).astype(str) + suffix).str.replace(".0","")
        start_count_df, start_count_ranking_order = make_count_df(df_filtered, start_date, end_date, 
        time_column, group, buffer_day=False, freq_min=unit_min)

        if method=="*(normal) bar":
            date_diff = (end_date - start_date).days
            if date_diff <=31:
                show_bar(start_count_df, start_count_ranking_order, group)
                # st.dataframe(start_count_df, hide_index=True)
            else:
                st.warning("The selected period exceeds 30 days. Select a time period within 30 days or another visualization method(line line) Please select.")
            
            total_df = start_count_df.groupby("Time")["index"].sum()//1
            
            # Calculating and displaying statistical information
            st.markdown("#### 📊 statistical information")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                max_value = total_df.max()
                st.metric("Max", f"{max_value:,.0f}")
            
            with col2:
                percentile_95 = total_df.quantile(0.95)
                st.metric("95% upper value", f"{percentile_95:,.0f}")
            
            with col3:
                percentile_90 = total_df.quantile(0.90)
                st.metric("90% upper value", f"{percentile_90:,.0f}")
            
            with col4:
                mean_value = total_df.mean()
                st.metric("average", f"{mean_value:,.0f}")
            
            st.markdown("#### 📋 Hourly data")
            st.table(total_df.T)
            return total_df
            
        elif method=="(normal) line":
            df_show = start_count_df.groupby("Time")["index"].agg("sum")
            fig = px.line(df_show, x=df_show.index, y='index')
            st.plotly_chart(fig)
            st.caption("※ Graph data")
            st.dataframe(df_show)

        elif method=="(compare) daily":
            df_show = start_count_df.groupby("Time")["index"].agg("sum").to_frame()
            df_show['date'] = df_show.index.date
            df_show['time'] = df_show.index.strftime('%H:%M')
            fig = px.line(df_show, x='time', y='index', color='date')
            df_show_download = df_show.set_index(["time","date"]).unstack()
            df_show_download.columns = [f'{col[1]}' for col in df_show_download.columns]
            st.plotly_chart(fig)
            self.heatmap(df_show_download)
            st.caption("※ Graph data")
            st.dataframe(df_show_download.T, use_container_width=True)

            
        elif method=="(compare) yearly_average":
            df_show = start_count_df.groupby("Time")["index"].agg("sum").to_frame()
            df_show['year'] = df_show.index.year
            df_show['time'] = df_show.index.strftime('%H:%M')
            df_show = df_show.groupby(['year', 'time'])['index'].mean().reset_index()


            fig = px.line(df_show, x='time', y='index', color='year',
                        labels={'index': 'Average Count', 'year': 'year'},
                        title='yearly_average Average Pattern')
            df_show_download = df_show.set_index(["time","year"]).unstack()
            df_show_download.columns = [f'{col[1]}' for col in df_show_download.columns]
            st.plotly_chart(fig)
            self.heatmap(df_show_download)
            st.caption("※ Graph data")
            st.dataframe(df_show_download.apply(np.ceil).T, use_container_width=True)


        elif method=="(compare) monthly_average":
            df_show = start_count_df.groupby("Time")["index"].agg("sum").to_frame()
            df_show['month'] = df_show.index.month
            df_show['time'] = df_show.index.strftime('%H:%M')
            df_show = df_show.groupby(['month', 'time'])['index'].mean().reset_index()
            fig = px.line(df_show, x='time', y='index', color='month',
                        labels={'index': 'Average Count', 'month': 'Month'},
                        title='Monthly Average Pattern')
            df_show_download = df_show.set_index(["time","month"]).unstack()
            df_show_download.columns = [f'{col[1]}' for col in df_show_download.columns]
            st.plotly_chart(fig)
            self.heatmap(df_show_download)
            st.caption("※ Graph data")
            st.dataframe(df_show_download.apply(np.ceil).T, use_container_width=True)


        elif method == "(compare) yearly_monthly_average":
            df_show = start_count_df.groupby("Time")['index'].sum().to_frame()
            df_show['year-month'] = df_show.index.strftime('%Y-%m')
            df_show['time'] = df_show.index.strftime('%H:%M')

            df_show = df_show.groupby(['year-month', 'time'])['index'].mean().reset_index()

            fig = px.line(df_show, x='time', y='index', color='year-month',
                        labels={'index': 'Average Count', 'year-month': 'Year-Month'},
                        title='Yearly & Monthly Average Pattern')

            df_show_download = df_show.set_index(["time", "year-month"])['index'].unstack()

            st.plotly_chart(fig)
            self.heatmap(df_show_download)
            st.caption("※ Graph data")
            st.dataframe(df_show_download.apply(np.ceil).T, use_container_width=True)


        elif method == "(compare) weekday_average":  # Add pattern for each day of the week
            df_show = start_count_df.groupby("Time")["index"].agg("sum").to_frame()
            df_show['weekday'] = df_show.index.weekday  # 0=Monday, 6=Sunday
            df_show['time'] = df_show.index.strftime('%H:%M')

            weekday_names = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 
                            4:'Friday', 5:'Saturday', 6:'Sunday'}
            df_show["weekday"] = df_show["weekday"].map(weekday_names)

            
            df_show = df_show.groupby(['weekday', 'time'])['index'].mean().reset_index()
            fig = px.line(df_show, x='time', y='index', color='weekday',
                            labels={'index': 'Average Count', 'weekday': 'Day of Week', 'time': 'Time'},
                            title='Weekday Average Pattern')
            st.plotly_chart(fig)

            df_show_download = df_show.set_index(["time","weekday"]).unstack()
            df_show_download.columns = [f'{col[1]}' for col in df_show_download.columns]

            column_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            existing_columns = [col for col in column_order if col in df_show_download.columns]
            df_show_download = df_show_download[existing_columns]


            self.heatmap(df_show_download)
            st.caption("※ Graph data")
            st.dataframe(df_show_download.apply(np.ceil).T, use_container_width=True)

        elif method=="*(compare) yearly_peak":
            df_show = start_count_df.groupby("Time")["index"].agg("sum").to_frame()
            df_show['year'] = df_show.index.year
            df_show['time'] = df_show.index.strftime('%H:%M')
            df_show = df_show.groupby(['year', 'time'])['index'].max().reset_index()
            fig = px.line(df_show, x='time', y='index', color='year',
                        labels={'index': 'Average Count', 'year': 'year'},
                        title='yearly Average Pattern')
            df_show_download = df_show.set_index(["time","year"]).unstack()
            df_show_download.columns = [f'{col[1]}' for col in df_show_download.columns]
            st.plotly_chart(fig)
            self.heatmap(df_show_download)
            st.caption("※ Graph data")
            st.dataframe(df_show_download.T, use_container_width=True)

        elif method=="(compare) monthly_peak":
            df_show = start_count_df.groupby("Time")["index"].agg("sum").to_frame()
            df_show['month'] = df_show.index.month
            df_show['time'] = df_show.index.strftime('%H:%M')
            df_show = df_show.groupby(['month', 'time'])['index'].max().reset_index()
            fig = px.line(df_show, x='time', y='index', color='month',
                        labels={'index': 'Average Count', 'month': 'Month'},
                        title='Monthly Peak Pattern')
            df_show_download = df_show.set_index(["time","month"]).unstack()
            df_show_download.columns = [f'{col[1]}' for col in df_show_download.columns]
            st.plotly_chart(fig)
            self.heatmap(df_show_download)
            st.dataframe(df_show_download.T, use_container_width=True)

        elif method == "*(compare) weekday_peak":  # Add pattern for each day of the week
            df_show = start_count_df.groupby("Time")["index"].agg("sum").to_frame()
            df_show['weekday'] = df_show.index.weekday  # 0=Monday, 6=Sunday
            df_show['time'] = df_show.index.strftime('%H:%M')

            weekday_names = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 
                            4:'Friday', 5:'Saturday', 6:'Sunday'}
            df_show["weekday"] = df_show["weekday"].map(weekday_names)

            df_show = df_show.groupby(['weekday', 'time'])['index'].max().reset_index()
            fig = px.line(df_show, x='time', y='index', color='weekday',
                            labels={'index': 'Average Count', 'weekday': 'Day of Week', 'time': 'Time'},
                            title='Weekday Peak Pattern')
            st.plotly_chart(fig)
            
            df_show_download = df_show.set_index(["time","weekday"]).unstack()
            df_show_download.columns = [f'{col[1]}' for col in df_show_download.columns]

            column_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            existing_columns = [col for col in column_order if col in df_show_download.columns]
            df_show_download = df_show_download[existing_columns]


            self.heatmap(df_show_download)
            st.caption("※ Graph data")
            st.dataframe(df_show_download.T, use_container_width=True)

        elif method=="(peak time) daily":
            df_show = (start_count_df
                        .groupby(start_count_df['Time'].dt.date)['index']
                        .max()
                        .reset_index())
            fig = px.bar(df_show, 
                        x='Time', 
                        y="index",
                        labels={'Time': 'Date', 'index': 'Peak Count'},
                        title='Daily Peak Values')
            st.plotly_chart(fig)
            st.caption("※ Graph data")
            st.dataframe(df_show, hide_index=True)
            ########################################
            # with st.expander("Workers"):
            #     capacity_col, workers_col = st.columns(2)
            #     with capacity_col : 
            #         capacity_min=st.number_input(
            #                 "**1EA Capacity (passengers/min)**",
            #                 value=1.5,
            #                 min_value=0.0, 
            #                 max_value=10000000.0,
            #                 step=0.1,
            #                 format="%.2f",
            #                 key="1Capacity per unit",
            #             )
            #     with workers_col:
            #         workers=st.number_input(
            #                 "**Staff per 1EA**",
            #                 value=5.0,
            #                 min_value=0.0, 
            #                 max_value=10000000.0,
            #                 step=0.1,
            #                 format="%.2f",
            #                 key="workers",
            #             )
            #     df_show = df_show.rename({"index":category}, axis=1)
            #     df_show["Required number of EA"]=(df_show[category]/(capacity_min*unit_min)).round()
            #     df_show["Total Required Staff"]=df_show["Required number of EA"]*workers
            #     st.info(f"""
            #             1EA Capacity : {capacity_min}passengers/min \n
            #             Required number of EA : {category} / ({capacity_min}passengers/min*{unit_min}min) \n
            #             Staff per 1EA : {workers}persons \n
            #             Total Required Staff : {workers}persons * Required number of EA
            #             """)
            #     st.dataframe(df_show, hide_index=True)
            ########################################

        elif method=="(peak time) weekly":
            # Display parking information more clearly
            df_show = (start_count_df
                        .assign(
                            year=start_count_df['Time'].dt.year,
                            week=start_count_df['Time'].dt.isocalendar().week
                        )
                        .groupby(['year', 'week'])['index']
                        .max()
                        .reset_index())
            df_show['period'] = df_show['year'].astype(str) + '-W' + df_show['week'].astype(str).str.zfill(2)
            fig = px.bar(df_show, 
                        x='period', 
                        y='index',
                        labels={'period': 'Year-Week', 'index': 'Peak Count'},
                        title='Weekly Peak Values')
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig)
            st.caption("※ Graph data")
            st.dataframe(df_show, hide_index=True)

            ########################################
            # with st.expander("Workers"):
            #     capacity_col, workers_col = st.columns(2)
            #     with capacity_col : 
            #         capacity_min=st.number_input(
            #                 "**1EA Capacity (passengers/min)**",
            #                 value=1.5,
            #                 min_value=0.0, 
            #                 max_value=10000000.0,
            #                 step=0.1,
            #                 format="%.2f",
            #                 key="1Capacity per unit",
            #             )
            #     with workers_col:
            #         workers=st.number_input(
            #                 "**Staff per 1EA**",
            #                 value=5.0,
            #                 min_value=0.0, 
            #                 max_value=10000000.0,
            #                 step=0.1,
            #                 format="%.2f",
            #                 key="workers",
            #             )
            #     df_show = df_show.rename({"index":category}, axis=1)
            #     df_show["Required number of EA"]=(df_show[category]/(capacity_min*unit_min)).round()
            #     df_show["Total Required Staff"]=df_show["Required number of EA"]*workers
            #     st.info(f"""
            #             1EA Capacity : {capacity_min}passengers/min \n
            #             Required number of EA : {category} / ({capacity_min}passengers/min*{unit_min}min) \n
            #             Staff per 1EA : {workers}persons \n
            #             Total Required Staff : {workers}persons * Required number of EA
            #             """)
            #     st.dataframe(df_show, hide_index=True)
            ########################################

        elif method=="(peak time) monthly":
            df_show = (start_count_df
                        .assign(month=start_count_df['Time'].dt.strftime('%Y-%m'))
                        .groupby('month')['index']
                        .max()
                        .reset_index())
            fig = px.bar(df_show, 
                        x='month', 
                        y='index',
                        labels={'month': 'Month', 'index': 'Peak Count'},
                        title='Monthly Peak Values')
            st.plotly_chart(fig)
            st.caption("※ Graph data")
            st.dataframe(df_show, hide_index=True)
            # ########################################
            # with st.expander("Workers"):
            #     capacity_col, workers_col = st.columns(2)
            #     with capacity_col : 
            #         capacity_min=st.number_input(
            #                 "**1EA Capacity (passengers/min)**",
            #                 value=1.5,
            #                 min_value=0.0, 
            #                 max_value=10000000.0,
            #                 step=0.1,
            #                 format="%.2f",
            #                 key="1Capacity per unit",
            #             )
            #     with workers_col:
            #         workers=st.number_input(
            #                 "**Staff per 1EA**",
            #                 value=5.0,
            #                 min_value=0.0, 
            #                 max_value=10000000.0,
            #                 step=0.1,
            #                 format="%.2f",
            #                 key="workers",
            #             )
            #     df_show = df_show.rename({"index":category}, axis=1)
            #     df_show["Required number of EA"]=(df_show[category]/(capacity_min*unit_min)).round()
            #     df_show["Total Required Staff"]=df_show["Required number of EA"]*workers
            #     st.info(f"""
            #             1EA Capacity : {capacity_min}passengers/min \n
            #             Required number of EA : {category} / ({capacity_min}passengers/min*{unit_min}min) \n
            #             Staff per 1EA : {workers}persons \n
            #             Total Required Staff : {workers}persons * Required number of EA
            #             """)
            #     st.dataframe(df_show, hide_index=True)
            # ########################################


    def get_airport_passenger_data(self, iata_code):
        sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
        query = f"""
        SELECT ?date ?year ?month ?passengers WHERE {{
        ?airport wdt:P238 "{iata_code}".
        ?airport p:P3872 ?statements.
        ?statements pq:P585 ?date.
        ?statements ps:P3872 ?passengers.
        BIND(YEAR(?date) AS ?year)
        BIND(MONTH(?date) AS ?month)
        }}
        ORDER BY ?date
        """
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()

        data = []
        for r in results["results"]["bindings"]:
            date = r["date"]["value"]
            year = int(r["year"]["value"])
            month = int(r["month"]["value"])
            passengers = int(r["passengers"]["value"])

            # Annual total if date is January 1st, otherwise monthly data

            data.append({
                "date":date,
                "Year": year,
                "Month": month,
                "Passengers": passengers
            })

        df = pd.DataFrame(data)
        return df


    def classify_passenger_data(self, df):
        df['date'] = pd.to_datetime(df['date'])

        def classify(group):
            group = group.sort_values('date')
            year = group['Year'].iloc[0]
            jan_data = group[group['Month'] == 1]
            other_months = group[group['Month'] != 1]

            types = ['dummy'] * len(group)

            if len(group) == 1 and jan_data.iloc[0]['date'].day == 1:
                types[0] = 'year'

            elif len(jan_data) >= 2:
                min_idx = jan_data['Passengers'].idxmin()
                max_idx = jan_data['Passengers'].idxmax()
                types[group.index.get_loc(min_idx)] = 'month'
                types[group.index.get_loc(max_idx)] = 'year'

            elif len(jan_data) == 1:
                jan_value = jan_data.iloc[0]['Passengers']
                if len(other_months) == 0:
                    types[group.index.get_loc(jan_data.index[0])] = 'year'
                else:
                    max_other = other_months['Passengers'].max()
                    if jan_value > max_other * 5:
                        types[group.index.get_loc(jan_data.index[0])] = 'year'
                    else:
                        types[group.index.get_loc(jan_data.index[0])] = 'month'

            for idx in other_months.index:
                types[group.index.get_loc(idx)] = 'month'

            group['Type'] = types

            if 'year' not in types:
                year_total = group[group['Type'] == 'month']['Passengers'].sum()
                year_row = pd.DataFrame({
                    'date': [pd.Timestamp(year=year, month=1, day=1)],
                    'Year': [year],
                    'Month': [None],
                    'Passengers': [year_total],
                    'Type': ['year']
                })
                group = pd.concat([group, year_row], ignore_index=True)

            return group

        df = df.groupby('Year', group_keys=False).apply(classify).reset_index(drop=True)
        return df



    @st.fragment
    def airport_basic_info(self):
        layout_facility_tab, traffic_tab, others_tab = st.tabs(["**Basic Info**","**Detail**","**Others**"])
        with layout_facility_tab:
            self.show_earth_chart(self.df_orig)
            self.show_bubble_map(self.df_orig, "carto-positron") # carto-positron / carto-darkmatter

            st.subheader(f"**Layout**")
            

            if self.iata_code!=None:
                    st.caption(f"Latitude : {self.airport_lat} / Longitude : {self.airport_long}")
                    src = f"https://openairportmap.org/{self.icao_code}#map=13.5/{self.airport_lat}/{self.airport_long}"
                    src2 = f"https://www.google.com/maps/@{self.airport_lat},{self.airport_long},500m"
                    src3 = f"https://earth.google.com/web/@{self.airport_lat},{self.airport_long},25.72a,133464.677d,1y,91.211h,60.984t,0r/"
                    st.markdown(
                        f'<div style="width: 100%; height: 100vh;"><iframe src="{src}" width="100%" height="100%" style="border: none;"></iframe></div>',
                        unsafe_allow_html=True
                    )
                    st.write("*:blue[Source : Open airport] →*", src)
                    st.write("*:blue[Google Map] →*", src2)
                    st.write("*:blue[Google Earth] →*", src3)
                    st.divider()


            elif self.iata_code==None:
                    tab_list=st.tabs(self.iata_code_list)
                    for tab, iata_code, icao_code in zip(tab_list, self.iata_code_list, self.icao_code_list):
                        with tab:
                            selected_airport = self.df_airport[self.df_airport["airport_id"] == iata_code]
                            lat = selected_airport["lat"].values[0]
                            lon = selected_airport["lon"].values[0]
                            st.caption(f"Latitude : {lat} / Longitude : {lon}")
                            src = f"https://openairportmap.org/{icao_code}#map=10/{lat}/{lon}"
                            src2 = f"https://www.google.com/maps/@{lat},{lon},500m"
                            src3 = f"https://earth.google.com/web/@{lat},{lon},25.72a,133464.677d,1y,91.211h,60.984t,0r/"

                            st.markdown(
                                f'<div style="width: 100%; height: 100vh;"><iframe src="{src}" width="100%" height="100%" style="border: none;"></iframe></div>',
                                unsafe_allow_html=True
                            )
                            st.write("*:blue[Source : Open airport] →*", src)
                            st.write("*:blue[Google Map] →*", src2)
                            st.write("*:blue[Google Earth] →*", src3)
                            st.divider()



        with traffic_tab:
            peak_profile_tab, annual_tab, peak_date_tab = st.tabs(["Daily Profile", "Year-to-Year Profile","Annual Profile"])

            with peak_profile_tab:
                self.show_profile(self.df_orig, key="basic")
                
            with annual_tab:
                c1,c2=st.columns(2)
                with c1:
                    group_col = st.selectbox('**Category**', 
                                            ["region_name",
                                            "country_code",
                                            "dep/arr_airport",
                                            "operating_carrier_iata",
                                            "operating_carrier_name",
                                            "primary_usage",
                                            "International/Domestic",
                                            "aircraft_type",
                                            "aircraft_code_iata",
                                            "aircraft_class",
                                            "terminal",
                                            "origin_airport"],
                                                index=0,
                                                key="groupby_category_select"
                                                )
                with c2:
                    count_col = st.selectbox('**Move/Pax/Seat/Num of Routes...**', ["movement","total_pax","od_pax","tr_pax","total_seat_count", "Num of Routes", "Num of Airlines"], index=0, key="count_select")
                if count_col =="Num of Routes":
                    grouped_df = self.df_orig.groupby([group_col,"year"])["dep/arr_airport"].nunique().unstack().fillna(0).astype(int).sort_values(by=2024,ascending=False)
                elif count_col =="Num of Airlines":
                    grouped_df = self.df_orig.groupby([group_col,"year"])["operating_carrier_iata"].nunique().unstack().fillna(0).astype(int).sort_values(by=2024,ascending=False)
                else:
                    grouped_df=self.df_orig.groupby([group_col,"year"])[count_col].agg("sum").unstack().fillna(0).astype(int).sort_values(by=2024,ascending=False)
                
                df_melted = grouped_df.reset_index().melt(id_vars=group_col, var_name='Year', value_name='Value')
                grouped_df.loc["※ Total ※"]=grouped_df.sum(axis=0)
                st.dataframe(grouped_df, use_container_width=True)

                fig = px.bar(
                    df_melted, 
                    x='Year', 
                    y='Value', 
                    color=group_col,
                    title="Regional data by year"
                )
                st.plotly_chart(fig)

            with peak_date_tab:
                select_target_day(self.df_orig)


        with others_tab:
            population_tab, airport_wiki_tab , economic_tab= st.tabs(["Catchment Area","Wikipedia", "Economic & Country"])
            with population_tab:
                color_col, radius,_ = st.columns([0.15,0.15,0.7])
                with color_col:
                    color = st.selectbox(
                        "**Bubble Color**",
                        ["country","city"],
                        key="color",    
                    )
                with radius:
                    radius_distance = st.selectbox("Population Radius(km)", [30,50,100,200,300,400,500,700,1000,2000,3000],index=6,
                    key="select_box_catchment")
                self.show_cities_location(map_style="carto-positron", color=color, radius_distance=radius_distance)
            # with modal_competition_tab:
            #     st.header(f"**{self.iata_code}**")
            #     src=f"https://www.openrailwaymap.org/?lat={self.airport_lat}&lon={self.airport_long}&zoom=14"
            #     st.markdown(
            #     f'<div style="width: 100%; height: 100vh;"><iframe src="{src}" width="100%" height="100%" style="border: none;"></iframe></div>',
            #     unsafe_allow_html=True
            #     )
            #     st.write("*:blue[Source → Open Railway]*", src)


            with airport_wiki_tab:
                if self.iata_code!=None:
                    st.header(f"**{self.iata_code}**")
                    selected_airport_open_data = self.airport_open_data[self.airport_open_data["ident"]==self.icao_code]
                    try:
                        src=selected_airport_open_data["wikipedia_link"].values[0]
                        st.markdown(
                        f'<div style="width: 100%; height: 180vh;"><iframe src="{src}" width="100%" height="100%" style="border: none;"></iframe></div>',
                        unsafe_allow_html=True
                        )
                        st.write("*:blue[→ WikiPedia]*", src)
                    except:
                        pass
                
                elif self.iata_code==None:
                    tab_list=st.tabs(self.iata_code_list)
                    for tab, iata_code, icao_code in zip(tab_list, self.iata_code_list, self.icao_code_list):
                        with tab:
                            st.header(f"**{iata_code}**")
                            selected_airport_open_data = self.airport_open_data[self.airport_open_data["ident"]==icao_code]
                            try:
                                src=selected_airport_open_data["wikipedia_link"].values[0]
                                st.markdown(
                                f'<div style="width: 100%; height: 180vh;"><iframe src="{src}" width="100%" height="100%" style="border: none;"></iframe></div>',
                                unsafe_allow_html=True
                                )
                                st.write("*:blue[→ WikiPedia]*", src)
                            except:
                                pass




            with economic_tab:
                econ_col, country_info_col = st.columns(2)
                country_adj = (self.country).replace(" ","-")
                country_dict = {
                    "country-rating":f"https://tradingeconomics.com/{country_adj}/rating",
                    "corruption-index":f"https://tradingeconomics.com/{country_adj}/corruption-index",
                    "currency":f"https://tradingeconomics.com/{country_adj}/currency",
                    "government-bond-yield":f"https://tradingeconomics.com/{country_adj}/government-bond-yield",
                    "risk-premium(damodaran)":f"https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/ctryprem.html",

                    "inflation-cpi":f"https://tradingeconomics.com/{country_adj}/inflation-cpi",
                    "Unemployment rate":f"https://tradingeconomics.com/{country_adj}/unemployment-rate",

                    "gdp":f"https://tradingeconomics.com/{country_adj}/gdp",
                    "gdp-per-capita":f"https://tradingeconomics.com/{country_adj}/gdp-per-capita",
                    "full-year-gdp-growth":f"https://tradingeconomics.com/{country_adj}/full-year-gdp-growth",
                    "gdp-constant-price":f"https://tradingeconomics.com/{country_adj}/gdp-constant-price",

                    "population":f"https://tradingeconomics.com/{country_adj}/population",
                    "corporate-tax-rate":f"https://tradingeconomics.com/{country_adj}/corporate-tax-rate",
                    "Labor Flexibility Index":f"https://www.theworldranking.com/statistics/90/labor-freedom/",
                    "foreign-direct-investment":f"https://tradingeconomics.com/{country_adj}/foreign-direct-investment",

                    "Religions_by_country":f"https://en.wikipedia.org/wiki/Religions_by_country",
                    "territory_area":"https://en.wikipedia.org/wiki/List_of_countries_and_dependencies_by_area",
                    "system_of_government":"https://en.wikipedia.org/wiki/List_of_countries_by_system_of_government",
                    "general_country_info":f"https://en.wikipedia.org/wiki/{self.country}"
                    }

                with econ_col:
                    c1,_=st.columns([0.3,0.7])
                    with c1:
                        economic_value = st.selectbox(
                            "**Economic Index**",
                            ["country-rating","corruption-index","currency","government-bond-yield","risk-premium(damodaran)",
                            "gdp", "full-year-gdp-growth", "gdp-per-capita", "gdp-constant-price",
                            "inflation-cpi","Unemployment rate",
                            "population","corporate-tax-rate","foreign-direct-investment","Labor Flexiblilty Index",
                            ],
                            key="economic_value",    
                        )
                    st.markdown(
                    f'<div style="width: 100%; height: 100vh;"><iframe src="{country_dict[economic_value]}" width="100%" height="100%" style="border: none;"></iframe></div>',
                    unsafe_allow_html=True
                    )
                    st.write(country_dict[economic_value])

                with country_info_col:
                    c1,_=st.columns([0.3,0.7])
                    with c1:
                        economic_value = st.selectbox(
                            "**Country Index**",
                            ["general_country_info",
                                "Religions_by_country",
                                "territory_area",
                                "system_of_government",
                            ],
                            key="country_info_value",    
                        )

                    st.markdown(
                    f'<div style="width: 100%; height: 100vh;"><iframe src="{country_dict[economic_value]}" width="100%" height="100%" style="border: none;"></iframe></div>',
                    unsafe_allow_html=True
                    )
                    st.write(country_dict[economic_value])

    @st.fragment
    def masterplan_func(self):
        set_years, filter_flights, forecast, analyse, compare= st.tabs(["**📅[M1] SET YEAR**","**✈️[M2] FILTER**","**📈[M3] FORECAST**","**🧮[M4] ANALYSE**", "**🟢🟡🔴[M5] CAMPARE**"])
        with set_years:
            self.select_period(self.df_orig)
        with filter_flights:
            self.df, self.filter_dict = self.select_filter(self.df_orig, key="masterplan")
        with forecast :
            self.forecast()
        with analyse :
            if st.button(
                "**RUN**",
                type="primary",
                use_container_width=True,
                key="MasterPlan",
            ):
                self.analyse_performance()
                self.apply_return_df_to_excel()
                download_excel_tab, = st.tabs(["**DOWNLOAD**"])
                with download_excel_tab:
                    download_excel()
        with compare:
            self.compare_scenario()

    def select_period(self, df):
        min_year = int(df['scheduled_gate_local'].dt.year.min())


        select_period_tab, reference_table_tab = st.tabs(["**Select Period**","**Reference Table**"])
        with select_period_tab:
            st.subheader('**Select Period**')
            c1, c2 = st.columns(2)
            with c1:
                period_containder1 = st.container(border=True)
                with period_containder1:
                    st.write("**Reference Year**")

                    start_, end_= st.columns(2)
                    with start_:

                        self.start_year = int(st.number_input(
                                "**Start**",
                                value=max(2015, min_year),
                                min_value=max(2015, min_year), # Analysis since at least 2018(I did not receive data prior to 2018 from Cirium.)
                                max_value=2024,
                                key="Start Year",
                            ))
                    with end_:
                        self.end_year = int(st.number_input(
                                "**End**",
                                value=2024,
                                key="End Year",
                            ))
            with c2:
                period_containder2 = st.container(border=True)
                with period_containder2:
                    st.write("**Predict Year**")

                    start_, end_ = st.columns(2)
                    with start_:
                        st.selectbox("**Start**",
                            [self.end_year+1],)
                    with end_:
                        self.predict_end_year = int(st.number_input(
                                "**End**",
                                value=int(self.end_year)+36,
                                key="Predict End Year",
                            ))
                    
        with reference_table_tab:
            st.subheader('**Reference Table**')
            idx_list=[
                # 'Demand',
                'Load_Factor',
                'Transfer_Rate',
                'FleetMix_Rate',
                'Seats_Per_Aircraft',
                'Load_Factor_By_Class',
                'Transfer_Rate_By_Class',
                'Crew_Per_Aircraft',
                'Movement_PHF_Adjustment',
                'Movement_D_Factor',
                'Passenger_PHF_Adjustment',
                'Passenger_D_Factor',
                'Directional_Factor',
                'Maximum_Takeoff_Weight',
                'PeakTime Standing',
            ]
            year_col_list=[int(year) for year in range(self.start_year, self.end_year+1)]
            self.ref_year_table = pd.DataFrame(index=idx_list, columns=year_col_list)
            self.ref_year_table = self.ref_year_table.fillna(0).astype(bool)


            first_col=self.ref_year_table.columns[0]
            self.ref_year_table[first_col]=1
            self.ref_year_table[self.end_year]=1

            self.ref_year_table = st.data_editor(self.ref_year_table.fillna(0).astype(bool), height=530, use_container_width=True)
            self.ref_year_table.columns=year_col_list # data_editorIf passed, the column changes to a string, but the column must be an int.
            st.info("""
            The checkboxes indicate which year's data will be used as a reference for future predictions. \n
            ✅ If you select 2019, future estimates will be based on 2019 data. \n
            ✅ If you select both 2018 and 2019, estimates will be based on the average of these two years.
            """)


    def select_filter(self, df_orig, key):
        start_year_mask = (df_orig['scheduled_gate_local'].dt.year>=self.start_year)
        end_year_mask = (df_orig['scheduled_gate_local'].dt.year<=self.end_year)
        df = df_orig[start_year_mask & end_year_mask]

        st.subheader("Filter Criteria")
        filter_cols = st.multiselect(
            "****",
            ["International/Domestic","primary_usage","terminal","operating_carrier_iata", "region_name","dep/arr_airport","aircraft_class","origin_airport"],
            default=["International/Domestic"],
            key=f"filter_col{key}",
        )
        
        airport_list = df["origin_airport"].unique()
        airport_df = df[df["origin_airport"].isin(airport_list)]

        filter_dict={}
        for idx, st_tab in enumerate(st.tabs(filter_cols)):
            with st_tab:
                filter_col = filter_cols[idx]
                

                # Filter data only for that airport
                
                # For each airport ratio_df generation
                ratio_df = pd.DataFrame({
                    'Category': airport_df[filter_col].value_counts().index.tolist(),
                    'Ratio': [100] * (len(airport_df[filter_col].value_counts())),
                })

                ratio_col, year_col = st.columns([0.4,0.6])
                with ratio_col:
                    # For each airport data_editor
                    editor_key = f"box_{filter_col}_{airport_list}_{idx}_{key}"
                    ratio_df = st.data_editor(
                        ratio_df,
                        hide_index=True,
                        use_container_width=True,
                        key=editor_key,
                        column_config={
                            "Category": st.column_config.TextColumn(
                                "Category",
                                disabled=True
                            ),
                            "Ratio": st.column_config.NumberColumn(
                                "📝 Sampling(%)",
                                min_value=0,
                                max_value=1000,
                                step=1,
                            )
                        }
                    )
                    # choice_df = choice_df.copy()

                with year_col:
                    # choice_df instead choice_df use
                    filtered_df=[]
                    for category, ratio_100 in zip(ratio_df['Category'], 
                                            ratio_df['Ratio']):
                        ratio=ratio_100/100
                        
                        if pd.notna(ratio) and ratio > 0:
                            category_df = airport_df[airport_df[filter_col] == category]
                            if ratio > 1:
                                whole_number = int(ratio)
                                fraction = ratio - whole_number
                                
                                repeated_df = pd.concat([category_df] * whole_number, ignore_index=True)
                                
                                if fraction > 0:
                                    sampled_df = category_df.sample(frac=fraction, random_state=42)
                                    category_result = pd.concat([repeated_df, sampled_df], ignore_index=True)
                                else:
                                    category_result = repeated_df
                            else:
                                category_result = category_df.sample(frac=ratio, random_state=42)
                            filtered_df+=[category_result]

                    airport_df = pd.concat(filtered_df)
                    ratio_df_grouped = airport_df.groupby([filter_col, "year"]).size().unstack().reset_index()
                    ratio_df_grouped = pd.merge(ratio_df, 
                                                ratio_df_grouped, 
                                                left_on="Category", 
                                                right_on=filter_col, 
                                                how="left").iloc[:,3:].fillna(0)
                    ratio_df_grouped.index=ratio_df["Category"].tolist()
                    st.dataframe(ratio_df_grouped, use_container_width=True)

                unique_list=(ratio_df[ratio_df["Ratio"]>0]["Category"].unique().tolist())
                ratio_list=(ratio_df[ratio_df["Ratio"]>0]["Ratio"].tolist())
                unique_ratio_list=[]
                for u,r in zip(unique_list, ratio_list):
                    unique_ratio_list+=[f"{u}({r}%)"]

                filter_dict[filter_col]={"ratio_df":ratio_df, "ratio_df_grouped":ratio_df_grouped, "unique_ratio_list":unique_ratio_list,"length":len(airport_df)}


        df = airport_df
        for key, value in filter_dict.items():
            filtered_length = value["length"]
            unique_ratio_list = value["unique_ratio_list"]
            st.caption(f"* **[{key}]** = {unique_ratio_list} : {filtered_length} movements")
        st.caption(f"* **[Final Filtered]**  : **{int(len(df))} movements**")
        filter_dict = pd.DataFrame(filter_dict)
        return df, filter_dict

    def analyse_performance(self):
        if "is_turnaround" not in self.df.columns:
            self.df = make_ground_time_col(self.df)

        self.df['crew']=(self.df['total_pax']*0.068)//1
        self.df['hand_carry_kg'] = self.df['total_pax']*12.3
        self.df['scheduled_gate_year'] = self.df['scheduled_gate_local'].dt.year
        self.df['scheduled_gate_month'] = self.df['scheduled_gate_local'].dt.month
        self.df['scheduled_gate_date'] = self.df['scheduled_gate_local'].dt.date
        self.df['scheduled_gate_hour'] = self.df['scheduled_gate_local'].dt.hour
        self.df['scheduled_gate_minute'] = self.df['scheduled_gate_local'].dt.minute
        self.df['move_category']='normal'
        self.df['Long/Short_haul']=np.where(self.df['flight_distance_km']>4000, 'Long_haul', 'Short_haul')
        self.df_cargo=self.df[self.df['primary_usage']=='Freight / Cargo']
        self.df_pax=self.df[self.df['primary_usage']!='Freight / Cargo']

        self.return_df = self.analyse_performance_detail(df=self.df, 
                                        category="airside", 
                                        peak_list_1=['movement'], 
                                        peak_list_2=['movement']
                                        )
        self.return_df_pax = self.analyse_performance_detail(df=self.df_pax, 
                                        category="passenger", 
                                        peak_list_1=['movement','total_pax','od_pax','tr_pax','total_seat_count'], 
                                        peak_list_2=['movement','total_pax','od_pax','tr_pax']
                                        )
        if len(self.df_cargo)>0:
            self.return_df_cargo = self.analyse_performance_detail(df=self.df_cargo, 
                                            category="cargo", 
                                            peak_list_1=['movement'], 
                                            peak_list_2=['movement']
                                            )
        else :
            self.return_df_cargo=pd.DataFrame()

        ############################################################################################################################
    def analyse_performance_detail(self, df, category, peak_list_1, peak_list_2):
        result_list=[]
        progress_text = f"{category} planning..."
        progress_bar = st.progress(0)
        total_years = self.end_year - self.start_year + 1

        for idx, year in enumerate(range(self.start_year,self.end_year+1)):
            Analysis=[]
            progress = (idx + 1) / total_years
            progress_bar.progress(progress, text=f"{progress_text} ({year} year)")
            
            df_one_year = df[df['scheduled_gate_year']==year]
            FleetMix = df_one_year.groupby(['aircraft_class'])[['total_seat_count','total_pax','movement','crew','tr_pax','maximum_payload_lb','operating_maximum_takeoff_weight_lb',
                                            'hand_carry_kg','actual_taxi_time','standing_time','ground_time','flight_distance_km']].sum().reset_index()  # ['ODfreight','total mail','Total transshipment cargo']
            
            if (category=="passenger") : 
                passenger_analysis=[
                    [f"Annual Passengers", df_one_year['total_pax'].sum()],
                    [f"Internationalpassenger", df_one_year[df_one_year['International/Domestic']=="international"]['total_pax'].sum()],
                    [f"Domesticpassenger", df_one_year[df_one_year['International/Domestic']=="domestic"]['total_pax'].sum()],

                    [f"Long_haulpassenger", df_one_year[df_one_year['Long/Short_haul']== 'Long_haul' ]['total_pax'].sum()],
                    [f"Short_haulpassenger", df_one_year[df_one_year['Long/Short_haul']== 'Short_haul' ]['total_pax'].sum()],
                    [f"regular passenger", df_one_year[df_one_year['move_category']=='normal']['total_pax'].sum()],
                    [f"Charter passenger", df_one_year[df_one_year['move_category']=='charter']['total_pax'].sum()],
                    
                    [f"annualODpassenger", df_one_year['total_pax'].sum() - df_one_year['tr_pax'].sum()],  
                    [f"Annual transfer passengers", df_one_year['tr_pax'].sum()],  
                    [f"annual crew", df_one_year['crew'].sum()],  

                    [f"Number of seats per party", FleetMix['total_seat_count'].sum()/FleetMix['movement'].sum()], 
                    [f"Passengers per car", FleetMix['total_pax'].sum()/FleetMix['movement'].sum()], 
                    [f"load factor", FleetMix['total_pax'].sum()/FleetMix['total_seat_count'].sum()], 
                    [f"transfer rate", FleetMix['tr_pax'].sum()/FleetMix['total_pax'].sum()], 
                    [f"Great transfer passengers", FleetMix['tr_pax'].sum()/FleetMix['movement'].sum()], 
                    [f"crew member", FleetMix['crew'].sum()/FleetMix['movement'].sum()], 
                    [f"Crew ratio", FleetMix['crew'].sum()/FleetMix['total_pax'].sum()], 
                    [f"Myeongdang baggage", FleetMix['hand_carry_kg'].sum()/FleetMix['total_pax'].sum()], 
                    [f"annual baggage", df_one_year['hand_carry_kg'].sum()], 
                    ]
                Analysis +=passenger_analysis


            
            if (category=="passenger") | (category=="cargo") |(category=="airside"): 
                movement_analysis=[
                    [f"Annual operation", len(df_one_year)],
                    [f"Internationaloperation", df_one_year[df_one_year['International/Domestic']=="international"]['movement'].sum()],
                    [f"Domesticoperation", df_one_year[df_one_year['International/Domestic']=="domestic"]['movement'].sum()],

                    [f"Long_hauloperation", df_one_year[df_one_year['Long/Short_haul']== 'Long_haul' ]['movement'].sum()],
                    [f"Short_hauloperation", df_one_year[df_one_year['Long/Short_haul']== 'Short_haul' ]['movement'].sum()],
                    [f"Regular operation", len(df_one_year[df_one_year['move_category']=='normal'])],
                    [f"Charter flight operation", len(df_one_year[df_one_year['move_category']=='charter'])], 
                    ]
                Analysis +=movement_analysis



            if (category=="passenger") | (category=="cargo") |(category=="airside"): 
                cargo_analysis=[
                    [f"Maximum annual takeoff weight", df_one_year['operating_maximum_takeoff_weight_lb'].sum()], 
                    [f"annualLong_haulMaximum takeoff weight", df_one_year[df_one_year['Long/Short_haul']== 'Long_haul' ]['operating_maximum_takeoff_weight_lb'].sum()],
                    [f"annualShort_haulMaximum takeoff weight", df_one_year[df_one_year['Long/Short_haul']== 'Short_haul' ]['operating_maximum_takeoff_weight_lb'].sum()],
                    
                    [f"Annual paid payload", df_one_year['maximum_payload_lb'].sum()], 
                    [f"annualLong_haulPaid payload", df_one_year[df_one_year['Long/Short_haul']== 'Long_haul' ]['maximum_payload_lb'].sum()],
                    [f"annualShort_haulPaid payload", df_one_year[df_one_year['Long/Short_haul']== 'Short_haul' ]['maximum_payload_lb'].sum()],
                    [f"Maximum takeoff weight per unit", FleetMix['operating_maximum_takeoff_weight_lb'].sum()/FleetMix['movement'].sum()], 

                    [f"Taxi average per vehicle", FleetMix['actual_taxi_time'].sum()/FleetMix['movement'].sum()], 
                    [f"Flight distance average(km)", FleetMix['flight_distance_km'].sum()/FleetMix['movement'].sum()], 
                    [f"Paid payload per unit", FleetMix['maximum_payload_lb'].sum()/FleetMix['movement'].sum()], 
                    ]
                Analysis +=cargo_analysis
            
            # Standing / Ground / Trunaround
            # Masks
            peak_mask=((df_one_year["scheduled_gate_local"].dt.hour).isin([8,9,10,11,12,13,14,15,16,17,18,19]))
            limit_trunaround_mask=(df_one_year["standing_time"]<420)
            limit_mask=(df_one_year["standing_time"]<1440*2)


            is_turnaround=(df_one_year["is_turnaround"]==1)

            # Series
            peak_limit_turn_standing = df_one_year[peak_mask & limit_trunaround_mask & is_turnaround]["standing_time"].mean()
            peak_limit_trun_ground = df_one_year[peak_mask & limit_trunaround_mask & is_turnaround]["ground_time"].mean()
            turn_standing = df_one_year[is_turnaround]["standing_time"].mean()
            turn_ground = df_one_year[is_turnaround]["ground_time"].mean()
            peak_standing = df_one_year[peak_mask & limit_mask]["standing_time"].mean()
            standing = df_one_year[limit_mask]["standing_time"].mean()
            ground = df_one_year[limit_mask]["ground_time"].mean()

            Analysis+=  [
                        ["Peak time Turnaround average cycle time(minute)", peak_limit_turn_standing],
                        ["Peak time Turnaround average ground time(minute)", peak_limit_trun_ground],
                        ["Turnaround average cycle time(minute)",turn_standing],
                        ["Turnaround average ground time(minute)",turn_ground],
                        ["(Overnight, including Sosan) Peak time average cycle time(minute)",peak_standing],
                        ["(Overnight, including Sosan) average cycle time(minute)",standing],
                        ["(Overnight, including Sosan) average ground time(minute)",ground],
                        ]

            peak_limit_turn_standing_byclass = df_one_year[peak_mask & limit_trunaround_mask & is_turnaround].groupby(["aircraft_class"])["standing_time"].agg('mean')
            peak_limit_trun_ground_byclass = df_one_year[peak_mask & limit_trunaround_mask & is_turnaround].groupby(["aircraft_class"])["ground_time"].agg('mean')
            turn_standing_byclass = df_one_year[is_turnaround].groupby(["aircraft_class"])["standing_time"].agg('mean')
            turn_ground_byclass = df_one_year[is_turnaround].groupby(["aircraft_class"])["ground_time"].agg('mean')
            peak_standing_byclass = df_one_year[peak_mask & limit_mask].groupby(["aircraft_class"])["ground_time"].agg('mean')
            standing_byclass = df_one_year[limit_mask].groupby(["aircraft_class"])["standing_time"].agg('mean')
            ground_byclass = df_one_year[limit_mask].groupby(["aircraft_class"])["ground_time"].agg('mean')

            
            # Add Analysis
            for series, col_name in zip([
                                        peak_limit_turn_standing_byclass,
                                        peak_limit_trun_ground_byclass,
                                        turn_standing_byclass,
                                        turn_ground_byclass,
                                        peak_standing_byclass,
                                        standing_byclass,
                                        ground_byclass,
                                        ],
                                        [
                                        "Peak time Turnaround average cycle time(minute)",
                                        "Peak time Turnaround average ground time(minute)",
                                        "Turnaround average cycle time(minute)",
                                        "Turnaround average ground time(minute)",
                                        "(Overnight, including Sosan) Peak time average cycle time(minute)",
                                        "(Overnight, including Sosan) average cycle time(minute)",
                                        "(Overnight, including Sosan) average ground time(minute)",
                                        ]
                                        ):
                for aircraft_class in series.index:
                    Analysis.append([
                        f"({aircraft_class}){col_name}", series[aircraft_class]
                    ])



            FleetMix['FleetMix'] = FleetMix['movement']/FleetMix['movement'].sum()
            FleetMix['seats_per_aircraft'] = FleetMix['total_seat_count']/FleetMix['movement']
            FleetMix['payload_per_aircraft'] = FleetMix['maximum_payload_lb']/FleetMix['movement']
            FleetMix['taxi_time_per_aircraft'] = FleetMix['actual_taxi_time']/FleetMix['movement'] ######
            FleetMix['standing_time_per_aircraft'] = FleetMix['standing_time']/FleetMix['movement'] ######
            FleetMix['ground_time_per_aircraft'] = FleetMix['ground_time']/FleetMix['movement'] ######
            # FleetMix['aircraft_age_month_per_aircraft'] = FleetMix['aircraft_age_months']/FleetMix['movement'] ######
            FleetMix['flight_distance_km_per_aircraft'] = FleetMix['flight_distance_km']/FleetMix['movement'] ######

            FleetMix['load_factor'] = FleetMix['total_pax']/FleetMix['total_seat_count']
            FleetMix['tr_ratio'] = FleetMix['tr_pax']/FleetMix['total_pax'] 
            FleetMix['crew_ratio'] = FleetMix['crew']/FleetMix['total_pax'] 
            FleetMix['pax_per_aircraft'] = FleetMix['seats_per_aircraft']*FleetMix['load_factor']
            FleetMix['trpax_per_aircraft'] = FleetMix['tr_pax']/FleetMix['movement'] 
            FleetMix['crew_per_aircraft'] = FleetMix['crew']/FleetMix['movement']
            FleetMix['hand_carry_per_pax'] = FleetMix['hand_carry_kg']/FleetMix['total_pax']
            FleetMix['mtow_per_aircraft'] = FleetMix['operating_maximum_takeoff_weight_lb']/FleetMix['movement']

            
            for i in range(len(FleetMix)):

                if (category=="passenger") | (category=="cargo") |(category=="airside"): 
                    Analysis.append([f"({FleetMix['aircraft_class'][i]})Number of flights", FleetMix['movement'][i]]) 
                    Analysis.append([f"({FleetMix['aircraft_class'][i]})mixing ratio", FleetMix['FleetMix'][i]]) 

                    Analysis.append([f"({FleetMix['aircraft_class'][i]})Maximum takeoff weight per unit", FleetMix['mtow_per_aircraft'][i]]) 
                    Analysis.append([f"({FleetMix['aircraft_class'][i]})Taxi average per vehicle", FleetMix['taxi_time_per_aircraft'][i]]) 
                    Analysis.append([f"({FleetMix['aircraft_class'][i]})Flight distance average(km)", FleetMix['flight_distance_km_per_aircraft'][i]]) 
                    Analysis.append([f"({FleetMix['aircraft_class'][i]})Paid payload per unit", FleetMix['payload_per_aircraft'][i]]) 

                if (category=="passenger"): 
                    Analysis.append([f"({FleetMix['aircraft_class'][i]})Number of seats per party", FleetMix['seats_per_aircraft'][i]]) 
                    Analysis.append([f"({FleetMix['aircraft_class'][i]})Passengers per car", FleetMix['pax_per_aircraft'][i]]) 
                    Analysis.append([f"({FleetMix['aircraft_class'][i]})load factor", FleetMix['load_factor'][i]]) 
                    Analysis.append([f"({FleetMix['aircraft_class'][i]})transfer rate", FleetMix['tr_ratio'][i]]) 
                    Analysis.append([f"({FleetMix['aircraft_class'][i]})Great transfer passengers", FleetMix['trpax_per_aircraft'][i]]) 
                    Analysis.append([f"({FleetMix['aircraft_class'][i]})crew member", FleetMix['crew_per_aircraft'][i]]) 
                    Analysis.append([f"({FleetMix['aircraft_class'][i]})Crew ratio", FleetMix['crew_ratio'][i]]) 
                    Analysis.append([f"({FleetMix['aircraft_class'][i]})Myeongdang baggage", FleetMix['hand_carry_per_pax'][i]]) 


            
            if len(df_one_year) >100 : 
                DepartureArrival = ['all',"d","a"]
                # The reason for setting two start and end points is as follows.
                # 1, 30If set to 1~30on average SBRThis comes out
                starting point = 140 
                end point = 160
                df_glance = df_one_year.groupby(['scheduled_gate_date','scheduled_gate_hour'])[peak_list_1].sum().sort_values(by=peak_list_1[0], ascending=False).reset_index()
                df_glance2 = df_one_year.groupby(['scheduled_gate_date','scheduled_gate_hour','flight_io'])[peak_list_1].sum().sort_values(by=peak_list_1[0], ascending=False).reset_index()

                Peak_list = []
                Peak_list.append([f"SBR/start(Nth)",f"{starting point}th"]) 
                Peak_list.append([f"SBR/end(Nth)",f"{end point}th"]) 
                for standard in (peak_list_1):
                    for Departure and Arrival in (DepartureArrival):
                        H_Factor, D_Factor, PHF, SUM = H_D(df_one_year, Standard, departure and arrival)
                        Peak_list.append([f"ADPM/{standard}/{Departure and Arrival}/Hfactor",H_Factor]) 
                        Peak_list.append([f"ADPM/{standard}/{Departure and Arrival}/Dfactor",D_Factor]) 
                        Peak_list.append([f"ADPM/{standard}/{Departure and Arrival}/PHF",PHF]) 
                        Peak_list.append([f"ADPM/{standard}/{Departure and Arrival}/peak hour",PHF*SUM]) 
                        if Departure and Arrival=='all':
                            df_ranking = df_glance.nlargest(End point, reference point)[Start point-1:End point]
                            entire = df_glance[standard].sum()
                            Nth = df_glance.sort_values(standard, ascending=False)
                        else:
                            df_glance3 = df_glance2[df_glance2['flight_io'] == Departure and Arrival]
                            df_ranking = df_glance3.nlargest(End point, reference point)[Start point-1:End point]
                            entire = df_glance3[standard].sum()
                            Nth = df_glance2.sort_values(standard, ascending=False)
                        peak = df_ranking[standard].sum() / (End point-start point+1)
                        Peak_list.append([f"SBR/{standard}/{Departure and Arrival}/PHF",peak/entire]) 
                        Peak_list.append([f"SBR/{standard}/{Departure and Arrival}/peak hour",peak]) 
                        Peak_list.append([f"ADPM_SBR/{standard}/{Departure and Arrival}/peak hour_Nth",len(Nth[Nth[standard] > PHF*SUM]) if PHF!=0 else 0]) 
                Analysis += Peak_list
                result = pd.DataFrame(Analysis)

                result.columns = ['variable name','result']
                middle direction = []
                for Target in (peak_list_2):
                    for standard in (['ADPM']): # ['ADPM','SBR']
                        if (Target !='movement') & (Target !='total_pax'):
                            middle direction molecule = result.loc[result['variable name']==f'{standard}/{Target}/all/peak hour']['result'].values[0] 
                            middle directional denominator = result.loc[result['variable name']==f'{standard}/total_pax/all/peak hour']['result'].values[0] 
                            middle direction.append([f"{standard}/{Target}/all/Middle direction coefficient",middle direction molecule/middle directional denominator if middle directional denominator !=0 else 0]) 
                        for Departure and Arrival in (["d","a"]):
                            middle direction molecule = result.loc[result['variable name']==f'{standard}/{Target}/{Departure and Arrival}/peak hour']['result'].values[0] 
                            middle directional denominator = result.loc[result['variable name']==f'{standard}/{Target}/all/peak hour']['result'].values[0] 
                            middle direction.append([f"{standard}/{Target}/{Departure and Arrival}/Middle direction coefficient",middle direction molecule/middle directional denominator if middle directional denominator !=0 else 0]) 
                Analysis += middle direction
            else : 
                pass

            Analysis.append([f"Airline placement", list(df_one_year['operating_carrier_id'].unique())]) 
            result = pd.DataFrame(Analysis)
            result.columns = ['variable name',year]
            result=result.set_index('variable name')
            result_list+=[result]
        result_df = pd.concat(result_list, axis=1)
        return result_df


    def virtual_airport_block(self):
        st.subheader("Virtual Aiport Builder")
        c1, _ = st.columns([0.2,0.8])
        with c1:
            airport_number = st.number_input(
                    "**Num of Source Airport**",
                    value=2,
                    min_value=1, 
                    max_value=10,
                    key="Airport Number",
                )
        data_source_list=[]
        open_year_list=[]
        df_orig_list=[]
        iata_code_list=[]
        icao_code_list=[]

        tab_name_list = [f"**Airport #{idx+1}**" for idx in range(airport_number)]+["**→ Demand Mix**"] +["**→ Select Location**"] 
        tab_list = st.tabs(tab_name_list)
        for idx, tab in enumerate(tab_list):
            with tab:
                if tab_name_list[idx]=="**→ Demand Mix**":
                    virtual_demand_tab=tab
                    continue
                if tab_name_list[idx]=="**→ Select Location**":
                    virtual_location_tab=tab
                    continue

                iata_code, icao_code, data_source, open_year, df_orig = self.select_airport(key=f"{self.real_virtual}*^*{idx}", multiple_airport=True)
                data_source_list+=[data_source]
                # open_year_list+=[open_year.set_index("year").rename({"total_pax":f"Source #{idx+1}"}, axis=1)]
                open_year_list+=[open_year.rename({"total_pax":f"Source #{idx+1}"}, axis=1)]
                # df_orig["origin_airport"]=iata_code
                df_orig_list+=[df_orig]
                iata_code_list+=[iata_code]
                icao_code_list+=[icao_code]
        self.data_source = data_source_list[0]
        self.open_year = pd.concat(open_year_list, axis=1)
        self.open_year["total_pax"]=self.open_year.sum(axis=1)
        self.open_year = self.open_year.reset_index()



        self.df_orig = pd.concat(df_orig_list, axis=0)
        self.iata_code_list=iata_code_list
        self.icao_code_list=icao_code_list
        self.iata_code=None
        self.icao_code=None
        with virtual_demand_tab:
            c1,c2=st.columns(2)
            seat_year = self.df_orig.groupby(["year","origin_airport"])["total_seat_count"].agg("sum").unstack().sort_index(ascending=False)//1
            seat_year["total_seat_count"]=seat_year.sum(axis=1)
            min_year=seat_year.index.min()
            pax_hist_year=self.open_year[["year","total_pax"]]
            pax_hist_year = pax_hist_year[pax_hist_year["year"]<min_year]
            pax_hist_year = pax_hist_year.set_index("year")
            pax_year = self.df_orig.groupby(["year","origin_airport"])["total_pax"].agg("sum").unstack().sort_index(ascending=False)//1
            pax_year = pd.concat([pax_year, pax_hist_year]).sort_index(ascending=False)//1
            pax_year["total_pax"]=pax_year.sum(axis=1)
            pax_year["load_factor(%)"]=pax_year["total_pax"]/seat_year["total_seat_count"]
            with c1:
                st.subheader("Seat Count")
                st.dataframe(seat_year, use_container_width=True)
            with c2:
                st.subheader("Passenger & Load-Factor(%)")
                st.dataframe(pax_year, use_container_width=True)

            self.open_year = pax_year.drop("load_factor(%)", axis=1).copy()



        with virtual_location_tab:
            st.subheader("Select Virtual Airport Location")

            # base map
            first_id = self.iata_code_list[0]
            first_row = self.df_airport[self.df_airport["airport_id"] == first_id]
            init_lat = first_row["lat"].values[0]
            init_lon = first_row["lon"].values[0]

            # map creation (CartoDB dark_matter style)
            m = folium.Map(
                location=[init_lat, init_lon],
                zoom_start=4,
                # tiles='CartoDB dark_matter'
            )

            m.add_child(folium.LatLngPopup())

            # Add airport point
            for airport_id in self.iata_code_list:
                row = self.df_airport[self.df_airport["airport_id"] == airport_id]
                if not row.empty:
                    lat = row["lat"].values[0]
                    lon = row["lon"].values[0]
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=6,
                        color='#FF0066',
                        fill=True,
                        fill_color='#FF0066',
                        fill_opacity=0.7,
                        popup=airport_id,
                        tooltip=airport_id
                    ).add_to(m)

            st_data = st_folium(m, width="100%", height=700)

                
            # Extract clicked coordinates and display country code
            if st_data and st_data['last_clicked']:
                self.airport_lat = st_data['last_clicked']['lat']
                self.airport_long = st_data['last_clicked']['lng']

                # Get country code
                from geopy.geocoders import Nominatim   
                geolocator = Nominatim(user_agent="virtual-airport")
                location = geolocator.reverse((self.airport_lat, self.airport_long), language='en')
                try :
                    country_code = location.raw['address'].get('country_code', '').upper()
                    st.info(f"""
                            Country Code: {country_code} \n
                            latitude {self.airport_lat:.7f}, longitude {self.airport_long:.7f} \n
                            """)




                    selected_airport=self.df_airport[self.df_airport["country_code"]==country_code]
                    self.country=selected_airport["country_name"].replace(self.country_mapper).values[0]
                    self.country_code=selected_airport["country_code"].replace(self.country_mapper).values[0]
                except:
                    st.write("please select other sites")            




        if st.button(f"✅ Apply Changes", type="primary", key=f"changes"):
            st.rerun()




        # if len(self.df_cargo)==0:
        #     st.write("break at total")
        #     self.return_df_cargo = pd.DataFrame()
        #     return None


        # result_list=[]
        # progress_text = "Processing Cargo Flight..."
        # progress_bar = st.progress(0)
        # total_years = self.end_year - self.start_year + 1
        # for idx, year in enumerate(range(self.start_year,self.end_year+1)):
        #     progress = (idx + 1) / total_years
        #     progress_bar.progress(progress, text=f"{progress_text} ({year} year)")
        #     df_cargo = self.df_cargo[self.df_cargo['scheduled_gate_year']==year]
        #     if len(df_cargo)==0:
        #         st.write("break at year")
        #         break

        #     FleetMix = df_cargo.groupby(['aircraft_class'])[['total_seat_count','total_pax','movement','crew','tr_pax','maximum_payload_lb','operating_maximum_takeoff_weight_lb','hand_carry_kg','actual_taxi_time','standing_time','ground_time','flight_distance_km']].sum().reset_index()  # ['ODfreight','total mail','Total transshipment cargo']


        #     Analysis = [
        #         # operation
        #         [f"Annual operation", len(df_cargo)],
        #         [f"Internationaloperation", df_cargo[df_cargo['International/Domestic']=="international"]['movement'].sum()],
        #         [f"Domesticoperation", df_cargo[df_cargo['International/Domestic']=="domestic"]['movement'].sum()],

        #         [f"Long_hauloperation", df_cargo[df_cargo['Long/Short_haul']== 'Long_haul' ]['movement'].sum()],
        #         [f"Short_hauloperation", df_cargo[df_cargo['Long/Short_haul']== 'Short_haul' ]['movement'].sum()],
        #         [f"Regular operation", len(df_cargo[df_cargo['move_category']=='normal'])],
        #         [f"Charter flight operation", len(df_cargo[df_cargo['move_category']=='charter'])],
                
        #         # freight
        #         [f"annual baggage", df_cargo['hand_carry_kg'].sum()], 
        #         [f"Maximum annual takeoff weight", df_cargo['operating_maximum_takeoff_weight_lb'].sum()], 
        #         [f"annualLong_haulMaximum takeoff weight", df_cargo[df_cargo['Long/Short_haul']== 'Long_haul' ]['operating_maximum_takeoff_weight_lb'].sum()],
        #         [f"annualShort_haulMaximum takeoff weight", df_cargo[df_cargo['Long/Short_haul']== 'Short_haul' ]['operating_maximum_takeoff_weight_lb'].sum()],
                
        #         [f"Annual paid payload", df_cargo['maximum_payload_lb'].sum()], 
        #         [f"annualLong_haulPaid payload", df_cargo[df_cargo['Long/Short_haul']== 'Long_haul' ]['maximum_payload_lb'].sum()],
        #         [f"annualShort_haulPaid payload", df_cargo[df_cargo['Long/Short_haul']== 'Short_haul' ]['maximum_payload_lb'].sum()],

        #         [f"Maximum takeoff weight per unit", FleetMix['operating_maximum_takeoff_weight_lb'].sum()/FleetMix['movement'].sum()], 

        #         [f"Taxi average per vehicle", FleetMix['actual_taxi_time'].sum()/FleetMix['movement'].sum()], 
        #         [f"Flight distance average(km)", FleetMix['flight_distance_km'].sum()/FleetMix['movement'].sum()], 
        #         [f"Paid payload per unit", FleetMix['maximum_payload_lb'].sum()/FleetMix['movement'].sum()], 
        #     ]


        #     Analysis.append([f"Airline placement", list(df_cargo['operating_carrier_id'].unique())]) 
        #     result = pd.DataFrame(Analysis)
        #     result.columns = ['variable name',year]
        #     result=result.set_index('variable name')
        #     result_list+=[result]

        # self.return_df_cargo=pd.concat(result_list, axis=1)


    def apply_return_df_to_excel(self):
        with pd.ExcelWriter('data/raw/excel_utility/In production_copy.xlsx', mode='a', if_sheet_exists='replace') as writer:
            self.final_predict.to_excel(writer, sheet_name=f"Final ({self.y_variable})")
            self.reg_df.to_excel(writer, sheet_name=f"Reg ({self.y_variable})")
            self.choice_df.to_excel(writer, sheet_name=f"Country Ratio ({self.y_variable})")
            
            self.filter_dict.to_excel(writer, sheet_name='filter_history')
            self.return_df.to_excel(writer, sheet_name='raw_data_airside')
            self.return_df_pax.to_excel(writer, sheet_name='raw_data_pax')
            self.return_df_cargo.to_excel(writer, sheet_name='raw_data_cargo')
            self.ref_year_table.to_excel(writer, sheet_name='ref', 
                                    startrow=11,  # A6of row location (0starting from)
                                    startcol=0,  # Aheat (0starting from) 
                                    ) # index exclude columns



        workbook = load_workbook('data/raw/excel_utility/In production_copy.xlsx')
        sheet = workbook['ref']

        sheet[f'A2'] = "iata_code"; sheet[f'B2'] = str(self.iata_code)
        sheet[f'A3'] = "start_year"; sheet[f'B3'] = self.start_year
        sheet[f'A4'] = "end_year"; sheet[f'B4'] = self.end_year
        sheet[f'A5'] = "predict_end_year"; sheet[f'B5'] = self.predict_end_year

        workbook.save('data/raw/excel_utility/In production_copy.xlsx')


    @st.fragment
    def generate_schedule(self):
        ref_year_col, pred_year_col, _, _ = st.columns(4)
        with ref_year_col:
            ref_year = st.number_input(
                "Reference Year",
                min_value=self.start_year,
                max_value=self.end_year,
                value=2019,
                step=1
            )

        with pred_year_col:
            pred_year = st.number_input(
                "Generate Schedule",
                min_value=self.end_year+1,
                max_value=self.predict_end_year,
                value=self.end_year+1,
                step=1
            )

            import time as tm
            tm.sleep(3)
        start_date = f"{pred_year}-01-01"
        end_date = f"{pred_year}-12-31"
        st.write(f"From: {start_date} To: {end_date}")
        st.divider()
        ref_df = self.df[self.df["scheduled_gate_local"].dt.year==ref_year].sort_values(by="scheduled_gate_local").reset_index(drop=True)

        st.header(f"Reference Year : {ref_year}")
        st.dataframe(ref_df)


        st.header(f"Predicted Year : {pred_year}")
        pred_df = ref_df.copy()
        from datetime import datetime
        diff_days = (datetime(pred_year, 1, 1) - datetime(ref_year, 1, 1)).days
        pred_df["scheduled_gate_local"] = pred_df["scheduled_gate_local"]+pd.to_timedelta( diff_days,unit="d")
        pred_df["actual_gate_local"] = pred_df["actual_gate_local"]+pd.to_timedelta( diff_days,unit="d")
        pred_df["scheduled_gate_local"] = pred_df["scheduled_gate_local"]+pd.to_timedelta( diff_days,unit="d")
        
        st.dataframe(pred_df)
        st.write(len(pred_df))

        import io
        buffer = io.BytesIO()
        pred_df.to_parquet(buffer)
        # Create download button
        st.download_button(
            label="Download Schedule",
            data=buffer.getvalue(),
            file_name='schedule.parquet',
            mime='application/octet-stream'
        )

    def apply_css(self):
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

    def demand_selector(self, use_data="Cirium_Status", y_variable="total_pax"):
        if use_data==self.data_source:
            demand = self.df.groupby(["year"])["total_pax"].agg("sum").to_frame()
            demand = demand.rename({"total_pax":"total_pax"}, axis=1)   
            
            aci_demand = self.aci_airport[self.aci_airport["airport_code_iata"]==self.iata_code].groupby(["year"])["total_passenger"].agg("sum").to_frame().sort_index()
            st.write(demand)
            # if self.data_source=="Cirium":
            #     ratio = cirium_demand.loc[2018:2019].sum()/aci_demand.loc[2018:2019].sum()
            #     aci_demand = aci_demand*ratio//1
            #     demand=pd.concat([aci_demand.loc[:2017],cirium_demand.loc[2018:]])
            # elif (self.data_source=="OAG")|(self.data_source=="Direct"):
        elif use_data=="ACI":
            aci_demand = self.aci_airport[self.aci_airport["airport_code_iata"]==self.iata_code].groupby(["year"])[y_variable].agg("sum").to_frame()
            demand=aci_demand
        elif use_data=="OPEN_DATA":
            # self.start_yearfrom self.end_yearFilter only data corresponding to the period up to
            start_year_mask = (self.df_orig['scheduled_gate_local'].dt.year >= self.start_year)
            end_year_mask = (self.df_orig['scheduled_gate_local'].dt.year <= self.end_year)
            filtered_df_orig = self.df_orig[start_year_mask & end_year_mask]
            
            start_year_mask_after = (self.df['scheduled_gate_local'].dt.year >= self.start_year)
            end_year_mask_after = (self.df['scheduled_gate_local'].dt.year <= self.end_year)
            filtered_df = self.df[start_year_mask_after & end_year_mask_after]
            
            before_pax = filtered_df_orig["total_pax"].sum()
            after_pax = filtered_df["total_pax"].sum() 


            filtered_ratio = after_pax/before_pax
            # demand = (self.open_year).set_index("year").sort_index()
            demand = (self.open_year).sort_index()


            demand["total_pax"] = (demand["total_pax"]*filtered_ratio) //1


        elif use_data=="DIRECT_INPUT":
            demand=pd.DataFrame(index=[year for year in range(2018,2025)], columns=[y_variable]).fillna(0)
        return demand

    def regression_predict(self, dataset_category, y_variable):
        var_num, _= st.columns([0.15,0.85])
        with var_num:
            variable_number = st.number_input("**How Many X-Variables**", min_value=2,max_value=10)

        tab_list = ["**variable " + str(i+1) + "**" for i in range(variable_number)] + ["**→ RESULT**"]
        variable_df=[]
        already_selected_list=[]
        for tab_name, tab in zip(tab_list, st.tabs(tab_list)):
            with tab:
                if tab_name!="**→ RESULT**":
                    variable_series, variable = economic_df(df=self.econ_df, country=self.country, already_selected_list = already_selected_list, ref_country="G20 emerging economies", key=tab_name)
                    st.dataframe(variable_series, hide_index=True)
                    variable_df+=[variable_series]
                    already_selected_list+=[variable]

                elif tab_name=="**→ RESULT**": 
                    variable_df=pd.concat(variable_df)
                    variable_df=variable_df.set_index("Variable").iloc[:,3:]
                    variable_df.columns = [int(col) for col in variable_df.columns]
                    variable_df=variable_df.T
                    variable_df=variable_df.astype(float)
                    variable_df = variable_df.dropna()
                    demand_df = self.demand_selector(use_data=dataset_category, y_variable=y_variable)




                    demand_start_year = demand_df.index.min()
                    variable_start_year = variable_df.index.min()
                    start_year = max(demand_start_year, variable_start_year)
                    demand_df = demand_df.loc[start_year:][["total_pax"]]


                    if dataset_category=="DIRECT_INPUT":
                        variable_col , direct_col, predict_col = st.columns([0.6,0.2,0.2])
                    else:
                        variable_col , predict_col = st.columns([0.8,0.2])

                    if dataset_category=="DIRECT_INPUT":
                        with direct_col:
                            demand_df=st.data_editor(demand_df,  key="direct_input_regression")

                    with variable_col:
                        variable_df = variable_df.loc[start_year:self.predict_end_year]
                        variable_df_=st.data_editor(
                                                    variable_df,
                                                    use_container_width=True,
                                                    height=1800,
                                                    )
                        demand_df = pd.concat([variable_df_, demand_df], axis=1)
                    with predict_col:
                        from sklearn.linear_model import LinearRegression
                        from sklearn.metrics import r2_score
                        def create_linear_regression(demand_df, x_variables, y_variable):
                            """
                            Function to create a linear regression model and make predictions
                            
                            Parameters:
                            demand_df (DataFrame): input data frame
                            x_variables (list): Column name list of independent variables
                            
                            Returns:
                            tuple: (model, R² Data frame with added scores and predicted values)
                            """

                            # Prepare training data
                            train_mask = demand_df[y_variable].notna()
                            X_train = demand_df[train_mask][x_variables]
                            y_train = demand_df[train_mask][y_variable]
                            
                            # Prepare your data for prediction
                            predict_mask = demand_df[y_variable].isna()
                            X_predict = demand_df[predict_mask][x_variables]
                            # model training
                            model = LinearRegression()
                            model.fit(X_train, y_train)
                            
                            # Model performance evaluation
                            y_pred_train = model.predict(X_train)
                            r2 = r2_score(y_train, y_pred_train)
                            
                            # Create a regression equation
                            equation_terms = []
                            for coef, var in zip(model.coef_, x_variables):
                                equation_terms.append(f"{coef:.2f} * ({var})")
                            equation = "y = " + " + ".join(equation_terms) + f" + {model.intercept_:.2f}"
                            

                            # predict future value
                            future_predictions = model.predict(X_predict)
                            
                            # Add predicted values ​​to original dataframe
                            demand_df.loc[predict_mask, y_variable] = future_predictions//1
                            demand_df[y_variable]=demand_df[y_variable].fillna(demand_df[y_variable])
                            return model, r2, demand_df, equation, r2

                        # Example of function usage
                        x_variables=demand_df.columns[:-1].tolist()
  

                        model, r2, reg_df, equation, r2 = create_linear_regression(demand_df, x_variables, y_variable)
                        # Result output
                        st.dataframe(reg_df[[y_variable]]//1, height=1800)
                    st.info(f"""
                    📊 Regression Model:
                    {equation} \n
                    📈 R² score: {r2:.4f}
                    """)
                    
                    # data_guide = f"""
                    # [provided data] \n
                    # {self.airport_iata_code_list} This is the future demand forecast table for. \n
                    # Indicates variables and final predicted values. \n
                    # x_variables : {x_variables}\n
                    # y_variable : {y_variable} \n
                    # r_square value : {r2} >> Generally, anything above 0.85 is a good prediction level. \n
                    # equation : {equation} >> prediction formula \n

                    # [Insight]
                    # 1) How predictors affect the final prediction
                    # 2) If there are dummy variables, an explanation of why.
                    # 3) Description of trends of each predictor, etc.
                    # 4) There is a need to add qualitative variables that can include the country's political situation and national policy direction as a dummy.
                    # """
                    # answer_guide="""
                    # Please write it in report form., \n
                    # The target audience is airport workers who review airport construction and growth rates. \n
                    # """
                    # example_question=f"""
                    # 1. Explain the demand forecasting process and results. \n
                    # 2. What are some additional things to consider??
                    # """
                    # self.ai_insight(system_prompt=self.system_prompt, 
                    #             data_list=[reg_df.reset_index()], 
                    #             data_guide=data_guide,
                    #             answer_guide=answer_guide, 
                    #             example_question=example_question, 
                    #             key="regression predict")

                    return reg_df


    def airport_choice_predict(self, dataset_category, y_variable):

        c1,c2=st.columns(2)
        with c1:
            st.caption(
                f"""
                ✈️ airport_traffic : airport real passenger count \n
                🌏 country_traffic : ACI World Airport Traffic Forecast \n
                """
            )
        with c2:
            st.caption(
                f"""
                ✈️/🌏 ratio = (airport_traffic) / (country_traffic) \n
                📈 {y_variable} = (country_traffic) * (ratio)
                """
            )
        if dataset_category=="DIRECT_INPUT":
            variable_col , direct_col, predict_col = st.columns([0.6,0.2,0.2])
        else:
            variable_col , predict_col = st.columns([0.8,0.2])
        watf_series=self.aci_watf[(self.aci_watf["Metric"]==y_variable) & (self.aci_watf["Entity"]==self.country)].set_index(["Metric","Region","Entity"])//1
        watf_series=watf_series.T
        if len(watf_series.sum())==0:
            st.warning(f"Data for **{self.country}** is not available.")
            return pd.DataFrame()

        # Growth rate in the last year
        last_growth_rate = watf_series.iloc[-1] / watf_series.iloc[-2]
        for year in range(watf_series.index[-1] + 1, self.predict_end_year + 1):
            watf_series.loc[year] = int(watf_series.loc[year - 1] * last_growth_rate)


        demand_df = self.demand_selector(use_data=dataset_category, y_variable=y_variable)

        if dataset_category=="DIRECT_INPUT":
            with direct_col:
                demand_df=st.data_editor(demand_df, key="direct_input_country_choice")
        demand_df = pd.concat([demand_df,watf_series], axis=1)

        demand_df=demand_df.sort_index().loc[:self.predict_end_year]
        demand_df = demand_df[demand_df.columns[-2:]]
        demand_df.columns=["airport_traffic","country_traffic"]
        demand_df["airport_traffic"] = demand_df["airport_traffic"]//1
        demand_df["ratio"]=demand_df["airport_traffic"]/demand_df["country_traffic"]
        demand_df["ratio"]=demand_df["ratio"].fillna(method="ffill")

        with variable_col:
            choice_df = st.data_editor(
                demand_df,
                height=1500,
                column_config={
                    "ratio": st.column_config.NumberColumn(
                        "ratio",
                        help="Airport Real / Country Pred ratio",
                        min_value=0,
                        format="%.4f",
                        disabled=False
                    ),
                    # "airport_traffic": st.column_config.NumberColumn(
                    #     "airport_traffic",
                    #     disabled=True
                    # ),
                    "country_traffic": st.column_config.NumberColumn(
                        "country_traffic",
                        disabled=True
                    ),
            },
                use_container_width=True
            )

        with predict_col:
            choice_df[y_variable]=choice_df["airport_traffic"].fillna(choice_df["country_traffic"]*choice_df["ratio"])//1
            st.dataframe(choice_df[[y_variable]], height=1500)

        return choice_df



    def forecast(self):
        dataset_category = st.selectbox(
            "**which historical data use?**",
            ["OPEN_DATA",self.data_source,"DIRECT_INPUT"],
            key="dataset_category",
        )

        # dataset_category = self.data_source
        y_variable = "total_pax"

        regression, country_ratio, final_forecast=st.tabs(["**(1) Linear Regressions**","**(2) Country Ratio Model**","**(1)+(2) Final Forecast**"])
        with regression:
            reg_df =self.regression_predict(dataset_category, y_variable)
            reg_df=reg_df.rename({y_variable:"regression"}, axis=1)
        with country_ratio:
            choice_df = self.airport_choice_predict(dataset_category, y_variable)
            choice_df=choice_df.rename({y_variable:"country ratio"}, axis=1)

        with final_forecast:
            if len(choice_df)==0:
                final_predict=reg_df[["regression"]]
                final_predict["country ratio"]=0
                final_predict["Final Forecast"] = final_predict["regression"]
                reg_ratio=1
            else:
                reg_ratio = st.number_input("regression ratio", 
                                            value=0.25,  # Change default to 0.5
                                            min_value=0.0, 
                                            max_value=1.0,
                                            step=0.01,
                                            format="%.2f")
                final_predict = pd.concat([reg_df[["regression"]]//1, choice_df[["country ratio"]]//1], axis=1)
                final_predict["Final Forecast"] = (final_predict["regression"]*reg_ratio + final_predict["country ratio"]*(1-reg_ratio))//1
                final_predict = final_predict.sort_index()
            st.dataframe(final_predict.T, use_container_width=True)
            st.caption(f"* Fianl Forecast = (regression x {reg_ratio}) + (country ratio x {1-reg_ratio})")

            self.final_predict=final_predict.T
            self.reg_df=reg_df
            self.choice_df=choice_df.T
            self.y_variable=y_variable
            show_graph(df=final_predict, method='line', height=1200)


            # data_guide = f"""
            # [Dataset1 : regression]
            # {self.airport_iata_code_list} This is the future demand forecast table for.
            # Indicates variables and final predicted values.
            # y_variable : {y_variable} \n

            # [Dataset2 : country ratio]
            # {self.airport_iata_code_list} This is the future demand forecast table. This prediction method is ACI(Airport Council International) Based on the agency's forecast of future demand by country. 

            # airport_traffic : {self.airport_iata_code_list} refers to the actual demand at the airport
            # country_traffic : ACIannounced in {self.country} whole country watf(world airport traffic forecast) These are predicted figures.
            # ratio : airport_traffic / country_traffic It's a ratio
            # y_variable : {y_variable} = country_traffic * ratioIt is predicted as \n
            
            # [Dataset3 : weighted average]
            # regression Forecast and country ratio Weighted average of forecasts weighted_average calculate. regression  ratio={reg_ratio}, country_ratio ratio={1-reg_ratio}(1-regression ratio) \n
            # """
            # answer_guide="""
            # Please explain the difference in growth rate by forecast and the trend.
            # Please write it in report form., \n
            # """
            # example_question=f"""
            # 1. Please summarize this forecast into a one-page report.
            # """
            # self.ai_insight(system_prompt=self.system_prompt, 
            #             data_list=[reg_df.reset_index(), choice_df.reset_index(), final_predict.reset_index()], 
            #             data_guide=data_guide,
            #             answer_guide=answer_guide, 
            #             example_question=example_question, 
            #             key="final forecast")

    def ai_insight(self, system_prompt, data_list, data_guide, answer_guide, example_question, default_question="", key="default"):
        # Sort a list and convert it to a string (For consistent comparison)
        current_codes = ','.join(sorted(self.airport_iata_code_list))
        
        # currently selected airport_iata_codetrack
        if 'current_airport_code' not in st.session_state:
            st.session_state.current_airport_code = current_codes
        
        # airport_iata_codeCheck if has changed
        if st.session_state.current_airport_code != current_codes:
            # Initialize all relevant session state
            keys_to_delete = [k for k in st.session_state.keys() if k.startswith('messages_')]
            for k in keys_to_delete:
                del st.session_state[k]
            
            # Initialize the input fields as well
            if key in st.session_state:
                del st.session_state[key]
                
            # Update current airport codes
            st.session_state.current_airport_code = current_codes
            
            # Refresh page
            st.rerun()

            

        # data_listIf is not a list, convert it to a list
        if not isinstance(data_list, list):
            data_list = [data_list]
        
        # Convert each data frame to a string and join them
        data_string = ""
        for i, data in enumerate(data_list):
            data_string += f"\nDataset {i+1}:\n" + data.to_string(index=False) + "\n"
        
        data_guide += data_string
        
        if f"messages_{key}" not in st.session_state:
            st.session_state[f"messages_{key}"] = [
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": answer_guide},
                {"role": "system", "content": data_guide}
            ]

        def handle_input():
            user_question = st.session_state[key]  
            
            if user_question:  # Process only when there is input
                st.session_state[f"messages_{key}"].append(
                    {"role": "user", "content": user_question}
                )
                
                response = openai.ChatCompletion.create(
                    # model="gpt-3.5-turbo",
                    model="gpt-4o",

                    messages=st.session_state[f"messages_{key}"]
                )
                
                reply = response["choices"][0]["message"]["content"]
                st.session_state[f"messages_{key}"].append(
                    {"role": "assistant", "content": reply}
                )
                
                st.session_state[key] = ""  # Initialize input fields

        container = st.container(border=True)
        with container:
            try : 
                st.subheader("🌟 AI Insight")
                st.info(example_question)

                messages_container = st.empty()
                
                def display_messages():
                    with messages_container.container():
                        for message in st.session_state[f"messages_{key}"]:
                            if message["role"] == "user":
                                c1,_=st.columns([0.6,0.4])
                                with c1:
                                    st.markdown(f"""
                                    <div style="background-color: #007aff; padding: 10px; border-radius: 14px;">
                                        🤔 {message["content"]}
                                    </div>
                                    """, unsafe_allow_html=True)
                                    st.write("");st.write("")
                            elif message["role"] == "assistant":
                                _,c1=st.columns([0.05,0.95])
                                with c1:
                                    content = message["content"]
                                    st.info(content) 
                display_messages()
                st.text_input("", 
                            key=key,
                            on_change=handle_input,
                            placeholder="Ask anything...")
            except :
                st.write("Re run!!") 

    @st.fragment
    def compare_scenario(self):
        st.subheader("**Scenario Compare**")
        c1, _ = st.columns([0.2,0.8])
        with c1:
            scenario_num = st.number_input(
                            "**Howmany Scenarios**",
                            value=2,
                            min_value=1,
                            max_value=5,
                            key="scenario number",
                        )
        with_col_list = st.columns(scenario_num)

        total_scenario_file = {}
        scenario_names = set()
        for idx, with_col in enumerate(with_col_list):
            with with_col: 
                container = st.container(border=True)
                with container:
                    st.subheader(f"# Scenario {idx+1}")
                    scenario_file_object = st.file_uploader("", type=["xlsx"], key=f"scenario{idx+1}")
                    scenario_name = st.text_input(
                        "**Set Scenario Name**",
                        value=f"Scenario {idx+1}",
                        key=f"scenario_name{idx+1}",
                    )
                    if scenario_file_object is not None : 
                        sheet_names = ["filter_history", "ref", "Final (total_pax)"]  # Set desired sheet name
                        dfs = pd.read_excel(scenario_file_object, sheet_name=sheet_names)
                        filter_series=dfs["filter_history"].set_index("Unnamed: 0").loc["unique_ratio_list"]
                        ref_series=dfs["ref"][["Unnamed: 0","Unnamed: 1"]].set_index("Unnamed: 0").head(4)
                        forecast_series=dfs["Final (total_pax)"].set_index("Unnamed: 0").tail(1)
                        forecast_series.index.name=""

                        container = st.container(border=True)
                        with container:
                            st.write("**● Basic Info**")
                            for index, value in zip(ref_series.index, ref_series.values):
                                st.caption(f"**[{index}]** = {value[0]}")
                            st.write("**● Forecast Info**")
                            st.dataframe(forecast_series, hide_index=True)
                            st.write("**● Filter Info**")
                            for index, value in zip(filter_series.index, filter_series.values):
                                st.caption(f"**[{index}]** = {value}")

                    # Check for name duplicates
                    if scenario_name in scenario_names:
                        st.warning(f"'{scenario_name}' Duplicate name. Please enter a different name.")
                    else:
                        scenario_names.add(scenario_name)
                    scenario_info = {
                        "object":scenario_file_object,
                        "file_exist": scenario_file_object is not None,
                    }
                    total_scenario_file[scenario_name] = scenario_info


        if all(info["file_exist"] for info in total_scenario_file.values()):
            container = st.container(border=True)
            with container:
                df_summary=[]
                for key, value in total_scenario_file.items():
                    df_summary_one = pd.read_excel(value["object"], sheet_name="Summary")
                    df_summary_one.insert(0, "scenario_name", key)
                    df_summary+=[df_summary_one]
                df_summary = pd.concat(df_summary, axis=0)
                df_summary = df_summary[df_summary["Main Category"].notna()]
                c1,c2,c3,c4=st.columns([0.15,0.15, 0.4,0.3])
                with c1:
                    main_category = st.selectbox(
                        "**Main Category**",
                        df_summary["Main Category"].unique(),
                        key="Main Category",
                    )
                    df_summary_filtered = df_summary[df_summary["Main Category"]==main_category]
                with c2:
                    sub_category = st.selectbox(
                        "**Sub Category**",
                        df_summary_filtered["Category"].unique(),
                        key="Sub Category",
                    )
                    df_summary_filtered = df_summary_filtered[df_summary_filtered["Category"]==sub_category]
                with c3:
                    value_list = st.multiselect(
                        "**Values**",
                        df_summary_filtered["Definition"].unique(),
                        default=df_summary_filtered["Definition"].unique()[0],
                        key="value_list",
                    )
                    df_summary_filtered = df_summary_filtered[df_summary_filtered["Definition"].isin(value_list)]
                with c4:
                    scenario_list = st.multiselect(
                        "**Scenario**",
                        df_summary_filtered["scenario_name"].unique(),
                        default=df_summary_filtered["scenario_name"].unique(),
                        key="scenario_list",
                    )
                    df_summary_filtered = df_summary_filtered[df_summary_filtered["scenario_name"].isin(scenario_list)]

                st.info(
                    f"""
                💡 Main Category : **{main_category}** \n
                💡 Sub Category : **{sub_category}** \n
                💡 Values : **{value_list}** \n
                """
                )
                df_summary_show = df_summary_filtered.copy()
                df_summary_show = df_summary_show.iloc[:,6:]
                df_summary_show.index=df_summary_filtered["scenario_name"] + "(" + df_summary_filtered["Acronym"] + ")"
                if len(value_list)==1:
                    title_name = value_list[0]
                else: 
                    title_name = "Line Chart"
                fig = px.line(df_summary_show.T, title=title_name)
                fig.update_layout(xaxis_title='Year', yaxis_title='Value', height=600)
                st.plotly_chart(fig)
                st.dataframe(df_summary_show)







def economic_df(df, country, ref_country, already_selected_list, key):
    country_list = sorted(df['Country'].unique())
    variable_list = sorted(df['Variable'].unique())
    variable_list = [x for x in variable_list if x not in already_selected_list]
    oecd_country_list=df[df["Scenario"]=="OECD"]["Country"].unique().tolist()

    valid_variable_df=df[(df['Variable'].isin(variable_list))&(df['Country']==country)][["Variable","2020"]]
    # st.write(df[(df['Variable'].isin(variable_list))&(df['Country']==country)])
    variable_list=valid_variable_df[valid_variable_df["2020"].notna()]["Variable"].unique().tolist()
    for var in ['GDP per capita, constant', 'Working-age Population'][::-1]:
        if var in variable_list:
            variable_list.insert(0, variable_list.pop(variable_list.index(var)))

    variable_list=variable_list+["Dummy"]
    # Select country
    c1,c2,c3=st.columns(3)
    with c1:
        country = st.selectbox('**Predict Country**', country_list,country_list.index(country), key=f"country{key}")
    with c2:
        variable = st.selectbox('**Variable**', variable_list, index=0, key=f"variable{key}")
    # data filtering
    if variable!="Dummy":
        filtered_df = df[
            (df['Country']==country) &
            (df['Variable']==variable)
        ].head(1)
    elif variable=="Dummy":
        with c3 : 
            variable = st.text_input("Type Dummy Name", value=f"Dummy_variable", key=f"variable{key}_dummy")
        filtered_df=pd.DataFrame(data={"LOCATION":[""], "Country": [country], "Variable":[variable],"Scenario":[""]})
        for year in range(1980,2061):
            filtered_df[str(year)]=0
        

    if (len(filtered_df)==1) & (filtered_df["Scenario"].values[0]=="IMF") & (variable not in ["Working-age Population", "Total Population"]):
        with c3:
            if variable == "Inflation_consumer_price":
                oecd_country_list = [x for x in oecd_country_list if x not in ["G20 advanced economies", "Euro area(17 countries)", "G20 emerging economies", "OECD-Total","OECE and G20 total", "G7"]]
                ref_country = "Indonesia"
                ref_country = st.selectbox('**Reference Country**', oecd_country_list, oecd_country_list.index(ref_country), key=f"ref{key}")

            else : 
                ref_country = st.selectbox('**Reference Country**', oecd_country_list, oecd_country_list.index(ref_country), key=f"ref{key}")
    
            ref_df = df[
                (df['Country']==ref_country) &
                (df['Variable']==variable) & (df["Scenario"]=="OECD")
            ]
        def find_last_notnull_year(df):
            for year in range(2029, 1999, -1):
                if not pd.isna(df[str(year)].values[0]):
                    return year
        ref_start=find_last_notnull_year(filtered_df)

        future_years = [str(year) for year in range(ref_start+1, 2061)]


        ratio = float(filtered_df[str(ref_start)].values[0]) / float(ref_df[str(ref_start)].values[0])

        st.caption(f"""
        * For trends before **{ref_start}**, referred to [IMF] **{country}** trend \n
        * For trends after **{ref_start+1}**, referred to [OECD] **{ref_country}** trend \n
        """)
        
        for year in future_years:
            filtered_df[year] = ref_df[year].values[0] * ratio
    # Data reconstruction
    plot_df = filtered_df.melt(
        id_vars=['Scenario'], 
        value_vars=[col for col in filtered_df.columns if str(col).isdigit()],
        var_name='Year',
        value_name='Value'
    )
    plot_df['Year'] = plot_df['Year'].astype(int)

    # Plotly Graph creation
    c1, c2=st.columns(2)
    with c1:
        st.write(variable)
        fig = px.line(plot_df, x='Year', y='Value', color='Scenario')
        fig.update_layout(xaxis_title='Year', hovermode='x')
        st.plotly_chart(fig)
    with c2:
        st.write(f"{variable} Growth rate(%)")
        plot_df['growth_rate'] = plot_df['Value'].astype(float) / plot_df['Value'].shift(1).astype(float) - 1
        fig = px.bar(plot_df, x='Year', y='growth_rate', color='Scenario')
        fig.update_layout(xaxis_title='Year', hovermode='x')
        st.plotly_chart(fig)
    return filtered_df, variable


@st.fragment
def download_excel(scenario_name="Analysis"):
    shutil.copy('data/raw/excel_utility/In production_copy.xlsx', 'data/raw/excel_utility/download.xlsx')

    # Create download button
    with open('data/raw/excel_utility/download.xlsx', 'rb') as f:
        excel_data = f.read()

    st.download_button(
        label="📥 Excel Download",
        data=excel_data,
        file_name=f'{scenario_name}.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

def show_graph(df, method, height=650):
    if (method=='bar_group') | (method=='bar_stack'):
        fig = px.bar(
        data_frame=df, 
        x=df.index,
        y=df.columns.tolist(),
        barmode=method.split('_')[1]
        )
        fig.update_layout(
            xaxis_title="",
            yaxis_title="",
            yaxis_tickformat=",",
        )
        fig.update_traces(
            texttemplate="%{y:,}",
            textposition="outside",
            textfont=dict(color="white", family="Arial"),
        )
    if method=='line':
        fig = go.Figure(layout=go.Layout(template="plotly_dark"))
        for col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col],
                    mode="lines",
                    name=col,
                    line=dict(dash="dashdot", shape="spline"),
                )
            )
        fig.update_layout(
            legend=dict(
                # title="",
                orientation="h",
                yanchor="bottom",
                y=1.05,
                xanchor="right",
                x=1,
            )
        )
    fig.update_layout(height=height)
    st.plotly_chart(fig, use_container_width=True)

def H_D(df, agg_col='movement', flight_io='all', peak_date_number=10, n_min=15, default_cols=['movement','total_seat_count','total_pax','od_pax','tr_pax']):
    if flight_io!='all':
        df2 = df[df['flight_io'] == flight_io]
    else:
        df2 = df
    target_date = df2.groupby(['scheduled_gate_date'])[default_cols].sum().sort_values(by=agg_col, ascending=False)[:peak_date_number].index
    target_date_H_factor = []
    if len(df2) ==  0:
        return 0, 0, 0, 0
    elif len(df2) < peak_date_number:
        peak_date_number = len(df2)
    for n in range(peak_date_number):
        df_target_date = df2[df2['scheduled_gate_date'] == target_date[n]]
        df_target_date['n_min'] = round(df_target_date['scheduled_gate_hour'] + (df_target_date['scheduled_gate_minute']//n_min) * (n_min/60),2)
        df_cumulative = pd.DataFrame([{'n_min':0}])
        for i in range(24*int(60/n_min)):
            df_cumulative.loc[i] = [round(i*(n_min/60),2)]
        df_cumulative = df_cumulative.merge(df_target_date.groupby(['n_min'])[agg_col].sum().reset_index(name=agg_col), on='n_min', how='left').fillna(0)
        df_cumulative[f'cumulative_{agg_col}']=0
        for i in range(int(60/n_min)):
            df_cumulative[f'cumulative_{agg_col}'] += df_cumulative[agg_col].shift(i)
        target_date_H_factor.append(df_target_date[agg_col].sum()/df_cumulative[f'cumulative_{agg_col}'].max())
    H_Factor = sum(target_date_H_factor)/peak_date_number
    df_month = df2.groupby(['scheduled_gate_month'])[default_cols].sum().reset_index()
    if len(df_month) !=12:
        return 0, 0, 0, 0
    df_month['scheduled_gate_date'] = [31,28,31,30,31,30,31,31,30,31,30,31]
    for col in (list(df_month.columns[1:])):
        exec(f"df_month[col+'_daily average'] = df_month[col] / df_month['scheduled_gate_date']")
    
    df_peak month = df_month.nlargest(2,[agg_col+'_daily average'])
    Peak daily average number of flights = df_peak month[agg_col].sum()/df_peak month['scheduled_gate_date'].sum()
    D_Factor = df2[agg_col].sum()/Peak daily average number of flights
    PHF = (H_Factor*D_Factor)**(-1)
    SUM = df2[agg_col].sum()
    return H_Factor, D_Factor, PHF, SUM



import base64
def set_bg_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()

    css = f"""
    <style>
    [data-testid="stAppViewContainer"], 
    [data-testid="stHeader"], 
    [data-testid="stSidebar"], 
    [data-testid="stToolbar"] {{
        background-image: url("data:image/png;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """

    st.markdown(css, unsafe_allow_html=True)
