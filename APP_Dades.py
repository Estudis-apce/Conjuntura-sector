import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import base64
from streamlit_option_menu import option_menu
import io
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import plotly.graph_objects as go
import matplotlib.colors as colors
# import streamlit.components.v1 as components
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from datetime import datetime
import plotly.graph_objs as go
import json

path = ""

st.set_page_config(
    page_title="Conjuntura de sector",
    page_icon="""data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAMAAAAoLQ9TAAAA1VBMVEVHcEylpKR6eHaBgH9GREGenJxRT06op6evra2Qj49kYWCbmpqdnJyWlJS+vb1CPzyurKyHhYWMiYl7eXgOCgiPjY10cnJZV1WEgoKCgYB9fXt
    /fHyzsrGUk5OTkZGlo6ONioqko6OLioq7urqysbGdnJuurazCwcHLysp+fHx9fHuDgYGJh4Y4NTJcWVl9e3uqqalcWlgpJyacm5q7urrJyMizsrLS0tKIhoaMioqZmJiTkpKgn5+Bf36WlZWdnJuFg4O4t7e2tbXFxMR3dXTg39/T0dLqKxxpAAAAOHRSTlMA/WCvR6hq/
    v7+OD3U9/1Fpw+SlxynxXWZ8yLp+IDo2ufp9s3oUPII+jyiwdZ1vczEli7waWKEmIInp28AAADMSURBVBiVNczXcsIwEAVQyQZLMrYhQOjV1DRKAomKJRkZ+P9PYpCcfbgze+buAgDA5nf1zL8TcLNamssiPG/
    vt2XbwmA8Rykqton/XVZAbYKTSxzVyvVlPMc4no2KYhFaePvU8fDHmGT93i47Xh8ijPrB/0lTcA3lcGQO7otPmZJfgwhhoytPeKX5LqxOPA9i7oDlwYwJ3p0iYaEqWDdlRB2nkDjgJPA7nX0QaVq3kPGPZq/V6qUqt9BAmVaCUcqEdACzTBFCpcyvFfAAxgMYYVy1sTwAAAAASUVORK5CYII=""",
    layout="wide"
)
def load_css_file(css_file_path):
    with open(css_file_path) as f:
        return st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css_file(path + "main.css")

# left_col, right_col, margin_right = st.columns((0.25, 1, 0.25))
# with right_col:
#     with open(path + "APCE_mod.png", "rb") as f:
#         data_uri = base64.b64encode(f.read()).decode("utf-8")
#     markdown = f"""
#     <div class="image">
#     <img src="data:image/png;base64, {data_uri}" alt="image" />
#     </div>
#     """
#     st.markdown(markdown, unsafe_allow_html=True)


# Creating a dropdown menu with options and icons, and customizing the appearance of the menu using CSS styles.
left_col, right_col, margin_right = st.columns((0.15, 1, 0.15))
with right_col:
    selected = option_menu(
        menu_title=None,  # required
        options=["Espanya","Catalunya","Províncies i àmbits", "Comarques", "Municipis", "Districtes de Barcelona"],  # Dropdown menu
        icons=[None, None, "map", "map","house-fill", "house-fill"],  # Icons for dropdown menu
        menu_icon="cast",  # optional
        default_index=0,  # optional
        orientation="horizontal",
        styles={
            "container": {"padding": "0px important!", "background-color": "#fcefdc", "align":"center", "overflow":"hidden"},
            "icon": {"color": "#bf6002", "font-size": "17px"},
            "nav-link": {
                "font-size": "17px",
                "text-align": "center",
                "font-weight": "bold",
                "color":"#363534",
                "padding": "5px",
                "--hover-color": "#fcefdc",
                "background-color": "#fcefdc",
                "overflow":"hidden"},
            "nav-link-selected": {"background-color": "#de7207"}
            })

##@st.cache_data(show_spinner="**Carregant les dades... Esperi, siusplau**", max_entries=500)
@st.cache_resource
def import_data(trim_limit, month_limit):
    with open('DT_simple.json', 'r') as outfile:
        list_of_df = [pd.DataFrame.from_dict(item) for item in json.loads(outfile.read())]
    DT_terr= list_of_df[0].copy()
    DT_mun= list_of_df[1].copy()
    DT_mun_aux= list_of_df[2].copy()
    DT_mun_aux2= list_of_df[3].copy()
    DT_mun_aux3= list_of_df[4].copy()
    DT_dis= list_of_df[5].copy()
    DT_terr_y= list_of_df[6].copy()
    DT_mun_y= list_of_df[7].copy()
    DT_mun_y_aux= list_of_df[8].copy()
    DT_mun_y_aux2= list_of_df[9].copy()
    DT_mun_y_aux3= list_of_df[10].copy()
    DT_dis_y= list_of_df[11].copy()
    DT_monthly= list_of_df[12].copy()
    DT_monthly["Fecha"] = DT_monthly["Fecha"].astype("datetime64[ns]")
    maestro_mun= list_of_df[13].copy()
    maestro_dis= list_of_df[14].copy()


    DT_monthly = DT_monthly[DT_monthly["Fecha"]<=month_limit]
    DT_terr = DT_terr[DT_terr["Fecha"]<=trim_limit]
    DT_mun = DT_mun[DT_mun["Fecha"]<=trim_limit]
    DT_mun_aux = DT_mun_aux[DT_mun_aux["Fecha"]<=trim_limit]
    DT_mun_aux2 = DT_mun_aux2[DT_mun_aux2["Fecha"]<=trim_limit]
    DT_mun_aux3 = DT_mun_aux3[DT_mun_aux3["Fecha"]<=trim_limit]
    DT_mun_pre = pd.merge(DT_mun, DT_mun_aux, how="left", on=["Trimestre","Fecha"])
    DT_mun_pre2 = pd.merge(DT_mun_pre, DT_mun_aux2, how="left", on=["Trimestre","Fecha"])
    DT_mun_def = pd.merge(DT_mun_pre2, DT_mun_aux3, how="left", on=["Trimestre","Fecha"])
    DT_dis = DT_dis[DT_dis["Fecha"]<=trim_limit]
    DT_mun_y_pre = pd.merge(DT_mun_y, DT_mun_y_aux, how="left", on="Fecha")
    DT_mun_y_pre2 = pd.merge(DT_mun_y_pre, DT_mun_y_aux2, how="left", on="Fecha")
    DT_mun_y_def = pd.merge(DT_mun_y_pre2, DT_mun_y_aux3, how="left", on="Fecha")    

    return([DT_monthly, DT_terr, DT_terr_y, DT_mun_def, DT_mun_y_def, DT_dis, DT_dis_y, maestro_mun, maestro_dis])

DT_monthly, DT_terr, DT_terr_y, DT_mun, DT_mun_y, DT_dis, DT_dis_y, maestro_mun, maestro_dis = import_data("2023-10-01", "2023-12-01")

##@st.cache_data(show_spinner="**Carregant les dades... Esperi, siusplau**", max_entries=500)
@st.cache_resource
def tidy_Catalunya_m(data_ori, columns_sel, fecha_ini, fecha_fin, columns_output):
    output_data = data_ori[["Fecha"] + columns_sel][(data_ori["Fecha"]>=fecha_ini) & (data_ori["Fecha"]<=fecha_fin)]
    output_data.columns = ["Fecha"] + columns_output
    output_data["Month"] = output_data['Fecha'].dt.month
    output_data = output_data.dropna()
    output_data = output_data[(output_data["Month"]<=output_data['Month'].iloc[-1])]
    return(output_data.drop(["Data", "Month"], axis=1))

##@st.cache_data(show_spinner="**Carregant les dades... Esperi, siusplau**", max_entries=500)
@st.cache_resource
def tidy_Catalunya(data_ori, columns_sel, fecha_ini, fecha_fin, columns_output):
    output_data = data_ori[["Trimestre"] + columns_sel][(data_ori["Fecha"]>=fecha_ini) & (data_ori["Fecha"]<=fecha_fin)]
    output_data.columns = ["Trimestre"] + columns_output

    return(output_data.set_index("Trimestre").drop("Data", axis=1))

##@st.cache_data(show_spinner="**Carregant les dades... Esperi, siusplau**", max_entries=500)
@st.cache_resource
def tidy_Catalunya_anual(data_ori, columns_sel, fecha_ini, fecha_fin, columns_output):
    output_data = data_ori[columns_sel][(data_ori["Fecha"]>=fecha_ini) & (data_ori["Fecha"]<=fecha_fin)]
    output_data.columns = columns_output
    output_data["Any"] = output_data["Any"].astype(str)
    return(output_data.set_index("Any"))

##@st.cache_data(show_spinner="**Carregant les dades... Esperi, siusplau**", max_entries=500)
@st.cache_resource
def tidy_Catalunya_mensual(data_ori, columns_sel, fecha_ini, fecha_fin, columns_output):
    output_data = data_ori[["Fecha"] + columns_sel][(data_ori["Fecha"]>=fecha_ini) & (data_ori["Fecha"]<=fecha_fin)]
    output_data.columns = ["Fecha"] + columns_output
    output_data["Fecha"] = output_data["Fecha"].astype(str)
    return(output_data.set_index("Fecha"))

##@st.cache_data(show_spinner="**Carregant les dades... Esperi, siusplau**", max_entries=500)
@st.cache_resource
def tidy_present(data_ori, columns_sel, year):
    output_data = data_ori[data_ori[columns_sel]!=0][["Trimestre"] + [columns_sel]].dropna()
    output_data["Trimestre_aux"] = output_data["Trimestre"].str[-1]
    output_data = output_data[(output_data["Trimestre_aux"]<=output_data['Trimestre_aux'].iloc[-1])]
    output_data["Any"] = output_data["Trimestre"].str[0:4]
    output_data = output_data.drop(["Trimestre", "Trimestre_aux"], axis=1)
    output_data = output_data.groupby("Any").mean().pct_change().mul(100).reset_index()
    output_data = output_data[output_data["Any"]==str(year)]
    output_data = output_data.set_index("Any")
    return(output_data.values[0][0])

##@st.cache_data(show_spinner="**Carregant les dades... Esperi, siusplau**", max_entries=500)
@st.cache_resource
def tidy_present_monthly(data_ori, columns_sel, year):
    output_data = data_ori[["Fecha"] + [columns_sel]]
    output_data["Any"] = output_data["Fecha"].dt.year
    output_data = output_data.drop_duplicates(["Fecha", columns_sel])
    output_data = output_data.groupby("Any").sum().pct_change().mul(100).reset_index()
    output_data = output_data[output_data["Any"]==int(year)].set_index("Any")
    return(output_data.values[0][0])

##@st.cache_data(show_spinner="**Carregant les dades... Esperi, siusplau**", max_entries=500)
@st.cache_resource
def tidy_present_monthly_aux(data_ori, columns_sel, year):
    output_data = data_ori[["Fecha"] + columns_sel].dropna(axis=0)
    output_data["month_aux"] = output_data["Fecha"].dt.month
    output_data = output_data[(output_data["month_aux"]<=output_data['month_aux'].iloc[-1])]
    output_data["Any"] = output_data["Fecha"].dt.year
    output_data = output_data.drop_duplicates(["Fecha"] + columns_sel)
    output_data = output_data.groupby("Any").sum().pct_change().mul(100).reset_index()
    output_data = output_data[output_data["Any"]==int(year)].set_index("Any")
    return(output_data.values[0][0])

##@st.cache_data(show_spinner="**Carregant les dades... Esperi, siusplau**", max_entries=500)
@st.cache_resource
def tidy_present_monthly_diff(data_ori, columns_sel, year):
    output_data = data_ori[["Fecha"] + columns_sel].dropna(axis=0)
    output_data["month_aux"] = output_data["Fecha"].dt.month
    output_data = output_data[(output_data["month_aux"]<=output_data['month_aux'].iloc[-1])]
    output_data["Any"] = output_data["Fecha"].dt.year
    output_data = output_data.drop_duplicates(["Fecha"] + columns_sel)
    output_data = output_data.groupby("Any").mean().diff().mul(100).reset_index()
    output_data = output_data[output_data["Any"]==int(year)].set_index("Any")
    return(output_data.values[0][0])

##@st.cache_data(show_spinner="**Carregant les dades... Esperi, siusplau**", max_entries=500)
@st.cache_resource
def indicator_year(df, df_aux, year, variable, tipus, frequency=None):
    if (year==str(datetime.now().year-1) and (frequency=="month") and ((tipus=="var") or (tipus=="diff"))):
        return(round(tidy_present_monthly(df_aux, variable, year),2))
    if (year==str(datetime.now().year-1) and (frequency=="month_aux") and (tipus=="var")):
        return(round(tidy_present_monthly_aux(df_aux, variable, year),2))
    if (year==str(datetime.now().year-1) and (frequency=="month_aux") and ((tipus=="diff"))):
        return(round(tidy_present_monthly_diff(df_aux, variable, year),2))
    if (year==str(datetime.now().year-1) and ((tipus=="var") or (tipus=="diff"))):
        return(round(tidy_present(df_aux.reset_index(), variable, year),2))
    if tipus=="level":
        df = df[df.index==year][variable]
        return(round(df.values[0],2))
    if tipus=="var":
        df = df[variable].pct_change().mul(100)
        df = df[df.index==year]
        return(round(df.values[0],2))
    if tipus=="diff":
        df = df[variable].diff().mul(100)
        df = df[df.index==year]
        return(round(df.values[0],2))

##@st.cache_data(show_spinner="**Carregant les dades... Esperi, siusplau**", max_entries=500)
@st.cache_resource
def concatenate_lists(list1, list2):
    result_list = []
    for i in list1:
        result_element = i+ list2
        result_list.append(result_element)
    return(result_list)


def filedownload(df, filename):
    towrite = io.BytesIO()
    df.to_excel(towrite, encoding='latin-1', index=True, header=True)
    towrite.seek(0)
    b64 = base64.b64encode(towrite.read()).decode("latin-1")
    href = f"""<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">
    <button class="download-button">Descarregar</button></a>"""
    return href

#@st.cache_data(show_spinner="**Carregant les dades... Esperi, siusplau**", max_entries=500)
@st.cache_resource
def line_plotly(table_n, selection_n, title_main, title_y, title_x="Trimestre", replace_0=False):
    plot_cat = table_n[selection_n]
    if replace_0==True:
        plot_cat = plot_cat.replace(0, np.NaN)
    colors = ['#2d538f', '#de7207', '#385723']
    traces = []
    for i, col in enumerate(plot_cat.columns):
        trace = go.Scatter(
            x=plot_cat.index,
            y=plot_cat[col],
            mode='lines',
            name=col,
            line=dict(color=colors[i % len(colors)])
        )
        traces.append(trace)
    layout = go.Layout(
        title=dict(text=title_main, font=dict(size=13)),
        xaxis=dict(title=title_x),
        yaxis=dict(title=title_y, tickformat=",d"),
        legend=dict(x=0, y=1.15, orientation="h"),
        paper_bgcolor = "#fcefdc",
        plot_bgcolor='#fcefdc'
    )
    fig = go.Figure(data=traces, layout=layout)
    return fig

#@st.cache_data(show_spinner="**Carregant les dades... Esperi, siusplau**", max_entries=500)
@st.cache_resource
def bar_plotly(table_n, selection_n, title_main, title_y, year_ini, year_fin=datetime.now().year-1):
    table_n = table_n.reset_index()
    table_n["Any"] = table_n["Any"].astype(int)
    plot_cat = table_n[(table_n["Any"] >= year_ini) & (table_n["Any"] <= year_fin)][["Any"] + selection_n].set_index("Any")
    colors = ['#2d538f', '#de7207', '#385723']
    traces = []
    for i, col in enumerate(plot_cat.columns):
        trace = go.Bar(
            x=plot_cat.index,
            y=plot_cat[col],
            name=col,
            marker=dict(color=colors[i % len(colors)])
        )
        traces.append(trace)
    layout = go.Layout(
        title=dict(text=title_main, font=dict(size=13)),
        xaxis=dict(title="Any"),
        yaxis=dict(title=title_y, tickformat=",d"),
        legend=dict(x=0, y=1.15, orientation="h"),
        paper_bgcolor = "#fcefdc",
        plot_bgcolor='#fcefdc'
    )
    fig = go.Figure(data=traces, layout=layout)
    return fig
#@st.cache_data(show_spinner="**Carregant les dades... Esperi, siusplau**", max_entries=500)
@st.cache_resource
def stacked_bar_plotly(table_n, selection_n, title_main, title_y, year_ini, year_fin=datetime.now().year-1):
    table_n = table_n.reset_index()
    table_n["Any"] = table_n["Any"].astype(int)
    plot_cat = table_n[(table_n["Any"] >= year_ini) & (table_n["Any"] <= year_fin)][["Any"] + selection_n].set_index("Any")
    colors = ['#2d538f', '#de7207', '#385723']
    
    traces = []
    for i, col in enumerate(plot_cat.columns):
        trace = go.Bar(
            x=plot_cat.index,
            y=plot_cat[col],
            name=col,
            marker=dict(color=colors[i % len(colors)])
        )
        traces.append(trace)
    
    layout = go.Layout(
        title=dict(text=title_main, font=dict(size=13)),
        xaxis=dict(title="Any"),
        yaxis=dict(title=title_y, tickformat=",d"),
        legend=dict(x=0, y=1.15, orientation="h"),
        barmode='stack',
        paper_bgcolor = "#fcefdc",
        plot_bgcolor='#fcefdc'
    )
    
    fig = go.Figure(data=traces, layout=layout)
    return fig
#@st.cache_data(show_spinner="**Carregant les dades... Esperi, siusplau**", max_entries=500)
@st.cache_resource
def area_plotly(table_n, selection_n, title_main, title_y, trim):
    plot_cat = table_n[table_n.index>=trim][selection_n]
    fig = px.area(plot_cat, x=plot_cat.index, y=plot_cat.columns, title=title_main)
    fig.for_each_trace(lambda trace: trace.update(fillcolor = trace.line.color))
    fig.update_layout(xaxis_title="Trimestre", yaxis=dict(title=title_y, tickformat=",d"), barmode='stack')
    fig.update_traces(opacity=0.4)  # Change opacity to 0.8
    fig.update_layout(legend_title_text="")
    fig.update_layout(
        title=dict(text=title_main, font=dict(size=13), y=0.97),
        legend=dict(x=-0.15, y=1.25, orientation="h"),  # Adjust the x and y values for the legend position
        paper_bgcolor = "#fcefdc",
        plot_bgcolor='#fcefdc'
    )
    return fig

##@st.cache_data(show_spinner="**Carregant les dades... Esperi, siusplau**", max_entries=500)
@st.cache_resource
def table_monthly(data_ori, year_ini, rounded=True):
    data_ori = data_ori.reset_index()
    month_mapping_catalan = {
        1: 'Gener',
        2: 'Febrer',
        3: 'Març',
        4: 'Abril',
        5: 'Maig',
        6: 'Juny',
        7: 'Juliol',
        8: 'Agost',
        9: 'Setembre',
        10: 'Octubre',
        11: 'Novembre',
        12: 'Desembre'
    }

    try:
        output_data = data_ori[data_ori["Data"]>=pd.to_datetime(str(year_ini)+"/01/01", format="%Y/%m/%d")]
        output_data['Mes'] = output_data['Data'].dt.month.map(month_mapping_catalan)
        if rounded==True:
            numeric_columns = output_data.select_dtypes(include=['float64', 'int64']).columns
            output_data[numeric_columns] = output_data[numeric_columns].applymap(lambda x: round(x, 1))
        output_data = output_data.drop(["Fecha", "Data"], axis=1).set_index("Mes").reset_index().T
        output_data.columns = output_data.iloc[0,:]
        output_data = output_data.iloc[1:,:]
    except KeyError:
        output_data = data_ori[data_ori["Fecha"]>=pd.to_datetime(str(year_ini)+"/01/01", format="%Y/%m/%d")]
        output_data['Mes'] = output_data['Fecha'].dt.month.map(month_mapping_catalan)
        if rounded==True:
            numeric_columns = output_data.select_dtypes(include=['float64', 'int64']).columns
            output_data[numeric_columns] = output_data[numeric_columns].applymap(lambda x: round(x, 1))
        output_data = output_data.drop(["Fecha", "index"], axis=1).set_index("Mes").reset_index().T
        output_data.columns = output_data.iloc[0,:]
        output_data = output_data.iloc[1:,:]
    return(output_data)

def format_dataframes(df, style_n):
    if style_n==True:
        return(df.style.format("{:,.0f}"))
    else:
        return(df.style.format("{:,.1f}"))



def table_trim(data_ori, year_ini, rounded=False, formated=True):
    data_ori = data_ori.reset_index()
    data_ori["Any"] = data_ori["Trimestre"].str.split("T").str[0]
    data_ori["Trimestre"] = data_ori["Trimestre"].str.split("T").str[1]
    data_ori["Trimestre"] = data_ori["Trimestre"] + "T"
    data_ori = data_ori[data_ori["Any"]>=str(year_ini)]
    data_ori = data_ori.replace(0, np.NaN)
    if rounded==True:
        numeric_columns = data_ori.select_dtypes(include=['float64', 'int64']).columns
        data_ori[numeric_columns] = data_ori[numeric_columns].applymap(lambda x: round(x, 1))
    output_data = data_ori.set_index(["Any", "Trimestre"]).T.dropna(axis=1, how="all")
    if formated==True:   
        return(format_dataframes(output_data, True))
    else:
        return(format_dataframes(output_data, False))


def table_year(data_ori, year_ini, rounded=False, formated=True):
    data_ori = data_ori.reset_index()
    if rounded==True:
        numeric_columns = data_ori.select_dtypes(include=['float64', 'int64']).columns
        data_ori[numeric_columns] = data_ori[numeric_columns].applymap(lambda x: round(x, 1))
    data_output = data_ori[data_ori["Any"]>=str(year_ini)].T
    data_output.columns = data_output.iloc[0,:]
    data_output = data_output.iloc[1:,:]
    if formated==True:   
        return(format_dataframes(data_output, True))
    else:
        return(format_dataframes(data_output, False))
    
if selected == "Espanya":
    st.sidebar.header("**ESPANYA**")
    selected_type = st.sidebar.radio("", ("Sector residencial","Indicadors econòmics"))
    if selected_type=="Indicadors econòmics":
        selected_index = st.sidebar.selectbox("**Selecciona un indicador:**", ["Índex de Preus al Consum (IPC)", "Consum de ciment","Tipus d'interès", "Hipoteques"])
        available_years = list(range(2018, datetime.now().year))
        selected_year_n = st.sidebar.selectbox("**Selecciona un any:**", available_years, available_years.index(datetime.now().year-1))
        if selected_index=="Índex de Preus al Consum (IPC)":
            st.subheader("ÍNDEX DE PREUS AL CONSUM (IPC)")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            min_year=2002
            max_year=datetime.now().year
            table_espanya_m = tidy_Catalunya_mensual(DT_monthly, ["Fecha", "IPC_Nacional_x", "IPC_subyacente", "IGC_Nacional"], f"{str(min_year)}-01-01", f"{str(max_year)}-01-01",["Data","IPC (Base 2021)","IPC subjacent", "IGC"])
            table_espanya_m["Inflació"] = table_espanya_m["IPC (Base 2021)"].pct_change(12).mul(100)
            table_espanya_m["Inflació subjacent"] = round(table_espanya_m["IPC subjacent"].mul(100),1)
            table_espanya_m["Índex de Garantia de Competitivitat (IGC)"] = round(table_espanya_m["IGC"],1)
            table_espanya_m = table_espanya_m.drop(["IPC subjacent", "IGC"], axis=1)
            table_espanya_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha","IPC_Nacional_x", "IPC_subyacente", "IGC_Nacional"], min_year, max_year,["Any", "IPC (Base 2021)","IPC subjacent", "IGC"])
            table_espanya_y["Inflació"] = table_espanya_y["IPC (Base 2021)"].pct_change(1).mul(100)
            table_espanya_y["Inflació subjacent"] = round(table_espanya_y["IPC subjacent"].mul(100),1)
            table_espanya_y["Índex de Garantia de Competitivitat (IGC)"] = round(table_espanya_y["IGC"],1)
            table_espanya_y = table_espanya_y.drop(["IPC subjacent", "IGC"], axis=1)
            # if selected_year_n==max_year:
            #     left, center, right= st.columns((1,1,1))
            #     with left:
            #         st.metric(label="**Inflació** (var. anual)", value=f"""{round(table_espanya_m["Inflació"][-1],1)}%""")
            #     with center:
            #         st.metric(label="**Inflació subjacent** (var. anual)", value=f"""{round(table_espanya_m["Inflació subjacent"][-1],1)}%""")
            #     with right:
            #         st.metric(label="**Índex de Garantia de Competitivitat** (var. anual)", value=f"""{round(table_espanya_m["Índex de Garantia de Competitivitat (IGC)"][-1],1)}%""")
            # if selected_year_n!=max_year:
            left, center, right= st.columns((1,1,1))
            with left:
                st.metric(label="**Inflació** (var. anual mitjana)", value=f"""{round(table_espanya_y[table_espanya_y.index==str(selected_year_n)]["Inflació"].values[0], 1)}%""")
            with center:
                st.metric(label="**Inflació subjacent** (var. anual mitjana)", value=f"""{round(table_espanya_y[table_espanya_y.index==str(selected_year_n)]["Inflació subjacent"].values[0], 1)}%""")
            with right:
                st.metric(label="**Índex de Garantia de Competitivitat** (var. anual mitjana)", value=f"""{round(table_espanya_y[table_espanya_y.index==str(selected_year_n)]["Índex de Garantia de Competitivitat (IGC)"].values[0], 1)}%""")
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_monthly(table_espanya_m, 2023).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_monthly(table_espanya_m, 2023), f"{selected_index}_Espanya.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_espanya_y, 2008, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_espanya_y, 2008, True, False), f"{selected_index}_Espanya_anual.xlsx"), unsafe_allow_html=True)
            st.plotly_chart(line_plotly(table_espanya_m[table_espanya_m.index>="2015-01-01"], ["Inflació", "Inflació subjacent", "Índex de Garantia de Competitivitat (IGC)"], "Evolució mensual de la inflació (variació anual del IPC) i l'IGC (Índex de Garantia de Competitivitat)", "%",  "Any"), use_container_width=True, responsive=True)
        if selected_index=="Consum de ciment":
            st.subheader("CONSUM DE CIMENT")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            min_year=2008
            max_year=datetime.now().year-1
            table_espanya_m = tidy_Catalunya_m(DT_monthly, ["Fecha"] + ["cons_ciment_Espanya"], f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Consum de ciment"])
            table_espanya_q = tidy_Catalunya(DT_terr, ["Fecha","cons_ciment_Espanya"],  f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Consum de ciment"])
            table_espanya_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha","cons_ciment_Espanya"], min_year, max_year,["Any", "Consum de ciment"])
            table_espanya_q = table_espanya_q.dropna(axis=0)
            table_espanya_y = table_espanya_y.dropna(axis=0)
            st.metric(label="**Consum de ciment** (Milers de tones)", value=f"""{indicator_year(table_espanya_y, table_espanya_q, str(selected_year_n), "Consum de ciment", "level"):,.0f}""", delta=f"""{indicator_year(table_espanya_y, table_espanya_m, str(selected_year_n), "Consum de ciment", "var", "month")}%""")
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_espanya_q, 2020, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_espanya_q, 2012), f"{selected_index}_Espanya.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_espanya_y, 2008, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_espanya_y, 2008, True, False), f"{selected_index}_Espanya_anual.xlsx"), unsafe_allow_html=True)
            left, right = st.columns((1,1))
            with left:
                st.plotly_chart(line_plotly(table_espanya_q, ["Consum de ciment"], "Consum de ciment (Milers T.)", "Milers de T."), use_container_width=True, responsive=True)
            with right:
                st.plotly_chart(bar_plotly(table_espanya_y.pct_change(1).mul(100).dropna(axis=0), ["Consum de ciment"], "Variació anual del consum de ciment (%)", "%", 2012), use_container_width=True, responsive=True)     
        if selected_index=="Tipus d'interès":
            min_year=2008
            max_year=datetime.now().year-1
            st.subheader("TIPUS D'INTERÈS I POLÍTICA MONETÀRIA")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            table_espanya_m = tidy_Catalunya_mensual(DT_monthly, ["Fecha", "Euribor_1m", "Euribor_3m",	"Euribor_6m", "Euribor_1y", "tipo_hipo"], f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data","Euríbor a 1 mes","Euríbor a 3 mesos","Euríbor a 6 mesos","Euríbor a 1 any", "Tipus d'interès d'hipoteques"])
            table_espanya_m = table_espanya_m[["Data","Euríbor a 1 mes","Euríbor a 3 mesos","Euríbor a 6 mesos","Euríbor a 1 any", "Tipus d'interès d'hipoteques"]].reset_index(drop=True).rename(columns={"Data":"Fecha"})
            table_espanya_q = tidy_Catalunya(DT_terr, ["Fecha", "Euribor_1m", "Euribor_3m","Euribor_6m", "Euribor_1y", "tipo_hipo"],  f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Euríbor a 1 mes","Euríbor a 3 mesos","Euríbor a 6 mesos", "Euríbor a 1 any", "Tipus d'interès d'hipoteques"])
            table_espanya_q = table_espanya_q[["Euríbor a 1 mes","Euríbor a 3 mesos","Euríbor a 6 mesos", "Euríbor a 1 any", "Tipus d'interès d'hipoteques"]]
            table_espanya_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha", "Euribor_1m", "Euribor_3m","Euribor_6m", "Euribor_1y", "tipo_hipo"], min_year, max_year,["Any", "Euríbor a 1 mes","Euríbor a 3 mesos","Euríbor a 6 mesos", "Euríbor a 1 any", "Tipus d'interès d'hipoteques"])
            table_espanya_y = table_espanya_y[["Euríbor a 1 mes","Euríbor a 3 mesos","Euríbor a 6 mesos","Euríbor a 1 any", "Tipus d'interès d'hipoteques"]]
            if selected_year_n==2023:
                left, left_center, right_center, right = st.columns((1,1,1,1))
                with left:
                    st.metric(label="**Euríbor a 3 mesos** (%)", value=f"""{indicator_year(table_espanya_y, table_espanya_q, str(selected_year_n), "Euríbor a 3 mesos", "level")}""", delta=f"""{indicator_year(table_espanya_y, table_espanya_m, str(selected_year_n), ["Euríbor a 3 mesos"], "diff", "month_aux")} p.b.""")
                with left_center:
                    st.metric(label="**Euríbor a 6 mesos** (%)", value=f"""{indicator_year(table_espanya_y, table_espanya_q, str(selected_year_n), "Euríbor a 6 mesos", "level")}""", delta=f"""{indicator_year(table_espanya_y, table_espanya_m, str(selected_year_n), ["Euríbor a 6 mesos"], "diff", "month_aux")} p.b.""")
                with right_center:
                    st.metric(label="**Euríbor a 1 any** (%)", value=f"""{indicator_year(table_espanya_y, table_espanya_q, str(selected_year_n), "Euríbor a 1 any", "level")}""", delta=f"""{indicator_year(table_espanya_y, table_espanya_m, str(selected_year_n), ["Euríbor a 1 any"], "diff", "month_aux")} p.b.""")
                with right:
                    st.metric(label="**Tipus d'interès d'hipoteques** (%)", value=f"""{indicator_year(table_espanya_y, table_espanya_q, str(selected_year_n), "Tipus d'interès d'hipoteques", "level")}""", delta=f"""{indicator_year(table_espanya_y, table_espanya_m, str(selected_year_n), ["Tipus d'interès d'hipoteques"], "diff", "month_aux")} p.b.""")
            if selected_year_n!=2023:
                left, left_center, right_center, right = st.columns((1,1,1,1))
                with left:
                    st.metric(label="**Euríbor a 3 mesos** (%)", value=f"""{indicator_year(table_espanya_y, table_espanya_q, str(selected_year_n), "Euríbor a 3 mesos", "level")}""", delta=f"""{indicator_year(table_espanya_y, table_espanya_m, str(selected_year_n), "Euríbor a 3 mesos", "diff", "month")} p.b.""")
                with left_center:
                    st.metric(label="**Euríbor a 6 mesos** (%)", value=f"""{indicator_year(table_espanya_y, table_espanya_q, str(selected_year_n), "Euríbor a 6 mesos", "level")}""", delta=f"""{indicator_year(table_espanya_y, table_espanya_m, str(selected_year_n), "Euríbor a 6 mesos", "diff", "month")} p.b.""")
                with right_center:
                    st.metric(label="**Euríbor a 1 any** (%)", value=f"""{indicator_year(table_espanya_y, table_espanya_q, str(selected_year_n), "Euríbor a 1 any", "level")}""", delta=f"""{indicator_year(table_espanya_y, table_espanya_m, str(selected_year_n), "Euríbor a 1 any", "diff", "month")} p.b.""")
                with right:
                    st.metric(label="**Tipus d'interès d'hipoteques** (%)", value=f"""{indicator_year(table_espanya_y, table_espanya_q, str(selected_year_n), "Tipus d'interès d'hipoteques", "level")}""", delta=f"""{indicator_year(table_espanya_y, table_espanya_m, str(selected_year_n), "Tipus d'interès d'hipoteques", "diff", "month")} p.b.""")
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_monthly(table_espanya_m, 2023).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_monthly(table_espanya_m, 2023), f"{selected_index}_Espanya.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_espanya_y, 2014, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_espanya_y, 2014, True, False), f"{selected_index}_Espanya_anual.xlsx"), unsafe_allow_html=True)
            selected_columns = ["Euríbor a 3 mesos","Euríbor a 6 mesos","Euríbor a 1 any", "Tipus d'interès d'hipoteques"]
            left, right = st.columns((1,1))
            with left:
                st.plotly_chart(line_plotly(table_espanya_m.set_index("Fecha"), selected_columns, "Evolució mensual dels tipus d'interès (%)", "Tipus d'interès (%)",  "Fecha"), use_container_width=True, responsive=True)
            with right:
                st.plotly_chart(bar_plotly(table_espanya_y, ["Euríbor a 1 any", "Tipus d'interès d'hipoteques"], "Evolució anual dels tipus d'interès (%)", "Tipus d'interès (%)",  2005), use_container_width=True, responsive=True)
        if selected_index=="Hipoteques":
            st.subheader("IMPORT I NOMBRE D'HIPOTEQUES INSCRITES EN ELS REGISTRES DE PROPIETAT")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            min_year=2008
            max_year=datetime.now().year-1
            table_espanya_m = tidy_Catalunya_mensual(DT_monthly, ["Fecha", "hipon_Nacional", "hipoimp_Nacional"], f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data","Nombre d'hipoteques", "Import d'hipoteques"])
            table_espanya_m = table_espanya_m[["Data", "Nombre d'hipoteques", "Import d'hipoteques"]].rename(columns={"Data":"Fecha"})
            table_espanya_q = tidy_Catalunya(DT_terr, ["Fecha", "hipon_Nacional", "hipoimp_Nacional"],  f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Nombre d'hipoteques", "Import d'hipoteques"])
            table_espanya_q = table_espanya_q[["Nombre d'hipoteques", "Import d'hipoteques"]]
            table_espanya_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha","hipon_Nacional", "hipoimp_Nacional"], min_year, max_year,["Any", "Nombre d'hipoteques", "Import d'hipoteques"])
            table_espanya_y = table_espanya_y[["Nombre d'hipoteques", "Import d'hipoteques"]]
            if selected_year_n==2023:
                left, right = st.columns((1,1))
                with left:
                    st.metric(label="**Nombre d'hipoteques**", value=f"""{indicator_year(table_espanya_y, table_espanya_q, str(selected_year_n), "Nombre d'hipoteques", "level"):,.0f}""", delta=f"""{indicator_year(table_espanya_y, table_espanya_m, str(selected_year_n), ["Nombre d'hipoteques"], "var", "month_aux")}%""")
                with right:
                    st.metric(label="**Import d'hipoteques** (Milers d'euros)", value=f"""{indicator_year(table_espanya_y, table_espanya_q, str(selected_year_n), "Import d'hipoteques", "level"):,.0f}""", delta=f"""{indicator_year(table_espanya_y, table_espanya_m, str(selected_year_n), ["Import d'hipoteques"], "var", "month_aux")}%""")
            if selected_year_n!=2023:
                left, right = st.columns((1,1))
                with left:
                    st.metric(label="**Nombre d'hipoteques**", value=f"""{indicator_year(table_espanya_y, table_espanya_q, str(selected_year_n), "Nombre d'hipoteques", "level"):,.0f}""", delta=f"""{indicator_year(table_espanya_y, table_espanya_m, str(selected_year_n), "Nombre d'hipoteques", "var")}%""")
                with right:
                    st.metric(label="**Import d'hipoteques** (Milers d'euros)", value=f"""{indicator_year(table_espanya_y, table_espanya_q, str(selected_year_n), "Import d'hipoteques", "level"):,.0f}""", delta=f"""{indicator_year(table_espanya_y, table_espanya_m, str(selected_year_n), "Import d'hipoteques", "var")}%""")

            selected_columns = ["Nombre d'hipoteques", "Import d'hipoteques"]
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_espanya_q, 2020).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_espanya_q, 2008), f"{selected_index}_Espanya.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_espanya_y, 2009, rounded=False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_espanya_y, 2008, rounded=False), f"{selected_index}_Espanya_anual.xlsx"), unsafe_allow_html=True)
            left, right = st.columns((1,1))
            with left:
                st.plotly_chart(line_plotly(table_espanya_m, ["Nombre d'hipoteques"], "Evolució mensual del nombre d'hipoteques", "Nombre d'hipoteques",  "Data"), use_container_width=True, responsive=True)
                st.plotly_chart(line_plotly(table_espanya_m, ["Import d'hipoteques"], "Evolució mensual de l'import d'hipoteques (Milers €)", "Import d'hipoteques",  "Data"), use_container_width=True, responsive=True)
            with right:
                st.plotly_chart(bar_plotly(table_espanya_y, ["Nombre d'hipoteques"], "Evolució anual del nombre d'hipoteques", "Nombre d'hipoteques",  2005), use_container_width=True, responsive=True)
                st.plotly_chart(bar_plotly(table_espanya_y, ["Import d'hipoteques"], "Evolució anual de l'import d'hipoteques (Milers €)", "Import d'hipoteques",  2005), use_container_width=True, responsive=True)

    if selected_type=="Sector residencial":
        selected_index = st.sidebar.selectbox("**Selecciona un indicador:**", ["Producció", "Compravendes", "Preus"])
        available_years = list(range(2018, datetime.now().year))
        selected_year_n = st.sidebar.selectbox("**Selecciona un any:**", available_years, available_years.index(datetime.now().year-1))
        if selected_index=="Producció":
            min_year=2008
            max_year=datetime.now().year-1
            st.subheader("PRODUCCIÓ D'HABITATGES A ESPANYA")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            table_esp_m = tidy_Catalunya_m(DT_monthly, ["Fecha"] + concatenate_lists(["iniviv_","finviv_"], "Nacional"), f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Habitatges iniciats", "Habitatges acabats"])                                                                                                                                                                                                                                                                                                                     
            table_esp = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["iniviv_","finviv_"], "Nacional") + concatenate_lists(["calprov_", "calprovpub_", "calprovpriv_", "caldef_", "caldefpub_", "caldefpriv_"], "Espanya"), f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Habitatges iniciats", "Habitatges acabats", 
                                                                                                                                                                                                                                                                                            "Qualificacions provisionals d'HPO", "Qualificacions provisionals d'HPO (Promotor públic)", "Qualificacions provisionals d'HPO (Promotor privat)", 
                                                                                                                                                                                                                                                                                            "Qualificacions definitives d'HPO",  "Qualificacions definitives d'HPO (Promotor públic)", "Qualificacions definitives d'HPO (Promotor privat)"])
            table_esp_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"] + concatenate_lists(["iniviv_","finviv_"], "Nacional")+ concatenate_lists(["calprov_", "calprovpub_", "calprovpriv_", "caldef_", "caldefpub_", "caldefpriv_"], "Espanya"), min_year, max_year,["Any", "Habitatges iniciats", "Habitatges acabats", 
                                                                                                                                                                                                                                                        "Qualificacions provisionals d'HPO", "Qualificacions provisionals d'HPO (Promotor públic)", "Qualificacions provisionals d'HPO (Promotor privat)", 
            
                                                                                                                                                                                                                                                    "Qualificacions definitives d'HPO",  "Qualificacions definitives d'HPO (Promotor públic)", "Qualificacions definitives d'HPO (Promotor privat)"])
            left, right = st.columns((1,1))
            with left:
                st.metric(label="**Habitatges iniciats**", value=f"""{indicator_year(table_esp_y, table_esp, str(selected_year_n), "Habitatges iniciats", "level"):,.0f}""", delta=f"""{indicator_year(table_esp_y, table_esp_m, str(selected_year_n), "Habitatges iniciats", "var", "month")}%""")
                st.metric(label="**Qualificacions provisionals d'HPO**", value=f"""{indicator_year(table_esp_y, table_esp, str(selected_year_n), "Qualificacions provisionals d'HPO", "level"):,.0f}""", delta=f"""{indicator_year(table_esp_y, table_esp, str(selected_year_n), "Qualificacions provisionals d'HPO", "var")}%""")
                st.metric(label="**Qualificacions provisionals d'HPO** (Promotor públic)", value=f"""{indicator_year(table_esp_y, table_esp, str(selected_year_n), "Qualificacions provisionals d'HPO (Promotor públic)", "level"):,.0f}""", delta=f"""{indicator_year(table_esp_y, table_esp, str(selected_year_n), "Qualificacions provisionals d'HPO (Promotor públic)", "var")}%""")
                st.metric(label="**Qualificacions definitives d'HPO** (Promotor públic)", value=f"""{indicator_year(table_esp_y, table_esp, str(selected_year_n), "Qualificacions definitives d'HPO (Promotor públic)", "level"):,.0f}""", delta=f"""{indicator_year(table_esp_y, table_esp, str(selected_year_n), "Qualificacions definitives d'HPO (Promotor públic)", "var")}%""")
            with right:
                st.metric(label="**Habitatges acabats**", value=f"""{indicator_year(table_esp_y, table_esp, str(selected_year_n), "Habitatges acabats", "level"):,.0f}""", delta=f"""{indicator_year(table_esp_y, table_esp_m, str(selected_year_n), "Habitatges acabats", "var","month")}%""")
                st.metric(label="**Qualificacions definitives d'HPO**", value=f"""{indicator_year(table_esp_y, table_esp, str(selected_year_n), "Qualificacions definitives d'HPO", "level"):,.0f}""", delta=f"""{indicator_year(table_esp_y, table_esp, str(selected_year_n), "Qualificacions definitives d'HPO", "var")}%""")
                st.metric(label="**Qualificacions provisionals d'HPO** (Promotor privat)", value=f"""{indicator_year(table_esp_y, table_esp, str(selected_year_n), "Qualificacions provisionals d'HPO (Promotor privat)", "level"):,.0f}""", delta=f"""{indicator_year(table_esp_y, table_esp, str(selected_year_n), "Qualificacions provisionals d'HPO (Promotor privat)", "var")}%""")
                st.metric(label="**Qualificacions definitives d'HPO** (Promotor privat)", value=f"""{indicator_year(table_esp_y, table_esp, str(selected_year_n), "Qualificacions definitives d'HPO (Promotor privat)", "level"):,.0f}""", delta=f"""{indicator_year(table_esp_y, table_esp, str(selected_year_n), "Qualificacions definitives d'HPO (Promotor privat)", "var")}%""")
            selected_columns_aux = ["Habitatges iniciats", "Habitatges acabats"]
            selected_columns_aux1 = ["Qualificacions provisionals d'HPO (Promotor públic)", "Qualificacions provisionals d'HPO (Promotor privat)"]
            selected_columns_aux2 = ["Qualificacions definitives d'HPO (Promotor públic)", "Qualificacions definitives d'HPO (Promotor privat)"]
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_esp, 2020).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_esp, 2008), f"{selected_index}_Espanya.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_esp_y, 2008).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_esp_y, 2008), f"{selected_index}_Espanya_anual.xlsx"), unsafe_allow_html=True)
            left, right = st.columns((1,1))
            with left:
                st.plotly_chart(line_plotly(table_esp, selected_columns_aux, "Evolució trimestral de la producció d'habitatges", "Nombre d'habitatges"), use_container_width=True, responsive=True)
                st.plotly_chart(stacked_bar_plotly(table_esp_y, selected_columns_aux1, "Qualificacions provisionals de protecció oficial segons tipus de promotor", "Nombre d'habitatges", 2014), use_container_width=True, responsive=True)
            with right:
                st.plotly_chart(bar_plotly(table_esp_y, selected_columns_aux, "Evolució anual de la producció d'habitatges", "Nombre d'habitatges", 2005), use_container_width=True, responsive=True)
                st.plotly_chart(stacked_bar_plotly(table_esp_y, selected_columns_aux2, "Qualificacions definitives de protecció oficial segons tipus de promotor", "Nombre d'habitatges", 2014), use_container_width=True, responsive=True)
        if selected_index=="Compravendes":
            min_year=2008
            max_year=datetime.now().year-1
            st.subheader("COMPRAVENDES D'HABITATGES A ESPANYA")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            table_esp_m = tidy_Catalunya_m(DT_monthly, ["Fecha"] + ["trvivses", "trvivnes"], f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"])
            table_esp_m["Compravendes d'habitatge total"] = table_esp_m["Compravendes d'habitatge de segona mà"] + table_esp_m["Compravendes d'habitatge nou"]
            table_esp = tidy_Catalunya(DT_terr, ["Fecha", "trvivses", "trvivnes"], f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data","Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"])
            table_esp["Compravendes d'habitatge total"] = table_esp["Compravendes d'habitatge de segona mà"] + table_esp["Compravendes d'habitatge nou"]
            table_esp = table_esp[["Compravendes d'habitatge total","Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"]]
            table_esp_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha", "trvivses", "trvivnes"], min_year, max_year,["Any", "Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"])
            table_esp_y["Compravendes d'habitatge total"] = table_esp_y["Compravendes d'habitatge de segona mà"] + table_esp_y["Compravendes d'habitatge nou"]
            table_esp_y = table_esp_y[["Compravendes d'habitatge total","Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"]]

            left, center, right = st.columns((1,1,1))
            with left:
                st.metric(label="**Compravendes d'habitatge total**", value=f"""{indicator_year(table_esp_y, table_esp_m, str(selected_year_n), "Compravendes d'habitatge total", "level"):,.0f}""", delta=f"""{indicator_year(table_esp_y, table_esp_m, str(selected_year_n), "Compravendes d'habitatge total", "var", "month")}%""")
            with center:
                st.metric(label="**Compravendes d'habitatge de segona mà**", value=f"""{indicator_year(table_esp_y, table_esp_m, str(selected_year_n), "Compravendes d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_esp_y, table_esp_m, str(selected_year_n), "Compravendes d'habitatge de segona mà", "var", "month")}%""")
            with right:
                st.metric(label="**Compravendes d'habitatge nou**", value=f"""{indicator_year(table_esp_y, table_esp_m, str(selected_year_n), "Compravendes d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_esp_y, table_esp_m, str(selected_year_n), "Compravendes d'habitatge nou", "var", "month")}%""")
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_esp, 2020).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_esp, 2008), f"{selected_index}_Espanya.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_esp_y, 2008).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_esp_y, 2008), f"{selected_index}_Espanya_anual.xlsx"), unsafe_allow_html=True)
            left_col, right_col = st.columns((1,1))
            with left_col:
                st.plotly_chart(line_plotly(table_esp[table_esp.notna()], table_esp.columns.tolist(), "Evolució trimestral de les compravendes d'habitatge per tipologia d'habitatge", "Nombre de compravendes"), use_container_width=True, responsive=True)
            with right_col:
                st.plotly_chart(stacked_bar_plotly(table_esp_y[table_esp_y.notna()], table_esp.columns.tolist()[1:3], "Evolució anual de les compravendes d'habitatge per tipologia d'habitatge", "Nombre de compravendes", 2008), use_container_width=True, responsive=True)
        if selected_index=="Preus":
                min_year=2008
                max_year=datetime.now().year-1  
                st.subheader("VALOR TASAT MITJÀ D'HABITATGE LLIURE €/M\u00b2 (MITMA)")
                st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
                table_esp = tidy_Catalunya(DT_terr, ["Fecha", "prvivlfom_Nacional", "prvivlnfom_Nacional"], f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Preu de l'habitatge lliure", "Preu de l'habitatge lliure nou"])
                table_esp_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha", "prvivlfom_Nacional", "prvivlnfom_Nacional"], min_year, max_year,["Any", "Preu de l'habitatge lliure", "Preu de l'habitatge lliure nou"])
                left, right = st.columns((1,1))
                with left:
                    st.metric(label=f"""**Preu de l'habitatge lliure** (€/m\u00b2)""", value=f"""{indicator_year(table_esp_y, table_esp, str(selected_year_n), "Preu de l'habitatge lliure", "level"):,.0f}""")
                with right:
                    st.metric(label=f"""**Preu de l'habitatge lliure nou** (€/m\u00b2)""", value=f"""{round(indicator_year(table_esp_y, table_esp, str(selected_year_n), "Preu de l'habitatge lliure nou", "level"),1):,.0f}""")               
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
                st.markdown(table_trim(table_esp, 2020, True, False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_trim(table_esp, 2008, True, False), f"{selected_index}_Espanya.xlsx"), unsafe_allow_html=True)
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES ANUALS**")
                st.markdown(table_year(table_esp_y, 2008, True, False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_year(table_esp_y, 2008, True, False), f"{selected_index}_Espanya_anual.xlsx"), unsafe_allow_html=True)
                left_col, right_col = st.columns((1,1))
                with left_col:
                    st.plotly_chart(line_plotly(table_esp, table_esp.columns.tolist(), "Preus per m\u00b2 de tasació per tipologia d'habitatge", "€/m\u00b2"), use_container_width=True, responsive=True)
                with right_col:
                    st.plotly_chart(bar_plotly(table_esp_y, table_esp.columns.tolist(), "Preus per m\u00b2 de tasació per tipologia d'habitatge", "€/m\u00b2", 2010), use_container_width=True, responsive=True)
                st.subheader("VARIACIONS ANUALS DE L'ÍNDEX DEL PREU DE L'HABITATGE (INE)")
                st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
                table_esp = tidy_Catalunya(DT_terr, ["Fecha", "ipves", "ipvses", "ipvnes"], f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Preu d'habitatge total", "Preus d'habitatge de segona mà", "Preus d'habitatge nou"])
                table_esp_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha", "ipves", "ipvses", "ipvnes"], min_year, max_year,["Any", "Preu d'habitatge total", "Preus d'habitatge de segona mà", "Preus d'habitatge nou"])
                left, center, right = st.columns((1,1,1))
                with left:
                    st.metric(label=f"""**Preu d'habitatge total** (var. anual)""", value=f"""{indicator_year(table_esp_y, table_esp, str(selected_year_n), "Preu d'habitatge total", "level")} %""")
                with center:
                    st.metric(label=f"""**Preu d'habitatge de segona mà** (var. anual)""", value=f"""{indicator_year(table_esp_y, table_esp, str(selected_year_n), "Preus d'habitatge de segona mà", "level")} %""")
                with right:
                    st.metric(label=f"""**Preu d'habitatge nou** (var. anual)""", value=f"""{round(indicator_year(table_esp_y, table_esp, str(selected_year_n), "Preus d'habitatge nou", "level"),1)} %""")                
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
                st.markdown(table_trim(table_esp, 2020, True, False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_trim(table_esp, 2008, True, False), f"{selected_index}_Espanya.xlsx"), unsafe_allow_html=True)
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES ANUALS**")
                st.markdown(table_year(table_esp_y, 2008, True, False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_year(table_esp_y, 2008, True, False), f"{selected_index}_Espanya_anual.xlsx"), unsafe_allow_html=True)
                left_col, right_col = st.columns((1,1))
                with left_col:
                    st.plotly_chart(line_plotly(table_esp, table_esp.columns.tolist(), "Índex trimestral de preus per tipologia d'habitatge (variació anual %)", "%"), use_container_width=True, responsive=True)
                with right_col:
                    st.plotly_chart(bar_plotly(table_esp_y, table_esp.columns.tolist(), "Índex anual de preus per tipologia d'habitatge (variació anual %)", "%", 2007), use_container_width=True, responsive=True)

if selected == "Catalunya":
    st.sidebar.header("**CATALUNYA**")
    selected_indicator = st.sidebar.radio("", ("Sector residencial", "Indicadors econòmics"))
    if selected_indicator=="Indicadors econòmics":
        selected_index = st.sidebar.selectbox("**Selecciona un indicador:**", ["Costos de construcció", "Mercat laboral", "Consum de Ciment", "Hipoteques"])
        available_years = list(range(2014, datetime.now().year))
        selected_year_n = st.sidebar.selectbox("**Selecciona un any:**", available_years, available_years.index(datetime.now().year-1))
        if selected_index=="Mercat laboral":
            st.subheader("MERCAT LABORAL DEL SECTOR DE LA CONSTRUCCIÓ")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            min_year=2008
            max_year=datetime.now().year-1
            table_catalunya_m = tidy_Catalunya_m(DT_monthly, ["Fecha"] + ["unempcons_Catalunya", "aficons_Catalunya"], f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Atur registrat del sector de la construcció", "Afiliats del sector de la construcció"])
            table_catalunya_q = tidy_Catalunya(DT_terr, ["Fecha", "emptot_Catalunya", "empcons_Catalunya", "unempcons_Catalunya", "aficons_Catalunya"],  f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Total població ocupada", "Ocupació del sector de la construcció","Atur registrat del sector de la construcció", "Afiliats del sector de la construcció"])
            table_catalunya_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha","emptot_Catalunya", "empcons_Catalunya", "unempcons_Catalunya", "aficons_Catalunya"], min_year, max_year,["Any", "Total població ocupada", "Ocupació del sector de la construcció","Atur registrat del sector de la construcció", "Afiliats del sector de la construcció"])
            table_catalunya_q = table_catalunya_q.dropna(axis=0)
            table_catalunya_y = table_catalunya_y.dropna(axis=0)
            left, right = st.columns((1,1))
            with left:
                st.metric(label="**Total població ocupada** (Milers)", value=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Total població ocupada", "level"):,.0f}""", delta=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Total població ocupada", "var")}%""")
                st.metric(label="**Atur registrat del sector de la construcció**", value=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Atur registrat del sector de la construcció", "level"):,.0f}""", delta=f"""{indicator_year(table_catalunya_y, table_catalunya_m, str(selected_year_n), "Atur registrat del sector de la construcció", "var", "month")}%""")
            with right:
                st.metric(label="**Ocupació del sector de la construcció** (Milers)", value=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Ocupació del sector de la construcció", "level"):,.0f}""", delta=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Ocupació del sector de la construcció", "var")}%""")
                st.metric(label="**Afiliats del sector de la construcció**", value=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Afiliats del sector de la construcció", "level"):,.0f}""", delta=f"""{indicator_year(table_catalunya_y, table_catalunya_m, str(selected_year_n), "Afiliats del sector de la construcció", "var", "month")}%""")
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_catalunya_q, 2020, rounded=True).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_catalunya_q, 2012, rounded=True), f"{selected_index}_Catalunya.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_catalunya_y, 2008, rounded=True).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_catalunya_y, 2008, rounded=True), f"{selected_index}_Catalunya_anual.xlsx"), unsafe_allow_html=True)

            
            left, right = st.columns((1,1))
            with left:
                st.plotly_chart(stacked_bar_plotly(table_catalunya_y, ["Total població ocupada", "Ocupació del sector de la construcció"], "Ocupats totals i del sector de la construcció (milers)", "Milers de persones", 2014), use_container_width=True, responsive=True)
            with right:
                st.plotly_chart(bar_plotly(table_catalunya_y, ["Afiliats del sector de la construcció", "Atur registrat del sector de la construcció"], "Afiliats i aturats del sector de la construcció", "Persones", 2014), use_container_width=True, responsive=True)

        if selected_index=="Costos de construcció":
            st.subheader("COSTOS DE CONSTRUCCIÓ PER TIPOLOGIA EDIFICATÒRIA")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            min_year=2013
            max_year=datetime.now().year-1
            table_catalunya_q = tidy_Catalunya(DT_terr, ["Fecha", "Costos_edificimitjaneres", "Costos_Unifamiliar2plantes", "Costos_nauind", "Costos_edificioficines"],  f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Edifici renda normal entre mitjaneres", "Unifamiliar de dos plantes entre mitjaneres", "Nau industrial", "Edifici d’oficines entre mitjaneres"])
            table_catalunya_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha","Costos_edificimitjaneres", "Costos_Unifamiliar2plantes", "Costos_nauind", "Costos_edificioficines"], min_year, max_year,["Any", "Edifici renda normal entre mitjaneres", "Unifamiliar de dos plantes entre mitjaneres", "Nau industrial", "Edifici d’oficines entre mitjaneres"])
            table_catalunya_q = table_catalunya_q.dropna(axis=0)
            table_catalunya_y = table_catalunya_y.dropna(axis=0)
            left, right = st.columns((1,1))
            with left:
                st.metric(label="**Edifici renda normal entre mitjaneres** (€/m\u00b2)", value=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Edifici renda normal entre mitjaneres", "level"):,.0f}""", delta=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Edifici renda normal entre mitjaneres", "var")}%""")
                st.metric(label="**Nau industrial** (€/m\u00b2)", value=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Nau industrial", "level"):,.0f}""", delta=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Nau industrial", "var")}%""")
            with right:
                st.metric(label="**Unifamiliar de dos plantes entre mitjaneres** (€/m\u00b2)", value=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Unifamiliar de dos plantes entre mitjaneres", "level"):,.0f}""", delta=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Unifamiliar de dos plantes entre mitjaneres", "var")}%""")
                st.metric(label="**Edifici d’oficines entre mitjaneres** (€/m\u00b2)", value=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Edifici d’oficines entre mitjaneres", "level"):,.0f}""", delta=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Edifici d’oficines entre mitjaneres", "var")}%""")
            desc_bec_aux = """Els preus per m² construït inclouen l’estudi de seguretat i salut, els honoraris tècnics i permisos d’obra amb un benefici industrial del 20% i despeses generals. Addicionalment, 
            cal comentar que aquests preus fan referència a la província de Barcelona. Si la ubicació de l'obra es troba en una província diferent, la disminució dels preus serà d'un 6% a 8% a Girona, 8% a 10% a Tarragona i del 12% a 15% a Lleida. 
            Pot consultar l'última edició del Butlletí Econòmic de la Costrucció (BEC) fent click sobre el link a continuació: """
            desc_bec = f'<div style="text-align: justify">{desc_bec_aux}</div>'
            st.markdown(desc_bec, unsafe_allow_html=True)
            st.markdown("")
            st.markdown(f"""<a href="https://drive.google.com/file/d/1OfCCmtxe92THECNRkFEU_bNb4Q_eGzPw/view?usp=sharing" target="_blank"><button class="download-button">Descarregar BEC</button></a>""", unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_catalunya_q, 2020).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_catalunya_q, 2013), f"{selected_index}_Catalunya.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_catalunya_y, 2013, rounded=True).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_catalunya_y, 2013, rounded=True), f"{selected_index}_Catalunya_anual.xlsx"), unsafe_allow_html=True)
            left, right = st.columns((1,1))
            with left:
                st.plotly_chart(line_plotly(table_catalunya_q, ["Edifici renda normal entre mitjaneres", "Unifamiliar de dos plantes entre mitjaneres", "Nau industrial", "Edifici d’oficines entre mitjaneres"], "Costos de construcció per tipologia (€/m\u00b2)", "€/m\u00b2 construït"), use_container_width=True, responsive=True)
            with right:
                st.plotly_chart(line_plotly(table_catalunya_q.pct_change(4).mul(100).iloc[4:,:], ["Edifici renda normal entre mitjaneres", "Unifamiliar de dos plantes entre mitjaneres", "Nau industrial", "Edifici d’oficines entre mitjaneres"], "Costos de construcció per tipologia (% var. anual)", "%"), use_container_width=True, responsive=True)

        if selected_index=="Consum de Ciment":
            st.subheader("CONSUM DE CIMENT")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            min_year=2012
            max_year=datetime.now().year-1
            table_catalunya_m = tidy_Catalunya_m(DT_monthly, ["Fecha"] + ["cons_ciment_Catalunya"], f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Consum de ciment"])
            table_catalunya_q = tidy_Catalunya(DT_terr, ["Fecha","cons_ciment_Catalunya"],  f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Consum de ciment"])
            table_catalunya_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha","cons_ciment_Catalunya"], min_year, max_year,["Any", "Consum de ciment"])
            table_catalunya_q = table_catalunya_q.dropna(axis=0)
            table_catalunya_y = table_catalunya_y.dropna(axis=0)
            st.metric(label="**Consum de ciment** (Milers de tones)", value=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Consum de ciment", "level"):,.0f}""", delta=f"""{indicator_year(table_catalunya_y, table_catalunya_m, str(selected_year_n), "Consum de ciment", "var", "month")}%""")
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_catalunya_q, 2018).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_catalunya_q, 2014), f"{selected_index}_Espanya.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_catalunya_y, 2014, True).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_catalunya_y, 2014, True), f"{selected_index}_Espanya_anual.xlsx"), unsafe_allow_html=True)
            left, right = st.columns((1,1))
            with left:
                st.plotly_chart(line_plotly(table_catalunya_q, ["Consum de ciment"], "Consum de ciment (Milers T.)", "Milers de T."), use_container_width=True, responsive=True)
            with right:
                st.plotly_chart(bar_plotly(table_catalunya_y.pct_change(1).mul(100).dropna(axis=0), ["Consum de ciment"], "Variació anual del consum de ciment (Milers T.)", "%", 2012), use_container_width=True, responsive=True)
        if selected_index=="Hipoteques":
            st.subheader("IMPORT I NOMBRE D'HIPOTEQUES INSCRITES EN ELS REGISTRES DE PROPIETAT")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            min_year=2008
            max_year=datetime.now().year-1
            table_catalunya_m = tidy_Catalunya_mensual(DT_monthly, ["Fecha", "hipon_Catalunya", "hipoimp_Catalunya"], f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data","Nombre d'hipoteques", "Import d'hipoteques"])
            table_catalunya_m = table_catalunya_m[["Data","Nombre d'hipoteques", "Import d'hipoteques"]].rename(columns={"Data":"Fecha"})
            table_catalunya_q = tidy_Catalunya(DT_terr, ["Fecha", "hipon_Catalunya", "hipoimp_Catalunya"],  f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Nombre d'hipoteques", "Import d'hipoteques"])
            table_catalunya_q = table_catalunya_q[["Nombre d'hipoteques", "Import d'hipoteques"]]
            table_catalunya_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha","hipon_Catalunya", "hipoimp_Catalunya"], min_year, max_year,["Any", "Nombre d'hipoteques", "Import d'hipoteques"])
            table_catalunya_y = table_catalunya_y[["Nombre d'hipoteques", "Import d'hipoteques"]]
            if selected_year_n==2023:
                left, right = st.columns((1,1))
                with left:
                    st.metric(label="**Nombre d'hipoteques**", value=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Nombre d'hipoteques", "level"):,.0f}""", delta=f"""{indicator_year(table_catalunya_y, table_catalunya_m, str(selected_year_n), ["Nombre d'hipoteques"], "var", "month_aux")}%""")
                with right:
                    st.metric(label="**Import d'hipoteques** (Milers €)", value=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Import d'hipoteques", "level"):,.0f}""", delta=f"""{indicator_year(table_catalunya_y, table_catalunya_m, str(selected_year_n), ["Import d'hipoteques"], "var", "month_aux")}%""")
            if selected_year_n!=2023:
                left, right = st.columns((1,1))
                with left:
                    st.metric(label="**Nombre d'hipoteques**", value=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Nombre d'hipoteques", "level"):,.0f}""", delta=f"""{indicator_year(table_catalunya_y, table_catalunya_m, str(selected_year_n), "Nombre d'hipoteques", "var")}%""")
                with right:
                    st.metric(label="**Import d'hipoteques** (Milers €)", value=f"""{indicator_year(table_catalunya_y, table_catalunya_q, str(selected_year_n), "Import d'hipoteques", "level"):,.0f}""", delta=f"""{indicator_year(table_catalunya_y, table_catalunya_m, str(selected_year_n), "Import d'hipoteques", "var")}%""")

            selected_columns = ["Nombre d'hipoteques", "Import d'hipoteques"]
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_catalunya_q, 2020).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_catalunya_q, 2014), f"{selected_index}_Catalunya.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_catalunya_y, 2014, rounded=False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_catalunya_y, 2014, rounded=False), f"{selected_index}_Catalunya_anual.xlsx"), unsafe_allow_html=True)

            left, right = st.columns((1,1))
            with left:
                st.plotly_chart(line_plotly(table_catalunya_m, ["Nombre d'hipoteques"], "Evolució mensual del nombre d'hipoteques", "Nombre d'hipoteques",  "Data"), use_container_width=True, responsive=True)
                st.plotly_chart(line_plotly(table_catalunya_m, ["Import d'hipoteques"], "Evolució mensual de l'import d'hipoteques (Milers €)", "Import d'hipoteques",  "Data"), use_container_width=True, responsive=True)
            with right:
                st.plotly_chart(bar_plotly(table_catalunya_y, ["Nombre d'hipoteques"], "Evolució anual del nombre d'hipoteques", "Nombre d'hipoteques",  2005), use_container_width=True, responsive=True)
                st.plotly_chart(bar_plotly(table_catalunya_y, ["Import d'hipoteques"], "Evolució anual de l'import d'hipoteques (Milers €)", "Import d'hipoteques",  2005), use_container_width=True, responsive=True)

    if selected_indicator=="Sector residencial":
        selected_type = st.sidebar.radio("**Mercat de venda o lloguer**", ("Venda", "Lloguer"))
        if selected_type=="Venda":
            index_names = ["Producció", "Compravendes", "Preus", "Superfície"]
            selected_index = st.sidebar.selectbox("**Selecciona un indicador:**", index_names)
            max_year=datetime.now().year-1
            available_years = list(range(2018, datetime.now().year))
            selected_year_n = st.sidebar.selectbox("****Selecciona un any:****", available_years, available_years.index(2023))
            if selected_index=="Producció":
                min_year=2008
                st.subheader("PRODUCCIÓ D'HABITATGES A CATALUNYA")
                st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
                table_cat_m = tidy_Catalunya_m(DT_monthly, ["Fecha"] + concatenate_lists(["iniviv_","finviv_"], "Catalunya"), f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Habitatges iniciats", "Habitatges acabats"])     
                table_Catalunya = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["iniviv_","iniviv_uni_", "iniviv_pluri_","finviv_","finviv_uni_", "finviv_pluri_"], "Catalunya") + concatenate_lists(["calprov_", "calprovpub_", "calprovpriv_", "caldef_", "caldefpub_", "caldefpriv_"], "Cataluña"), f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Habitatges iniciats","Habitatges iniciats unifamiliars", "Habitatges iniciats plurifamiliars", "Habitatges acabats", "Habitatges acabats unifamiliars", "Habitatges acabats plurifamiliars",
                                                                                                                                                                                                                                                                                                                                   "Qualificacions provisionals d'HPO", "Qualificacions provisionals d'HPO (Promotor públic)", "Qualificacions provisionals d'HPO (Promotor privat)", 
                                                                                                                                                                                                                                                                                                                                    "Qualificacions definitives d'HPO",  "Qualificacions definitives d'HPO (Promotor públic)", "Qualificacions definitives d'HPO (Promotor privat)"])
                table_Catalunya_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"] + concatenate_lists(["iniviv_","iniviv_uni_", "iniviv_pluri_","finviv_","finviv_uni_", "finviv_pluri_"], "Catalunya") + concatenate_lists(["calprov_", "calprovpub_", "calprovpriv_", "caldef_", "caldefpub_", "caldefpriv_"], "Cataluña"), min_year, max_year,["Any","Habitatges iniciats","Habitatges iniciats unifamiliars", "Habitatges iniciats plurifamiliars", "Habitatges acabats", "Habitatges acabats unifamiliars", "Habitatges acabats plurifamiliars",
                                                                                                                                                                                                                                                                                                                                              "Qualificacions provisionals d'HPO", "Qualificacions provisionals d'HPO (Promotor públic)", "Qualificacions provisionals d'HPO (Promotor privat)", 
                                                                                                                                                                                                                                                                                                                                                "Qualificacions definitives d'HPO",  "Qualificacions definitives d'HPO (Promotor públic)", "Qualificacions definitives d'HPO (Promotor privat)"])
                table_Catalunya_pluri = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["iniviv_pluri_50m2_","iniviv_pluri_5175m2_", "iniviv_pluri_76100m2_","iniviv_pluri_101125m2_", "iniviv_pluri_126150m2_", "iniviv_pluri_150m2_"], "Catalunya"), f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Plurifamiliar fins a 50m2","Plurifamiliar entre 51m2 i 75 m2", "Plurifamiliar entre 76m2 i 100m2","Plurifamiliar entre 101m2 i 125m2", "Plurifamiliar entre 126m2 i 150m2", "Plurifamiliar de més de 150m2"])
                table_Catalunya_uni = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["iniviv_uni_50m2_","iniviv_uni_5175m2_", "iniviv_uni_76100m2_","iniviv_uni_101125m2_", "iniviv_uni_126150m2_", "iniviv_uni_150m2_"], "Catalunya"), f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Unifamiliar fins a 50m2","Unifamiliar entre 51m2 i 75 m2", "Unifamiliar entre 76m2 i 100m2","Unifamiliar entre 101m2 i 125m2", "Unifamiliar entre 126m2 i 150m2", "Unifamiliar de més de 150m2"])
                left, center, right = st.columns((1,1,1))
                with left:
                    st.metric(label="**Habitatges iniciats**", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Habitatges iniciats", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_cat_m, str(selected_year_n), "Habitatges iniciats", "var", "month")}%""")
                with center:
                    try:
                        st.metric(label="**Habitatges iniciats plurifamiliars**", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Habitatges iniciats plurifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Habitatges iniciats plurifamiliars", "var")}%""")
                    except IndexError:
                        st.metric(label="**Habitatges iniciats plurifamiliars**", value="Pendent")
                with right:
                    try:
                        st.metric(label="**Habitatges iniciats unifamiliars**", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Habitatges iniciats unifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Habitatges iniciats unifamiliars", "var")}%""")
                    except IndexError:
                        st.metric(label="**Habitatges iniciats unifamiliars**", value="Pendent")
                left, center, right = st.columns((1,1,1))
                with left:
                    st.metric(label="**Habitatges acabats**", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Habitatges acabats", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_cat_m, str(selected_year_n), "Habitatges acabats", "var", "month")}%""")
                with center:
                    try:
                        st.metric(label="**Habitatges acabats plurifamiliars**", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Habitatges acabats plurifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Habitatges acabats plurifamiliars", "var")}%""")
                    except IndexError:
                        st.metric(label="**Habitatges acabats plurifamiliars**", value="Pendent")
                with right:
                    try:
                        st.metric(label="**Habitatges acabats unifamiliars**", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Habitatges acabats unifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Habitatges acabats unifamiliars", "var")}%""")
                    except IndexError:
                        st.metric(label="**Habitatges acabats unifamiliars**", value="Pendent")
                left, center, right = st.columns((1,1,1))
                with left:
                    st.metric(label="**Qualificacions provisionals d'HPO**", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Qualificacions provisionals d'HPO", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Qualificacions provisionals d'HPO", "var")}%""")
                with center:
                    try:
                        st.metric(label="**Qualificacions provisionals d'HPO** (Promotor públic)", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Qualificacions provisionals d'HPO (Promotor públic)", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Qualificacions provisionals d'HPO (Promotor públic)", "var")}%""")
                    except IndexError:
                        st.metric(label="**Qualificacions provisionals d'HPO** (Promotor públic)", value="Pendent")
                with right:
                    try:
                        st.metric(label="**Qualificacions provisionals d'HPO** (Promotor privat)", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Qualificacions provisionals d'HPO (Promotor privat)", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Qualificacions provisionals d'HPO (Promotor privat)", "var")}%""")
                    except IndexError:
                        st.metric(label="**Qualificacions provisionals d'HPO** (Promotor privat)", value="Pendent")
                left, center, right = st.columns((1,1,1))
                with left:
                    st.metric(label="**Qualificacions definitives d'HPO**", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Qualificacions definitives d'HPO", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Qualificacions definitives d'HPO", "var")}%""")
                with center:
                    try:
                        st.metric(label="**Qualificacions definitives d'HPO** (Promotor públic)", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Qualificacions definitives d'HPO (Promotor públic)", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n),  "Qualificacions definitives d'HPO (Promotor públic)", "var")}%""")
                    except IndexError:
                        st.metric(label="**Qualificacions definitives d'HPO** (Promotor públic)", value="Pendent")
                with right:
                    try:
                        st.metric(label="**Qualificacions definitives d'HPO** (Promotor privat)", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Qualificacions definitives d'HPO (Promotor privat)", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n),  "Qualificacions definitives d'HPO (Promotor privat)", "var")}%""")
                    except IndexError:
                        st.metric(label="**Qualificacions definitives d'HPO** (Promotor privat)", value="Pendent")
                # st.markdown("La producció d'habitatge a Catalunya al 2022")
                
                # selected_columns = st.multiselect("**Selecció d'indicadors:**", table_Catalunya.columns.tolist(), default=table_Catalunya.columns.tolist())
                selected_columns_ini = [col for col in table_Catalunya.columns.tolist() if col.startswith("Habitatges iniciats ")]
                selected_columns_fin = [col for col in table_Catalunya.columns.tolist() if col.startswith("Habitatges acabats ")]
                selected_columns_aux = ["Habitatges iniciats", "Habitatges acabats"]
                selected_columns_aux1 = ["Qualificacions provisionals d'HPO (Promotor públic)", "Qualificacions provisionals d'HPO (Promotor privat)"]
                selected_columns_aux2 = ["Qualificacions definitives d'HPO (Promotor públic)", "Qualificacions definitives d'HPO (Promotor privat)"]
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
                st.markdown(table_trim(table_Catalunya, 2020).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_trim(table_Catalunya, 2008), f"{selected_index}_Catalunya.xlsx"), unsafe_allow_html=True)
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES ANUALS**")
                st.markdown(table_year(table_Catalunya_y, 2008).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_year(table_Catalunya_y, 2008), f"{selected_index}_Catalunya_anual.xlsx"), unsafe_allow_html=True)

                left_col, right_col = st.columns((1,1))
                with left_col:
                    st.plotly_chart(line_plotly(table_Catalunya, selected_columns_aux, "Evolució trimestral de la producció d'habitatges", "Nombre d'habitatges"), use_container_width=True, responsive=True)
                    st.plotly_chart(stacked_bar_plotly(table_Catalunya_y, selected_columns_aux1, "Qualificacions provisionals de protecció oficial segons tipus de promotor", "Nombre d'habitatges", 2014), use_container_width=True, responsive=True)
                    st.plotly_chart(area_plotly(table_Catalunya[selected_columns_ini], selected_columns_ini, "Habitatges iniciats per tipologia", "Habitatges iniciats", "2013T1"), use_container_width=True, responsive=True)
                    st.plotly_chart(area_plotly(table_Catalunya_pluri, table_Catalunya_pluri.columns.tolist(), "Habitatges iniciats plurifamiliars per superfície construida", "Habitatges iniciats", "2014T1"), use_container_width=True, responsive=True)
                with right_col:
                    st.plotly_chart(bar_plotly(table_Catalunya_y, selected_columns_aux, "Evolució anual de la producció d'habitatges", "Nombre d'habitatges", 2005), use_container_width=True, responsive=True) 
                    st.plotly_chart(stacked_bar_plotly(table_Catalunya_y, selected_columns_aux2, "Qualificacions definitives de protecció oficial segons tipus de promotor", "Nombre d'habitatges", 2014), use_container_width=True, responsive=True)
                    st.plotly_chart(area_plotly(table_Catalunya[selected_columns_fin], selected_columns_fin, "Habitatges acabats per tipologia", "Habitatges acabats", "2013T1"), use_container_width=True, responsive=True)
                    st.plotly_chart(area_plotly(table_Catalunya_uni, table_Catalunya_uni.columns.tolist(), "Habitatges iniciats unifamiliars per superfície construida", "Habitatges iniciats", "2014T1"), use_container_width=True, responsive=True)

            if selected_index=="Compravendes":
                min_year=2014
                st.subheader("COMPRAVENDES D'HABITATGES A CATALUNYA")
                st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
                table_Catalunya = tidy_Catalunya(DT_terr, ["Fecha", "trvivt_Catalunya", "trvivs_Catalunya", "trvivn_Catalunya"], f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Compravendes d'habitatge total", "Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"])
                table_Catalunya_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha", "trvivt_Catalunya", "trvivs_Catalunya", "trvivn_Catalunya"], min_year, max_year,["Any", "Compravendes d'habitatge total", "Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"])
                left, center, right = st.columns((1,1,1))
                with left:
                    st.metric(label="**Compravendes d'habitatge total**", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Compravendes d'habitatge total", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Compravendes d'habitatge total", "var")}%""")
                with center:
                    st.metric(label="**Compravendes d'habitatge de segona mà**", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Compravendes d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Compravendes d'habitatge de segona mà", "var")}%""")
                with right:
                    st.metric(label="**Compravendes d'habitatge nou**", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Compravendes d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Compravendes d'habitatge nou", "var")}%""")
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
                st.markdown(table_trim(table_Catalunya, 2020).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_trim(table_Catalunya, 2014), f"{selected_index}_Catalunya.xlsx"), unsafe_allow_html=True)
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES ANUALS**")
                st.markdown(table_year(table_Catalunya_y, 2014).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_year(table_Catalunya_y, 2014), f"{selected_index}_Catalunya_anual.xlsx"), unsafe_allow_html=True)
                left_col, right_col = st.columns((1,1))
                with left_col:
                    st.plotly_chart(line_plotly(table_Catalunya,  table_Catalunya.columns.tolist(), "Evolució trimestral de les compravendes d'habitatge per tipologia", "Nombre de compravendes"), use_container_width=True, responsive=True)
                with right_col:
                    st.plotly_chart(stacked_bar_plotly(table_Catalunya_y,  table_Catalunya.columns.tolist()[1:3], "Evolució anual de les compravendes d'habitatge per tipologia", "Nombre de compravendes", 2014), use_container_width=True, responsive=True)
            if selected_index=="Preus":
                min_year=2014
                st.subheader("PREUS PER M\u00b2 CONSTRUÏT")
                st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
                table_Catalunya = tidy_Catalunya(DT_terr, ["Fecha", "prvivt_Catalunya", "prvivs_Catalunya", "prvivn_Catalunya"], f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Preu d'habitatge total", "Preus d'habitatge de segona mà", "Preus d'habitatge nou"])
                table_Catalunya_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha", "prvivt_Catalunya", "prvivs_Catalunya", "prvivn_Catalunya"], min_year, max_year,["Any", "Preu d'habitatge total", "Preus d'habitatge de segona mà", "Preus d'habitatge nou"])
                left, center, right = st.columns((1,1,1))
                with left:
                    st.metric(label="**Preu d'habitatge total** (€/m\u00b2)", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Preu d'habitatge total", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Preu d'habitatge total", "var")}%""")
                with center:
                    st.metric(label="**Preu d'habitatge de segona mà** (€/m\u00b2)", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Preus d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Preus d'habitatge de segona mà", "var")}%""")
                with right:
                    st.metric(label="**Preu d'habitatge nou** (€/m\u00b2)", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Preus d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Preus d'habitatge nou", "var")}%""")
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
                st.markdown(table_trim(table_Catalunya, 2020, True, False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_trim(table_Catalunya, 2014, True, False), f"{selected_index}_Catalunya.xlsx"), unsafe_allow_html=True)
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES ANUALS**")
                st.markdown(table_year(table_Catalunya_y, 2014, True, False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_year(table_Catalunya_y, 2014, True, False), f"{selected_index}_Catalunya_anual.xlsx"), unsafe_allow_html=True)
                left_col, right_col = st.columns((1,1))
                with left_col:
                    st.plotly_chart(line_plotly(table_Catalunya, table_Catalunya.columns.tolist(), "Evolució trimestral dels preus per m\u00b2 construït per tipologia d'habitatge", "€/m\u00b2 construït"), use_container_width=True, responsive=True)
                with right_col:
                    st.plotly_chart(bar_plotly(table_Catalunya_y, table_Catalunya.columns.tolist(), "Evolució anual dels preus per m\u00b2 construït per tipologia d'habitatge", "€/m\u00b2 construït", 2014), use_container_width=True, responsive=True)
            if selected_index=="Superfície":
                min_year=2014
                st.subheader("SUPERFÍCIE EN M\u00b2 CONSTRUÏTS")
                st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
                table_Catalunya = tidy_Catalunya(DT_terr, ["Fecha", "supert_Catalunya", "supers_Catalunya", "supern_Catalunya"], f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Superfície mitjana total", "Superfície mitjana d'habitatge de segona mà", "Superfície mitjana d'habitatge nou"])
                table_Catalunya_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha", "supert_Catalunya", "supers_Catalunya", "supern_Catalunya"], min_year, max_year,["Any", "Superfície mitjana total", "Superfície mitjana d'habitatge de segona mà", "Superfície mitjana d'habitatge nou"])
                left, center, right = st.columns((1,1,1))
                with left:
                    st.metric(label="**Superfície mitjana** (m\u00b2)", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Superfície mitjana total", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Superfície mitjana total", "var")}%""")
                with center:
                    st.metric(label="**Superfície d'habitatges de segona mà** (m\u00b2)", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Superfície mitjana d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Superfície mitjana d'habitatge de segona mà", "var")}%""")
                with right:
                    st.metric(label="**Superfície d'habitatges nous** (m\u00b2)", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Superfície mitjana d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Superfície mitjana d'habitatge nou", "var")}%""")                
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
                st.markdown(table_trim(table_Catalunya, 2020, True, False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_trim(table_Catalunya, 2014, True, False), f"{selected_index}_Catalunya.xlsx"), unsafe_allow_html=True)
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES ANUALS**")
                st.markdown(table_year(table_Catalunya_y, 2014, True, False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_year(table_Catalunya_y, 2014, True, False), f"{selected_index}_Catalunya_anual.xlsx"), unsafe_allow_html=True)
                left_col, right_col = st.columns((1,1))
                with left_col:
                    st.plotly_chart(line_plotly(table_Catalunya, table_Catalunya.columns.tolist(), "Evolució trimestral de la superfície mitjana per tipologia d'habitatge", "m\u00b2 construïts"), use_container_width=True, responsive=True)
                with right_col:
                    st.plotly_chart(bar_plotly(table_Catalunya_y, table_Catalunya.columns.tolist(), "Evolució anual de la superfície mitjana per tipologia d'habitatge", "m\u00b2 construïts", 2014), use_container_width=True, responsive=True)   
        if selected_type=="Lloguer":
            st.subheader("MERCAT DE LLOGUER")
            max_year=datetime.now().year-1
            available_years = list(range(2018, max_year+1))
            selected_year_n = st.sidebar.selectbox("****Selecciona un any:****", available_years, available_years.index(max_year))
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            min_year=2014
            table_Catalunya = tidy_Catalunya(DT_terr, ["Fecha", "trvivalq_Catalunya", "pmvivalq_Catalunya"], f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Nombre de contractes de lloguer", "Rendes mitjanes de lloguer"])
            table_Catalunya_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha", "trvivalq_Catalunya",  "pmvivalq_Catalunya"], min_year, max_year,["Any", "Nombre de contractes de lloguer", "Rendes mitjanes de lloguer"])
            left_col, right_col = st.columns((1,1))
            with left_col:
                st.metric(label="**Nombre de contractes de lloguer**", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Nombre de contractes de lloguer", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Nombre de contractes de lloguer", "var")}%""")
            with right_col:
                st.metric(label="**Rendes mitjanes de lloguer** (€/mes)", value=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Rendes mitjanes de lloguer", "level"):,.0f}""", delta=f"""{indicator_year(table_Catalunya_y, table_Catalunya, str(selected_year_n), "Rendes mitjanes de lloguer", "var")}%""")
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_Catalunya, 2020, True).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_Catalunya, 2014, True), f"{selected_type}_Catalunya.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_Catalunya_y, 2014, True).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_Catalunya_y, 2014, True), f"{selected_type}_Catalunya_anual.xlsx"), unsafe_allow_html=True)
            left_col, right_col = st.columns((1,1))
            with left_col:
                st.plotly_chart(line_plotly(table_Catalunya, ["Rendes mitjanes de lloguer"], "Evolució trimestral de les rendes mitjanes de lloguer a Catalunya", "€/mes"), use_container_width=True, responsive=True)
                st.plotly_chart(line_plotly(table_Catalunya, ["Nombre de contractes de lloguer"], "Evolució trimestral dels contractes registrats d'habitatges en lloguer a Catalunya", "Nombre de contractes de lloguer"), use_container_width=True, responsive=True)
            with right_col:
                st.plotly_chart(bar_plotly(table_Catalunya_y, ["Rendes mitjanes de lloguer"], "Evolució anual de les rendes mitjanes de lloguer a Catalunya", "€/mes", 2005), use_container_width=True, responsive=True)   
                st.plotly_chart(bar_plotly(table_Catalunya_y, ["Nombre de contractes de lloguer"], "Evolució anual dels contractes registrats d'habitatges en lloguer a Catalunya", "Nombre de contractes de lloguer", 2005), use_container_width=True, responsive=True)  
if selected == "Províncies i àmbits":
    st.sidebar.header("**PROVÍNCIES I ÀMBITS TERRITORIALS DE CATALUNYA**")
    selected_type = st.sidebar.radio("**Mercat de venda o lloguer**", ("Venda", "Lloguer"))
    selected_option = st.sidebar.radio("**Selecciona un tipus d'àrea geogràfica:**", ["Províncies", "Àmbits territorials"])
    if selected_type=="Venda":
        st.sidebar.header("")
        prov_names = ["Barcelona", "Girona", "Tarragona", "Lleida"]
        ambit_names = ["Alt Pirineu i Aran","Camp de Tarragona","Comarques centrals","Comarques gironines","Metropolità","Penedès","Ponent","Terres de l'Ebre"]
        ambit_names_aux = ["Alt Pirineu i Aran","Camp de Tarragona","Comarques Centrals","Comarques Gironines","Metropolità","Penedès","Ponent","Terres de l'Ebre"]
        if selected_option=="Àmbits territorials":
            selected_geo = st.sidebar.selectbox('**Selecciona un àmbit territorial:**', ambit_names, index= ambit_names.index("Metropolità"))
            index_indicator = ["Producció", "Compravendes", "Preus", "Superfície"]
            selected_index = st.sidebar.selectbox("**Selecciona un indicador:**", index_indicator)
            max_year=datetime.now().year-1
            available_years = list(range(2018, datetime.now().year))
            selected_year_n = st.sidebar.selectbox("**Selecciona un any:**", available_years, available_years.index(2023))
            if selected_index=="Producció":
                min_year=2008
                st.subheader(f"PRODUCCIÓ D'HABITATGES A L'ÀMBIT: {selected_geo.upper()}")
                st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
                table_province = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["iniviv_","iniviv_uni_", "iniviv_pluri_","finviv_","finviv_uni_", "finviv_pluri_"], selected_geo), f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Habitatges iniciats","Habitatges iniciats unifamiliars", "Habitatges iniciats plurifamiliars", "Habitatges acabats", "Habitatges acabats unifamiliars", "Habitatges acabats plurifamiliars"])
                table_province_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"] + concatenate_lists(["iniviv_","iniviv_uni_", "iniviv_pluri_","finviv_","finviv_uni_", "finviv_pluri_"], selected_geo), min_year, max_year,["Any","Habitatges iniciats","Habitatges iniciats unifamiliars", "Habitatges iniciats plurifamiliars", "Habitatges acabats", "Habitatges acabats unifamiliars", "Habitatges acabats plurifamiliars"])
                table_province_pluri = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["iniviv_pluri_50m2_","iniviv_pluri_5175m2_", "iniviv_pluri_76100m2_","iniviv_pluri_101125m2_", "iniviv_pluri_126150m2_", "iniviv_pluri_150m2_"], selected_geo), f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Plurifamiliar fins a 50m2","Plurifamiliar entre 51m2 i 75 m2", "Plurifamiliar entre 76m2 i 100m2","Plurifamiliar entre 101m2 i 125m2", "Plurifamiliar entre 126m2 i 150m2", "Plurifamiliar de més de 150m2"])
                table_province_uni = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["iniviv_uni_50m2_","iniviv_uni_5175m2_", "iniviv_uni_76100m2_","iniviv_uni_101125m2_", "iniviv_uni_126150m2_", "iniviv_uni_150m2_"], selected_geo), f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Unifamiliar fins a 50m2","Unifamiliar entre 51m2 i 75 m2", "Unifamiliar entre 76m2 i 100m2","Unifamiliar entre 101m2 i 125m2", "Unifamiliar entre 126m2 i 150m2", "Unifamiliar de més de 150m2"])
                left, center, right = st.columns((1,1,1))
                with left:
                    st.metric(label="**Habitatges iniciats**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges iniciats", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges iniciats", "var")}%""")
                with center:
                    st.metric(label="**Habitatges iniciats plurifamiliars**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges iniciats plurifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges iniciats plurifamiliars", "var")}%""")
                with right:
                    st.metric(label="**Habitatges iniciats unifamiliars**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges iniciats unifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges iniciats unifamiliars", "var")}%""")
                left, center, right = st.columns((1,1,1))
                with left:
                    st.metric(label="**Habitatges acabats**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges acabats", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges acabats", "var")}%""")
                with center:
                    st.metric(label="**Habitatges acabats plurifamiliars**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges acabats plurifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges acabats plurifamiliars", "var")}%""")
                with right:
                    st.metric(label="**Habitatges acabats unifamiliars**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges acabats unifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges acabats unifamiliars", "var")}%""")
                selected_columns_ini = [col for col in table_province.columns.tolist() if col.startswith("Habitatges iniciats ")]
                selected_columns_fin = [col for col in table_province.columns.tolist() if col.startswith("Habitatges acabats ")]
                selected_columns_aux = ["Habitatges iniciats", "Habitatges acabats"]
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
                st.markdown(table_trim(table_province, 2020).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_trim(table_province, 2008), f"{selected_index}_{selected_geo}.xlsx"), unsafe_allow_html=True)
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES ANUALS**")
                st.markdown(table_year(table_province_y, 2008, rounded=False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_year(table_province_y, 2008, rounded=False), f"{selected_index}_{selected_geo}_anual.xlsx"), unsafe_allow_html=True)
                left_col, right_col = st.columns((1,1))
                with left_col:
                    st.plotly_chart(line_plotly(table_province, selected_columns_aux, "Evolució trimestral de la producció d'habitatges", "Nombre d'habitatges"), use_container_width=True, responsive=True)
                    st.plotly_chart(area_plotly(table_province[selected_columns_ini], selected_columns_ini, "Habitatges iniciats per tipologia", "Habitatges iniciats", "2013T1"), use_container_width=True, responsive=True)
                    st.plotly_chart(area_plotly(table_province_pluri, table_province_pluri.columns.tolist(), "Habitatges iniciats plurifamiliars per superfície construida", "Habitatges iniciats", "2014T1"), use_container_width=True, responsive=True)
                with right_col:
                    st.plotly_chart(bar_plotly(table_province_y, selected_columns_aux, "Evolució anual de la producció d'habitatges", "Nombre d'habitatges", 2005), use_container_width=True, responsive=True) 
                    st.plotly_chart(area_plotly(table_province[selected_columns_fin], selected_columns_fin, "Habitatges acabats per tipologia", "Habitatges acabats", "2013T1"), use_container_width=True, responsive=True)
                    st.plotly_chart(area_plotly(table_province_uni, table_province_uni.columns.tolist(), "Habitatges iniciats unifamiliars per superfície construida", "Habitatges iniciats", "2014T1"), use_container_width=True, responsive=True)

            if selected_index=="Compravendes":
                min_year=2014
                st.subheader(f"COMPRAVENDES D'HABITATGE A L'ÀMBIT: {selected_geo.upper()}")
                st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
                table_province = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["trvivt_", "trvivs_", "trvivn_"], selected_geo), f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Compravendes d'habitatge total", "Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"])
                table_province_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"] + concatenate_lists(["trvivt_", "trvivs_", "trvivn_"], selected_geo), min_year, max_year,["Any", "Compravendes d'habitatge total", "Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"])
                left, center, right = st.columns((1,1,1))
                with left:
                    st.metric(label="**Compravendes d'habitatge total**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Compravendes d'habitatge total", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Compravendes d'habitatge total", "var")}%""")
                with center:
                    st.metric(label="**Compravendes d'habitatge de segona mà**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Compravendes d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Compravendes d'habitatge de segona mà", "var")}%""")
                with right:
                    st.metric(label="**Compravendes d'habitatge nou**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Compravendes d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Compravendes d'habitatge nou", "var")}%""") 
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
                st.markdown(table_trim(table_province, 2020).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_trim(table_province, 2014), f"{selected_index}_{selected_geo}.xlsx"), unsafe_allow_html=True)
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES ANUALS**")
                st.markdown(table_year(table_province_y, 2014, rounded=False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_year(table_province_y, 2014, rounded=False), f"{selected_index}_{selected_geo}_anual.xlsx"), unsafe_allow_html=True)

                left_col, right_col = st.columns((1,1))
                with left_col:
                    st.plotly_chart(line_plotly(table_province, table_province.columns.tolist(), "Evolució trimestral de les compravendes d'habitatge per tipologia", "Nombre de compravendes"), use_container_width=True, responsive=True)
                with right_col:
                    st.plotly_chart(bar_plotly(table_province_y, table_province.columns.tolist(), "Evolució anual de les compravendes d'habitatge per tipologia", "Nombre de compravendes", 2005), use_container_width=True, responsive=True) 
            if selected_index=="Preus":
                min_year=2014
                st.subheader(f"PREUS PER M\u00b2 CONSTRUÏT D'HABITATGE A L'ÀMBIT: {selected_geo.upper()}")
                st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
                
                table_province = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["prvivt_", "prvivs_", "prvivn_"], selected_geo), f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Preu d'habitatge total", "Preus d'habitatge de segona mà", "Preus d'habitatge nou"])
                table_province_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"] + concatenate_lists(["prvivt_", "prvivs_", "prvivn_"], selected_geo), min_year, max_year,["Any", "Preu d'habitatge total", "Preus d'habitatge de segona mà", "Preus d'habitatge nou"])
                left, center, right = st.columns((1,1,1))
                with left:
                    st.metric(label="**Preu d'habitatge total** (€/m\u00b2)", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Preu d'habitatge total", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Preu d'habitatge total", "var")}%""")
                with center:
                    st.metric(label="**Preus d'habitatge de segona mà** (€/m\u00b2)", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Preus d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Preus d'habitatge de segona mà", "var")}%""")
                with right:
                    st.metric(label="**Preus d'habitatge nou** (€/m\u00b2)", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Preus d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Preus d'habitatge nou", "var")}%""") 
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
                st.markdown(table_trim(table_province, 2020, True, False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_trim(table_province, 2014, True, False), f"{selected_index}_{selected_geo}.xlsx"), unsafe_allow_html=True)
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES ANUALS**")
                st.markdown(table_year(table_province_y, 2014, True, False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_year(table_province_y, 2014, True, False), f"{selected_index}_{selected_geo}_anual.xlsx"), unsafe_allow_html=True)

                left_col, right_col = st.columns((1,1))
                with left_col:
                    st.plotly_chart(line_plotly(table_province, table_province.columns.tolist(), "Evolució trimestral dels preus per m\u00b2 construït per tipologia d'habitatge", "€/m\u00b2 construït"), use_container_width=True, responsive=True)
                with right_col:
                    st.plotly_chart(bar_plotly(table_province_y, table_province.columns.tolist(), "Evolució anual dels preus per m\u00b2 construït per tipologia d'habitatge", "€/m\u00b2 construït", 2005), use_container_width=True, responsive=True) 
            if selected_index=="Superfície":
                min_year=2014
                st.subheader(f"SUPERFÍCIE EN M\u00b2 CONSTRUÏTS D'HABITATGE A L'ÀMBIT: {selected_geo.upper()}")
                st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
                
                table_province = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["supert_", "supers_", "supern_"], selected_geo), f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Superfície mitjana total", "Superfície mitjana d'habitatge de segona mà", "Superfície mitjana d'habitatge nou"])
                table_province_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"]  + concatenate_lists(["supert_", "supers_", "supern_"], selected_geo), min_year, max_year,["Any","Superfície mitjana total", "Superfície mitjana d'habitatge de segona mà", "Superfície mitjana d'habitatge nou"])
                left, center, right = st.columns((1,1,1))
                with left:
                    st.metric(label="**Superfície mitjana** (m\u00b2)", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Superfície mitjana total", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Superfície mitjana total", "var")}%""")
                with center:
                    st.metric(label="**Superfície d'habitatges de segona mà** (m\u00b2)", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Superfície mitjana d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Superfície mitjana d'habitatge de segona mà", "var")}%""")
                with right:
                    st.metric(label="**Superfície d'habitatges nous** (m\u00b2)", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Superfície mitjana d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Superfície mitjana d'habitatge nou", "var")}%""") 
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
                st.markdown(table_trim(table_province, 2020, True, False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_trim(table_province, 2014, True, False), f"{selected_index}_{selected_geo}.xlsx"), unsafe_allow_html=True)
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES ANUALS**")
                st.markdown(table_year(table_province_y, 2014, True, False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_year(table_province_y, 2014, True, False), f"{selected_index}_{selected_geo}_anual.xlsx"), unsafe_allow_html=True)
                left_col, right_col = st.columns((1,1))
                with left_col:
                    st.plotly_chart(line_plotly(table_province, table_province.columns.tolist(), "Evolució trimestral de la superfície mitjana en m\u00b2 construïts per tipologia d'habitatge", "m\u00b2 construït"), use_container_width=True, responsive=True)
                with right_col:
                    st.plotly_chart(bar_plotly(table_province_y, table_province.columns.tolist(), "Evolució anual de la superfície mitjana en m\u00b2 construïts per tipologia d'habitatge", "m\u00b2 construït", 2005), use_container_width=True, responsive=True) 
        if selected_option=="Províncies":
            selected_geo = st.sidebar.selectbox('**Selecciona una província:**', prov_names, index= prov_names.index("Barcelona"))
            index_indicator = ["Producció", "Compravendes", "Preus", "Superfície"]
            selected_index = st.sidebar.selectbox("**Selecciona un indicador:**", index_indicator)
            max_year=datetime.now().year-1
            available_years = list(range(2018, datetime.now().year))
            selected_year_n = st.sidebar.selectbox("**Selecciona un any:**", available_years, available_years.index(2023))
            if selected_index=="Producció":
                min_year=2008
                st.subheader(f"PRODUCCIÓ D'HABITATGES A {selected_geo.upper()}")
                st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
                table_province_m = tidy_Catalunya_m(DT_monthly, ["Fecha"] + concatenate_lists(["iniviv_","finviv_"], selected_geo), f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Habitatges iniciats", "Habitatges acabats"])     
                table_province = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["iniviv_","iniviv_uni_", "iniviv_pluri_","finviv_","finviv_uni_", "finviv_pluri_"], selected_geo), f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Habitatges iniciats","Habitatges iniciats unifamiliars", "Habitatges iniciats plurifamiliars", "Habitatges acabats", "Habitatges acabats unifamiliars", "Habitatges acabats plurifamiliars"])
                table_province_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"] + concatenate_lists(["iniviv_","iniviv_uni_", "iniviv_pluri_","finviv_","finviv_uni_", "finviv_pluri_"], selected_geo), min_year, max_year,["Any","Habitatges iniciats","Habitatges iniciats unifamiliars", "Habitatges iniciats plurifamiliars", "Habitatges acabats", "Habitatges acabats unifamiliars", "Habitatges acabats plurifamiliars"])
                table_province_pluri = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["iniviv_pluri_50m2_","iniviv_pluri_5175m2_", "iniviv_pluri_76100m2_","iniviv_pluri_101125m2_", "iniviv_pluri_126150m2_", "iniviv_pluri_150m2_"], selected_geo), f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Plurifamiliar fins a 50m2","Plurifamiliar entre 51m2 i 75 m2", "Plurifamiliar entre 76m2 i 100m2","Plurifamiliar entre 101m2 i 125m2", "Plurifamiliar entre 126m2 i 150m2", "Plurifamiliar de més de 150m2"])
                table_province_uni = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["iniviv_uni_50m2_","iniviv_uni_5175m2_", "iniviv_uni_76100m2_","iniviv_uni_101125m2_", "iniviv_uni_126150m2_", "iniviv_uni_150m2_"], selected_geo), f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Unifamiliar fins a 50m2","Unifamiliar entre 51m2 i 75 m2", "Unifamiliar entre 76m2 i 100m2","Unifamiliar entre 101m2 i 125m2", "Unifamiliar entre 126m2 i 150m2", "Unifamiliar de més de 150m2"])
                left, center, right = st.columns((1,1,1))
                with left:
                    st.metric(label="**Habitatges iniciats**", value=f"""{indicator_year(table_province_y, table_province_m, str(selected_year_n), "Habitatges iniciats", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province_m, str(selected_year_n), "Habitatges iniciats", "var", "month")}%""")                
                with center:
                    try:
                        st.metric(label="**Habitatges iniciats plurifamiliars**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges iniciats plurifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges iniciats plurifamiliars", "var")}%""")
                    except IndexError:
                        st.metric(label="**Habitatges iniciats plurifamiliars**", value="Pendent")
                with right:
                    try:
                        st.metric(label="**Habitatges iniciats unifamiliars**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges iniciats unifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges iniciats unifamiliars", "var")}%""")
                    except IndexError:
                        st.metric(label="**Habitatges iniciats unifamiliars**", value="Pendent")
                left, center, right = st.columns((1,1,1))
                with left:
                    st.metric(label="**Habitatges acabats**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges acabats", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province_m, str(selected_year_n), "Habitatges acabats", "var", "month")}%""")
                with center:
                    try:
                        st.metric(label="**Habitatges acabats plurifamiliars**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges acabats plurifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges acabats plurifamiliars", "var")}%""")
                    except IndexError:
                        st.metric(label="**Habitatges iniciats unifamiliars**", value="Pendent")
                with right:
                    try:
                        st.metric(label="**Habitatges acabats unifamiliars**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges acabats unifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Habitatges acabats unifamiliars", "var")}%""")
                    except IndexError:
                        st.metric(label="**Habitatges iniciats unifamiliars**", value="Pendent")

                selected_columns_ini = [col for col in table_province.columns.tolist() if col.startswith("Habitatges iniciats ")]
                selected_columns_fin = [col for col in table_province.columns.tolist() if col.startswith("Habitatges acabats ")]
                selected_columns_aux = ["Habitatges iniciats", "Habitatges acabats"]
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
                st.markdown(table_trim(table_province, 2020).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_trim(table_province, 2008), f"{selected_index}_{selected_geo}.xlsx"), unsafe_allow_html=True)
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES ANUALS**")
                st.markdown(table_year(table_province_y, 2008, rounded=False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_year(table_province_y, 2008, rounded=False), f"{selected_index}_{selected_geo}_anual.xlsx"), unsafe_allow_html=True)
                left_col, right_col = st.columns((1,1))
                with left_col:
                    st.plotly_chart(line_plotly(table_province, selected_columns_aux, "Evolució trimestral de la producció d'habitatges", "Nombre d'habitatges"), use_container_width=True, responsive=True)
                    st.plotly_chart(area_plotly(table_province[selected_columns_ini], selected_columns_ini, "Habitatges iniciats per tipologia", "Habitatges iniciats", "2013T1"), use_container_width=True, responsive=True)
                    st.plotly_chart(area_plotly(table_province_pluri, table_province_pluri.columns.tolist(), "Habitatges iniciats plurifamiliars per superfície construida", "Habitatges iniciats", "2014T1"), use_container_width=True, responsive=True)
                with right_col:
                    st.plotly_chart(bar_plotly(table_province_y, selected_columns_aux, "Evolució anual de la producció d'habitatges", "Nombre d'habitatges", 2005), use_container_width=True, responsive=True)
                    st.plotly_chart(area_plotly(table_province[selected_columns_fin], selected_columns_fin, "Habitatges acabats per tipologia", "Habitatges acabats", "2013T1"), use_container_width=True, responsive=True)
                    st.plotly_chart(area_plotly(table_province_uni, table_province_uni.columns.tolist(), "Habitatges iniciats unifamiliars per superfície construida", "Habitatges iniciats", "2014T1"), use_container_width=True, responsive=True)

            if selected_index=="Compravendes":
                min_year=2014
                st.subheader(f"COMPRAVENDES D'HABITATGE A {selected_geo.upper()}")
                st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
                table_province = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["trvivt_", "trvivs_", "trvivn_"], selected_geo), f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Compravendes d'habitatge total", "Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"])
                table_province_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"]  + concatenate_lists(["trvivt_", "trvivs_", "trvivn_"], selected_geo), min_year, max_year,["Any","Compravendes d'habitatge total", "Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"])
                left, center, right = st.columns((1,1,1))
                with left:
                    st.metric(label="**Compravendes d'habitatge total**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Compravendes d'habitatge total", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Compravendes d'habitatge total", "var")}%""")
                with center:
                    st.metric(label="**Compravendes d'habitatge de segona mà**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Compravendes d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Compravendes d'habitatge de segona mà", "var")}%""")
                with right:
                    st.metric(label="**Compravendes d'habitatge nou**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Compravendes d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Compravendes d'habitatge nou", "var")}%""") 
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
                st.markdown(table_trim(table_province, 2020).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_trim(table_province, 2014), f"{selected_index}_{selected_geo}.xlsx"), unsafe_allow_html=True)
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES ANUALS**")
                st.markdown(table_year(table_province_y, 2014, rounded=False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_year(table_province_y, 2014, rounded=False), f"{selected_index}_{selected_geo}_anual.xlsx"), unsafe_allow_html=True)

                left_col, right_col = st.columns((1,1))
                with left_col:
                    st.plotly_chart(line_plotly(table_province, table_province.columns.tolist(), "Evolució trimestral de les compravendes d'habitatge per tipologia", "Nombre de compravendes"), use_container_width=True, responsive=True)
                with right_col:
                    st.plotly_chart(bar_plotly(table_province_y, table_province.columns.tolist(), "Evolució anual de les compravendes d'habitatge per tipologia", "Nombre de compravendes", 2005), use_container_width=True, responsive=True)     
            if selected_index=="Preus":
                min_year=2014
                st.subheader(f"PREUS PER M\u00b2 CONSTRUÏT D'HABITATGE A {selected_geo.upper()}")
                st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
                
                table_province = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["prvivt_", "prvivs_", "prvivn_"], selected_geo), f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Preu d'habitatge total", "Preus d'habitatge de segona mà", "Preus d'habitatge nou"])
                table_province_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"]  + concatenate_lists(["prvivt_", "prvivs_", "prvivn_"], selected_geo), min_year, max_year,["Any","Preu d'habitatge total", "Preus d'habitatge de segona mà", "Preus d'habitatge nou"])
                left, center, right = st.columns((1,1,1))
                with left:
                    st.metric(label="**Preu d'habitatge total** (€/m\u00b2)", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Preu d'habitatge total", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Preu d'habitatge total", "var")}%""")
                with center:
                    st.metric(label="**Preus d'habitatge de segona mà** (€/m\u00b2)", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Preus d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Preus d'habitatge de segona mà", "var")}%""")
                with right:
                    st.metric(label="**Preus d'habitatge nou** (€/m\u00b2)", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Preus d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Preus d'habitatge nou", "var")}%""") 
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
                st.markdown(table_trim(table_province, 2020, True, False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_trim(table_province, 2014, True, False), f"{selected_index}_{selected_geo}.xlsx"), unsafe_allow_html=True)
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES ANUALS**")
                st.markdown(table_year(table_province_y, 2014, True, False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_year(table_province_y, 2014, True, False), f"{selected_index}_{selected_geo}_anual.xlsx"), unsafe_allow_html=True)
                left_col, right_col = st.columns((1,1))
                with left_col:
                    st.plotly_chart(line_plotly(table_province, table_province.columns.tolist(), "Evolució trimestral dels preus per m\u00b2 construït per tipologia d'habitatge", "€/m\u00b2 construït"), use_container_width=True, responsive=True)
                with right_col:
                    st.plotly_chart(bar_plotly(table_province_y, table_province.columns.tolist(), "Evolució anual dels preus per m\u00b2 construït per tipologia d'habitatge", "€/m\u00b2 construït", 2005), use_container_width=True, responsive=True)     
                
            if selected_index=="Superfície":
                min_year=2014
                st.subheader(f"SUPERFÍCIE EN M\u00b2 CONSTRUÏTS D'HABITATGE A {selected_geo.upper()}")
                st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
                
                table_province = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["supert_", "supers_", "supern_"], selected_geo), f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Superfície mitjana total", "Superfície mitjana d'habitatge de segona mà", "Superfície mitjana d'habitatge nou"])
                table_province_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"]  + concatenate_lists(["supert_", "supers_", "supern_"], selected_geo), min_year, max_year,["Any","Superfície mitjana total", "Superfície mitjana d'habitatge de segona mà", "Superfície mitjana d'habitatge nou"])
                left, center, right = st.columns((1,1,1))
                with left:
                    st.metric(label="**Superfície mitjana** (m\u00b2)", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Superfície mitjana total", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Superfície mitjana total", "var")}%""")
                with center:
                    st.metric(label="**Superfície d'habitatges de segona mà** (m\u00b2)", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Superfície mitjana d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Superfície mitjana d'habitatge de segona mà", "var")}%""")
                with right:
                    st.metric(label="**Superfície d'habitatges nous** (m\u00b2)", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Superfície mitjana d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Superfície mitjana d'habitatge nou", "var")}%""") 
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
                st.markdown(table_trim(table_province, 2020, True, False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_trim(table_province, 2014, True, False), f"{selected_index}_{selected_geo}.xlsx"), unsafe_allow_html=True)
                st.markdown("")
                st.markdown("")
                # st.subheader("**DADES ANUALS**")
                st.markdown(table_year(table_province_y, 2014, True, False).to_html(), unsafe_allow_html=True)
                st.markdown(filedownload(table_year(table_province_y, 2014, True, False), f"{selected_index}_{selected_geo}_anual.xlsx"), unsafe_allow_html=True)
                left_col, right_col = st.columns((1,1))
                with left_col:
                    st.plotly_chart(line_plotly(table_province, table_province.columns.tolist(), "Evolució trimestral de la superfície mitjana per tipologia d'habitatge", "m\u00b2 construït"), use_container_width=True, responsive=True)
                with right_col:
                    st.plotly_chart(bar_plotly(table_province_y, table_province.columns.tolist(), "Evolució anual de la superfície mitjana per tipologia d'habitatge", "m\u00b2 construït", 2005), use_container_width=True, responsive=True)

    if selected_type=="Lloguer":
        st.sidebar.header("")
        prov_names = ["Barcelona", "Girona", "Tarragona", "Lleida"]
        ambit_names = ["Alt Pirineu i Aran","Camp de Tarragona","Comarques centrals","Comarques gironines","Metropolità","Penedès","Ponent","Terres de l'Ebre"]
        selected_option = st.sidebar.selectbox("**Selecciona un tipus d'àrea geogràfica:**", ["Províncies", "Àmbits territorials"])
        if selected_option=="Àmbits territorials":
            selected_geo = st.sidebar.selectbox('**Selecciona un àmbit territorial:**', ambit_names, index= ambit_names.index("Metropolità"))
            min_year=2014
            max_year=datetime.now().year-1
            available_years = list(range(2018, datetime.now().year))
            selected_year_n = st.sidebar.selectbox("**Selecciona un any:**", available_years, available_years.index(2023))
            st.subheader(f"MERCAT DE LLOGUER A L'ÀMBIT: {selected_geo.upper()}")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            table_province = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["trvivalq_", "pmvivalq_"], selected_geo), f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Nombre de contractes de lloguer", "Rendes mitjanes de lloguer"])
            table_province_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"]  + concatenate_lists(["trvivalq_", "pmvivalq_"], selected_geo), min_year, max_year,["Any","Nombre de contractes de lloguer", "Rendes mitjanes de lloguer"])
            left_col, right_col = st.columns((1,1))
            with left_col:
                st.metric(label="**Nombre de contractes de lloguer**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Nombre de contractes de lloguer", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Nombre de contractes de lloguer", "var")}%""")
            with right_col:
                st.metric(label="**Rendes mitjanes de lloguer** (€/mes)", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Rendes mitjanes de lloguer", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Rendes mitjanes de lloguer", "var")}%""")
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_province, 2020, rounded=True).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_province, 2014, rounded=True), f"{selected_type}_{selected_geo}.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_province_y, 2014, rounded=True).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_province_y, 2014, rounded=True), f"{selected_type}_{selected_geo}_anual.xlsx"), unsafe_allow_html=True)
            left_col, right_col = st.columns((1,1))
            with left_col:
                st.plotly_chart(line_plotly(table_province, ["Rendes mitjanes de lloguer"], "Evolució trimestral de les rendes mitjanes de lloguer", "€/mes"), use_container_width=True, responsive=True)
                st.plotly_chart(line_plotly(table_province, ["Nombre de contractes de lloguer"], "Evolució trimestral dels contractes registrats d'habitatges en lloguer", "Nombre de contractes"), use_container_width=True, responsive=True)
            with right_col:
                st.plotly_chart(bar_plotly(table_province_y, ["Rendes mitjanes de lloguer"], "Evolució anual de les rendes mitjanes de lloguer", "€/mes", 2005), use_container_width=True, responsive=True)
                st.plotly_chart(bar_plotly(table_province_y, ["Nombre de contractes de lloguer"], "Evolució anual dels contractes registrats d'habitatges en lloguer", "Nombre de contractes", 2005), use_container_width=True, responsive=True)
        if selected_option=="Províncies":
            selected_geo = st.sidebar.selectbox('**Selecciona una província:**', prov_names, index= prov_names.index("Barcelona"))
            min_year=2014
            max_year=datetime.now().year-1
            available_years = list(range(2018, datetime.now().year))
            selected_year_n = st.sidebar.selectbox("**Selecciona un any:**", available_years, available_years.index(2023))
            st.subheader(f"MERCAT DE LLOGUER A {selected_geo.upper()}")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            table_province = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["trvivalq_", "pmvivalq_"], selected_geo), f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Nombre de contractes de lloguer", "Rendes mitjanes de lloguer"])
            table_province_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"]  + concatenate_lists(["trvivalq_", "pmvivalq_"], selected_geo), min_year, max_year,["Any","Nombre de contractes de lloguer", "Rendes mitjanes de lloguer"])
            left_col, right_col = st.columns((1,1))
            with left_col:
                st.metric(label="**Nombre de contractes de lloguer**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Nombre de contractes de lloguer", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Nombre de contractes de lloguer", "var")}%""")
            with right_col:
                st.metric(label="**Rendes mitjanes de lloguer** (€/mes)", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Rendes mitjanes de lloguer", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Rendes mitjanes de lloguer", "var")}%""")
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_province, 2020, rounded=True).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_province, 2014, rounded=True), f"{selected_type}_{selected_geo}.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_province_y, 2014, rounded=True).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_province_y, 2014, rounded=True), f"{selected_type}_{selected_geo}_anual.xlsx"), unsafe_allow_html=True)
            left_col, right_col = st.columns((1,1))
            with left_col:
                st.plotly_chart(line_plotly(table_province, ["Rendes mitjanes de lloguer"], "Evolució trimestral de les rendes mitjanes de lloguer", "€/mes"), use_container_width=True, responsive=True)
                st.plotly_chart(line_plotly(table_province, ["Nombre de contractes de lloguer"], "Evolució trimestral dels contractes registrats d'habitatges en lloguer", "Nombre de contractes"), use_container_width=True, responsive=True)
            with right_col:
                st.plotly_chart(bar_plotly(table_province_y, ["Rendes mitjanes de lloguer"], "Evolució anual de les rendes mitjanes de lloguer", "€/mes", 2005), use_container_width=True, responsive=True)
                st.plotly_chart(bar_plotly(table_province_y, ["Nombre de contractes de lloguer"], "Evolució anual dels contractes registrats d'habitatges en lloguer", "Nombre de contractes", 2005), use_container_width=True, responsive=True)

if selected=="Comarques":
    st.sidebar.header("**COMARQUES DE CATALUNYA**")
    selected_type = st.sidebar.radio("**Mercat de venda o lloguer**", ("Venda", "Lloguer"))
    if selected_type=="Venda":
        selected_com = st.sidebar.selectbox("**Selecciona una comarca:**", sorted(maestro_mun["Comarca"].unique().tolist()), index= sorted(maestro_mun["Comarca"].unique().tolist()).index("Barcelonès"))
        index_names = ["Producció", "Compravendes", "Preus", "Superfície"]
        selected_index = st.sidebar.selectbox("**Selecciona un indicador:**", index_names)
        max_year=datetime.now().year-1
        available_years = list(range(2018, datetime.now().year))
        selected_year_n = st.sidebar.selectbox("**Selecciona un any:**", available_years, available_years.index(2023))
        if selected_index=="Producció":
            min_year=2008
            st.subheader(f"PRODUCCIÓ D'HABITATGES A LA COMARCA: {selected_com.upper()}")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            
            table_com_m = tidy_Catalunya_m(DT_monthly, ["Fecha"] + concatenate_lists(["iniviv_","finviv_"], selected_com), f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Habitatges iniciats", "Habitatges acabats"])     
            table_com = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["iniviv_","iniviv_uni_", "iniviv_pluri_","finviv_","finviv_uni_", "finviv_pluri_"], selected_com), f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Habitatges iniciats","Habitatges iniciats unifamiliars", "Habitatges iniciats plurifamiliars", "Habitatges acabats", "Habitatges acabats unifamiliars", "Habitatges acabats plurifamiliars"])
            table_com_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"] + concatenate_lists(["iniviv_","iniviv_uni_", "iniviv_pluri_","finviv_","finviv_uni_", "finviv_pluri_"], selected_com), min_year, max_year,["Any","Habitatges iniciats","Habitatges iniciats unifamiliars", "Habitatges iniciats plurifamiliars", "Habitatges acabats", "Habitatges acabats unifamiliars", "Habitatges acabats plurifamiliars"])
            table_com_pluri = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["iniviv_pluri_50m2_","iniviv_pluri_5175m2_", "iniviv_pluri_76100m2_","iniviv_pluri_101125m2_", "iniviv_pluri_126150m2_", "iniviv_pluri_150m2_"], selected_com), f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Plurifamiliar fins a 50m2","Plurifamiliar entre 51m2 i 75 m2", "Plurifamiliar entre 76m2 i 100m2","Plurifamiliar entre 101m2 i 125m2", "Plurifamiliar entre 126m2 i 150m2", "Plurifamiliar de més de 150m2"])
            table_com_uni = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["iniviv_uni_50m2_","iniviv_uni_5175m2_", "iniviv_uni_76100m2_","iniviv_uni_101125m2_", "iniviv_uni_126150m2_", "iniviv_uni_150m2_"], selected_com), f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Unifamiliar fins a 50m2","Unifamiliar entre 51m2 i 75 m2", "Unifamiliar entre 76m2 i 100m2","Unifamiliar entre 101m2 i 125m2", "Unifamiliar entre 126m2 i 150m2", "Unifamiliar de més de 150m2"])
            left, center, right = st.columns((1,1,1))
            with left:
                try:
                    st.metric(label="**Habitatges iniciats**", value=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Habitatges iniciats", "level"):,.0f}""", delta=f"""{indicator_year(table_com_y, table_com_m, str(selected_year_n), "Habitatges iniciats", "var", "month")}%""")
                except IndexError:
                    st.metric(label="**Habitatges iniciats**", value="Pendent")
            with center:
                try:
                    st.metric(label="**Habitatges iniciats plurifamiliars**", value=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Habitatges iniciats plurifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Habitatges iniciats plurifamiliars", "var")}%""")
                except IndexError:
                    st.metric(label="**Habitatges iniciats plurifamiliars**", value="Pendent")
            with right:
                try:
                    st.metric(label="**Habitatges iniciats unifamiliars**", value=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Habitatges iniciats unifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Habitatges iniciats unifamiliars", "var")}%""")
                except IndexError:
                    st.metric(label="**Habitatges iniciats unifamiliars**", value="Pendent")
            left, center, right = st.columns((1,1,1))
            with left:
                try:
                    st.metric(label="**Habitatges acabats**", value=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Habitatges acabats", "level"):,.0f}""", delta=f"""{indicator_year(table_com_y, table_com_m, str(selected_year_n), "Habitatges acabats", "var", "month")}%""")
                except IndexError:
                    st.metric(label="**Habitatges acabats**", value="Pendent")          
            with center:
                try:
                    st.metric(label="**Habitatges acabats plurifamiliars**", value=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Habitatges acabats plurifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Habitatges acabats plurifamiliars", "var")}%""")
                except IndexError:
                    st.metric(label="**Habitatges acabats unifamiliars**", value="Pendent")
            with right:
                try:
                    st.metric(label="**Habitatges acabats unifamiliars**", value=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Habitatges acabats unifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Habitatges acabats unifamiliars", "var")}%""")
                except IndexError:
                    st.metric(label="**Habitatges acabats unifamiliars**", value="Pendent")
            selected_columns_ini = [col for col in table_com.columns.tolist() if col.startswith("Habitatges iniciats ")]
            selected_columns_fin = [col for col in table_com.columns.tolist() if col.startswith("Habitatges acabats ")]
            selected_columns_aux = ["Habitatges iniciats", "Habitatges acabats"]
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_com, 2020).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_com, 2008), f"{selected_index}_{selected_com}.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_com_y, 2008, rounded=False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_com_y, 2008, rounded=False), f"{selected_index}_{selected_com}_anual.xlsx"), unsafe_allow_html=True)
            left_col, right_col = st.columns((1,1))
            with left_col:
                st.plotly_chart(line_plotly(table_com[selected_columns_aux], selected_columns_aux, "Evolució trimestral de la producció d'habitatges", "Indicador d'oferta en nivells"), use_container_width=True, responsive=True)
                st.plotly_chart(area_plotly(table_com[selected_columns_ini], selected_columns_ini, "Habitatges iniciats per tipologia", "Habitatges iniciats", "2011T1"), use_container_width=True, responsive=True)
                st.plotly_chart(area_plotly(table_com_pluri, table_com_pluri.columns.tolist(), "Habitatges iniciats plurifamiliars per superfície construida", "Habitatges iniciats", "2014T1"), use_container_width=True, responsive=True)
            with right_col:
                st.plotly_chart(bar_plotly(table_com_y[selected_columns_aux], selected_columns_aux, "Evolució anual de la produció d'habitatges", "Indicador d'oferta en nivells", 2005), use_container_width=True, responsive=True)
                st.plotly_chart(area_plotly(table_com[selected_columns_fin], selected_columns_fin, "Habitatges acabats per tipologia", "Habitatges acabats", "2011T1"), use_container_width=True, responsive=True)
                st.plotly_chart(area_plotly(table_com_uni, table_com_uni.columns.tolist(), "Habitatges iniciats unifamiliars per superfície construida", "Habitatges iniciats", "2014T1"), use_container_width=True, responsive=True)

        if selected_index=="Compravendes":
            min_year=2014
            st.subheader(f"COMPRAVENDES D'HABITATGE A LA COMARCA: {selected_com.upper()}")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            
            table_com = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["trvivt_", "trvivs_", "trvivn_"], selected_com), f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Compravendes d'habitatge total", "Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"])
            table_com_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"] + concatenate_lists(["trvivt_", "trvivs_", "trvivn_"], selected_com), min_year, max_year,["Any","Compravendes d'habitatge total", "Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"])
            left, center, right = st.columns((1,1,1))
            with left:
                st.metric(label="**Compravendes d'habitatge total**", value=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Compravendes d'habitatge total", "level"):,.0f}""", delta=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Compravendes d'habitatge total", "var")}%""")
            with center:
                st.metric(label="**Compravendes d'habitatge de segona mà**", value=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Compravendes d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Compravendes d'habitatge de segona mà", "var")}%""")
            with right:
                try:
                    st.metric(label="**Compravendes d'habitatge nou**", value=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Compravendes d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Compravendes d'habitatge nou", "var")}%""") 
                except IndexError:
                    st.metric(label="**Compravendes d'habitatge nou**", value=0)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_com, 2020).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_com, 2014), f"{selected_index}_{selected_com}.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_com_y, 2014, rounded=False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_com_y, 2014, rounded=False), f"{selected_index}_{selected_com}_anual.xlsx"), unsafe_allow_html=True)
            left_col, right_col = st.columns((1,1))
            with left_col:
                st.plotly_chart(line_plotly(table_com, table_com.columns.tolist(), "Evolució trimestral de les compravendes d'habitatge per tipologia", "Nombre de compravendes"), use_container_width=True, responsive=True)
            with right_col:
                st.plotly_chart(bar_plotly(table_com_y, table_com.columns.tolist(), "Evolució anual de les compravendes d'habitatge per tipologia", "Nombre de compravendes", 2005), use_container_width=True, responsive=True)
        if selected_index=="Preus":
            min_year=2014
            st.subheader(f"PREUS PER M\u00b2 CONSTRUÏT D'HABITATGE A LA COMARCA: {selected_com.upper()}")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            
            table_com = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["prvivt_", "prvivs_", "prvivn_"], selected_com), f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Preu d'habitatge total", "Preu d'habitatge de segona mà", "Preu d'habitatge nou"])
            table_com_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"] + concatenate_lists(["prvivt_", "prvivs_", "prvivn_"], selected_com), min_year, max_year,["Any","Preu d'habitatge total", "Preu d'habitatge de segona mà", "Preu d'habitatge nou"])
            left, center, right = st.columns((1,1,1))
            with left:
                st.metric(label="**Preu d'habitatge total** (€/m\u00b2)", value=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Preu d'habitatge total", "level"):,.0f}""", delta=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Preu d'habitatge total", "var")}%""")
            with center:
                st.metric(label="**Preu d'habitatge de segona mà** (€/m\u00b2)", value=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Preu d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Preu d'habitatge de segona mà", "var")}%""")
            with right:
                try:
                    st.metric(label="**Preu d'habitatge nou** (€/m\u00b2)", value=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Preu d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Preu d'habitatge nou", "var")}%""") 
                except IndexError:
                    st.metric(label="**Preu d'habitatge nou** (€/m\u00b2)", value=0)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_com, 2020,True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_com, 2014, True, False), f"{selected_index}_{selected_com}.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_com_y, 2014, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_com_y, 2014, True, False), f"{selected_index}_{selected_com}_anual.xlsx"), unsafe_allow_html=True)
            left_col, right_col = st.columns((1,1))
            with left_col:
                st.plotly_chart(line_plotly(table_com, table_com.columns.tolist(), "Evolució trimestral dels preus per m\u00b2 construït per tipologia d'habitatge", "€/m\u00b2 útil"), use_container_width=True, responsive=True)
            with right_col:
                st.plotly_chart(bar_plotly(table_com_y, table_com.columns.tolist(), "Evolució anual dels preus per m\u00b2 construït per tipologia d'habitatge", "€/m\u00b2 útil", 2005), use_container_width=True, responsive=True)
        if selected_index=="Superfície":
            min_year=2014
            st.subheader(f"SUPERFÍCIE EN M\u00b2 CONSTRUÏTS D'HABITATGE A LA COMARCA: {selected_com.upper()}")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            
            table_com = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["supert_", "supers_", "supern_"], selected_com), f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Superfície mitjana total", "Superfície mitjana d'habitatge de segona mà", "Superfície mitjana d'habitatge nou"])
            table_com_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"] + concatenate_lists(["supert_", "supers_", "supern_"], selected_com), min_year, max_year,["Any","Superfície mitjana total", "Superfície mitjana d'habitatge de segona mà", "Superfície mitjana d'habitatge nou"])
            left, center, right = st.columns((1,1,1))
            with left:
                st.metric(label="**Superfície mitjana** (m\u00b2)", value=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Superfície mitjana total", "level"):,.0f}""", delta=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Superfície mitjana total", "var")}%""")
            with center:
                st.metric(label="**Superfície d'habitatges de segona mà** (m\u00b2)", value=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Superfície mitjana d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Superfície mitjana d'habitatge de segona mà", "var")}%""")
            with right:
                try:
                    st.metric(label="**Superfície d'habitatges nous** (m\u00b2)", value=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Superfície mitjana d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_com_y, table_com, str(selected_year_n), "Superfície mitjana d'habitatge nou", "var")}%""") 
                except IndexError:
                    st.metric(label="**Superfície d'habitatges nous** (m\u00b2)", value=0)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_com, 2020, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_com, 2014, True, False), f"{selected_index}_{selected_com}.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_com_y, 2014, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_com_y, 2014, True, False), f"{selected_index}_{selected_com}_anual.xlsx"), unsafe_allow_html=True)

            left_col, right_col = st.columns((1,1))
            with left_col:
                st.plotly_chart(line_plotly(table_com, table_com.columns.tolist(), "Evolució trimestral de la superfície mitjana per tipologia d'habitatge", "m\u00b2 construït"), use_container_width=True, responsive=True)
            with right_col:
                st.plotly_chart(bar_plotly(table_com_y, table_com.columns.tolist(), "Evolució anual de la superfície mitjana per tipologia d'habitatge", "m\u00b2 construït", 2005), use_container_width=True, responsive=True)
    if selected_type=="Lloguer":
        selected_com = st.sidebar.selectbox("**Selecciona una comarca:**", sorted(maestro_mun["Comarca"].unique().tolist()), index= sorted(maestro_mun["Comarca"].unique().tolist()).index("Barcelonès"))
        available_years = list(range(2018, datetime.now().year))
        selected_year_n = st.sidebar.selectbox("**Selecciona un any:**", available_years, available_years.index(2023))
        min_year=2014
        max_year=datetime.now().year-1
        st.subheader(f"MERCAT DE LLOGUER A LA COMARCA: {selected_com.upper()}")
        st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
        table_province = tidy_Catalunya(DT_terr, ["Fecha"] + concatenate_lists(["trvivalq_", "pmvivalq_"], selected_com), f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Nombre de contractes de lloguer", "Rendes mitjanes de lloguer"])
        table_province_y = tidy_Catalunya_anual(DT_terr_y, ["Fecha"]  + concatenate_lists(["trvivalq_", "pmvivalq_"], selected_com), min_year, max_year,["Any","Nombre de contractes de lloguer", "Rendes mitjanes de lloguer"])
        left_col, right_col = st.columns((1,1))
        with left_col:
            st.metric(label="**Nombre de contractes de lloguer**", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Nombre de contractes de lloguer", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Nombre de contractes de lloguer", "var")}%""")
        with right_col:
            st.metric(label="**Rendes mitjanes de lloguer** (€/mes)", value=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Rendes mitjanes de lloguer", "level"):,.0f}""", delta=f"""{indicator_year(table_province_y, table_province, str(selected_year_n), "Rendes mitjanes de lloguer", "var")}%""")
        st.markdown("")
        st.markdown("")
        # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
        st.markdown(table_trim(table_province, 2020, rounded=True).to_html(), unsafe_allow_html=True)
        st.markdown(filedownload(table_trim(table_province, 2014, rounded=True), f"{selected_type}_{selected_com}.xlsx"), unsafe_allow_html=True)
        st.markdown("")
        st.markdown("")
        # st.subheader("**DADES ANUALS**")
        st.markdown(table_year(table_province_y, 2014, rounded=True).to_html(), unsafe_allow_html=True)
        st.markdown(filedownload(table_year(table_province_y, 2014, rounded=True), f"{selected_type}_{selected_com}_anual.xlsx"), unsafe_allow_html=True)
        left_col, right_col = st.columns((1,1))
        with left_col:
            st.plotly_chart(line_plotly(table_province, ["Rendes mitjanes de lloguer"], "Evolució trimestral de les rendes mitjanes de lloguer", "€/mes"), use_container_width=True, responsive=True)
            st.plotly_chart(line_plotly(table_province, ["Nombre de contractes de lloguer"], "Evolució trimestral del nombre de contractes de lloguer", "Nombre de contractes"), use_container_width=True, responsive=True)
        with right_col:
            st.plotly_chart(bar_plotly(table_province_y, ["Rendes mitjanes de lloguer"], "Evolució anual de les rendes mitjanes de lloguer", "€/mes", 2005), use_container_width=True, responsive=True)
            st.plotly_chart(bar_plotly(table_province_y, ["Nombre de contractes de lloguer"], "Evolució anual del nombre de contractes de lloguer", "Nombre de contractes", 2005), use_container_width=True, responsive=True)
if selected=="Municipis":
    st.sidebar.header("**MUNICIPIS DE CATALUNYA**")
    selected_type = st.sidebar.radio("**Mercat de venda o lloguer**", ("Venda", "Lloguer"))
    if selected_type=="Venda":
        selected_mun = st.sidebar.selectbox("**Selecciona un municipi:**", maestro_mun[maestro_mun["ADD"]=="SI"]["Municipi"].unique(), index= maestro_mun[maestro_mun["ADD"]=="SI"]["Municipi"].tolist().index("Barcelona"))
        index_names = ["Producció", "Compravendes", "Preus", "Superfície"]
        selected_index = st.sidebar.selectbox("**Selecciona un indicador:**", index_names)
        max_year=datetime.now().year-1
        available_years = list(range(2018, datetime.now().year))
        selected_year_n = st.sidebar.selectbox("**Selecciona un any:**", available_years, available_years.index(2023))
        if selected_index=="Producció":
            min_year=2008
            st.subheader(f"PRODUCCIÓ D'HABITATGES A {selected_mun.upper()}")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            
            table_mun = tidy_Catalunya(DT_mun, ["Fecha"] + concatenate_lists(["iniviv_","iniviv_uni_", "iniviv_pluri_","finviv_","finviv_uni_", "finviv_pluri_"], selected_mun), f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Habitatges iniciats","Habitatges iniciats unifamiliars", "Habitatges iniciats plurifamiliars", "Habitatges acabats", "Habitatges acabats unifamiliars", "Habitatges acabats plurifamiliars"])
            table_mun_y = tidy_Catalunya_anual(DT_mun_y, ["Fecha"] + concatenate_lists(["iniviv_","iniviv_uni_", "iniviv_pluri_","finviv_","finviv_uni_", "finviv_pluri_"], selected_mun), min_year, max_year,["Any","Habitatges iniciats","Habitatges iniciats unifamiliars", "Habitatges iniciats plurifamiliars", "Habitatges acabats", "Habitatges acabats unifamiliars", "Habitatges acabats plurifamiliars"])
            table_mun_pluri = tidy_Catalunya(DT_mun, ["Fecha"] + concatenate_lists(["iniviv_pluri_50m2_","iniviv_pluri_5175m2_", "iniviv_pluri_76100m2_","iniviv_pluri_101125m2_", "iniviv_pluri_126150m2_", "iniviv_pluri_150m2_"], selected_mun), f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Plurifamiliar fins a 50m2","Plurifamiliar entre 51m2 i 75 m2", "Plurifamiliar entre 76m2 i 100m2","Plurifamiliar entre 101m2 i 125m2", "Plurifamiliar entre 126m2 i 150m2", "Plurifamiliar de més de 150m2"])
            table_mun_uni = tidy_Catalunya(DT_mun, ["Fecha"] + concatenate_lists(["iniviv_uni_50m2_","iniviv_uni_5175m2_", "iniviv_uni_76100m2_","iniviv_uni_101125m2_", "iniviv_uni_126150m2_", "iniviv_uni_150m2_"], selected_mun), f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Unifamiliar fins a 50m2","Unifamiliar entre 51m2 i 75 m2", "Unifamiliar entre 76m2 i 100m2","Unifamiliar entre 101m2 i 125m2", "Unifamiliar entre 126m2 i 150m2", "Unifamiliar de més de 150m2"])
            left, center, right = st.columns((1,1,1))
            with left:
                try:
                    st.metric(label="**Habitatges iniciats**", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Habitatges iniciats", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Habitatges iniciats", "var")}%""")
                except IndexError:
                    st.metric(label="**Habitatges iniciats**", value=0, delta="-100%")
            with center:
                try:
                    st.metric(label="**Habitatges iniciats plurifamiliars**", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Habitatges iniciats plurifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Habitatges iniciats plurifamiliars", "var")}%""")
                except IndexError:
                    st.metric(label="**Habitatges iniciats plurifamiliars**", value=0, delta="-100%")
            with right:
                try:
                    st.metric(label="**Habitatges iniciats unifamiliars**", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Habitatges iniciats unifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Habitatges iniciats unifamiliars", "var")}%""")
                except IndexError:
                    st.metric(label="**Habitatges iniciats unifamiliars**", value=0, delta="-100%")
            left, center, right = st.columns((1,1,1))
            with left:
                try:
                    st.metric(label="**Habitatges acabats**", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Habitatges acabats", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Habitatges acabats", "var")}%""")
                except IndexError:
                    st.metric(label="**Habitatges acabats**", value=0, delta="-100%")
            with center:
                try:
                    st.metric(label="**Habitatges acabats plurifamiliars**", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Habitatges acabats plurifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Habitatges acabats plurifamiliars", "var")}%""")
                except IndexError:
                    st.metric(label="**Habitatges acabats plurifamiliars**", value=0, delta="-100%")
            with right:
                try:
                    st.metric(label="**Habitatges acabats unifamiliars**", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Habitatges acabats unifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Habitatges acabats unifamiliars", "var")}%""")
                except IndexError:
                    st.metric(label="**Habitatges acabats unifamiliars**", value=0, delta="-100%")
            selected_columns_ini = [col for col in table_mun.columns.tolist() if col.startswith("Habitatges iniciats ")]
            selected_columns_fin = [col for col in table_mun.columns.tolist() if col.startswith("Habitatges acabats ")]
            selected_columns_aux = ["Habitatges iniciats", "Habitatges acabats"]
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_mun, 2020).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_mun, 2008), f"{selected_index}_{selected_mun}.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_mun_y, 2008, rounded=False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_mun_y, 2008, rounded=False), f"{selected_index}_{selected_mun}_anual.xlsx"), unsafe_allow_html=True)
            left_col, right_col = st.columns((1,1))
            with left_col:
                st.plotly_chart(line_plotly(table_mun[selected_columns_aux], selected_columns_aux, "Evolució trimestral de la producció d'habitatges", "Indicador d'oferta en nivells"), use_container_width=True, responsive=True)
                st.plotly_chart(area_plotly(table_mun[selected_columns_ini], selected_columns_ini, "Habitatges iniciats per tipologia", "Habitatges iniciats", "2011T1"), use_container_width=True, responsive=True)
                st.plotly_chart(area_plotly(table_mun_pluri, table_mun_pluri.columns.tolist(), "Habitatges iniciats plurifamiliars per superfície construida", "Habitatges iniciats", "2014T1"), use_container_width=True, responsive=True)
            with right_col:
                st.plotly_chart(bar_plotly(table_mun_y[selected_columns_aux], selected_columns_aux, "Evolució anual de la producció d'habitatges", "Indicador d'oferta en nivells", 2005), use_container_width=True, responsive=True)
                st.plotly_chart(area_plotly(table_mun[selected_columns_fin], selected_columns_fin, "Habitatges acabats per tipologia", "Habitatges acabats", "2011T1"), use_container_width=True, responsive=True)
                st.plotly_chart(area_plotly(table_mun_uni, table_mun_uni.columns.tolist(), "Habitatges iniciats unifamiliars per superfície construida", "Habitatges iniciats", "2014T1"), use_container_width=True, responsive=True)
        if selected_index=="Compravendes":
            min_year=2014
            st.subheader(f"COMPRAVENDES D'HABITATGE A {selected_mun.upper()}")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            
            table_mun = tidy_Catalunya(DT_mun, ["Fecha"] + concatenate_lists(["trvivt_", "trvivs_", "trvivn_"], selected_mun), f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Compravendes d'habitatge total", "Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"])
            table_mun_y = tidy_Catalunya_anual(DT_mun_y, ["Fecha"] + concatenate_lists(["trvivt_", "trvivs_", "trvivn_"], selected_mun), min_year, max_year,["Any","Compravendes d'habitatge total", "Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"])
            left, center, right = st.columns((1,1,1))
            with left:
                try:
                    st.metric(label="**Compravendes d'habitatge total**", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Compravendes d'habitatge total", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Compravendes d'habitatge total", "var")}%""")
                except IndexError:
                    st.metric(label="**Compravendes d'habitatge total**", value=0, delta="-100%")
            with center:
                try:
                    st.metric(label="**Compravendes d'habitatge de segona mà**", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Compravendes d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Compravendes d'habitatge de segona mà", "var")}%""")
                except IndexError:
                    st.metric(label="**Compravendes d'habitatge de segona mà**", value=0, delta="-100%") 
            with right:
                try:
                    st.metric(label="**Compravendes d'habitatge nou**", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Compravendes d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Compravendes d'habitatge nou", "var")}%""") 
                except IndexError:
                    st.metric(label="**Compravendes d'habitatge nou**", value=0, delta="-100%") 
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_mun, 2020).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_mun, 2014), f"{selected_index}_{selected_mun}.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_mun_y, 2014, rounded=False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_mun_y, 2014, rounded=False), f"{selected_index}_{selected_mun}_anual.xlsx"), unsafe_allow_html=True)
            left_col, right_col = st.columns((1,1))
            with left_col:
                st.plotly_chart(line_plotly(table_mun, table_mun.columns.tolist(), "Evolució trimestral de les compravendes d'habitatge per tipologia", "Nombre de compravendes"), use_container_width=True, responsive=True)
            with right_col:
                st.plotly_chart(bar_plotly(table_mun_y, table_mun.columns.tolist(), "Evolució anual de les compravendes d'habitatge per tipologia", "Nombre de compravendes", 2005), use_container_width=True, responsive=True)
        if selected_index=="Preus":
            min_year=2014
            st.subheader(f"PREUS PER M\u00b2 CONSTRUÏT D'HABITATGE A {selected_mun.upper()}")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            
            table_mun = tidy_Catalunya(DT_mun, ["Fecha"] + concatenate_lists(["prvivt_", "prvivs_", "prvivn_"], selected_mun), f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Preu d'habitatge total", "Preu d'habitatge de segona mà", "Preu d'habitatge nou"])
            table_mun = table_mun.replace(0, np.NaN)
            table_mun_y = table_mun.reset_index().copy()
            table_mun_y["Any"] = table_mun_y["Trimestre"].str[:4]
            table_mun_y = table_mun_y.drop("Trimestre", axis=1)
            table_mun_y = table_mun_y.groupby("Any").mean()
            left, center, right = st.columns((1,1,1))
            with left:
                try:
                    st.metric(label="**Preu d'habitatge total** (€/m\u00b2 construït)", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Preu d'habitatge total", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Preu d'habitatge total", "var")}%""")
                except IndexError:
                    st.metric(label="**Preu d'habitatge total** (€/m\u00b2 construït)", value="n/a") 
            with center:
                try:
                    st.metric(label="**Preu d'habitatge de segona mà** (€/m\u00b2 construït)", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Preu d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Preu d'habitatge de segona mà", "var")}%""")
                except IndexError:
                    st.metric(label="**Preu d'habitatge de segona mà** (€/m\u00b2 construït)", value="n/a") 
            with right:
                try:
                    st.metric(label="**Preu d'habitatge nou** (€/m\u00b2 construït)", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Preu d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Preu d'habitatge nou", "var")}%""") 
                except IndexError:
                    st.metric(label="**Preu d'habitatge nou** (€/m\u00b2 construït)", value="n/a") 
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_mun, 2020, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_mun, 2014, True, False), f"{selected_index}_{selected_mun}.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_mun_y, 2014, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_mun_y, 2014, True, False), f"{selected_index}_{selected_mun}_anual.xlsx"), unsafe_allow_html=True)
            left_col, right_col = st.columns((1,1))
            with left_col:
                st.plotly_chart(line_plotly(table_mun, table_mun.columns.tolist(), "Evolució trimestral dels preus per m\u00b2 construït per tipologia d'habitatge", "€/m\u00b2 útil", True), use_container_width=True, responsive=True)
            with right_col:
                st.plotly_chart(bar_plotly(table_mun_y, table_mun.columns.tolist(), "Evolució anual dels preus per m\u00b2 construït per tipologia d'habitatge", "€/m\u00b2 útil", 2005), use_container_width=True, responsive=True)
        if selected_index=="Superfície":
            min_year=2014
            st.subheader(f"SUPERFÍCIE EN M\u00b2 CONSTRUÏTS D'HABITATGE A {selected_mun.upper()}")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            
            table_mun = tidy_Catalunya(DT_mun, ["Fecha"] + concatenate_lists(["supert_", "supers_", "supern_"], selected_mun), f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Superfície mitjana total", "Superfície mitjana d'habitatge de segona mà", "Superfície mitjana d'habitatge nou"])
            table_mun_y = tidy_Catalunya_anual(DT_mun_y, ["Fecha"] + concatenate_lists(["supert_", "supers_", "supern_"], selected_mun), min_year, max_year,["Any","Superfície mitjana total", "Superfície mitjana d'habitatge de segona mà", "Superfície mitjana d'habitatge nou"])
            left, center, right = st.columns((1,1,1))
            with left:
                try:
                    st.metric(label="**Superfície mitjana** (m\u00b2)", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Superfície mitjana total", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Superfície mitjana total", "var")}%""")
                except IndexError:
                    st.metric(label="**Superfície mitjana** (m\u00b2)", value="n/a")
            with center:
                try:
                    st.metric(label="**Superfície d'habitatges de segona mà** (m\u00b2)", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Superfície mitjana d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Superfície mitjana d'habitatge de segona mà", "var")}%""")
                except IndexError:
                    st.metric(label="**Superfície d'habitatges de segona mà** (m\u00b2)", value="n/a")
            with right:
                try:
                    st.metric(label="**Superfície d'habitatges nous** (m\u00b2)", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Superfície mitjana d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Superfície mitjana d'habitatge nou", "var")}%""")
                except IndexError:
                    st.metric(label="**Superfície d'habitatges nous** (m\u00b2)", value="n/a")
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_mun, 2020, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_mun, 2014, True, False), f"{selected_index}_{selected_mun}.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_mun_y, 2014, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_mun_y, 2014, True, False), f"{selected_index}_{selected_mun}_anual.xlsx"), unsafe_allow_html=True)
            left_col, right_col = st.columns((1,1))
            with left_col:
                st.plotly_chart(line_plotly(table_mun, table_mun.columns.tolist(), "Evolució trimestral de la superfície mitjana per tipologia d'habitatge", "m\u00b2 útil", True), use_container_width=True, responsive=True)
            with right_col:
                st.plotly_chart(bar_plotly(table_mun_y, table_mun.columns.tolist(), "Evolució anual de la superfície mitjana per tipologia d'habitatge", "m\u00b2 útil", 2005), use_container_width=True, responsive=True)
    if selected_type=="Lloguer":
        selected_mun = st.sidebar.selectbox("**Selecciona un municipi:**", maestro_mun[maestro_mun["ADD"]=="SI"]["Municipi"].unique(), index= maestro_mun[maestro_mun["ADD"]=="SI"]["Municipi"].tolist().index("Barcelona"))
        max_year=datetime.now().year-1
        available_years = list(range(2018, datetime.now().year))
        selected_year_n = st.sidebar.selectbox("**Selecciona un any:**", available_years, available_years.index(2023))
        min_year=2014
        st.subheader(f"MERCAT DE LLOGUER A {selected_mun.upper()}")
        st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
        table_mun = tidy_Catalunya(DT_mun, ["Fecha"] + concatenate_lists(["trvivalq_", "pmvivalq_"], selected_mun), f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Nombre de contractes de lloguer", "Rendes mitjanes de lloguer"])
        table_mun_y = tidy_Catalunya_anual(DT_mun_y, ["Fecha"] + concatenate_lists(["trvivalq_", "pmvivalq_"], selected_mun), min_year, max_year,["Any", "Nombre de contractes de lloguer", "Rendes mitjanes de lloguer"])
        left_col, right_col = st.columns((1,1))
        with left_col:
            try:
                st.metric(label="**Nombre de contractes de lloguer**", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Nombre de contractes de lloguer", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Nombre de contractes de lloguer", "var")}%""")
            except IndexError:
                st.metric(label="**Nombre de contractes de lloguer**", value="n/a")
        with right_col:
            try:
                st.metric(label="**Rendes mitjanes de lloguer** (€/mes)", value=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Rendes mitjanes de lloguer", "level"):,.0f}""", delta=f"""{indicator_year(table_mun_y, table_mun, str(selected_year_n), "Rendes mitjanes de lloguer", "var")}%""")
            except IndexError:
                st.metric(label="**Rendes mitjanes de lloguer** (€/mes)", value="n/a")
                st.markdown("")
        st.markdown("")
        # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
        st.markdown(table_trim(table_mun, 2020, rounded=True).to_html(), unsafe_allow_html=True)
        st.markdown(filedownload(table_trim(table_mun, 2014, rounded=True), f"{selected_type}_{selected_mun}.xlsx"), unsafe_allow_html=True)
        st.markdown("")
        st.markdown("")
        # st.subheader("**DADES ANUALS**")
        st.markdown(table_year(table_mun_y, 2014, rounded=True).to_html(), unsafe_allow_html=True)
        st.markdown(filedownload(table_year(table_mun_y, 2014, rounded=True), f"{selected_type}_{selected_mun}_anual.xlsx"), unsafe_allow_html=True)
        left_col, right_col = st.columns((1,1))
        with left_col:
            st.plotly_chart(line_plotly(table_mun, ["Rendes mitjanes de lloguer"], "Evolució trimestral de les rendes mitjanes de lloguer", "€/mes", True), use_container_width=True, responsive=True)
            st.plotly_chart(line_plotly(table_mun, ["Nombre de contractes de lloguer"], "Evolució trimestral del nombre de contractes de lloguer", "Nombre de contractes"), use_container_width=True, responsive=True)
        with right_col:
            st.plotly_chart(bar_plotly(table_mun_y, ["Rendes mitjanes de lloguer"], "Evolució anual de les rendes mitjanes de lloguer", "€/mes", 2005), use_container_width=True, responsive=True)
            st.plotly_chart(bar_plotly(table_mun_y, ["Nombre de contractes de lloguer"],  "Evolució anual del nombre de contractes de lloguer", "Nombre de contractes", 2005), use_container_width=True, responsive=True)

if selected=="Districtes de Barcelona":
    st.sidebar.header("**DISTRICTES DE BARCELONA**")
    selected_type = st.sidebar.radio("**Mercat de venda o lloguer**", ("Venda", "Lloguer"))
    if selected_type=="Venda":
        selected_dis = st.sidebar.selectbox("**Selecciona un districte de Barcelona:**", maestro_dis["Districte"].unique())
        index_names = ["Producció", "Compravendes", "Preus", "Superfície"]
        selected_index = st.sidebar.selectbox("**Selecciona un indicador**", index_names)
        max_year=datetime.now().year-1
        available_years = list(range(2018, datetime.now().year))
        selected_year_n = st.sidebar.selectbox("**Selecciona un any:**", available_years, available_years.index(2023))
        if selected_index=="Producció":
            min_year=2011
            st.subheader(f"PRODUCCIÓ D'HABITATGES A {selected_dis.upper()}")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            table_dis = tidy_Catalunya(DT_dis, ["Fecha"] + concatenate_lists(["iniviv_","iniviv_uni_", "iniviv_pluri_","finviv_","finviv_uni_", "finviv_pluri_"], selected_dis), f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Habitatges iniciats","Habitatges iniciats unifamiliars", "Habitatges iniciats plurifamiliars", "Habitatges acabats", "Habitatges acabats unifamiliars", "Habitatges acabats plurifamiliars"])
            table_dis_y = tidy_Catalunya_anual(DT_dis_y, ["Fecha"] + concatenate_lists(["iniviv_","iniviv_uni_", "iniviv_pluri_","finviv_","finviv_uni_", "finviv_pluri_"], selected_dis), min_year, max_year,["Any","Habitatges iniciats","Habitatges iniciats unifamiliars", "Habitatges iniciats plurifamiliars", "Habitatges acabats", "Habitatges acabats unifamiliars", "Habitatges acabats plurifamiliars"])
            # table_dis_pluri = tidy_Catalunya(DT_dis, ["Fecha"] + concatenate_lists(["iniviv_pluri_50m2_","iniviv_pluri_5175m2_", "iniviv_pluri_76100m2_","iniviv_pluri_101125m2_", "iniviv_pluri_126150m2_", "iniviv_pluri_150m2_"], selected_dis), f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Plurifamiliar fins a 50m2","Plurifamiliar entre 51m2 i 75 m2", "Plurifamiliar entre 76m2 i 100m2","Plurifamiliar entre 101m2 i 125m2", "Plurifamiliar entre 126m2 i 150m2", "Plurifamiliar de més de 150m2"])
            # table_dis_uni = tidy_Catalunya(DT_dis, ["Fecha"] + concatenate_lists(["iniviv_uni_50m2_","iniviv_uni_5175m2_", "iniviv_uni_76100m2_","iniviv_uni_101125m2_", "iniviv_uni_126150m2_", "iniviv_uni_150m2_"], selected_dis), f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Unifamiliar fins a 50m2","Unifamiliar entre 51m2 i 75 m2", "Unifamiliar entre 76m2 i 100m2","Unifamiliar entre 101m2 i 125m2", "Unifamiliar entre 126m2 i 150m2", "Unifamiliar de més de 150m2"])
            left, center, right = st.columns((1,1,1))
            with left:
                try:
                    st.metric(label="**Habitatges iniciats**", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Habitatges iniciats", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Habitatges iniciats", "var")}%""")
                except IndexError:
                    st.metric(label="**Habitatges iniciats**", value=0, delta="-100%")
            with center:
                try:
                    st.metric(label="**Habitatges iniciats plurifamiliars**", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Habitatges iniciats plurifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Habitatges iniciats plurifamiliars", "var")}%""")
                except IndexError:
                    st.metric(label="**Habitatges iniciats plurifamiliars**", value="Pendent", delta="N/A")
            with right:
                try:
                    st.metric(label="**Habitatges iniciats unifamiliars**", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Habitatges iniciats unifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Habitatges iniciats unifamiliars", "var")}%""")
                except IndexError:
                    st.metric(label="**Habitatges iniciats plurifamiliars**", value="Pendent", delta="N/A")
            with left:
                try:
                    st.metric(label="**Habitatges acabats**", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Habitatges acabats", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Habitatges acabats", "var")}%""")
                except IndexError:
                    st.metric(label="**Habitatges acabats**", value=0, delta="-100%")
            with center:
                try:
                    st.metric(label="**Habitatges acabats plurifamiliars**", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Habitatges acabats plurifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Habitatges acabats plurifamiliars", "var")}%""")
                except IndexError:
                    st.metric(label="**Habitatges acabats plurifamiliars**", value="Pendent", delta="N/A")           
            with right:
                try:
                    st.metric(label="**Habitatges acabats unifamiliars**", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Habitatges acabats unifamiliars", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Habitatges acabats unifamiliars", "var")}%""")
                except IndexError:
                    st.metric(label="**Habitatges acabats unifamiliars**", value="Pendent", delta="N/A")
            selected_columns_ini = [col for col in table_dis.columns.tolist() if col.startswith("Habitatges iniciats ")]
            selected_columns_fin = [col for col in table_dis.columns.tolist() if col.startswith("Habitatges acabats ")]
            selected_columns_aux = ["Habitatges iniciats", "Habitatges acabats"]
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_dis, 2020).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_dis, 2014), f"{selected_index}_{selected_dis}.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_dis_y, 2014, rounded=False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_dis_y, 2014, rounded=False), f"{selected_index}_{selected_dis}_anual.xlsx"), unsafe_allow_html=True)
            left_col, right_col = st.columns((1,1))
            with left_col:
                st.plotly_chart(line_plotly(table_dis[selected_columns_aux], selected_columns_aux, "Evolució trimestral de la producció d'habitatges", "Indicador d'oferta en nivells"), use_container_width=True, responsive=True)
                st.plotly_chart(area_plotly(table_dis[selected_columns_ini], selected_columns_ini, "Habitatges iniciats per tipologia", "Habitatges iniciats", "2011T1"), use_container_width=True, responsive=True)
            with right_col:
                st.plotly_chart(bar_plotly(table_dis_y[selected_columns_aux], selected_columns_aux, "Evolució anual de la producció d'habitatges", "Indicador d'oferta en nivells", 2005), use_container_width=True, responsive=True)
                st.plotly_chart(area_plotly(table_dis[selected_columns_fin], selected_columns_fin, "Habitatges acabats per tipologia", "Habitatges acabats", "2011T1"), use_container_width=True, responsive=True)
        if selected_index=="Compravendes":
            min_year=2014
            st.subheader(f"COMPRAVENDES D'HABITATGE A {selected_dis.upper()}")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            
            table_dis = tidy_Catalunya(DT_dis, ["Fecha"] + concatenate_lists(["trvivt_", "trvivs_", "trvivn_"], selected_dis), f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Compravendes d'habitatge total", "Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"])
            table_dis_y = tidy_Catalunya_anual(DT_dis_y, ["Fecha"] + concatenate_lists(["trvivt_", "trvivs_", "trvivn_"], selected_dis), min_year, max_year,["Any","Compravendes d'habitatge total", "Compravendes d'habitatge de segona mà", "Compravendes d'habitatge nou"])
            left, center, right = st.columns((1,1,1))
            with left:
                st.metric(label="**Compravendes d'habitatge total**", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Compravendes d'habitatge total", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Compravendes d'habitatge total", "var")}%""")
            with center:
                st.metric(label="**Compravendes d'habitatge de segona mà**", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Compravendes d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Compravendes d'habitatge de segona mà", "var")}%""")
            with right:
                st.metric(label="**Compravendes d'habitatge nou**", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Compravendes d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Compravendes d'habitatge nou", "var")}%""") 
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_dis, 2020).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_dis, 2017), f"{selected_index}_{selected_dis}.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_dis_y, 2017, rounded=False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_dis_y, 2017, rounded=False), f"{selected_index}_{selected_dis}_anual.xlsx"), unsafe_allow_html=True)
            left_col, right_col = st.columns((1,1))
            with left_col:
                st.plotly_chart(line_plotly(table_dis.iloc[12:,:], table_dis.columns.tolist(), "Evolució trimestral de les compravendes d'habitatge per tipologia", "Nombre de compravendes"), use_container_width=True, responsive=True)
            with right_col:
                st.plotly_chart(bar_plotly(table_dis_y, table_dis_y.columns.tolist(), "Evolució anual de les compravendes d'habitatge per tipologia", "Nombre de compravendes", 2017), use_container_width=True, responsive=True)
        if selected_index=="Preus":
            min_year=2014
            st.subheader(f"PREUS PER M\u00b2 CONSTRUÏTS D'HABITATGE A {selected_dis.upper()}")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            
            table_dis = tidy_Catalunya(DT_dis, ["Fecha"] + concatenate_lists(["prvivt_", "prvivs_", "prvivn_"], selected_dis), f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Preu d'habitatge total", "Preu d'habitatge de segona mà", "Preu d'habitatge nou"])
            table_dis_y = tidy_Catalunya_anual(DT_dis_y, ["Fecha"] + concatenate_lists(["prvivt_", "prvivs_", "prvivn_"], selected_dis), min_year, max_year,["Any","Preu d'habitatge total", "Preu d'habitatge de segona mà", "Preu d'habitatge nou"])
            left, center, right = st.columns((1,1,1))
            with left:
                st.metric(label="**Preu d'habitatge total** (€/m\u00b2)", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Preu d'habitatge total", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Preu d'habitatge total", "var")}%""")
            with center:
                st.metric(label="**Preu d'habitatge de segona mà** (€/m\u00b2)", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Preu d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Preu d'habitatge de segona mà", "var")}%""")
            with right:
                st.metric(label="**Preu d'habitatge nou** (€/m\u00b2)", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Preu d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Preu d'habitatge nou", "var")}%""") 
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_dis, 2020, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_dis, 2017, True, False), f"{selected_index}_{selected_dis}.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_dis_y, 2017, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_dis_y, 2017, True, False), f"{selected_index}_{selected_dis}_anual.xlsx"), unsafe_allow_html=True)
            left_col, right_col = st.columns((1,1))
            with left_col:
                st.plotly_chart(line_plotly(table_dis.iloc[12:,:], table_dis.columns.tolist(), "Evolució trimestral dels preus per m\u00b2 construït per tipologia d'habitatge", "€/m2 útil", True), use_container_width=True, responsive=True)
            with right_col:
                st.plotly_chart(bar_plotly(table_dis_y, table_dis.columns.tolist(), "Evolució anual dels preus per m\u00b2 construït per tipologia d'habitatge", "€/m2 útil", 2017), use_container_width=True, responsive=True)
        if selected_index=="Superfície":
            min_year=2014
            st.subheader(f"SUPERFÍCIE EN M\u00b2 CONSTRUÏTS D'HABITATGE A {selected_dis.upper()}")
            st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
            table_dis = tidy_Catalunya(DT_dis, ["Fecha"] + concatenate_lists(["supert_", "supers_", "supern_"], selected_dis), f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Superfície mitjana total", "Superfície mitjana d'habitatge de segona mà", "Superfície mitjana d'habitatge nou"])
            table_dis_y = tidy_Catalunya_anual(DT_dis_y, ["Fecha"] + concatenate_lists(["supert_", "supers_", "supern_"], selected_dis), min_year, max_year,["Any","Superfície mitjana total", "Superfície mitjana d'habitatge de segona mà", "Superfície mitjana d'habitatge nou"])
            left, center, right = st.columns((1,1,1))
            with left:
                st.metric(label="**Superfície mitjana** (m\u00b2)", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Superfície mitjana total", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Superfície mitjana total", "var")}%""")
            with center:
                st.metric(label="**Superfície d'habitatges de segona mà** (m\u00b2)", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Superfície mitjana d'habitatge de segona mà", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Superfície mitjana d'habitatge de segona mà", "var")}%""")
            with right:
                st.metric(label="**Superfície d'habitatges nous** (m\u00b2)", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Superfície mitjana d'habitatge nou", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Superfície mitjana d'habitatge nou", "var")}%""")
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
            st.markdown(table_trim(table_dis, 2020, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_trim(table_dis, 2017, True, False), f"{selected_index}_{selected_dis}.xlsx"), unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            # st.subheader("**DADES ANUALS**")
            st.markdown(table_year(table_dis_y, 2017, True, False).to_html(), unsafe_allow_html=True)
            st.markdown(filedownload(table_year(table_dis_y, 2017, True, False), f"{selected_index}_{selected_dis}_anual.xlsx"), unsafe_allow_html=True)
            left_col, right_col = st.columns((1,1))
            with left_col:
                st.plotly_chart(line_plotly(table_dis.iloc[12:,:], table_dis.columns.tolist(), "Evolució trimestral de la superfície mitjana per tipologia d'habitatge", "m2 útil", True), use_container_width=True, responsive=True)
            with right_col:
                st.plotly_chart(bar_plotly(table_dis_y, table_dis.columns.tolist(), "Evolució anual de la superfície mitjana per tipologia d'habitatge", "m2 útil", 2017), use_container_width=True, responsive=True)
    if selected_type=="Lloguer":
        selected_dis = st.sidebar.selectbox("**Selecciona un districte de Barcelona:**", maestro_dis["Districte"].unique())
        st.subheader(f"MERCAT DE LLOGUER A {selected_dis.upper()}")
        max_year=datetime.now().year-1
        available_years = list(range(2018, max_year+1))
        selected_year_n = st.sidebar.selectbox("**Selecciona un any:**", available_years, available_years.index(max_year))
        st.markdown(f'<div class="custom-box">ANY {selected_year_n}</div>', unsafe_allow_html=True)
        min_year=2014
        table_dis = tidy_Catalunya(DT_dis, ["Fecha"] + concatenate_lists(["trvivalq_", "pmvivalq_"], selected_dis), f"{str(min_year)}-01-01", f"{str(max_year+1)}-01-01",["Data", "Nombre de contractes de lloguer", "Rendes mitjanes de lloguer"])
        table_dis_y = tidy_Catalunya_anual(DT_dis_y, ["Fecha"] + concatenate_lists(["trvivalq_", "pmvivalq_"], selected_dis), min_year, max_year,["Any", "Nombre de contractes de lloguer", "Rendes mitjanes de lloguer"])
        left_col, right_col = st.columns((1,1))
        with left_col:
            st.metric(label="**Nombre de contractes de lloguer**", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Nombre de contractes de lloguer", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Nombre de contractes de lloguer", "var")}%""")
        with right_col:
            st.metric(label="**Rendes mitjanes de lloguer** (€/mes)", value=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Rendes mitjanes de lloguer", "level"):,.0f}""", delta=f"""{indicator_year(table_dis_y, table_dis, str(selected_year_n), "Rendes mitjanes de lloguer", "var")}%""")
        st.markdown("")
        st.markdown("")
        # st.subheader("**DADES TRIMESTRALS MÉS RECENTS**")
        st.markdown(table_trim(table_dis, 2020, True).to_html(), unsafe_allow_html=True)
        st.markdown(filedownload(table_trim(table_dis, 2014, True), f"{selected_type}_{selected_dis}.xlsx"), unsafe_allow_html=True)
        st.markdown("")
        st.markdown("")
        # st.subheader("**DADES ANUALS**")
        st.markdown(table_year(table_dis_y, 2014, rounded=True).to_html(), unsafe_allow_html=True)
        st.markdown(filedownload(table_year(table_dis_y, 2014, rounded=True), f"{selected_type}_{selected_dis}_anual.xlsx"), unsafe_allow_html=True)
        left_col, right_col = st.columns((1,1))
        with left_col:
            st.plotly_chart(line_plotly(table_dis, ["Rendes mitjanes de lloguer"], "Evolució trimestral de les rendes mitjanes de lloguer", "€/mes", True), use_container_width=True, responsive=True)
            st.plotly_chart(line_plotly(table_dis, ["Nombre de contractes de lloguer"], "Evolució trimestral del nombre de contractes de lloguer", "Nombre de contractes"), use_container_width=True, responsive=True)
        with right_col:
            st.plotly_chart(bar_plotly(table_dis_y, ["Rendes mitjanes de lloguer"], "Evolució anual de les rendes mitjanes de lloguer", "€/mes", 2005), use_container_width=True, responsive=True)
            st.plotly_chart(bar_plotly(table_dis_y, ["Nombre de contractes de lloguer"],  "Evolució anual del nombre de contractes de lloguer", "Nombre de contractes", 2005), use_container_width=True, responsive=True)
# if selected=="Contacte":
#     load_css_file(path + "main.css")
#     CONTACT_EMAIL = "estudis@apcecat.cat"
#     st.write("")
#     st.subheader(":mailbox: Contacteu-nos!")
#     contact_form = f"""
#     <form action="https://formsubmit.co/{CONTACT_EMAIL}" method="POST">
#         <input type="hidden" class="Contacte" name="_captcha" value="false">
#         <input type="text" class="Contacte" name="name" placeholder="Nom" required>
#         <input type="email" class="Contacte" name="email" placeholder="Correu electrónic" required>
#         <textarea class="Contacte" name="message" placeholder="La teva consulta aquí"></textarea>
#         <button type="submit" class="button">Enviar ✉</button>
#     </form>
#     """
#     st.markdown(contact_form, unsafe_allow_html=True)


# min_year, max_year = st.sidebar.slider("**Interval d'anys de la mostra**", value=[min_year, max_year], min_value=min_year, max_value=max_year)