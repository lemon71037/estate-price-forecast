from collections import Counter
import dash_core_components as dcc
from dash_core_components.Dropdown import Dropdown
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash_html_components.Button import Button
import dash_table as dt
from dash.dependencies import Input, State, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from dash_table.Format import Format, Scheme

from textwrap import dedent
# from collections import Counter

import plotly.express as px
from ..app import app
from .data_table_display import item_list
from . import df_convert

total_options = [
    {"label": "区县名 district_name", "value": "district_name"},
    {"label": "住房类型 house_type", "value": "house_type"},
    {"label": "装修类型 decoration", "value": "decoration"},
    {"label": "最小房屋面积 min_frame_area", "value": "min_frame_area"},
    {"label": "最大房屋面积 max_frame_area", "value": "max_frame_area"},
    {"label": "户型数 room_types", "value": "room_types"},
    {"label": "平均房间面积 avg_room_area", "value": "avg_room_area"},
    {"label": "经度 longitude", "value": "longitude"},
    {"label": "纬度 latitude", "value": "latitude"},
]

# 导入数据
df = pd.read_csv('E:/github/check-price/AnalysisSystem/dataset/raw_loupan_data.csv', header=0)
columns = [{"name": i, "id": i} for i in df.columns]
for i in range(len(columns)):
    if columns[i]["name"] == "longitude" or columns[i]["name"] == "latitude":
        columns[i]["type"] = "numeric"
        columns[i]["format"] = Format(precision=6, scheme=Scheme.fixed)

item_list = df.to_dict("records") # 将 dataframe 转为 dict_list 形式

district_list = [item['district_name'] for item in item_list]
district_type_list = list(set(district_list)) # 区县 list

district_dict = {} # 每个区县对应的项
for item in item_list:
    if item['district_name'] not in district_dict.keys():
        district_dict[item['district_name']] = [item, ]
    else:
        district_dict[item['district_name']].append(item)

def get_layout(args):
    
    return html.Div(children=[
        dcc.Markdown(
            dedent(
                """
                # 楼盘特征预处理与可视化
                
                Preprocessing and visualization of real estate features.
                """
            )
        ),
        html.Hr(),
        html.Div(children=[
            html.Hr(),
            dbc.FormGroup(
                [
                    dbc.Label("Selected Feature:", className="mr-2", width=2),
                    dbc.Col(
                        dcc.Dropdown(id="feature_list",
                            options=total_options,
                            value="district_name"
                        ),
                    )
                ], row=True
            ),
            html.Hr(), 
            html.Div("特征间的相关性分析"), 
            # html.Div("Correlation between Features and Price:"),
            html.P(),
            dbc.FormGroup(
                [
                    dbc.Label("Compared Feature:", className="mr-2", width=2),
                    dbc.Col(
                        dcc.Dropdown(id="feature_compare"),
                    )
                ], row=True
            ),

            # dbc.Button("Analyse", id="analyse_button", className="mr-2"),

            # html.Hr(),
            # html.Div(id="additional_info"),
            html.Div(id="relation_div", style={"text-align": "center"}),
            dcc.Graph(id="relation_graph"),
            
            html.Hr(), 
            html.Div(id="distribution_div", style={"text-align": "center"}),
            dcc.Graph(id="distribution_graph"),
            
            html.Hr(), 
            html.Div(id="statistic_div", style={"text-align": "center"}),
            dcc.Graph(id="statistic_graph"),

        ], style={
            'padding': '0px 10px 15px 10px',
            'marginLeft': 'auto', 'marginRight': 'auto', "width": "1000px",
            'boxShadow': '0px 0px 5px 5px rgba(204,204,204,0.4)'}),
    ])

@app.callback(
    [Output("feature_compare", "options"),
    Output("feature_compare", "value")],
    Input("feature_list", "value")
)
def update_feature_compared(value):
    new_options = [{"label": "平均价格 average_price", "value": "average_price"}]
    for option in total_options:
        if option["value"] != value:
            new_options.append(option)
    
    return new_options, "average_price"

@app.callback(
    [Output("relation_div", "children"),
    Output("relation_graph", "figure")],
    [Input("feature_list", "value"),
    Input("feature_compare", "value")]
)
def update_relation_graph(feature1, feature2):
    scatter_df = pd.DataFrame()

    feature1_list = df_convert[feature1]
    feature2_list = df_convert[feature2] if feature2 != "average_price" else df[feature2]

    ab_idx1 = [] if feature1 in ["district_name", "house_type", "decoration"] else \
        [i for i, x in enumerate(feature1_list) if x == 0] # 剔除缺失值
    
    ab_idx2 = [] if feature2 in ["district_name", "house_type", "decoration"] else \
        [i for i, x in enumerate(feature2_list) if x == 0]
    ab_idx = set(ab_idx1 + ab_idx2)

    scatter_df[feature1] = [feature1_list[i] for i in range(len(feature1_list)) \
        if i not in ab_idx]
    scatter_df[feature2] = [feature2_list[i] for i in range(len(feature2_list)) \
        if i not in ab_idx]
    
    fig = px.scatter(scatter_df, x=feature1, y=feature2, \
        color_discrete_sequence=["#7EC0EE"], \
            marginal_x="rug", marginal_y="violin")

    return "楼盘特征 "+feature1+" 与特征 "+feature2+" 的散点关系图", fig

@app.callback(
    [Output("distribution_div", "children"),
    Output("distribution_graph", "figure")],
    Input("feature_list", "value")
)
def update_distribution_graph(feature):
    temp_df = pd.DataFrame()
    temp_df[feature] = df_convert[feature] if feature in \
        ["district_name", "house_type", "decoration"] else \
            [i for i in df_convert[feature] if i != 0]

    fig = px.histogram(temp_df, x=feature, color_discrete_sequence=["#B0C4DE"],\
        marginal="box")
    # fig.add_trace(go.Layout())

    return "楼盘特征 "+feature+" 数值分布图", fig

@app.callback(
    [Output("statistic_div", "children"),
    Output("statistic_graph", "figure")],
    Input("feature_list", "value")
)
def update_distribution_graph(feature):

    if feature in ["district_name", "house_type", "decoration"]:
        feat_count = dict(Counter(df_convert[feature]))
        fig = px.pie(names=list(feat_count.keys()), values=list(feat_count.values()))
        
        return '离散特征 '+feature+' 统计分布图', fig

    else:
        temp_df = pd.DataFrame()
        temp_df[feature] = [i for i in df_convert[feature] if i != 0]
        df = temp_df.describe()
        df = df.drop(['count'])
        df['statistic'] = df.index
        
        fig = px.bar(df, x='statistic', y=feature, color_discrete_sequence=["#9BCD9B"])

        return '连续特征 '+feature+' 统计分布图', fig
