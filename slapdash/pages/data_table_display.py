import dash_core_components as dcc
from dash_core_components.Dropdown import Dropdown
import dash_html_components as html
import dash_bootstrap_components as dbc
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
from . import df

# mapbox的 token
token = "pk.eyJ1IjoibGVtb25ibG9ja3MiLCJhIjoiY2tvaTd0NHhiMHVrejJwcG4zdW5jc200OCJ9.Eixuo7pHjcquyX-DIK7EMA"

# 得到数据的列名
columns = [{"name": i, "id": i} for i in df.columns]
for i in range(len(columns)):
    if columns[i]["name"] == "longitude" or columns[i]["name"] == "latitude":
        columns[i]["type"] = "numeric"
        columns[i]["format"] = Format(precision=6, scheme=Scheme.fixed)

item_list = df.to_dict("records") # 将 dataframe 转为 dict_list 形式

district_list = df["district_name"]
district_type_list = list(set(district_list)) # 区县 list

district_dict = {} # 每个区县对应的项
for item in item_list:
    if item['district_name'] not in district_dict.keys():
        district_dict[item['district_name']] = [item, ]
    else:
        district_dict[item['district_name']].append(item)

def get_layout(args):
    initial_text = args.get("text", "Type some text into me!")

    # Note that if you need to access multiple values of an argument, you can
    # use args.getlist("param")
    return html.Div(children=[
        dcc.Markdown(
            dedent(
                """
                # 杭州在售楼盘数据
                
                Historical real estate price records in Hangzhou.
                """
            )
        ),
        html.Hr(),
        html.Div(children=[
            html.Hr(),
            dbc.FormGroup(
                [
                    dbc.Label("Visulaize Mode:", className="mr-2", width=2),
                    dbc.Col(
                        dcc.Dropdown(id="visual_mode",
                            options=[
                                {"label": "杭州各在售楼盘地图展示", "value": 1},
                                {"label": "杭州各区县在售楼盘数统计", "value": 2},
                                {"label": "杭州各区县平均房价统计", "value": 3},
                            ],
                            value=1
                        ),
                    )
                ], row=True
            ),
            
            dbc.FormGroup(
                [
                    dbc.Label("Optional:", className="mr-2", width=2),
                    dbc.Col(
                        dcc.Dropdown(id="sort_type",
                            value="price"
                        ),
                    )
                ], row=True
            ),

            html.Hr(),

            dcc.Graph(id="price_graph"),
            
            html.Hr(),
            
            dt.DataTable(id="datatable",
                columns=columns,
                fixed_rows={'headers': True},
                data=item_list,
                style_table={'height': '500px', 'overflowX': 'auto'},
                style_cell={
                    'textAlign': 'left', 'height': 'auto', 'whiteSpace': 'normal',
                    'minWidth': '160px', 'width': '160px', 'maxWidth': '160px',
                },
                style_as_list_view=True,
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                }
            ),

        ], style={
            'padding': '0px 10px 15px 10px',
            'marginLeft': 'auto', 'marginRight': 'auto', "width": "1000px",
            'boxShadow': '0px 0px 5px 5px rgba(204,204,204,0.4)'}),
    ])

@app.callback(
    [Output("sort_type", "disabled"),
    Output("sort_type", "options")],
    Input("visual_mode", "value")
)
def update_div(value):
    if value == 3:
        return False, \
            [{"label": "Sort by Average Price", "value": "price"},
            {"label": "Sort by District Name", "value": "district"}]
    else:
        return True, \
            [{"label": "No Optional Choice", "value": "price"}]

@app.callback(
    Output("price_graph", "figure"),
    Input("visual_mode", "value"),
    Input("sort_type", "value"), 
)
def update_graph(visual_mode, sort_type="price"):
    if visual_mode == 1:
        fig = px.scatter_mapbox(df, lat="latitude", lon="longitude", height=500, \
            size="average_price", color="average_price", title="杭州在售楼盘数据可视化",
            hover_name="title", hover_data=["district_name", "house_type", "decoration"],
            size_max=30, color_continuous_scale=px.colors.carto.Temps)
        
        fig.update_layout(mapbox={"accesstoken": token, "zoom": 9, \
            "center": {"lon": np.mean([item["longitude"] for item in item_list if item["longitude"] != 0]),
            "lat": np.mean([item["latitude"] for item in item_list if item["latitude"] != 0])}},
            title=dict(x=0.5, xref='paper'),
            margin={"l":10, "r":0, "t":50, "b":10})
        
        return fig

    elif visual_mode == 2:
        return {
            'data': [
                go.Pie(
                    labels=district_type_list,
                    values=[len(district_dict[d]) for d in district_type_list],
            )],

            'layout': go.Layout(
                title='杭州各区县楼盘数比例',
                font={"size": 16},
                height=500
            )}

    else:
        if sort_type == "price":
            sort_func = lambda x: -x[1]
        else:
            sort_func = lambda x: [ord(i) for i in x[0]]
        
        district_avgPrice = [(key, np.mean([float(i["average_price"]) for i in value])) \
            for key, value in district_dict.items()]
        x_data, y_data = zip(*sorted(district_avgPrice, key=sort_func))
        return {
            "data": [{"x": x_data, "y": y_data, "type": "bar", "name": "trace1"}],
            "layout": {
                "title": "杭州各区县平均房价示意图",
                "height": "500",
                "font": {"size": 16},
            },
        }
