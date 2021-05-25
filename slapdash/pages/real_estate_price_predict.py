from inspect import ismethoddescriptor
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table as dt
from dash.dependencies import Input, State, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import os
from dash_table.Format import Format, Scheme
import joblib

from textwrap import dedent
# from collections import Counter

from ..app import app
from . import df, df_processed, root_path, discrete_feat_map, \
    discrete_feat_type_list, new_tag
from .model_train_test import saved_model

distirct_map, district_type_list = discrete_feat_map["district_name"], \
    discrete_feat_type_list["district_name"]
house_type_map, house_type_list = discrete_feat_map["house_type"], \
    discrete_feat_type_list["house_type"]
raw_decoration_map, decoration_type_list = discrete_feat_map["decoration"], \
    discrete_feat_type_list["decoration"]

for i in range(len(decoration_type_list)):
    if not isinstance(decoration_type_list[i], str):
        decoration_type_list[i] = "[空]"

decoration_map = {}
for key, value in raw_decoration_map.items(): # decoration 是有空值的，需要特别处理
    if not isinstance(key, str):
        decoration_map["[空]"] = value
    else:
        decoration_map[key] = value

# saved_model = os.listdir(root_path + "saved_model")

def get_layout(args):
    
    return html.Div(children=[
        dcc.Markdown(
            dedent(
                """
                # 新盘限价预测
                
                Real Estate Price Predict.
                """
            )
        ),
        html.Hr(),
        html.Div(children=[
            html.Hr(),
            dbc.FormGroup(
                [
                    dbc.Label("区县名:", className="mr-2", width=2),
                    dbc.Col(
                        dcc.Dropdown(id="district_name",
                            options=[{"label": i, "value": i} for i in district_type_list],
                            value="萧山"
                        )
                    )
                ], row=True
            ),

            html.Div([
                html.Div([
                    dbc.FormGroup(
                        [
                            dbc.Label("经度:", className="mr-2", width=4),
                            dbc.Col(
                                dbc.Input(id="longitude", value=round(np.mean(df["longitude"]),4), \
                                    placeholder=str(round(np.min(df["longitude"]), 4))+"-"+\
                                        str(round(np.max(df["longitude"]), 4))),
                                width=6
                            )
                        ], row=True
                    ),
                ], style={'width': '48%', 'display': 'inline-block'}),

                html.Div([
                    dbc.FormGroup(
                        [
                            dbc.Label("纬度:", className="mr-2", width=4),
                            dbc.Col(
                                dbc.Input(id="latitude", value=round(np.mean(df["latitude"]),4),\
                                    placeholder=str(round(np.min(df["latitude"]), 4))+"-"+\
                                        str(round(np.max(df["latitude"]), 4))),
                                width=6
                            )
                        ], row=True
                    ),
                ], style={'width': '50%', 'display': 'inline-block'}),
            ]),

            html.Div([
                html.Div([
                    dbc.FormGroup(
                        [
                            dbc.Label("住房类型:", className="mr-2", width=4),
                            dbc.Col(
                                dcc.Dropdown(id="house_type",
                                    options=[{"label": i, "value": i} for i in house_type_list],
                                    value="住宅"
                                ),width=6
                            )
                        ], row=True
                    ),
                ], style={'width': '48%', 'display': 'inline-block'}),

                html.Div([
                    dbc.FormGroup(
                        [
                            dbc.Label("装修类型:", className="mr-2", width=4),
                            dbc.Col(
                                dcc.Dropdown(id="decoration",
                                    options=[{"label": i, "value": i} for i in decoration_type_list],
                                    value="精装修"
                                ),width=6
                            )
                        ], row=True
                    ),
                ], style={'width': '50%', 'display': 'inline-block'}),
            ]),

            html.Hr(),

            dbc.FormGroup(
                [
                    dbc.Label("选择户型:", className="mr-2", width=2),
                    dbc.Col(
                        dcc.Dropdown(
                            id = "selected_rooms", 
                            options=[{"label": str(i)+"室", "value": i} for i in range(1,9) if i != 7],
                            value = [3, ],
                            multi=True
                        )
                    ),
                ], row=True
            ),
            
            html.Div(id="addtional_roomtype"),
            
            html.Hr(),
            dbc.FormGroup(
                [
                    dbc.Label("特色标签:", className="mr-2", width=2),
                    dbc.Col(
                        dcc.Dropdown(
                            id = "selected_tags", 
                            options=[{"label": i, "value": i} for i in new_tag],
                            value = ["成熟商圈", ],
                            multi=True
                        )
                    ),
                ], row=True
            ),

            html.Hr(),

            html.Div([
                html.Div([
                    dbc.FormGroup(
                        [
                            dbc.Col(
                                dbc.Button("获取模型", id="get_model_btn", outline=True, color="secondary", \
                                    className="mr-1"),
                                width=4
                            ),
                            dbc.Col(
                                dcc.Dropdown(
                                    id = "selected_model", 
                                    # options=[{"label": i, "value": i} for i in saved_model],
                                    # value = saved_model[0],
                                ), width=6
                            ),
                        ], row=True
                    ),
                ], style={'width': '48%', 'display': 'inline-block'}),

                html.Div([dbc.Button("预测", id="predict_btn", outline=True, color="secondary", \
                    className="mr-1")], style={'width': '50%', 'display': 'inline-block'}),
            ]),

            dbc.Spinner(dbc.Collapse(\
                dbc.Card(dbc.CardBody("Predict Result")), id="predict_collapse"),
                size="sm"),

        ], style={
            'padding': '0px 10px 15px 10px',
            'marginLeft': 'auto', 'marginRight': 'auto', "width": "1000px",
            'boxShadow': '0px 0px 5px 5px rgba(204,204,204,0.4)'}),
    ])

@app.callback(
    [Output("selected_model", "options"),
    Output("selected_model", "value")],
    Input("get_model_btn", "n_clicks")
)
def update_selected_model(n_clicks):
    saved_model = os.listdir(root_path + "saved_model")
    # print(saved_model)
    options=[{"label": i, "value": i} for i in saved_model]
    value = saved_model[0]
    return options, value

@app.callback(
    Output("addtional_roomtype", "children"),
    Input("selected_rooms", "value")
)
def update_additional_rooms(value):
    total_value = [i for i in range(1,9) if i!=7]
    room_div = []
    for i in total_value:
        if i in value:
            room_div.append(html.Div([
                dbc.FormGroup(
                    [
                        dbc.Label("面积（{}室）:".format(i), className="mr-2", width=4),
                        dbc.Col(
                            dbc.Input(id="room_area"+str(i), placeholder="eg.32-68 (单位为m2)"),
                            width=6
                        ),
                    ], row=True
                ),
            ], style={'width': '48%', 'display': 'inline-block'}))
        else:
            room_div.append(html.Div(id="room_area"+str(i)))

    return room_div


@app.callback(
    [Output("predict_collapse", "children"),
    Output("predict_collapse", "is_open")],
    Input("predict_btn", "n_clicks"),
    [State("district_name", "value"),
    State("longitude", "value"),
    State("latitude", "value"),
    State("house_type", "value"),
    State("decoration", "value"),
    State("selected_rooms", "value"),
    State("room_area1", "value"),
    State("room_area2", "value"),
    State("room_area3", "value"),
    State("room_area4", "value"),
    State("room_area5", "value"),
    State("room_area6", "value"),
    State("room_area8", "value"),
    State("selected_tags", "value"),
    State("selected_model", "value")]
)
def update_predict_result(n_clicks, district_name, longitude, latitude, house_type, decoration, \
    selected_rooms, room_area1, room_area2, room_area3, room_area4, room_area5, room_area6, \
        room_area8, selected_tags, selected_model):
    
    x_df = {}
    room_area_list = [room_area1, room_area2, room_area3, room_area4, room_area5, room_area6, room_area8]
    room_num_list = [i for i in range(1, 9) if i != 7]

    x_df["district_name"] = distirct_map[district_name]
    x_df["house_type"] = house_type_map[house_type]
    x_df["decoration"] = decoration_map[decoration]

    # handle room type
    if len(selected_rooms) != 0:
        room_area_num_list = [ [float(area) for area in item.split("-")] for item in \
            room_area_list if item is not None]
        x_df["min_frame_area"] = min([min(area) for area in room_area_num_list])
        x_df["max_frame_area"] = max([max(area) for area in room_area_num_list])
        x_df["room_types"] = len(selected_rooms)
        avg_room_area = 0.0
        for i in range(len(room_area_num_list)):
            avg_room_area += np.mean(room_area_num_list[i]) / room_num_list[i]
        x_df["avg_room_area"] = avg_room_area / len(selected_rooms)
    
    else:
        x_df["min_frame_area"] = np.mean(df_processed["min_frame_area"])
        x_df["max_frame_area"] = np.mean(df_processed["max_frame_area"])
        x_df["room_types"] = int(np.mean(df_processed["room_types"]))
        x_df["avg_room_area"] = np.mean(df_processed["avg_room_area"])

    x_df["longitude"] = longitude if longitude is not None else np.mean(df_processed["longitude"])
    x_df["latitude"] = latitude if latitude is not None else np.mean(df_processed["latitude"])

    for tag in new_tag:
        x_df[tag] = 1 if tag in selected_tags else 0
    
    for key, value in x_df.items():
        x_df[key] = [value,]
    x_df = pd.DataFrame(x_df)
    columns = [{"name": i, "id": i} for i in x_df.columns] # 打印待预测的特征列表
    for i in range(len(columns)):
        if columns[i]["name"] == "longitude" or columns[i]["name"] == "latitude":
            columns[i]["type"] = "numeric"
            columns[i]["format"] = Format(precision=6, scheme=Scheme.fixed)

    model = joblib.load(root_path + "saved_model/" + selected_model)
    price = model.predict(x_df)
    is_open = True if n_clicks else False

    return [
        html.P(),
        dt.DataTable(id="example_table",
            columns=columns,
            fixed_rows={'headers': True},
            data=x_df.to_dict("records"),
            # style_table={'height': '500px', 'overflowX': 'auto'},
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
        html.P(),
        "预测价格为 " + str(round(price[0])) + "/m2"
    ], is_open

