import os
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table as dt
from dash.dependencies import Input, State, Output
import pandas as pd
import numpy as np
from dash_table.Format import Format, Scheme

from sklearn.linear_model import LinearRegression, ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

from textwrap import dedent
# from collections import Counter

import plotly.express as px
from ..app import app
from .data_table_display import item_list
from . import df_processed, df, root_path

df_processed = df_processed
df = df
saved_model = os.listdir(root_path + "saved_model")

def get_layout(args):
    
    return html.Div(children=[
        dcc.Markdown(
            dedent(
                """
                # 新盘限价预测模型训练与测试
                
                Training and testing of real estate price limit prediction model.
                """
            )
        ),
        html.Hr(),
        html.Div(children=[
            html.Hr(),
            dbc.FormGroup(
                [
                    dbc.Label("Algorithum:", className="mr-2", width=2),
                    dbc.Col(
                        dcc.Dropdown(id="alg_list",
                            options=[
                                {"label": "决策树回归 - Decision Tree Regression", "value": 0},
                                {"label": "线性回归 - Linear Regression", "value": 1},
                                {"label": "弹性网络回归 - ElasticNet Regression", "value": 2},
                                {"label": "KNN回归 - K-nearest Neighbors Regression", "value": 3},
                                {"label": "随机森林回归 - Random Forest Regression", "value": 4},
                                {"label": "AdaBoost回归 - AdaBoost Regression", "value": 5},
                                {"label": "GBRT回归 - Gradient Boosting Regressor", "value": 6},
                            ],
                            value=0
                        ),
                    )
                ], row=True
            ),

            html.Div(id="alg_config"),
            
            dbc.FormGroup(
                [
                    dbc.Label("TestSize:", html_for="slider", className="mr-2", width=2),
                    dcc.Slider(id="test_size", min=0.1, max=0.9, step=0.1, value=0.2),
                ]
            ),

            dbc.Form(
                [
                    dbc.Button("Train and Saved As", id="train_btn", outline=True, \
                        color="secondary", className="mr-1"),
                    dbc.Input(id="model_name", type="text")
                ], inline=True
            ),

            dbc.Spinner(dbc.Collapse(\
                dbc.Card(dbc.CardBody("This content is hidden in the collapse")), id="train_collapse"),
                size="sm"),
            
            html.Hr(), 
            html.Div("测试结果评估", style={"text-align": "center"}),
            dt.DataTable(id="metric_table",
                fixed_rows={'headers': True},
                style_table={'width': '500px', 'margin-top':'20px', 'margin-left':'250px'},
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

            html.Hr(), 
            html.Div("测试集上预测结果与真实结果对比图", style={"text-align": "center"}),
            dcc.Graph(id="test_graph"),

        ], style={
            'padding': '0px 10px 15px 10px',
            'marginLeft': 'auto', 'marginRight': 'auto', "width": "1000px",
            'boxShadow': '0px 0px 5px 5px rgba(204,204,204,0.4)'}),
    ])

@app.callback(
    [Output("alg_config", "children"),
    Output("model_name", "value")],
    Input("alg_list", "value")
)
def update_feature_compared(value):

    if value == 0:
        return [
            html.Div([
                dbc.FormGroup(
                    [
                        dbc.Label("Criterion:", className="mr-2", width=4),
                        dbc.Col(
                            dcc.Dropdown(id="config1",
                                options=[{"label": "mse", "value": "mse"},
                                    {"label": "friedman_mse", "value": "friedman_mse"},
                                    {"label": "mae", "value": "mae"},
                                    {"label": "poisson", "value": "poisson"}],
                                value="mse"
                            ),width=6
                        )
                    ], row=True
                ),
            ], style={'width': '48%', 'display': 'inline-block'}),

            html.Div([
                dbc.Form(
                    [
                        dbc.Label("Max_depth:", className="mr-2", width=4),
                        dbc.Input(id="config2", type="number", value=3),
                    ], inline=True
                ),
            ], style={'width': '50%', 'display': 'inline-block'}),
        ], "DecisionTreeModel"

    elif value == 1:
        return [html.Div(id="config1"), html.Div(id="config2")], "LinearRegressionModel"
    
    elif value == 2:
        return [
            dbc.FormGroup(
                [
                    dbc.Label("l1_ratio（0-Lasso回归 1-岭回归）:", html_for="config1", className="mr-2", width=2),
                    dcc.Slider(id="config1", min=0, max=1, step=0.1, value=0.5),
                ]
            ),
            html.Div(id='config2')
        ], "ElasticNetModel"
    
    elif value == 3:
        return [
            html.Div([
                dbc.FormGroup(
                    [
                        dbc.Label("Weights:", className="mr-2", width=4),
                        dbc.Col(
                            dcc.Dropdown(id="config1",
                                options=[{"label": "uniform", "value": "uniform"},
                                    {"label": "distance", "value": "distance"}],
                                value="uniform"
                            ),width=6
                        )
                    ], row=True
                ),
            ], style={'width': '48%', 'display': 'inline-block'}),

            html.Div([
                dbc.Form(
                    [
                        dbc.Label("N_neighbors:", className="mr-2", width=4),
                        dbc.Input(id="config2", type="number", value=5),
                    ], inline=True
                ),
            ], style={'width': '50%', 'display': 'inline-block'}),
        ], "KNNModel"
    
    elif value == 4:
        return [
            html.Div([
                dbc.FormGroup(
                    [
                        dbc.Label("Criterion:", className="mr-2", width=4),
                        dbc.Col(
                            dcc.Dropdown(id="config1",
                                options=[{"label": "mse", "value": "mse"},
                                    {"label": "mae", "value": "mae"}],
                                value="mse"
                            ),width=6
                        )
                    ], row=True
                ),
            ], style={'width': '48%', 'display': 'inline-block'}),

            html.Div([
                dbc.Form(
                    [
                        dbc.Label("N_estimators:", className="mr-2", width=4),
                        dbc.Input(id="config2", type="number", value=20),
                    ], inline=True
                ),
            ], style={'width': '50%', 'display': 'inline-block'}),
        ], "RandomForestModel"
    
    elif value == 5:
        return [
            html.Div([
                dbc.FormGroup(
                    [
                        dbc.Label("Loss:", className="mr-2", width=4),
                        dbc.Col(
                            dcc.Dropdown(id="config1",
                                options=[{"label": "linear", "value": "linear"},
                                    {"label": "square", "value": "square"},
                                    {"label": "exponential", "value": "exponential"}],
                                value="linear"
                            ),width=6
                        )
                    ], row=True
                ),
            ], style={'width': '48%', 'display': 'inline-block'}),

            html.Div([
                dbc.Form(
                    [
                        dbc.Label("N_estimators:", className="mr-2", width=4),
                        dbc.Input(id="config2", type="number", value=20),
                    ], inline=True
                ),
            ], style={'width': '50%', 'display': 'inline-block'}),
        ], "AdaBoostModel"

    elif value == 6:
        return [
            html.Div([
                dbc.FormGroup(
                    [
                        dbc.Label("Loss:", className="mr-2", width=4),
                        dbc.Col(
                            dcc.Dropdown(id="config1",
                                options=[{"label": "ls", "value": "ls"},
                                    {"label": "lad", "value": "lad"},
                                    {"label": "huber", "value": "huber"},
                                    {"label": "quantile", "value": "quantile"}],
                                value="ls"
                            ),width=6
                        )
                    ], row=True
                ),
            ], style={'width': '48%', 'display': 'inline-block'}),

            html.Div([
                dbc.Form(
                    [
                        dbc.Label("N_estimators:", className="mr-2", width=4),
                        dbc.Input(id="config2", type="number", value=3),
                    ], inline=True
                ),
            ], style={'width': '50%', 'display': 'inline-block'}),
        ], "GBRTModel"
    
    return [html.Div(id="config1"), html.Div(id="config2")], "NoneModel"
    

@app.callback(
    [Output("train_collapse", "children"),
    Output("train_collapse", "is_open"),
    Output("metric_table", "columns"), 
    Output("metric_table", "data"),
    Output("test_graph", "figure")],
    Input("train_btn", "n_clicks"),
    [State("alg_list", "value"),
    State("test_size", "value"),
    State("config1", "value"),
    State("config2", "value"),
    State("model_name", "value")]
)
def update_test_graph(n_clicks, value, test_size, config1, config2, model_name):

    train_x, test_x, train_y, test_y = \
        train_test_split(df_processed, df["average_price"], test_size=test_size)

    if value == 0:
        model = DecisionTreeRegressor(criterion=config1, max_depth=config2)
    elif value == 1:
        model = LinearRegression()
        # scaler = StandardScaler().fit(train_x)
        # train_x = scaler.transform(train_x)
        # test_x = scaler.transform(test_x)
    elif value == 2:
        model = ElasticNetCV(l1_ratio=float(config1))
    elif value == 3:
        model = KNeighborsRegressor(weights=config1, n_neighbors=config2)
    elif value == 4:
        model = RandomForestRegressor(criterion=config1, n_estimators=config2)
    elif value == 5:
        model = AdaBoostRegressor(loss=config1, n_estimators=config2)
    elif value == 6:
        model = GradientBoostingRegressor(loss=config1, n_estimators=config2)
    
    model.fit(train_x, train_y)
    joblib.dump(model, root_path + "saved_model/"+model_name+".pkl")

    train_score = model.score(train_x, train_y)
    test_score = model.score(test_x, test_y)
    train_result = dbc.Card(dbc.CardBody("Train score: {:.4g}, Test score: {:.4g}"\
        .format(train_score, test_score)))
    is_open = True if n_clicks else False

    result_df, eval_df = pd.DataFrame(), pd.DataFrame()
    pred = model.predict(test_x)

    result_df["pred"] = pred
    result_df["true"] = list(test_y)
    result_df["sample"] = list(range(1, len(test_y)+1))

    eval_df["metric"] = ["rmse", "mae", "r2_score"]
    eval_df["value"] = [np.sqrt(mean_squared_error(list(test_y), pred)), \
        mean_absolute_error(list(test_y), pred), \
            r2_score(list(test_y), pred)]
    
    eval_columns = [{"name": i, "id": i} for i in eval_df.columns]
    eval_columns[1]["format"] = Format(precision=6, scheme=Scheme.fixed)
    
    test_fig = px.line(result_df, x="sample", y=["true", "pred"], \
        color_discrete_sequence=["#9BCD9B", "#7EC0EE"])

    return train_result, is_open, eval_columns, eval_df.to_dict("records"), test_fig

