import dash_html_components as html

from .app import app
from .utils import DashRouter, DashNavBar
from .pages import data_table_display, feature_preprocess, \
        model_train_test, real_estate_price_predict
from .components import fa


# Ordered iterable of routes: tuples of (route, layout), where 'route' is a
# string corresponding to path of the route (will be prefixed with Dash's
# 'routes_pathname_prefix' and 'layout' is a Dash Component.
urls = (
    ("", data_table_display.get_layout),
    ("data_table_display", data_table_display.get_layout),
    ("feature_preprocess", feature_preprocess.get_layout),
    ("model_train_test", model_train_test.get_layout),
    ("real_estate_price_predict", real_estate_price_predict.get_layout)
)

# Ordered iterable of navbar items: tuples of `(route, display)`, where `route`
# is a string corresponding to path of the route (will be prefixed with
# URL_BASE_PATHNAME) and 'display' is a valid value for the `children` keyword
# argument for a Dash component (ie a Dash Component or a string).
nav_items = (
    ("data_table_display", html.Div([fa("fas fa-keyboard"), "历史楼盘数据"])),
    ("feature_preprocess", html.Div([fa("fas fa-chart-area"), "楼盘特征预处理"])),
    ("model_train_test", html.Div([fa("fas fa-chart-line"), "预测模型训练与测试"])),
    ("real_estate_price_predict", html.Div([fa("fas fa-address-book"), "新盘限价预测"]))
)

router = DashRouter(app, urls)
navbar = DashNavBar(app, nav_items)
