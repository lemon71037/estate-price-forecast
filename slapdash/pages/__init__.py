import dash_html_components as _html
from numpy.lib.arraysetops import isin
from numpy.lib.type_check import nan_to_num
import pandas as pd
import numpy as np
from collections import Counter
from math import isnan
# from .data_table_display import district_dict

from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

root_path = "E:/github/check-price/AnalysisSystem/"

df = pd.read_csv(root_path + "dataset/raw_loupan_data.csv", header=0)

district_type_list = list(set(df["district_name"]))
district_map = {}
for i in range(len(district_type_list)):
    district_map[district_type_list[i]] = i

item_list = df.to_dict("records") # 将 dataframe 转为 dict_list 形式
district_dict = {} # 每个区县对应的项
for item in item_list:
    if item['district_name'] not in district_dict.keys():
        district_dict[item['district_name']] = [item, ]
    else:
        district_dict[item['district_name']].append(item)

# feat_key = ["district_name", "house_type", "decoration", "min_frame_area", \
#     "max_frame_area", "room_types", "avg_room_area", "longitude", "latitude"]

def findnum(s):
    """找到字符串中的数字
    Input:
        s - 输入字符串
    Output:
        numbers - 返回数字列表
    """
    numbers = []
    l = len(s)
    i = 0

    while i < l:
        num = ''
        symbol = s[i]
        while '0' <= symbol <= '9': # symbol.isdigit()
            num += symbol
            i += 1
            if i < l:
                symbol = s[i]
            else:
                break
        i += 1
        if num != '':
            numbers.append(int(num))
    
    return numbers

def discrete_feat_process(df, column_name):
    """离散特征处理
    Input:
        df - total DataFrame
        column_name - 列名(特征名)
    Output:
        feat_map - 向数字特征转换的 map
        num_list - 转换后的数字特征
    """
    feat_list = df[column_name]
    type_list = list(set(feat_list))
    feat_map = {}
    for i in range(len(type_list)):
        feat_map[type_list[i]] = i

    num_list = [feat_map[i] for i in feat_list]
    return num_list, feat_map, type_list

def fill_missing_data(raw_list, district_list):
    """填充缺失值
    Input:
        raw_list - 原始数据列表
        district_list - 区县名列表
    Output:
        feat_list - 处理后的数据列表
    """
    if 0 in raw_list:
        # raw_list = df[column_name] # 原始数据列表
        feat_list = [] # 处理后的数据列表
        # district_list = df["district_name"]
        district_feat_dict = {}
        for i in range(len(raw_list)):
            if district_list[i] not in district_feat_dict.keys():
                district_feat_dict[district_list[i]] = [raw_list[i]]
            else:
                district_feat_dict[district_list[i]].append(raw_list[i])
        
        # 获取每个区县下该特征的均值作为填充值
        for key, value in district_feat_dict.items():
            mean_value = np.mean([i for i in value if i != 0])
            inter_value = mean_value if mean_value != 0 else \
                np.mean([i for i in raw_list if i != 0])
            district_feat_dict[key] = inter_value

        for i in range(len(raw_list)):
            if raw_list[i] == 0:
                # 缺失值，用该楼盘所在区县中其他楼盘在该特征下的均值填充
                feat_list.append(district_feat_dict[district_list[i]])
            else:
                feat_list.append(raw_list[i])
        
        return feat_list
    
    return raw_list

def handle_rooms(rooms_list):
    """处理户型特征
    Input:
        room_list - 户型数的特征列表
    """
    avg_room_area_list = [] # 每个房间平均面积
    num_room_list = [] # 有多少种户型

    # room_type_list = [] # 统计各个户型出现的频率
    
    for rooms in rooms_list:
        if isinstance(rooms, str):
            total_avg_area = 0.0
            room_items = rooms.split(",")

            for item in room_items:
                room_type, area = item.split(":")

                area_list = area.split("-")
                area_list[-1] = area_list[-1][:-1]
                area = [float(i) for i in area_list] # 获取每个户型对应的最小/最大面积

                room_type, area = findnum(room_type)[0], np.mean(area)
                room_type = room_type if room_type != 0 else 1 # 会出现 0室 的情况
                # room_type_list.append(room_type)
                total_avg_area += area / room_type
            
            num_room_list.append(len(room_items))
            avg_room_area_list.append(total_avg_area / len(room_items))
        
        else:
            avg_room_area_list.append(0.0) # 缺失值
            num_room_list.append(0)
    
    return num_room_list, avg_room_area_list  

def handle_tags(tag_list, freq=50):
    """处理特色标签
    Input:
        tag_list - DataFrame中的tag列

    """
    total_tag_list = []
    
    for tag in tag_list:
        if isinstance(tag, str):
            total_tag_list = total_tag_list + tag.split(",")
    
    new_tag = []
    for key, value in dict(Counter(total_tag_list)).items():
        if value > freq:
            new_tag.append(key)
    
    new_tag_list = [[] for _ in range(len(new_tag))]
    for tag in tag_list:
        for i in range(len(new_tag)):
            # print(i, tag_list[i])
            if isinstance(tag, str) and new_tag[i] in tag:
                new_tag_list[i].append(1)
            else:
                new_tag_list[i].append(0)
    
    return new_tag, new_tag_list

def data_convert(df):
    """特征转换，将一些字符串特征转为数值特征
    Input:
        df - raw DataFrame
    Output:
        new_df - 转换后的 DataFrame
        new_tag - 作为特征的特殊标签
    """
    new_df = pd.DataFrame()
    discrete_feat_map = {}
    discrete_feat_type_list = {}
    for c in df.columns:
        if c in ["district_name", "house_type", "decoration"]:
            new_df[c], discrete_feat_map[c], discrete_feat_type_list[c] = discrete_feat_process(df, c)
        
        elif c in ["min_frame_area", "max_frame_area", "longitude", "latitude"]:
            new_df[c] = df[c]
        
        elif c == "converged_rooms":
            new_df["room_types"], new_df["avg_room_area"] = handle_rooms(df[c])

        elif c == "tags":
            new_tag, new_tag_list = handle_tags(df["tags"])
            for i in range(len(new_tag)):
                new_df[new_tag[i]] = new_tag_list[i]

    return new_df, discrete_feat_map, discrete_feat_type_list, new_tag

def data_process(df):
    """特征处理，替换缺失值等操作
    Input:
        df - 经过数值转换后的 DataFrame
    Output:
        df - 经过缺失值填充等预处理手段后的 DataFrame
    """
    new_df = df.copy()
    for c in new_df.columns:
        if c in ["min_frame_area", "max_frame_area", "longitude", "latitude", \
            "avg_room_area"]:
            new_df[c] = fill_missing_data(new_df[c], new_df["district_name"])
        
        elif c == "room_types":
            new_df[c] = [int(i) for i in fill_missing_data(new_df[c], new_df["district_name"])]

    return new_df

def page_not_found(pathname):
    return _html.P("No page '{}'".format(pathname))


df_convert, discrete_feat_map, discrete_feat_type_list, new_tag = data_convert(df)
df_processed = data_process(df_convert)

# if __name__ == "__main__":
    # df_convert = data_convert(df)
    # df_processed = data_process(df_convert)
    
    # value = 2
    
    # train_x, test_x, train_y, test_y = \
    #     train_test_split(df_processed, df["average_price"], test_size=0.2)

    # if value == 0:
    #     model = DecisionTreeRegressor()
    # elif value == 1:
    #     model = LinearRegression()
    #     # scaler = StandardScaler().fit(train_x)
    #     # train_x = scaler.transform(train_x)
    #     # test_x = scaler.transform(test_x)
    
    # elif value == 2:
    #     model = RidgeCV()
    
    # model.fit(train_x, train_y)
    
    # train_score = model.score(train_x, train_y)
    # test_score = model.score(test_x, test_y)
    # print("Train score: {:.4g}, Test score: {:.4g}".format(train_score, test_score))

    # result_df = pd.DataFrame()
    # eval_df = pd.DataFrame()
    # pred = model.predict(test_x)

    # result_df["pred"] = pred
    # result_df["true"] = list(test_y)
    # result_df["sample"] = list(range(1, len(test_y)+1))

    # eval_df["metric"] = ["mse", "mae", "r2_score"]
    # eval_df["value"] = [mean_squared_error(list(test_y), pred), \
    #     mean_absolute_error(list(test_y), pred), r2_score(list(test_y), pred)]
    
    # # fig = px.line(result_df, x="sample", y=["true", "pred"])
    # print(eval_df)
    # print(discrete_feat_map["decoration"])
