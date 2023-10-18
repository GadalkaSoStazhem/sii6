import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

def shape (df):
    print(df.shape)
    return df.shape

def get_num_cols(df):
    data = {}
    df_num = pd.DataFrame(data)
    for col in df.columns:
        if type(df[col].to_dict()[0]) == int or type(df[col].to_dict()[0]) == float:
            df_num[df[col].name] = df[col].values
        else:
            print("Столбец ", df[col].name, " содержит нечисловые значения")
    return df_num

def mean_vals(df, rows):
    means = []
    for col in df.columns:
        means.append(df[col].sum() / rows)
    return means

def st_deviation(df, means, n):
    cnt = 0
    devs = []
    for col in df.columns:
        tmp = sum((x - means[cnt]) ** 2 for x in df[col].values) / n
        st_dev = np.sqrt(tmp)
        devs.append(st_dev)
        cnt += 1
    return devs

def printer (df, arr):
    cnt = 0
    for col in df.columns:
        print(df[col].name, ": ", arr[cnt])
        cnt += 1

def min_max(df):
    mins = []
    maxs = []
    print("\nМинимальные и максимальные значения столбцов")
    for col in df.columns:
        if type(df[col].to_dict()[0]) == int or type(df[col].to_dict()[0]) == float:
            print(df[col].name, ": Минимум: ", df[col].min(), " Максимум: ", df[col].max())

def quantiles(df, n):
    print("\nКвантили: ")
    for col in  df.columns:
        val_list = df[col].values
        val_list = sorted(val_list)
        q_25 = val_list[math.floor((n + 1) * 0.25)]
        q_50 = val_list[math.floor((n + 1) * 0.5)]
        q_75 = val_list[math.floor((n + 1) * 0.75)]
        print(df[col].name, " Q(0.25) = ", q_25, "; Q(0.5) = ", q_50, "; Q(0.75) = ", q_75)

def visualization (df):
    plt.figure(figsize = (20, 20))
    for i, col in enumerate(df.columns):
        plt.subplot(2, 3, i + 1)
        df[col].plot(kind = 'box')
        plt.title(df[col].name)
    plt.tight_layout()
    plt.show()

def get_characteristics(df):
    df_shape = shape(df)
    print("Характеристики числовых столбцов: ")
    df_nums = get_num_cols(df)
    print("\nСредние значения по колонкам: ")
    means = mean_vals(df_nums, df_shape[0])
    printer(df_nums, means)
    print("\nСтандартные отклонения: ")
    devs = st_deviation(df_nums, means, df_shape[0])
    printer(df_nums, devs)
    min_max(df_nums)
    quantiles(df_nums, df_shape[0])
    #visualization(df_nums)