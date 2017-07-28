#!/usr/bin/env python3
# -*- coding=utf-8 -*-
__author__ = 'skyeagle'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import datetime


class LearnPandas():
    def __init__(self):
        pass

    def createSeries(self):
        data = [1, 3, 5, np.nan, 6, 8]
        s = pd.Series(data)
        print(s)

    def createDataFrame(self):
        dates = pd.date_range(datetime.date.today(), periods=6)
        # print(dates)
        df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
        # print(df)

        df2 = pd.DataFrame({'A': 1.0,
                            'B': pd.date_range(datetime.date.today(), periods=4),
                            'C': np.random.rand(4),
                            'D': pd.Series(1, index=list(range(4)), dtype='float32'),
                            'E': np.array([3] * 4, dtype='int32'),
                            'F': pd.Categorical(["test", "train", "test", "train"]),
                            'G': 'foo'})
        # print(df2)
        # print(df2.dtypes)

        return df

    # 查看DateFrame数据
    def viewDateFrame(self, df):
        print(df.head(2))
        print("***********************")

        print(df.tail(2))
        print("***********************")

        print(df.index)
        print(df.columns)
        print("***********************")

        print(df.describe())
        print("***********************")

        print(df.T)
        print("***********************")

        print(df.sort_index(axis=1, ascending=False))
        print(df.sort_index(axis=0, ascending=False))
        print("***********************")

        print(df.sort_values(by='B'))

    # 获取DataFrame数据
    def getData(self, df):
        # 列
        print(df['A'])
        # 行
        print(df[4:])
        print(df[1:3])
        print(df['2017-07-29 ':'2017-08-01'])

        # 使用label，就是行和列的名字 location =loc
        dates = pd.date_range(datetime.date.today(), periods=6)
        print(df.loc[dates[0]])
        print(df.loc[:, ['B', 'C']])
        print(df.loc['2017-07-29 ':'2017-08-01', 'B':'C'])
        print(df.loc['2017-07-29 ', 'B':'C'])

        x = df.loc['2017-07-29 ', 'B']
        print(x)
        print(type(x))

        print(df.at[dates[0], 'B'])

        # 使用行和列的序号 index location =iloc
        print(df.iloc[3])
        print(df.iloc[3:5, 0:2])
        print(df.iloc[[1, 2, 4], [0, 2]])

        print(df.iloc[1:3, :])
        print(df.iloc[:, 1:3])

        print(df.iloc[3, 3])
        print(df.iat[3, 2])


if __name__ == "__main__":
    lp = LearnPandas()
    # lp.createDataFrame()
    df = lp.createDataFrame()
    # lp.viewDateFrame(df)
    lp.getData(df)
