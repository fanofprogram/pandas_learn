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

        # 逻辑操作
        print(df[df.B > 0])

        print(df[df > 0])

        df2 = df.copy()
        df2['E'] = ['one', 'one', 'two', 'three', 'four', 'three']
        print(df2[df2['E'].isin(['two', 'four'])])

    # 设置DataFrame里面的数据
    def setData(self, df):
        # 设置一列
        today = datetime.date.today()
        s = pd.Series(range(1, 7), index=pd.date_range(today, periods=6))
        df['F'] = s
        print(df)

        df.loc[:, 'D'] = np.array([5] * len(df))
        print(df)

        # 设置一个数据
        df.at['2017-07-29', 'B'] = 666.666
        print(df)
        df.iat[3, 3] = 888.888
        print(df)

        #
        df[df > 0] = -df
        print(df)

    def missingData(self, df):
        df1 = df.reindex(index=df.index[0:4], columns=list(df.columns) + ['E'])
        print(df1)
        df1.iloc[0:2, 4] = 1
        print(df1)
        df1.iloc[0, 3] = np.nan
        print(df1)
        print(df1.dropna(how='any'))
        print(df1.fillna(value=666))
        print(df1.isnull())

    def stats(self, df):
        df.loc[:, 'E'] = range(0, len(df))
        print(df)
        print(df.mean())
        print(df.mean(1))

    def applyfun(self, df):
        df.loc[:, 'E'] = range(0, len(df))
        print(df)
        print(df.apply(sum))
        print(df.apply(np.cumsum))
        ydf = df.apply(lambda x: x.max() - x.min())
        print(ydf)

    def histogramming(self):
        s = pd.Series(np.random.randint(0, 7, size=10))
        print(s)
        print(s.value_counts())

    def string(self):
        s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
        print(s.str.lower())
        print(s.str.upper())

    def merge(self):
        df = pd.DataFrame(np.random.randn(10, 4))
        print(df)
        a = df[:3]
        print(a)
        b = df[3:7]
        print(b)
        c = df[7:]
        print(c)
        x = pd.concat([a, b, c])
        print(x)

        left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})
        right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})

        print(left)
        print(right)
        print(pd.merge(left, right, on='key'))

    def plot(self, df):
        df.plot()
        plt.show()

if __name__ == "__main__":
        lp = LearnPandas()
        # lp.createDataFrame()
        df = lp.createDataFrame()
        # lp.viewDateFrame(df)
        # lp.getData(df)
        # lp.setData(df)
        # lp.missingData(df)
        # lp.stats(df)
        # lp.applyfun(df)
        # lp.plot(df)
        # lp.histogramming()
        # lp.string()
        lp.merge()
