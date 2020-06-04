import os
os.environ['PYSPARK_SUBMIT_ARGS'] = '--jars xgboost4j-spark-0.72.jar,xgboost4j-0.72.jar pyspark-shell'
# Default Java is 11 but need to use 8 for PySpark
java8_location= 'C:/Program Files/Java/jdk1.8.0_251'
os.environ['JAVA_HOME'] = java8_location

import pandas as pd
import numpy as np
import copy
import json
import sys
import operator
from datetime import datetime
import matplotlib.pyplot as plt
import random
import tqdm
import time
from datetime import datetime, timedelta

import pyspark
from pyspark.sql import types, SQLContext
from pyspark.sql.types import StructType, StructField

from pyspark import SparkConf
from pyspark.sql import SparkSession,Row
from pyspark.sql.functions import collect_list, struct, col, concat
from pyspark.sql.types import FloatType, StructField, StructType, StringType, TimestampType
from sklearn.metrics import mean_squared_error

from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
import xgboost

#os.environ['PYSPARK_PYTHON'] = './python_env/py27/bin/python2' 
#os.environ['PYSPARK_SUBMIT_ARGS'] = '--jars xgboost4j-spark-0.72.jar,xgboost4j-0.72.jar pyspark-shell'

#https://zhuanlan.zhihu.com/p/101997567

def mape(y_true, y_pred):
    """计算 mape"""
    return np.mean(np.abs((y_pred - y_true) / y_true))

def make_label_each_row(row):
    """制作标签"""
    model_serial = row.name[0]
    dt = row.name[1]
    if model_serial in fault_disk_dict:
        fault_time = fault_disk_dict[model_serial]
        # 如果在30天内, 标记为正样本
        data_date = datetime.strptime(str(dt), '%Y%m%d')
        fault_date = datetime.strptime(str(fault_time), "%Y-%m-%d")
        if (fault_date - data_date).days <= 30 & (fault_date - data_date).days > 0:
            return 1
    return 0

def normalize(df, df_mean=None, df_std=None):
    """
    describe: normalize data 
    parameter: DataFrame, df_mean, df_std
    return: DataFrame, df_mean, df_std
    """
    if (df_mean is None) or (df_std is None):
        df_mean = df.mean()
        df_std = df.std()
    else:
        print("Using passed df_mean and df_std")
    return (df - df_mean)/df_std, df_mean, df_std

def normalize_min_max(df, df_min=None, df_max=None):
    """
    describe: normalize data 
    parameter: DataFrame, df_mean, df_std
    return: DataFrame, df_mean, df_std
    """
    if (df_min is None) or (df_max is None):
        df_min = df.min()+1e-3
        df_max = df.max()
    else:
        print("Using passed df_mean and df_std")
    return (df - df_min)/(df_max-df_min), df_min, df_max

def shift_by_stock_code(df, tracking_days, mode, del_nan=True):
    '''
    :param df: 原始Dataframe
    :param tracking_days: 要进行回溯/延后的天数
    :param mode: 指定位移模式，可选"forward", "backward"
    
    :return: 一个新的dataframe. 加入位移后特征作为新列,索引与原始dataframe相同。
    '''
    
    #  判断位移模式
    if mode=="backward":
        shift_func=_shift_backward
    elif mode=="forward":
        shift_func=_shift_forward
    else:
        raise Exception("shift mode must be 'forward' or 'backward'")
    
    stock_code_list = df.index.levels[0].tolist()
    
    gen_data_block = list()
    
    #  对每只股票进行位移操作
    for sc in stock_code_list:
        tmp_df = df.loc[sc]

        # 小于track day的话回报奇怪的错
        if (tmp_df.shape[0]-2) < tracking_days:
            continue
        # 不做这步操作的的话，tmp_df仍然存在于原始df中，
        # 会拖慢shift效率
        tmp_df = pd.DataFrame(tmp_df)
        tmp_df = shift_func(tmp_df, tracking_days, sc)
        gen_data_block.append(tmp_df)
    #  拼接还原
    df = pd.concat(gen_data_block, axis=0, sort=True)
    # dropna
    if del_nan:
        df = df.dropna(axis=0)
    # 重设索引
    df.set_index(['code','dt'],inplace=True)    
    
    return df[sorted(df.columns)]

#  针对一支股票的backward
def _shift_backward(df, back_n_day, sc):
    num_of_columns = len(df.columns)
    #shift之前的index 还有 values
    original_value = df.values
    original_index = df.index
    # shift之后的column name
    backward_day_column = []
    # shift之后的数据array，init 为 nan
    
    if len(df) >= back_n_day:
        backward_day_data = np.empty(shape=(len(df), (back_n_day+1)*num_of_columns), dtype='object')
    else:
        backward_day_data = np.empty(shape=(back_n_day, (back_n_day+1)*num_of_columns), dtype='object')
        original_value = np.pad(original_value, ((0, back_n_day-len(df)), (0, 0)),'constant', constant_values=np.nan)
        original_index = list(original_index.values).extend([np.nan]*(back_n_day-len(df)))
    backward_day_data[:] = np.nan
    #不使用shift函数，直接用numpy赋值
    for i in range(0, back_n_day+1):
        tmp = [ "{}_{}_day_before".format(index, str(i).zfill(3)) for index in list(df.columns)]
        backward_day_column.extend(tmp)
        ##把数据填充到numpy里面
        #特殊情况i=0，就是[0:0]会出问题
        if i == 0:
            backward_day_data[:, i*num_of_columns:(i+1)*num_of_columns] = original_value
            continue
        backward_day_data[:, i*num_of_columns:(i+1)*num_of_columns] =  np.pad(original_value[:-i,:], ((i, 0), (0, 0)),                                                                            'constant', constant_values=np.nan)
    #合成dataframe
    new_df = pd.DataFrame(data=backward_day_data, columns=backward_day_column, index=original_index)
    new_df = new_df.reset_index()
    new_df['code'] = sc
    return new_df

#  针对一支股票的forward
def _shift_forward(df, forward_n_day, sc):
    num_of_columns = len(df.columns)
    #shift之前的index 还有 values
    original_value = df.values
    original_index = df.index
    # shift之后的column name
    forward_day_column = []
    # shift之后的数据array，init 为 nan
    if len(df) >= forward_n_day:
        forward_day_data = np.empty(shape=(len(df), (forward_n_day+1)*num_of_columns), dtype='object')
    else:
        forward_day_data = np.empty(shape=(forward_n_day, (forward_n_day+1)*num_of_columns), dtype='object')
        original_value = np.pad(original_value, ((0, forward_n_day-len(df)), (0, 0)),'constant', constant_values=np.nan)
        original_index = list(original_index.values).extend([np.nan]*(forward_n_day-len(df)))
    forward_day_data[:] = np.nan
    #不使用shift函数，直接用numpy赋值
    for i in range(0, forward_n_day+1):
        tmp = [ "{}_{}_day_before".format(index, str(i).zfill(3)) for index in list(df.columns)]
        forward_day_column.extend(tmp)
        ##把数据填充到numpy里面
        #print(forward_day_data[:, i*num_of_columns:(i+1)*num_of_columns].shape,\
        #     np.pad(original_value[i:,:], ((0, i), (0, 0)),'constant', constant_values=np.nan).shape)
        forward_day_data[:, i*num_of_columns:(i+1)*num_of_columns] =  np.pad(original_value[i:,:], ((0, i), (0, 0)),                                                                            'constant', constant_values=np.nan)
    #合成dataframe
    new_df = pd.DataFrame(data=forward_day_data, columns=forward_day_column, index=original_index)
    new_df = new_df.reset_index()
    new_df['code'] = sc
    return new_df


def Timedata(data):
    data['year'] = data['dt'].apply(lambda x:x.year)
    data['month'] = data['dt'].apply(lambda x:x.month)
    data['day'] = data['dt'].apply(lambda x:x.day)
    data['week'] = data['dt'].apply(lambda x:x.week)
    data['weekday'] = data['dt'].apply(lambda x:x.isoweekday())
    data['isoweekday'] = data['dt'].apply(lambda x:x.isoweekday())
    data['weekofyear'] = data['dt'].apply(lambda x:int(x.strftime("%W")))
    data['dayofyear'] = data['dt'].apply(lambda x:int(x.strftime("%j")))
    data['dayofmonth'] = data['dt'].apply(lambda x:int(x.strftime("%d")))
    return data

def feat_days2(line):
    "feature days2"
    return (line['year']-2018) + (line['month']-1)*30 + line['day']

def feat_generator(input_data):
    """Add features"""
    input_data = Timedata(input_data.reset_index())
    input_data['days2'] = input_data.apply(feat_days2, axis=1)
    if 'index' in input_data.columns:
        input_data.drop(['index'], axis=1, inplace=True)
    return input_data

if __name__ == '__main__':
    

    spark = SparkSession.builder.appName("SparkXgboostforNPA").enableHiveSupport().getOrCreate()
    # Enable Arrow-based columnar data transfers
    # spark.conf.set("spark.sql.execution.arrow.enabled", "true")
    print("Session created!")
    sc = spark.sparkContext
    sc.setLogLevel("INFO")
    sc.addPyFile("sparkxgb.zip")
    sqlContext = SQLContext(sc)

    from sparkxgb import XGBoostEstimator

    # dt, site_id, node_desc, us_load
    # Load from database
##    df0 = spark.sql("select dt, cast(site_id as String) as site_id, COALESCE(node_desc, 'null') as node_desc, us_load from inp.weekly_utilization_node_gnis where where us_load is not null")
##    df0.createOrReplaceTempView("tmp_df")
##
##    df1 = spark.sql("select dt, site_id, node_desc, sum(us_load) as value from tmp_df group by dt, site_id, node_desc")
##    df1.createOrReplaceTempView("tmp_df1")
##
##    df2 = spark.sql("select dt as date, concat(site_id,node_desc) as market, value from tmp_df1")

    # Load from csv
    file_location = 'usinput.csv/test_sample.csv'
    
    df = spark.read.format("csv").option("inferSchema", 
           True).option("header", True).load(file_location)

    # Concat sites and node_desc into market column
    df = df.withColumn("market",concat(col("sites"),col("node_desc"))).drop("sites").drop("node_desc")

    # Rename columns
    df = df.withColumnRenamed("time_key","date").withColumnRenamed("ds_load","value")

    # Drop rows with missing values
    df = df.dropna()

    # Select pertinent columns
    df = df.select(
        df['date'],
        df['market'],
        df['value']    
    )

    # Convert to pandas dataframe
    print('Starting conversion')
    input_data=df.select("*").toPandas()
    print('Converted')
    input_data['dt'] = pd.to_datetime(input_data.date)
    input_data.drop('date',axis=1,inplace=True)
    input_data.rename(columns={'market':'code', "value":"Y"}, inplace=True)
    # two level index
    input_data = input_data.set_index(['code','dt'])
    input_data.fillna(method='ffill', inplace=True)
    input_data = feat_generator(input_data)

    # shift生成时间序列
    shifted_input_data = shift_by_stock_code(input_data.set_index(['code','dt']), tracking_days=5, 
                        mode='backward', del_nan=True)

    shifted_input_data[(shifted_input_data<-10000000)]=np.nan

    shifted_input_data.dropna(axis=0,inplace=True)

    # make label
    shifted_input_data.rename(columns={'Y_000_day_before':'Y'}, inplace=True)

    ## start time
    train_start_t = shifted_input_data.reset_index()['dt'].min()
    ## split for test
    total_date_num = (shifted_input_data.reset_index()['dt'].max()-train_start_t).days
    train_date_num = int(total_date_num*0.75)
    ## end time
    train_end_t = train_start_t+timedelta(train_date_num); 
    test_start_t = train_end_t+timedelta(1)
    print("train time: [{}] to [{}]".format(train_start_t, train_end_t))
    print("test time: [{}] to [{}]".format(test_start_t, shifted_input_data.reset_index()['dt'].max()))
    print()
    print("train_datas_num:", train_date_num)
    print("test_datas_num:", total_date_num-train_date_num)
    print()

    # split according to date
    shifted_input_data = shifted_input_data.reset_index()
    train_df = shifted_input_data[shifted_input_data['dt'] <= train_end_t].copy()
    test_df = shifted_input_data[~(shifted_input_data['dt'] <= train_end_t)].copy()


    features = [col for col in train_df.columns if (col not in ['Y','code','dt'])]

    train_df.loc[:,features] = train_df.loc[:,features].astype(float)
    test_df.loc[:,features] = test_df.loc[:,features].astype(float)

    print(train_df)
    print(test_df)

    # Convert back to spark for training
    train_df = sqlContext.createDataFrame(train_df)
    test_df = sqlContext.createDataFrame(test_df)

    vectorAssembler = VectorAssembler()\
      .setInputCols(features)\
      .setOutputCol("features")

    xgboost = XGBoostEstimator(
        # evalMetric="rmse",
        featuresCol="features", 
        labelCol="Y",
        # numEarlyStoppingRounds=5,
        predictionCol="prediction",
    )

    pipeline = Pipeline().setStages([vectorAssembler, xgboost])

    model = pipeline.fit(train_df)

    model.transform(test_df).select(col("prediction")).show()

    
