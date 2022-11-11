import time
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm

#start time 开始时间
St = time.time()

#read data from csv file 从csv文件中读取数据
path = './dataset/'

#1. fund return
#基金复权净值收益率 测试集
fr_test  = pd.read_csv(path + 'test_fund_return.csv')
#基金复权净值收益率 训练集
fr_train = pd.read_csv(path + 'train_fund_return.csv')
# 合并训练数据和测试数据
fr = pd.merge(fr_train, fr_test, how='left')
# 读入数据时，发现都是<1的小数，因此将读入的数据都乘一个倍数，来提高精度 *10000
fr = pd.concat([fr["Unnamed: 0"], fr.drop(columns = "Unnamed: 0")*1000], axis=1)

#2. fund benchmark return
#基金业绩比较基准收益率 测试集
fbr_test  = pd.read_csv(path + 'test_fund_benchmark_return.csv')
#基金业绩比较基准收益率 训练集
fbr_train = pd.read_csv(path + 'train_fund_benchmark_return.csv')
# 合并训练数据和测试数据
fbr = pd.merge(fbr_train, fbr_test, how='left')
# 读入数据时，发现都是<1的小数，因此将读入的数据都乘一个倍数，来提高精度 *10000
fbr = pd.concat([fbr["Unnamed: 0"], fbr.drop(columns = "Unnamed: 0")*10000], axis=1)

#3. index return
#重要市场指数收益率 测试集
ir_test  = pd.read_csv(path + 'test_index_return.csv', encoding='GBK', index_col=0)
#重要市场指数收益率 训练集
ir_train =  pd.read_csv(path + 'train_index_return.csv', encoding='GBK', index_col=0)
# 读入数据时，发现都是<1的小数，因此将读入的数据都乘一个倍数，来提高精度 *10000
ir = pd.concat([ir_train, ir_test], axis=1)*10000

#4. correlation
#基金间的相关性 测试集
correlation_test  = pd.read_csv(path + 'test_correlation.csv')
#基金间的相关性 训练集
correlation_train = pd.read_csv(path + 'train_correlation.csv')
#合并
correlation = pd.merge(correlation_train, correlation_test, how='left')
#读入数据时，发现都是<1的小数，因此将读入的数据都乘一个倍数，来提高精度 *10000
correlation = pd.concat([correlation["Unnamed: 0"], correlation.drop(columns="Unnamed: 0")*10000], axis=1)

#training target: correlation between fundA ans fundB
#根据TargetID把基金对拆为两列ID 分别为基金A和基金B
ID = correlation["Unnamed: 0"]
ID = pd.concat([ID.map(lambda x:x.split('-')[0]), ID.map(lambda x:x.split('-')[1])], axis=1)
ID.columns = ['fundA', 'fundB']


#model evaluate
#模型评估 根据评分规则定义函数
from sklearn.metrics import mean_absolute_error
#相关性预测值 相关性真实值
def ModelEva(label, y):
    MAE = mean_absolute_error(label, y)
    TMAPE = cnt = 0
    for i in range(len(label)):
        TMAPE += abs((y[i] - label[i])/(1.5 - label[i]))
        cnt   += 1
    TMAPE = TMAPE / cnt
    score = (2 / (2 + MAE + TMAPE))**2
    print("Model Score: ", score)
    return score


#construct 1st layer train set
##data from t1 to t2
#定义TrainData函数：根据输入的数据集和起止时间，提取基金A和基金B的数据作为特征
#提取数据
def TrainData(dataset, t1, t2): #(DataFrame, column time, column time)
    data = pd.concat([dataset[dataset.columns[0]], dataset[dataset.columns[t1:t2]]], axis=1)
    #data1: fundA col
    data.rename(columns={data.columns[0]:"fundA"}, inplace=True)
    data1 = pd.merge(ID, data, how='left')
    data1 = data1[data1.columns[2:]]
    data1.columns = range(0, data1.shape[1])
    #data2: fundB col
    data.rename(columns={data.columns[0]:"fundB"}, inplace=True)
    data2 = pd.merge(ID, data, on='fundB', how='left')
    data2 = data2[data2.columns[2:]]
    data2.columns = range(0, data2.shape[1])
    return data1, data2

#提取第一层训练集特征共3组特征:
# 1.基金对的fund_return相关性
# 2.基金对的benchmark_return相关性
# 3.基金对相关性均值和分位数

##feature project
#提取特征 # 1.基金对的fund_return相关性 # 2.基金对的benchmark_return相关性
###construct feature for training set using data from t1 to t2
def TrainFeature(t1, t2):
    #特征1：基金对的fund_return相关性
    data1, data2 = TrainData(fr, t1, t2)
    # 用于计算data1和data2列与列之间的相关性
    fr_cor = data1.corrwith(data2, axis=1)
    #特征2：基金对的benchmark_return相关性
    data1, data2 = TrainData(fbr, t1, t2)
    # 用于计算data1和data2列与列之间的相关性
    fbr_cor= data1.corrwith(data2, axis=1)
    # 按垂直方向（行顺序）堆叠数组构成一个新的数组
    return np.vstack([fr_cor, fbr_cor]).T

#stacking就是用下一个模型去针对上一个模型预测的结果进行学习
#计算数据集的平均值，25%、50%、75%分位值，作为特征之一]
#定义函数：根据给定时间间隔和次数，叠加特征集，并增加一组特征：计算基金对相关性的平均值，25%、50%、75%分位值。
#stack feature from t1 to t2 many times to enlarge train set
def StackFeature(t1, t2, times):
    #特征3：基金A-基金B检验相关均值和分位数
    data = correlation.drop(columns="Unnamed: 0")
    fea1 = data.mean(axis=1)
    fea2 = data.quantile(0.25, axis=1)
    fea3 = data.quantile(0.5 , axis=1)
    fea4 = data.quantile(0.75, axis=1)
    f2 = np.vstack([fea1, fea2, fea3, fea4]).T
    #按垂直方向（行顺序）堆叠数组构成一个新的数组
    # tqdm 是python进度条库 可以在Python长循环中添加一个进度提示信息。用户只需要封装任意的迭代器，是一个快速、扩展性强的进度条工具库。
    for i in tqdm(range(times)):
        if i == 0:
            f1 = TrainFeature(t1, t2)
            xtrain = np.hstack([f2, f1])
        else:
            f1 = TrainFeature(t1 - 20*(i+1), t2)
            xtrain = np.hstack([xtrain, f1])
    return xtrain

###train set
for i in range(15):
    tsetx = StackFeature(-82, -62, 20)
    tsety = correlation[correlation.columns[-2-i]]
    if i == 0:
        xtrain = tsetx
        ytrain = tsety
    else:
        xtrain = np.vstack([xtrain, tsetx])
        ytrain = np.hstack([ytrain, tsety])




#1、定义训练目标和验证集目标
#验证集
xval = StackFeature(-81, -61, 20)
yval = correlation[correlation.columns[-1]]
#用于线下测试集，用于模型验证


#设定:间隔每20天提取一次上述三个特征特征，即0-20，0-40……0-400天的数据，生成训练、验证、测试数据集
#加上基金对相关性的 平均值，25%、50%、75%分位值共1004列特征
#test set for predict 测试
xtest = StackFeature(-20, None, 20)

#定义xgboost模型
# 训练集 相关性  验证集  测试集 参数
def XGB(xtrain, label, val, xtest, params):
    trainM = xgb.DMatrix(np.array(xtrain), label)
    valM   = xgb.DMatrix(np.array(val))
    testM  = xgb.DMatrix(np.array(xtest))
    model = xgb.train(params, trainM, params['nrounds'])
    return model.predict(valM), model.predict(testM)

#定义lgboost模型
def LGB(xtrain, label, val, xtest, params):
    trainM = lgb.Dataset(np.array(xtrain), label)
    model = lgb.train(params, trainM, params['nrounds'])
    #训练参数 要训练的数据 提升迭代次数
    return model.predict(val), model.predict(xtest)

#lgb
lgb_params = {
    'application':'regression_l1',
    'metric':'mae',
    'seed': 0,
    'learning_rate':0.04,
    'max_depth':1,
    'feature_fraction':0.7,
    'lambda_l1':2,
    'nrounds':900
}
lgbval, lgby = LGB(xtrain, ytrain, xval, xtest, lgb_params)

#xgb
xgb_params = {
    'objective':'reg:linear',
    'learning_rate':0.3,
    'max_depth':1,
    'subsample':1,
    'colsample_bytree':0.06,
    'alpha':50,
    'lambda':5,
    'nrounds':1800
}
# 5-fold
rows = xtrain.shape[0]
piece = int(rows/5)
xtrain_1 = xtrain[0:(piece)*4]
ytrain_1 = ytrain[0:(piece)*4]
xval_1   = xtrain[(piece)*4:]

xtrain_2 = np.vstack([xtrain[0:(piece)*3], xtrain[(piece)*4:]])
ytrain_2 = np.hstack([ytrain[0:(piece)*3], ytrain[(piece)*4:]])
xval_2   = xtrain[(piece)*3:(piece)*4]

xtrain_3 = np.vstack([xtrain[0:(piece)*2], xtrain[(piece)*3:]])
ytrain_3 = np.hstack([ytrain[0:(piece)*2], ytrain[(piece)*3:]])
xval_3   = xtrain[(piece)*2:(piece)*3]

xtrain_4 = np.vstack([xtrain[0:(piece)*1], xtrain[(piece)*2:]])
ytrain_4 = np.hstack([ytrain[0:(piece)*1], ytrain[(piece)*2:]])
xval_4   = xtrain[(piece)*1:(piece)*2]

xtrain_5 = xtrain[(piece)*1:]
ytrain_5 = ytrain[(piece)*1:]
xval_5   = xtrain[0:(piece)*1]

xgbval1, xgby1 = XGB(xtrain_1, ytrain_1, xval, xtest, xgb_params)
xgbval2, xgby2 = XGB(xtrain_2, ytrain_2, xval, xtest, xgb_params)
xgbval3, xgby3 = XGB(xtrain_3, ytrain_3, xval, xtest, xgb_params)
xgbval4, xgby4 = XGB(xtrain_4, ytrain_4, xval, xtest, xgb_params)
xgbval5, xgby5 = XGB(xtrain_5, ytrain_5, xval, xtest, xgb_params)

xgby = np.vstack([xgby1, xgby2, xgby3, xgby4, xgby5])
xgby = np.mean(xgby, axis=0)
xgbval = np.vstack([xgbval1, xgbval2, xgbval3, xgbval4, xgbval5])
xgbval = np.mean(xgbval, axis=0)


#第二层模型训练
xtrain2 = np.vstack([xgbval, lgbval])
date = [5, 30, 60, 90]
for i in tqdm(date):
    data1, data2 = TrainData(fr, -61-i, -61)
    #feature: fundA-fundB sum distance
    fea = abs(data1 - data2).sum(axis=1)  #计算基金对fund_return的曼哈顿距离并求和
    xtrain2 = np.vstack([xtrain2, fea])
xtrain2 = xtrain2.T
ytrain2 = correlation[correlation.columns[-1]]

#lgb train
lgbs_params = {
    'application':'regression_l1',
    'seed':0,
    'learning_rate': 0.01,
    'max_depth':1,
    'feature_fraction':0.8,
    'nrounds':1800
}

##第二层模型测试
xtest2  = np.vstack([lgby, lgby])
for i in tqdm(date):
    data1, data2 = TrainData(fr, -i, None)
    #feature: fundA-fundB sum distance
    fea = abs(data1 - data2).sum(axis=1)
    xtest2 = np.vstack([xtest2, fea])
xtest2 = xtest2.T

yval2, ypredict = LGB(xtrain2, ytrain2, xtrain2, xtest2, lgbs_params)

#模型评估
ModelEva(yval/10000, yval2/10000)
ModelEva(yval/10000, ypredict/10000)

#保存预测结果
res = pd.DataFrame({"ID":correlation["Unnamed: 0"], "value":ypredict/10000})
res.to_csv("predict.csv", index=None)

#训练模型时间
Ed = time.time()
print("The train takes " + str(Ed - St) + "second")