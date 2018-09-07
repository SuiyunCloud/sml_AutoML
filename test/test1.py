import matplotlib.pyplot as plt
import feature
import seaborn as sns
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)
from scipy import stats
from scipy.stats import norm, skew #for some statistics
from speedml import Speedml
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
 

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats 
sml = Speedml("C:\\Users\\suiyun.yang\\Desktop\\Kaggle\\datasets\\Kaggle\\Titanic\\train.csv","C:\\Users\\suiyun.yang\\Desktop\\Kaggle\\datasets\\Kaggle\\Titanic\\test.csv",target='Survived',uid='PassengerId')
train = sml.train
test = sml.test

ntrain = train.shape[0]
ntest = test.shape[0]
data = train.append(test)

sml.feature.outliers('Fare',upper=99)
sml.feature.outliers('Parch',upper=99)


### 使用 RandomForestClassifier 填补缺失的年龄属性
def set_missing_ages(df):    
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
    # y即目标年龄
    y = known_age[:, 0]
    # X即特征属性值
    X = known_age[:, 1:]
    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)    
    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])    
    # 用得到的预测结果填补原缺失数据
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges    
    return df, rfr
    #return rfr

#sml.train,rfr = set_missing_ages(sml.train)
df,rfr = set_missing_ages(sml.train)
sml.set('train',df)
testAgeNull = sml.test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']][sml.test.Age.isnull()] 
testAgeNull.Age = rfr.predict(testAgeNull.as_matrix()[:,1:])
sml.test.loc[(sml.test.Age.isnull()),"Age"] = testAgeNull

sml.feature.density(['Age','Ticket'])
ageDist = sml.train[['Age','Age_density']]
sml.feature.fillna(a='Cabin',new='Z')
sml.feature.extract(new='Deck',a='Cabin',regex='([A-Z]){1}')
sml.feature.mapping('Sex',{'male':0,'female':1})
sml.feature.sum(new='FamilySize',a='Parch',b='SibSp')
sml.feature.add('FamilySize',1)
sml.feature.drop(['Parch','SibSp'])
sml.train.info()
sml.feature.impute()
sml.feature.extract(new='Title',a='Name',regex='([A-Za-z]+)\.')
sml.feature.replace(a='Title',match=['Lady','Countess','Capt','Col','Don','Dr',"Major",'Rev','Sir','Jonkheer','Dona'],new='Rare')
sml.feature.replace('Title','Mlle','Miss')
sml.feature.replace("Title",'Mme','Mrs')
sml.feature.drop(['Name','Cabin','Ticket'])
sml.feature.labels(['Title','Embarked','Deck'])