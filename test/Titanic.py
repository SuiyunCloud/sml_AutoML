
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from collections import defaultdict, OrderedDict
from speedml.tuning import auto_tuning

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

from scipy import stats
from scipy.stats import norm, skew #for some statistics

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

if __name__ == "__main__":
    # Read the data
    train = pd.read_csv('C:\\Users\\suiyun.yang\\Desktop\\Kaggle\\datasets\\Kaggle\\HousePrice\\train.csv')
    # Read the test data
    test = pd.read_csv('C:\\Users\\suiyun.yang\\Desktop\\Kaggle\\datasets\\Kaggle\\HousePrice\\test.csv')
    
    
    
    #Deleting outliers
    train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
    y = train.SalePrice
    
    import seaborn as sns
    from scipy.stats import norm
    (mu,sigma) = norm.fit(y)
    
    
    from scipy import stats
    
    y = np.log1p(y)
    
    ntrain = train.shape[0]
    ntest = test.shape[0]
    y_train = train.SalePrice.values
    data = train.append(test).reset_index(drop=True) #两个数据集中的索引有重叠，重建索引
    data.drop(['SalePrice','Id'],axis=1,inplace=True)
    
    
    
    # Encode some categorical features as ordered numbers when there is information in the or
    data = data.replace({"Alley" : {"Grvl" : 1, "Pave" : 2},
                           "BsmtCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                           "BsmtExposure" : {"No" : 0, "Mn" : 1, "Av": 2, "Gd" : 3},
                           "BsmtFinType1" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 
                                             "ALQ" : 5, "GLQ" : 6},
                           "BsmtFinType2" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 
                                             "ALQ" : 5, "GLQ" : 6},
                           "BsmtQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},
                           "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                           "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                           "FireplaceQu" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                           "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, 
                                           "Min2" : 6, "Min1" : 7, "Typ" : 8},
                           "GarageCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                           "GarageQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                           "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                           "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                           "LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},
                           "LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},
                           "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},
                           "PoolQC" : {"No" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},
                           "Street" : {"Grvl" : 1, "Pave" : 2},
                           "Utilities" : {"ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4}}
                         )
    
    
    data_na = data.isnull().sum()/len(data) * 100
    data_na = data_na.drop(data_na[data_na == 0].index).sort_values(ascending=False)
    data_na[:30]
    
    
    def fill_na(items,value):
        for item in items:
            data[item] = data[item].fillna(value)
            
    def fill_na_with_mode(items):
        for item in items:
            data[item] = data[item].fillna(data[item].mode()[0])
    
    
    
    data_na = data.isnull().sum()/len(data) * 100
    data_na = data_na.drop(data_na[data_na == 0].index).sort_values(ascending=False)
    data_na[:30]
    
    
    items1 = ['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','Fence','BsmtCond', 'BsmtExposure',
              'BsmtFinType1', 'BsmtFinType2','BsmtQual','MasVnrType','GarageCond','GarageFinish',
              'GarageQual', 'GarageType']
    fill_na(items1,"None")
    items2 = ['GarageYrBlt','GarageArea', 'GarageCars','BsmtFinSF1', 'BsmtFinSF2',
           'TotalBsmtSF', 'BsmtFinType2', 'BsmtFullBath', 'BsmtHalfBath','MasVnrArea','BsmtUnfSF']
    fill_na(items2,0)
    #用中值填充
    data["LotFrontage"] = data.groupby('Neighborhood')["LotFrontage"].transform(lambda x :x.fillna(x.median()))
    #用众数填充
    #item3 = [ 'MSZoning','Electrical','Exterior1st','Exterior2nd','SaleType','KitchenQual']
    item3 = [ 'KitchenQual']
    fill_na_with_mode(item3)
    
    data['Functional'] = data['Functional'].fillna("Typ")
    #删除仅有一个值的,如何查找出某个特征中含有的值的类型较少？？
    data = data.drop(['Utilities'],axis=1)
    
    
    data_null = data.isnull().sum()/len(data) * 100
    data_null = data_null.drop(data_null[data_null == 0].index).sort_values()

    
    # Create new features
    # 3* Polynomials on the top 10 existing features
    data["OverallQual-s2"] = data["OverallQual"] ** 2
    data["OverallQual-s3"] = data["OverallQual"] ** 3
    data["OverallQual-Sq"] = np.sqrt(data["OverallQual"])
    data["GrLivArea-2"] = data["GrLivArea"] ** 2
    data["GrLivArea-3"] = data["GrLivArea"] ** 3
    data["GrLivArea-Sq"] = np.sqrt(data["GrLivArea"])
    data["ExterQual-2"] = data["ExterQual"] ** 2
    data["ExterQual-3"] = data["ExterQual"] ** 3
    data["ExterQual-Sq"] = np.sqrt(data["ExterQual"])
    data["GarageCars-2"] = data["GarageCars"] ** 2
    data["GarageCars-3"] = data["GarageCars"] ** 3
    data["GarageCars-Sq"] = np.sqrt(data["GarageCars"])
    data["KitchenQual-2"] = data["KitchenQual"] ** 2
    data["KitchenQual-3"] = data["KitchenQual"] ** 3
    data["KitchenQual-Sq"] = np.sqrt(data["KitchenQual"])
    
    
    
    
    def num_transform(items):
        for item in items:
            data[item] = data[item].apply(str)
    
    item4 = ['MSSubClass','OverallCond','YrSold','MoSold']
    num_transform(item4)   
    
    cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
            'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
            'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
            'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
            'YrSold', 'MoSold')
    
    for c in cols:
        lbl = LabelEncoder()
        data[c] =lbl.fit_transform(list(data[c].values))
        
    data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']
    
    numeric_feats = data.dtypes[data.dtypes != 'object'].index
    
    skewed_feats = data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewness = pd.DataFrame({'skew':skewed_feats})
    skewness.head(10)
    
    skewness = skewness[abs(skewness) > 0.75]
    
    from scipy.special import boxcox1p
    skewed_features = skewness.index
    lam = 0.15
    for feat in skewed_features:
        #all_data[feat] += 1
        data[feat] = boxcox1p(data[feat], lam)
    
    data = pd.get_dummies(data)
    
    train = data[:ntrain]
    test = data[ntrain:]
    y_train.shape
    
    
    
    data2 = train
    data2['SalePrice'] = y_train

   
    param_test3 = OrderedDict()
    param_test3['n_estimators']=(15, 200, 2, 10000)
    param_test3['max_depth']=(3, 10, 2, 1000)
    param_test3['max_leaf_nodes']=(2, 1000, -1, -1)
    param_test3['min_samples_split']=(2, 1000, -1, -1)
    param_test3['min_samples_leaf']=(2,1000, -1, 100000)
    param_test3['min_weight_fraction_leaf']=(0, 0.5,0, 0.5)
    #param_test3['max_features']=(0.1, 10, 0.001, 100)
    
    estimator = GradientBoostingClassifier
    init_param ={}
    ##init_param = { 'n_estimators': 15,
    #              'max_depth': 5, 'max_leaf_nodes': 0.9, 'min_samples_split': 0,
    #              'min_samples_leaf': 0.8, 'min_weight_fraction_leaf': 0.8,
    #              'min_weight_fraction_leaf': 'binary:logistic',
    #              'max_features': 4}
    
    #gbdt = auto_tuning(estimator, init_param, param_test3, data2,target='SalePrice')
    mode = GradientBoostingClassifier()
    mode.fit(train,y_train)
    print(data)
    mode.predict(test)



