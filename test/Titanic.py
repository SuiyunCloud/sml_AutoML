
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

from speedml import Speedml

sml = Speedml("C:\\Users\\suiyun.yang\\Desktop\\Kaggle\\datasets\\Kaggle\\Titanic\\train.csv","C:\\Users\\suiyun.yang\\Desktop\\Kaggle\\datasets\\Kaggle\\Titanic\\test.csv",target='Survived',uid='PassengerId')
bins = [20,30,40,50,60]
sml.feature.cut('Age',"AgeBin",bins)
print(sml.columns)

