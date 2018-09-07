"""
Speedml Feature component with methods that work on dataset features or the feature engineering workflow. Contact author https://twitter.com/manavsehgal. Code, docs and demos https://speedml.com.
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

from speedml.base import Base
from speedml.util import DataFrameImputer

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
import re

class Feature(Base):
 
    #def __init__(self):
        #Base.data = Base.train.append(Base.test)
    
    def drop(self, features):
        """
        Drop one or more list of strings naming ``features`` from train and test datasets.
        """
        start = Base.train.shape[1]

        Base.train = Base.train.drop(features, axis=1)
        Base.test = Base.test.drop(features, axis=1)

        end = Base.train.shape[1]
        message = 'Dropped {} features with {} features available.'
        return message.format(start - end, end)

    def impute(self):
        """
        Replace empty values in the entire dataframe with median value for numerical features and most common values for text features.
        """
        start = Base.train.isnull().sum().sum()

        Base.test[Base.target] = -1
        combine = Base.train.append(Base.test)
        combine = DataFrameImputer().fit_transform(combine)
        Base.train = combine[0:Base.train.shape[0]]
        Base.test = combine[Base.train.shape[0]::]
        Base.test = Base.test.drop([Base.target], axis=1)

        end = Base.train.isnull().sum().sum()
        message = 'Imputed {} empty values to {}.'
        return message.format(start, end)

    def mapping(self, a, data):
        """
        Convert values for categorical feature ``a`` using ``data`` dictionary. Use when number of categories are limited otherwise use labels.
        """
        Base.train[a] = Base.train[a].apply(lambda x: data[x])
        Base.test[a] = Base.test[a].apply(lambda x: data[x])

    def fillna(self, a, new):
        """
        Fills empty or null values in ``a`` feature name with ``new`` string value.
        """
        start = Base.train[a].isnull().sum() + Base.test[a].isnull().sum()

        Base.train[a] = Base.train[a].fillna(new)
        Base.test[a] = Base.test[a].fillna(new)

        message = 'Filled {} null values across test and train datasets.'
        return message.format(start)

    def replace(self, a, match, new):
        """
        In feature ``a`` values ``match`` string or list of strings and replace with a ``new`` string.
        """
        if type(match) is str:
            # [TODO] What is the performance cost of message ops?
            start = Base.train[Base.train[a] == match][a].shape[0] + Base.test[Base.test[a] == match][a].shape[0]
            message = 'Replaced {} matching values across train and test datasets.'
            message = message.format(start)
        else:
            # [TODO] Can we possibly use pandas.isin to check counts?
            message = 'Replaced matching list of strings across train and test datasets.'

        Base.train[a] = Base.train[a].replace(match, new)
        Base.test[a] = Base.test[a].replace(match, new)

        return message

    def outliers(self, a, lower = None, upper = None):
        """
        Fix outliers for ``lower`` or ``upper`` or both percentile of values within ``a`` feature.
        """
        if upper:
            upper_value = np.percentile(Base.train[a].values, upper)
            change = Base.train.loc[Base.train[a] > upper_value, a].shape[0]
            Base.train.loc[Base.train[a] > upper_value, a] = upper_value
            message = 'Fixed {} or {:.2f}% upper outliers. '.format(change, change/Base.train.shape[0]*100)

        if lower:
            lower_value = np.percentile(Base.train[a].values, lower)
            change = Base.train.loc[Base.train[a] < lower_value, a].shape[0]
            Base.train.loc[Base.train[a] < lower_value, a] = lower_value
            message = message + 'Fixed {} or {:.2f}% lower outliers.'.format(change, change/Base.train.shape[0]*100)

        return message

    def _density_by_feature(self, a):
        vals = Base.train[a].value_counts()
        dvals = vals.to_dict()
        Base.train[a + '_density'] = Base.train[a].apply(lambda x: dvals.get(x, vals.min()))
        Base.test[a + '_density'] = Base.test[a].apply(lambda x: dvals.get(x, vals.min()))

    def density(self, a):
        """
        Create new feature named ``a`` feature name + suffix '_density', based on density or value_counts for each unique value in ``a`` feature specified as a string or multiple features as a list of strings.
        """
        if isinstance(a, str):
            self._density_by_feature(a)

        if isinstance(a, list):
            for feature in a:
                self._density_by_feature(feature)

    def add(self, a, num):
        """
        Update ``a`` numeric feature by adding ``num`` number to each values.
        """
        Base.train[a] = Base.train[a] + num
        Base.test[a] = Base.test[a] + num

    def sum(self, new, a, b):
        """
        Create ``new`` numeric feature by adding ``a`` + ``b`` feature values.
        """
        Base.train[new] = Base.train[a] + Base.train[b]
        Base.test[new] = Base.test[a] + Base.test[b]

    def diff(self, new, a, b):
        """
        Create ``new`` numeric feature by subtracting ``a`` - ``b`` feature values.
        """
        Base.train[new] = Base.train[a] - Base.train[b]
        Base.test[new] = Base.test[a] - Base.test[b]

    def product(self, new, a, b):
        """
        Create ``new`` numeric feature by multiplying ``a`` * ``b`` feature values.
        """
        Base.train[new] = Base.train[a] * Base.train[b]
        Base.test[new] = Base.test[a] * Base.test[b]

    def divide(self, new, a, b):
        """
        Create ``new`` numeric feature by dividing ``a`` / ``b`` feature values. Replace division-by-zero with zero values.
        """
        Base.train[new] = Base.train[a] / Base.train[b]
        Base.test[new] = Base.test[a] / Base.test[b]
        # Histograms require finite values
        Base.train[new] = Base.train[new].replace([np.inf, -np.inf], 0)
        Base.test[new] = Base.test[new].replace([np.inf, -np.inf], 0)

    def round(self, new, a, precision):
        """
        Create ``new`` numeric feature by rounding ``a`` feature value to ``precision`` decimal places.
        """
        Base.train[new] = round(Base.train[a], precision)
        Base.test[new] = round(Base.test[a], precision)

    def concat(self, new, a, sep, b):
        """
        Create ``new`` text feature by concatenating ``a`` and ``b`` text feature values, using ``sep`` separator.
        """
        Base.train[new] = Base.train[a].astype(str) + sep + Base.train[b].astype(str)
        Base.test[new] = Base.test[a].astype(str) + sep + Base.test[b].astype(str)

    def list_len(self, new, a):
        """
        Create ``new`` numeric feature based on length or item count from ``a`` feature containing list object as values.
        """
        Base.train[new] = Base.train[a].apply(len)
        Base.test[new] = Base.test[a].apply(len)

    def word_count(self, new, a):
        """
        Create ``new`` numeric feature based on length or word count from ``a`` feature containing free-form text.
        """
        Base.train[new] = Base.train[a].apply(lambda x: len(x.split(" ")))
        Base.test[new] = Base.test[a].apply(lambda x: len(x.split(" ")))

    def _regex_text(self, regex, text):
        regex_search = re.search(regex, text)
        # If the word exists, extract and return it.
        if regex_search:
            return regex_search.group(1)
        return ""

    def extract(self, a, regex, new=None):
        """
        Match ``regex`` regular expression with ``a`` text feature values to update ``a`` feature with matching text if ``new`` = None. Otherwise create ``new`` feature based on matching text.
        """
        Base.train[new if new else a] = Base.train[a].apply(lambda x: self._regex_text(regex=regex, text=x))
        Base.test[new if new else a] = Base.test[a].apply(lambda x: self._regex_text(regex=regex, text=x))

    def labels(self, features):
        """
        Generate numerical labels replacing text values from list of categorical ``features``.
        """
        Base.test[Base.target] = -1
        combine = Base.train.append(Base.test)

        le = LabelEncoder()
        for feature in features:
            combine[feature] = le.fit_transform(combine[feature])

        Base.train = combine[0:Base.train.shape[0]]
        Base.test = combine[Base.train.shape[0]::]
        Base.test = Base.test.drop([Base.target], axis=1)

    def cut(self,refFeature, newFeature, bins, right=True, labels=None, retbins=False, precision=3,
        include_lowest=False):
        """
        Return indices of half-open bins to which each value of `x` belongs.

        Parameters
        ----------
        refFeature: the feature which will be cutted
        newFeature: new name of the feature which is the result of pd.cut
        bins : int, sequence of scalars, or IntervalIndex
            If `bins` is an int, it defines the number of equal-width bins in the
            range of `x`. However, in this case, the range of `x` is extended
            by .1% on each side to include the min or max values of `x`. If
            `bins` is a sequence it defines the bin edges allowing for
            non-uniform bin width. No extension of the range of `x` is done in
            this case.
        right : bool, optional
            Indicates whether the bins include the rightmost edge or not. If
            right == True (the default), then the bins [1,2,3,4] indicate
            (1,2], (2,3], (3,4].
        labels : array or boolean, default None
            Used as labels for the resulting bins. Must be of the same length as
            the resulting bins. If False, return only integer indicators of the
            bins.
        retbins : bool, optional
            Whether to return the bins or not. Can be useful if bins is given
            as a scalar.
        precision : int, optional
            The precision at which to store and display the bins labels
        include_lowest : bool, optional
            Whether the first interval should be left-inclusive or not.

        Returns
        -------
        out : Categorical or Series or array of integers if labels is False
            The return type (Categorical or Series) depends on the input: a Series
            of type category if input is a Series else Categorical. Bins are
            represented as categories when categorical data is returned.
        bins : ndarray of floats
            Returned only if `retbins` is True.

        Notes
        -----
        The `cut` function can be useful for going from a continuous variable to
        a categorical variable. For example, `cut` could convert ages to groups
        of age ranges.

        Any NA values will be NA in the result.  Out of bounds values will be NA in
        the resulting Categorical object


        Examples
        --------
        >>> pd.cut(np.array([.2, 1.4, 2.5, 6.2, 9.7, 2.1]), 3, retbins=True)
        ... # doctest: +ELLIPSIS
        ([(0.19, 3.367], (0.19, 3.367], (0.19, 3.367], (3.367, 6.533], ...
        Categories (3, interval[float64]): [(0.19, 3.367] < (3.367, 6.533] ...

        >>> pd.cut(np.array([.2, 1.4, 2.5, 6.2, 9.7, 2.1]),
        ...        3, labels=["good", "medium", "bad"])
        ... # doctest: +SKIP
        [good, good, good, medium, bad, good]
        Categories (3, object): [good < medium < bad]

        >>> pd.cut(np.ones(5), 4, labels=False)
        array([1, 1, 1, 1, 1])
        """
        Base.test[Base.target] = -1
        combine = Base.train.append(Base.test)

        combine[newFeature] = pd.cut(combine[refFeature], bins, right, labels, retbins, precision,include_lowest)
        
        Base.train = combine[0:Base.train.shape[0]]
        Base.test = combine[Base.train.shape[0]::]
        Base.test = Base.test.drop([Base.target], axis=1)

    def mapFunction(self,fun,feature1,feature2=None):
        if feature2 == None:
            feature2 = feature1
        Base.train[feature2] = Base.train[feature1].map(fun)
        Base.test[feature2] = Base.test[feature1].map(fun)

    def get_dummies(self, prefix=None, prefix_sep='_', dummy_na=False,
                columns=None, sparse=False, drop_first=False):
        Base.test[Base.target] = -1
        combine = Base.train.append(Base.test)
        combine = pd.get_dummies(combine, prefix=prefix, prefix_sep=prefix_sep, dummy_na=dummy_na,
                columns=columns, sparse=sparse, drop_first=drop_first)

        Base.train = combine[0:Base.train.shape[0]]
        Base.test = combine[Base.train.shape[0]::]
        Base.test = Base.test.drop([Base.target], axis=1)


    
