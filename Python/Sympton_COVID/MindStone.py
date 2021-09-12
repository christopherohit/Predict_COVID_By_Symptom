import pandas as pd
import numpy as np
import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler , Normalizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import norm
from scipy import stats
from sklearn import metrics
import warnings


warnings.filterwarnings('ignore')

df_ame = pd.read_csv('Cleaned-Data.csv')
pd.pandas.set_option('display.max_columns', None)
display("Peeking into data", df_ame)