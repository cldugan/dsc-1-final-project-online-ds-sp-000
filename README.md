## Final Project Submission

Please fill out:
* Student name: Crystal Dugan
* Student pace: self-paced 
* Scheduled project review date/time: date/time: 07-09-19 6:15pm EST
* Instructor name: Eli Thomas
* Blog post URL: https://cldugan.github.io/feature_selection_-_module_1_project


#### My Approach  
My goal was to model the King County Housing dataset with a multivariate linear regression in order to predict the sale price of houses as accurately as possible. First I put the data in a pandas dataframe and looked over the summary. A few columns that were obviously not useful for my model were deleted. Then I cleaned the data and removed independent variables that had too much missing data. I looked at correlation for feature collinearity. I also looked at histograms and scatterplots to get a better understanding of my data. I separated out features that were categorical to look at later while I explored the continuous independent variables. Next I scaled and normalized (where necessary) the data. I ran an OLS model in StatsModel and looked at performance and quality. I tried a few iterations to see if I could improve my model. Then I explored the categorical variables, one-hot encoded the one I thought would be appropriate and added it to my model. I ran a final OLS model and checked the performance with k-folds cross validation and test-train-split.

# Import Libraries


```python
# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE
from sklearn import preprocessing
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
import scipy.stats as stats
plt.style.use('bmh')
%matplotlib inline
```

# Obtain Data


```python
kc_df = pd.read_csv("kc_house_data.csv")
```


```python
kc_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>...</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>7129300520</td>
      <td>10/13/2014</td>
      <td>221900.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>...</td>
      <td>7</td>
      <td>1180</td>
      <td>0.0</td>
      <td>1955</td>
      <td>0.0</td>
      <td>98178</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>1340</td>
      <td>5650</td>
    </tr>
    <tr>
      <td>1</td>
      <td>6414100192</td>
      <td>12/9/2014</td>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>7</td>
      <td>2170</td>
      <td>400.0</td>
      <td>1951</td>
      <td>1991.0</td>
      <td>98125</td>
      <td>47.7210</td>
      <td>-122.319</td>
      <td>1690</td>
      <td>7639</td>
    </tr>
    <tr>
      <td>2</td>
      <td>5631500400</td>
      <td>2/25/2015</td>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>10000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>6</td>
      <td>770</td>
      <td>0.0</td>
      <td>1933</td>
      <td>NaN</td>
      <td>98028</td>
      <td>47.7379</td>
      <td>-122.233</td>
      <td>2720</td>
      <td>8062</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2487200875</td>
      <td>12/9/2014</td>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>7</td>
      <td>1050</td>
      <td>910.0</td>
      <td>1965</td>
      <td>0.0</td>
      <td>98136</td>
      <td>47.5208</td>
      <td>-122.393</td>
      <td>1360</td>
      <td>5000</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1954400510</td>
      <td>2/18/2015</td>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>8</td>
      <td>1680</td>
      <td>0.0</td>
      <td>1987</td>
      <td>0.0</td>
      <td>98074</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>1800</td>
      <td>7503</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



# Scrub / Explore Data

I have identified columns to delete before further exploration of the data.  
id- will have no bearing on house value  
date- not useful for linear regression


```python
kc_df = kc_df.drop(['id', 'date'], axis=1)
kc_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>221900.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1180</td>
      <td>0.0</td>
      <td>1955</td>
      <td>0.0</td>
      <td>98178</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>1340</td>
      <td>5650</td>
    </tr>
    <tr>
      <td>1</td>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>2170</td>
      <td>400.0</td>
      <td>1951</td>
      <td>1991.0</td>
      <td>98125</td>
      <td>47.7210</td>
      <td>-122.319</td>
      <td>1690</td>
      <td>7639</td>
    </tr>
    <tr>
      <td>2</td>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>10000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>6</td>
      <td>770</td>
      <td>0.0</td>
      <td>1933</td>
      <td>NaN</td>
      <td>98028</td>
      <td>47.7379</td>
      <td>-122.233</td>
      <td>2720</td>
      <td>8062</td>
    </tr>
    <tr>
      <td>3</td>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>1050</td>
      <td>910.0</td>
      <td>1965</td>
      <td>0.0</td>
      <td>98136</td>
      <td>47.5208</td>
      <td>-122.393</td>
      <td>1360</td>
      <td>5000</td>
    </tr>
    <tr>
      <td>4</td>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1680</td>
      <td>0.0</td>
      <td>1987</td>
      <td>0.0</td>
      <td>98074</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>1800</td>
      <td>7503</td>
    </tr>
  </tbody>
</table>
</div>




```python
kc_df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>2.159700e+04</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>2.159700e+04</td>
      <td>21597.000000</td>
      <td>19221.000000</td>
      <td>21534.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>17755.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
      <td>21597.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>5.402966e+05</td>
      <td>3.373200</td>
      <td>2.115826</td>
      <td>2080.321850</td>
      <td>1.509941e+04</td>
      <td>1.494096</td>
      <td>0.007596</td>
      <td>0.233863</td>
      <td>3.409825</td>
      <td>7.657915</td>
      <td>1788.596842</td>
      <td>1970.999676</td>
      <td>83.636778</td>
      <td>98077.951845</td>
      <td>47.560093</td>
      <td>-122.213982</td>
      <td>1986.620318</td>
      <td>12758.283512</td>
    </tr>
    <tr>
      <td>std</td>
      <td>3.673681e+05</td>
      <td>0.926299</td>
      <td>0.768984</td>
      <td>918.106125</td>
      <td>4.141264e+04</td>
      <td>0.539683</td>
      <td>0.086825</td>
      <td>0.765686</td>
      <td>0.650546</td>
      <td>1.173200</td>
      <td>827.759761</td>
      <td>29.375234</td>
      <td>399.946414</td>
      <td>53.513072</td>
      <td>0.138552</td>
      <td>0.140724</td>
      <td>685.230472</td>
      <td>27274.441950</td>
    </tr>
    <tr>
      <td>min</td>
      <td>7.800000e+04</td>
      <td>1.000000</td>
      <td>0.500000</td>
      <td>370.000000</td>
      <td>5.200000e+02</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>370.000000</td>
      <td>1900.000000</td>
      <td>0.000000</td>
      <td>98001.000000</td>
      <td>47.155900</td>
      <td>-122.519000</td>
      <td>399.000000</td>
      <td>651.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>3.220000e+05</td>
      <td>3.000000</td>
      <td>1.750000</td>
      <td>1430.000000</td>
      <td>5.040000e+03</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>1190.000000</td>
      <td>1951.000000</td>
      <td>0.000000</td>
      <td>98033.000000</td>
      <td>47.471100</td>
      <td>-122.328000</td>
      <td>1490.000000</td>
      <td>5100.000000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>4.500000e+05</td>
      <td>3.000000</td>
      <td>2.250000</td>
      <td>1910.000000</td>
      <td>7.618000e+03</td>
      <td>1.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>1560.000000</td>
      <td>1975.000000</td>
      <td>0.000000</td>
      <td>98065.000000</td>
      <td>47.571800</td>
      <td>-122.231000</td>
      <td>1840.000000</td>
      <td>7620.000000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>6.450000e+05</td>
      <td>4.000000</td>
      <td>2.500000</td>
      <td>2550.000000</td>
      <td>1.068500e+04</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>8.000000</td>
      <td>2210.000000</td>
      <td>1997.000000</td>
      <td>0.000000</td>
      <td>98118.000000</td>
      <td>47.678000</td>
      <td>-122.125000</td>
      <td>2360.000000</td>
      <td>10083.000000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>7.700000e+06</td>
      <td>33.000000</td>
      <td>8.000000</td>
      <td>13540.000000</td>
      <td>1.651359e+06</td>
      <td>3.500000</td>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>13.000000</td>
      <td>9410.000000</td>
      <td>2015.000000</td>
      <td>2015.000000</td>
      <td>98199.000000</td>
      <td>47.777600</td>
      <td>-121.315000</td>
      <td>6210.000000</td>
      <td>871200.000000</td>
    </tr>
  </tbody>
</table>
</div>



observations: 
>price- This is the dependent variable. House prices run 78,000 to 7,700,000. High max value and mean larger than median, data probably skewed.  
bedrooms- 33 max value, possible outlier. closer look needed.  
sqft_living- possible skew or outliers on the high end.  
waterfront- categorical.  
veiw- not sure this is useful.  
yr_built and yr_renovated are dates, change to ages or min-max scaling will deal with issues?  
zip code- will need to one-hot encode in order to use.  
lat and long- interesting, but are they useful in a linear regression?  




```python
kc_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 21597 entries, 0 to 21596
    Data columns (total 19 columns):
    price            21597 non-null float64
    bedrooms         21597 non-null int64
    bathrooms        21597 non-null float64
    sqft_living      21597 non-null int64
    sqft_lot         21597 non-null int64
    floors           21597 non-null float64
    waterfront       19221 non-null float64
    view             21534 non-null float64
    condition        21597 non-null int64
    grade            21597 non-null int64
    sqft_above       21597 non-null int64
    sqft_basement    21597 non-null object
    yr_built         21597 non-null int64
    yr_renovated     17755 non-null float64
    zipcode          21597 non-null int64
    lat              21597 non-null float64
    long             21597 non-null float64
    sqft_living15    21597 non-null int64
    sqft_lot15       21597 non-null int64
    dtypes: float64(8), int64(10), object(1)
    memory usage: 3.1+ MB


Observations- 
>waterfront, yr_renovated, and view have some missing data.  
sqft_basement type indicates string which is weird and needs further examination.   






```python
kc_df.view.unique()
```




    array([ 0., nan,  3.,  4.,  2.,  1.])




```python
# deleting view because I don't think it has much influence on price, and it is mostly 0 or nan.
kc_df = kc_df.drop(['view'], axis=1)
```


```python
# changing nan to 0 in waterfront because I think it is reasonable 
# to assume that if the house has water views it would be noted.

kc_df.waterfront.fillna(0, inplace=True)
```


```python
kc_df.waterfront.unique()
```




    array([0., 1.])




```python
# examining sqft_basement because data type seems fishy. 
kc_df['sqft_basement'].unique()
```




    array(['0.0', '400.0', '910.0', '1530.0', '?', '730.0', '1700.0', '300.0',
           '970.0', '760.0', '720.0', '700.0', '820.0', '780.0', '790.0',
           '330.0', '1620.0', '360.0', '588.0', '1510.0', '410.0', '990.0',
           '600.0', '560.0', '550.0', '1000.0', '1600.0', '500.0', '1040.0',
           '880.0', '1010.0', '240.0', '265.0', '290.0', '800.0', '540.0',
           '710.0', '840.0', '380.0', '770.0', '480.0', '570.0', '1490.0',
           '620.0', '1250.0', '1270.0', '120.0', '650.0', '180.0', '1130.0',
           '450.0', '1640.0', '1460.0', '1020.0', '1030.0', '750.0', '640.0',
           '1070.0', '490.0', '1310.0', '630.0', '2000.0', '390.0', '430.0',
           '850.0', '210.0', '1430.0', '1950.0', '440.0', '220.0', '1160.0',
           '860.0', '580.0', '2060.0', '1820.0', '1180.0', '200.0', '1150.0',
           '1200.0', '680.0', '530.0', '1450.0', '1170.0', '1080.0', '960.0',
           '280.0', '870.0', '1100.0', '460.0', '1400.0', '660.0', '1220.0',
           '900.0', '420.0', '1580.0', '1380.0', '475.0', '690.0', '270.0',
           '350.0', '935.0', '1370.0', '980.0', '1470.0', '160.0', '950.0',
           '50.0', '740.0', '1780.0', '1900.0', '340.0', '470.0', '370.0',
           '140.0', '1760.0', '130.0', '520.0', '890.0', '1110.0', '150.0',
           '1720.0', '810.0', '190.0', '1290.0', '670.0', '1800.0', '1120.0',
           '1810.0', '60.0', '1050.0', '940.0', '310.0', '930.0', '1390.0',
           '610.0', '1830.0', '1300.0', '510.0', '1330.0', '1590.0', '920.0',
           '1320.0', '1420.0', '1240.0', '1960.0', '1560.0', '2020.0',
           '1190.0', '2110.0', '1280.0', '250.0', '2390.0', '1230.0', '170.0',
           '830.0', '1260.0', '1410.0', '1340.0', '590.0', '1500.0', '1140.0',
           '260.0', '100.0', '320.0', '1480.0', '1060.0', '1284.0', '1670.0',
           '1350.0', '2570.0', '1090.0', '110.0', '2500.0', '90.0', '1940.0',
           '1550.0', '2350.0', '2490.0', '1481.0', '1360.0', '1135.0',
           '1520.0', '1850.0', '1660.0', '2130.0', '2600.0', '1690.0',
           '243.0', '1210.0', '1024.0', '1798.0', '1610.0', '1440.0',
           '1570.0', '1650.0', '704.0', '1910.0', '1630.0', '2360.0',
           '1852.0', '2090.0', '2400.0', '1790.0', '2150.0', '230.0', '70.0',
           '1680.0', '2100.0', '3000.0', '1870.0', '1710.0', '2030.0',
           '875.0', '1540.0', '2850.0', '2170.0', '506.0', '906.0', '145.0',
           '2040.0', '784.0', '1750.0', '374.0', '518.0', '2720.0', '2730.0',
           '1840.0', '3480.0', '2160.0', '1920.0', '2330.0', '1860.0',
           '2050.0', '4820.0', '1913.0', '80.0', '2010.0', '3260.0', '2200.0',
           '415.0', '1730.0', '652.0', '2196.0', '1930.0', '515.0', '40.0',
           '2080.0', '2580.0', '1548.0', '1740.0', '235.0', '861.0', '1890.0',
           '2220.0', '792.0', '2070.0', '4130.0', '2250.0', '2240.0',
           '1990.0', '768.0', '2550.0', '435.0', '1008.0', '2300.0', '2610.0',
           '666.0', '3500.0', '172.0', '1816.0', '2190.0', '1245.0', '1525.0',
           '1880.0', '862.0', '946.0', '1281.0', '414.0', '2180.0', '276.0',
           '1248.0', '602.0', '516.0', '176.0', '225.0', '1275.0', '266.0',
           '283.0', '65.0', '2310.0', '10.0', '1770.0', '2120.0', '295.0',
           '207.0', '915.0', '556.0', '417.0', '143.0', '508.0', '2810.0',
           '20.0', '274.0', '248.0'], dtype=object)




```python
# Aha, there is "?" hiding in the data as placeholder.
(kc_df['sqft_basement']== '?').sum()
```




    454



454 mising data points. I can either drop basement or drop the rows with '?'. I am going with dropping the column.
Dropping rows would lose ~2% of the data.  
Also I suspect it is colinear with other size features.


```python
kc_df = kc_df.drop(['sqft_basement'], axis=1)
```


```python
kc_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>221900.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1180</td>
      <td>1955</td>
      <td>0.0</td>
      <td>98178</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>1340</td>
      <td>5650</td>
    </tr>
    <tr>
      <td>1</td>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>2170</td>
      <td>1951</td>
      <td>1991.0</td>
      <td>98125</td>
      <td>47.7210</td>
      <td>-122.319</td>
      <td>1690</td>
      <td>7639</td>
    </tr>
    <tr>
      <td>2</td>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>10000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>6</td>
      <td>770</td>
      <td>1933</td>
      <td>NaN</td>
      <td>98028</td>
      <td>47.7379</td>
      <td>-122.233</td>
      <td>2720</td>
      <td>8062</td>
    </tr>
    <tr>
      <td>3</td>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>1050</td>
      <td>1965</td>
      <td>0.0</td>
      <td>98136</td>
      <td>47.5208</td>
      <td>-122.393</td>
      <td>1360</td>
      <td>5000</td>
    </tr>
    <tr>
      <td>4</td>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1680</td>
      <td>1987</td>
      <td>0.0</td>
      <td>98074</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>1800</td>
      <td>7503</td>
    </tr>
  </tbody>
</table>
</div>




```python
kc_df.isna().sum()
```




    price               0
    bedrooms            0
    bathrooms           0
    sqft_living         0
    sqft_lot            0
    floors              0
    waterfront          0
    condition           0
    grade               0
    sqft_above          0
    yr_built            0
    yr_renovated     3842
    zipcode             0
    lat                 0
    long                0
    sqft_living15       0
    sqft_lot15          0
    dtype: int64




```python
#Dropping yr_renovated, too much missing data.

kc_df = kc_df.drop('yr_renovated', axis = 1)
```


```python
# lets see what these features look like.
kc_df.hist(figsize = (12,12));
```


![png](student_files/student_25_0.png)


Observations - Some data looks skewed, confirming earlier observations. Some data looks categorical,
once data is cleaned I will look at linear relationships to see if I should treat as categorical or continuous


```python
#dropping the row that has the 33 bedroom outier

kc_df = kc_df[kc_df.bedrooms != 33]
kc_df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>yr_built</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>2.159600e+04</td>
      <td>21596.000000</td>
      <td>21596.000000</td>
      <td>21596.000000</td>
      <td>2.159600e+04</td>
      <td>21596.000000</td>
      <td>21596.000000</td>
      <td>21596.000000</td>
      <td>21596.000000</td>
      <td>21596.000000</td>
      <td>21596.000000</td>
      <td>21596.000000</td>
      <td>21596.000000</td>
      <td>21596.000000</td>
      <td>21596.000000</td>
      <td>21596.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>5.402920e+05</td>
      <td>3.371828</td>
      <td>2.115843</td>
      <td>2080.343165</td>
      <td>1.509983e+04</td>
      <td>1.494119</td>
      <td>0.006761</td>
      <td>3.409752</td>
      <td>7.657946</td>
      <td>1788.631506</td>
      <td>1971.000787</td>
      <td>98077.950685</td>
      <td>47.560087</td>
      <td>-122.213977</td>
      <td>1986.650722</td>
      <td>12758.656649</td>
    </tr>
    <tr>
      <td>std</td>
      <td>3.673760e+05</td>
      <td>0.904114</td>
      <td>0.768998</td>
      <td>918.122038</td>
      <td>4.141355e+04</td>
      <td>0.539685</td>
      <td>0.081946</td>
      <td>0.650471</td>
      <td>1.173218</td>
      <td>827.763251</td>
      <td>29.375460</td>
      <td>53.514040</td>
      <td>0.138552</td>
      <td>0.140725</td>
      <td>685.231768</td>
      <td>27275.018316</td>
    </tr>
    <tr>
      <td>min</td>
      <td>7.800000e+04</td>
      <td>1.000000</td>
      <td>0.500000</td>
      <td>370.000000</td>
      <td>5.200000e+02</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>370.000000</td>
      <td>1900.000000</td>
      <td>98001.000000</td>
      <td>47.155900</td>
      <td>-122.519000</td>
      <td>399.000000</td>
      <td>651.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>3.220000e+05</td>
      <td>3.000000</td>
      <td>1.750000</td>
      <td>1430.000000</td>
      <td>5.040000e+03</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>1190.000000</td>
      <td>1951.000000</td>
      <td>98033.000000</td>
      <td>47.471100</td>
      <td>-122.328000</td>
      <td>1490.000000</td>
      <td>5100.000000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>4.500000e+05</td>
      <td>3.000000</td>
      <td>2.250000</td>
      <td>1910.000000</td>
      <td>7.619000e+03</td>
      <td>1.500000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>1560.000000</td>
      <td>1975.000000</td>
      <td>98065.000000</td>
      <td>47.571800</td>
      <td>-122.231000</td>
      <td>1840.000000</td>
      <td>7620.000000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>6.450000e+05</td>
      <td>4.000000</td>
      <td>2.500000</td>
      <td>2550.000000</td>
      <td>1.068550e+04</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>8.000000</td>
      <td>2210.000000</td>
      <td>1997.000000</td>
      <td>98118.000000</td>
      <td>47.678000</td>
      <td>-122.125000</td>
      <td>2360.000000</td>
      <td>10083.000000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>7.700000e+06</td>
      <td>11.000000</td>
      <td>8.000000</td>
      <td>13540.000000</td>
      <td>1.651359e+06</td>
      <td>3.500000</td>
      <td>1.000000</td>
      <td>5.000000</td>
      <td>13.000000</td>
      <td>9410.000000</td>
      <td>2015.000000</td>
      <td>98199.000000</td>
      <td>47.777600</td>
      <td>-121.315000</td>
      <td>6210.000000</td>
      <td>871200.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# lets take a closer look at continuous variables
column_list = ['price', 'bathrooms', 'bedrooms', 'condition', 'floors', 'grade', 'sqft_above', 'sqft_living', 'sqft_living15', 'sqft_lot', 'sqft_lot15','yr_built']
for col in column_list:
    sns.distplot(kc_df[col])
    plt.title(col)
    plt.show();
```


![png](student_files/student_28_0.png)



![png](student_files/student_28_1.png)



![png](student_files/student_28_2.png)



![png](student_files/student_28_3.png)



![png](student_files/student_28_4.png)



![png](student_files/student_28_5.png)



![png](student_files/student_28_6.png)



![png](student_files/student_28_7.png)



![png](student_files/student_28_8.png)



![png](student_files/student_28_9.png)



![png](student_files/student_28_10.png)



![png](student_files/student_28_11.png)


I think some of the variables are too skewed to use as-is. Some also have a lot of "peakedness". Some look 
categorical. I will try log-transformations and look at scatterplots to check for linear relationships with the target.

Some features are discrete and not continuous varibles. I am trying to decide whether to handle them as continuous or categorical varibles. 


```python
column_list = ['bathrooms', 'bedrooms', 'condition', 'floors', 'grade']
for col in column_list:
    f, ax = plt.subplots(figsize=(12,6))
    sns.violinplot(x = kc_df[col], y = kc_df['price'])
    plt.title(col)
    plt.show();
```


![png](student_files/student_31_0.png)



![png](student_files/student_31_1.png)



![png](student_files/student_31_2.png)



![png](student_files/student_31_3.png)



![png](student_files/student_31_4.png)


'grade' and 'bathrooms' seem to show the strongest positive correlations with price. I am suprised the other features don't show a stronger relationship. But nothing is screaming categorical to me at this stage so I will keep them in as continuous variables for now.


```python
# Making a DF of cleaned features that I will normalize, scale, and then model.
# Dropping zipcode, lat, and long for now, these will have to be one-hot encoded if used.  
# Will explore later on and add back to model.

data_pred= kc_df.drop([ 'zipcode', 'lat', 'long'], axis =1)
data_pred.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>yr_built</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>221900.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1180</td>
      <td>1955</td>
      <td>1340</td>
      <td>5650</td>
    </tr>
    <tr>
      <td>1</td>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>2170</td>
      <td>1951</td>
      <td>1690</td>
      <td>7639</td>
    </tr>
    <tr>
      <td>2</td>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>10000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>6</td>
      <td>770</td>
      <td>1933</td>
      <td>2720</td>
      <td>8062</td>
    </tr>
    <tr>
      <td>3</td>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>1050</td>
      <td>1965</td>
      <td>1360</td>
      <td>5000</td>
    </tr>
    <tr>
      <td>4</td>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1680</td>
      <td>1987</td>
      <td>1800</td>
      <td>7503</td>
    </tr>
  </tbody>
</table>
</div>



### Correlation and Collinearity

### How well are the variables correlated with the target and is there any collinearity to be worried about?


```python
# plotting heatmap of variables for correlation and collinearity
plt.figure(figsize=(12,10))
corr = abs(data_pred.corr())
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap');
```


![png](student_files/student_36_0.png)


Target variable is 'price'. The features that are most highly correlated with 'price' are:  
'sqft_living', 'grade', and 'sqft_above'. The least correlated:  'condition', 'yr_built', and 'sqft_lot15'.

Dropping 'sqft_above' because it is highly colinear with 'sqft_living', and will bias my model.  
It also seems like they are measuring almost the same thing.  
I am changing 'waterfront's type to category.



```python
data_pred = data_pred.drop(['sqft_above'], axis=1)
```


```python
data_pred['waterfront'] = data_pred.waterfront.astype('category')
```

### Scaling and Normalization


```python
# Looking at how skewed the data is.
data_pred.skew()
```




    price             4.023329
    bedrooms          0.551382
    bathrooms         0.519644
    sqft_living       1.473143
    sqft_lot         13.072315
    floors            0.614427
    waterfront       12.039300
    condition         1.036107
    grade             0.788166
    yr_built         -0.469549
    sqft_living15     1.106828
    sqft_lot15        9.524159
    dtype: float64



I am log-transforming the obviously skewed data. (-1 < Skew < 1)  
(Except 'waterfront' because it is now categorical.)


```python
# updating my exploration df with transformed variables
data_pred["sqft_living"] = np.log(data_pred["sqft_living"])
data_pred["sqft_lot"] = np.log(data_pred["sqft_lot"])
data_pred["sqft_lot15"] = np.log(data_pred["sqft_lot15"])
data_pred["price"] = np.log(data_pred["price"])
data_pred.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>condition</th>
      <th>grade</th>
      <th>yr_built</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>12.309982</td>
      <td>3</td>
      <td>1.00</td>
      <td>7.073270</td>
      <td>8.639411</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1955</td>
      <td>1340</td>
      <td>8.639411</td>
    </tr>
    <tr>
      <td>1</td>
      <td>13.195614</td>
      <td>3</td>
      <td>2.25</td>
      <td>7.851661</td>
      <td>8.887653</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1951</td>
      <td>1690</td>
      <td>8.941022</td>
    </tr>
    <tr>
      <td>2</td>
      <td>12.100712</td>
      <td>2</td>
      <td>1.00</td>
      <td>6.646391</td>
      <td>9.210340</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>6</td>
      <td>1933</td>
      <td>2720</td>
      <td>8.994917</td>
    </tr>
    <tr>
      <td>3</td>
      <td>13.311329</td>
      <td>4</td>
      <td>3.00</td>
      <td>7.580700</td>
      <td>8.517193</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>1965</td>
      <td>1360</td>
      <td>8.517193</td>
    </tr>
    <tr>
      <td>4</td>
      <td>13.142166</td>
      <td>3</td>
      <td>2.00</td>
      <td>7.426549</td>
      <td>8.997147</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1987</td>
      <td>1800</td>
      <td>8.923058</td>
    </tr>
  </tbody>
</table>
</div>



### Did the log-transfomations improve distributions?


```python
# plots of transformed variables to see if improved by log-transformation.
column_list = ['price', 'sqft_living', 'sqft_living15', 'sqft_lot', 'sqft_lot15']
for col in column_list:
    sns.distplot(data_pred[col])
    plt.title(col)
    plt.show();
```


![png](student_files/student_45_0.png)



![png](student_files/student_45_1.png)



![png](student_files/student_45_2.png)



![png](student_files/student_45_3.png)



![png](student_files/student_45_4.png)


These features now have a more normal distibution. A bit "peaky" but I think good enough to move on to checking visually for linear relationships with the target.

### Do the features have a linear relationships with the target?


```python
# With jointplots we can look at the transformed histograms and KDE as well as linear relationships with target.
for column in data_pred.drop('price', axis=1):
    sns.jointplot(x=column, y='price',
                  data=data_pred, 
                  kind='reg',
                  space=0.0,
                  label=column,
                  joint_kws={'line_kws':{'color':'green'}})
    #plt.title("Price vs " + column)
    plt.legend()
    plt.show()
```


![png](student_files/student_48_0.png)



![png](student_files/student_48_1.png)



![png](student_files/student_48_2.png)



![png](student_files/student_48_3.png)



![png](student_files/student_48_4.png)



![png](student_files/student_48_5.png)



![png](student_files/student_48_6.png)



![png](student_files/student_48_7.png)



![png](student_files/student_48_8.png)



![png](student_files/student_48_9.png)



![png](student_files/student_48_10.png)


Looking at the histograms with density estimates, we can see that data that was log-transformed now have a less skewed distribution. With the scatterplots with the best-fit line drawn, I am satisfied that the independent variables have a good enough linear relationship with the target as to continue with modeling. The best linear relationships are with sqft_living and grade.  I am going to treat the discrete data variables as continuous data (except for waterfront which is categorical) as they appear to have a linear relationship with the target.

### MinMax scaling the data


```python
# scaling the data. I am chosing to min-max scale the data so everything will be on the same scale of 0-1.  


data_minMax = data_pred.drop(['waterfront', 'price'], axis=1)

for column in data_minMax:
    data_minMax[column] = (data_minMax[column]-min(data_minMax[column]))/(max(data_minMax[column])-min(data_minMax[column]))

data_df = pd.concat([data_minMax, data_pred[['price','waterfront']]], axis=1)    
data_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>condition</th>
      <th>grade</th>
      <th>yr_built</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
      <th>price</th>
      <th>waterfront</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.2</td>
      <td>0.066667</td>
      <td>0.322166</td>
      <td>0.295858</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.4</td>
      <td>0.478261</td>
      <td>0.161934</td>
      <td>0.300162</td>
      <td>12.309982</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.2</td>
      <td>0.233333</td>
      <td>0.538392</td>
      <td>0.326644</td>
      <td>0.4</td>
      <td>0.5</td>
      <td>0.4</td>
      <td>0.443478</td>
      <td>0.222165</td>
      <td>0.342058</td>
      <td>13.195614</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.1</td>
      <td>0.066667</td>
      <td>0.203585</td>
      <td>0.366664</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.3</td>
      <td>0.286957</td>
      <td>0.399415</td>
      <td>0.349544</td>
      <td>12.100712</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.3</td>
      <td>0.333333</td>
      <td>0.463123</td>
      <td>0.280700</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.4</td>
      <td>0.565217</td>
      <td>0.165376</td>
      <td>0.283185</td>
      <td>13.311329</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.2</td>
      <td>0.200000</td>
      <td>0.420302</td>
      <td>0.340224</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.756522</td>
      <td>0.241094</td>
      <td>0.339562</td>
      <td>13.142166</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# sanity check

data_df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>condition</th>
      <th>grade</th>
      <th>yr_built</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>21596.000000</td>
      <td>21596.000000</td>
      <td>21596.000000</td>
      <td>21596.000000</td>
      <td>21596.000000</td>
      <td>21596.000000</td>
      <td>21596.000000</td>
      <td>21596.000000</td>
      <td>21596.000000</td>
      <td>21596.000000</td>
      <td>21596.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>0.237183</td>
      <td>0.215446</td>
      <td>0.454797</td>
      <td>0.339315</td>
      <td>0.197648</td>
      <td>0.602438</td>
      <td>0.465795</td>
      <td>0.617398</td>
      <td>0.273215</td>
      <td>0.344802</td>
      <td>13.048196</td>
    </tr>
    <tr>
      <td>std</td>
      <td>0.090411</td>
      <td>0.102533</td>
      <td>0.117836</td>
      <td>0.111877</td>
      <td>0.215874</td>
      <td>0.162618</td>
      <td>0.117322</td>
      <td>0.255439</td>
      <td>0.117920</td>
      <td>0.112878</td>
      <td>0.526562</td>
    </tr>
    <tr>
      <td>min</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>11.264464</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>0.200000</td>
      <td>0.166667</td>
      <td>0.375546</td>
      <td>0.281688</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>0.400000</td>
      <td>0.443478</td>
      <td>0.187747</td>
      <td>0.285936</td>
      <td>12.682307</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>0.200000</td>
      <td>0.233333</td>
      <td>0.455945</td>
      <td>0.332938</td>
      <td>0.200000</td>
      <td>0.500000</td>
      <td>0.400000</td>
      <td>0.652174</td>
      <td>0.247978</td>
      <td>0.341712</td>
      <td>13.017003</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>0.300000</td>
      <td>0.266667</td>
      <td>0.536222</td>
      <td>0.374886</td>
      <td>0.400000</td>
      <td>0.750000</td>
      <td>0.500000</td>
      <td>0.843478</td>
      <td>0.337463</td>
      <td>0.380616</td>
      <td>13.377006</td>
    </tr>
    <tr>
      <td>max</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>15.856731</td>
    </tr>
  </tbody>
</table>
</div>




```python
# setting waterfront as type category.
data_df['waterfront'] = data_df.waterfront.astype('category')
```


```python
#making sure everything looks right
data_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 21596 entries, 0 to 21596
    Data columns (total 12 columns):
    bedrooms         21596 non-null float64
    bathrooms        21596 non-null float64
    sqft_living      21596 non-null float64
    sqft_lot         21596 non-null float64
    floors           21596 non-null float64
    condition        21596 non-null float64
    grade            21596 non-null float64
    yr_built         21596 non-null float64
    sqft_living15    21596 non-null float64
    sqft_lot15       21596 non-null float64
    price            21596 non-null float64
    waterfront       21596 non-null category
    dtypes: category(1), float64(11)
    memory usage: 2.6 MB


# Modeling the Data

### Running Ordinary Least Squares regression experiments in Statsmodels
I am using StatsModels because the summay contains a lot of information and the layout is easy to read.


```python
outcome = 'price'
predictors = data_df.drop(['price'], axis=1)
pred_sum = "+".join(predictors.columns)
formula = outcome + "~" + pred_sum
model = ols(formula= formula, data=data_df).fit()
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th> <td>   0.658</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.657</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   3767.</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 01 Mar 2020</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>16:34:33</td>     <th>  Log-Likelihood:    </th> <td> -5221.1</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td> 21596</td>      <th>  AIC:               </th> <td>1.047e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 21584</td>      <th>  BIC:               </th> <td>1.056e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>    11</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
          <td></td>             <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>         <td>   11.7254</td> <td>    0.015</td> <td>  766.698</td> <td> 0.000</td> <td>   11.695</td> <td>   11.755</td>
</tr>
<tr>
  <th>waterfront[T.1.0]</th> <td>    0.5532</td> <td>    0.026</td> <td>   21.375</td> <td> 0.000</td> <td>    0.503</td> <td>    0.604</td>
</tr>
<tr>
  <th>bedrooms</th>          <td>   -0.4225</td> <td>    0.031</td> <td>  -13.482</td> <td> 0.000</td> <td>   -0.484</td> <td>   -0.361</td>
</tr>
<tr>
  <th>bathrooms</th>         <td>    0.6303</td> <td>    0.036</td> <td>   17.288</td> <td> 0.000</td> <td>    0.559</td> <td>    0.702</td>
</tr>
<tr>
  <th>sqft_living</th>       <td>    1.2824</td> <td>    0.040</td> <td>   31.807</td> <td> 0.000</td> <td>    1.203</td> <td>    1.361</td>
</tr>
<tr>
  <th>sqft_lot</th>          <td>   -0.1437</td> <td>    0.049</td> <td>   -2.956</td> <td> 0.003</td> <td>   -0.239</td> <td>   -0.048</td>
</tr>
<tr>
  <th>floors</th>            <td>    0.1158</td> <td>    0.013</td> <td>    8.900</td> <td> 0.000</td> <td>    0.090</td> <td>    0.141</td>
</tr>
<tr>
  <th>condition</th>         <td>    0.1716</td> <td>    0.014</td> <td>   12.188</td> <td> 0.000</td> <td>    0.144</td> <td>    0.199</td>
</tr>
<tr>
  <th>grade</th>             <td>    2.0690</td> <td>    0.031</td> <td>   65.887</td> <td> 0.000</td> <td>    2.007</td> <td>    2.131</td>
</tr>
<tr>
  <th>yr_built</th>          <td>   -0.6877</td> <td>    0.011</td> <td>  -63.881</td> <td> 0.000</td> <td>   -0.709</td> <td>   -0.667</td>
</tr>
<tr>
  <th>sqft_living15</th>     <td>    0.7683</td> <td>    0.030</td> <td>   25.985</td> <td> 0.000</td> <td>    0.710</td> <td>    0.826</td>
</tr>
<tr>
  <th>sqft_lot15</th>        <td>   -0.3664</td> <td>    0.048</td> <td>   -7.646</td> <td> 0.000</td> <td>   -0.460</td> <td>   -0.272</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>48.465</td> <th>  Durbin-Watson:     </th> <td>   1.965</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  56.757</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.056</td> <th>  Prob(JB):          </th> <td>4.74e-13</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 3.225</td> <th>  Cond. No.          </th> <td>    51.7</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



An adjusted R-squared of 0.657 is pretty good, considering I still have data to add to the model.  
All p-values are better than 0.05.
Skew shows not much tailing. A kurtosis of 3.225 is close to expected value of 3 for normal distibution.  
However, the high JB score tells me that the data may not be normally distributed. 
'bedrooms' , sqft_lot', 'yr_built', and 'sqft_lot15' have a negative coefficient. something to keep an eye on.

### Are the residuals consistent with a normal distribution?


```python
# q-q plot to visualize the residuals

residuals = model.resid
fig = sm.graphics.qqplot(residuals, dist=stats.norm, line='45', fit=True)
plt.title("Q-Q plot of Residuals");

```


![png](student_files/student_60_0.png)


The Q-Q plot shows some peakyness or tailing perhaps, but for the most part seems close to a normal distibution. Real life data  
won't be perfect!


```python
# Checking model performance with cross-validaton
linreg = LinearRegression()

X = data_df.drop(['price'], axis=1)
y = data_df.price

cv_results = cross_val_score(linreg, X, y, cv=10, scoring="neg_mean_squared_error")
cv_results
```




    array([-0.09485161, -0.09953494, -0.09634111, -0.09786468, -0.09079897,
           -0.09538301, -0.09304397, -0.10168181, -0.09753282, -0.09143151])




```python
np.mean(cv_results)
```




    -0.09584644256982355



Results look pretty consistent. A good sign.


```python
# Running recursive feature elimination to look for candidate variables to drop to see if I can improve model.

predictors = data_df.drop(['price', 'waterfront'], axis=1)

linreg = LinearRegression()
selector = RFE(linreg, n_features_to_select = 1)
selector = selector.fit(predictors, data_df["price"])
list(zip(predictors.columns,selector.ranking_))
```




    [('bedrooms', 7),
     ('bathrooms', 3),
     ('sqft_living', 2),
     ('sqft_lot', 8),
     ('floors', 10),
     ('condition', 9),
     ('grade', 1),
     ('yr_built', 4),
     ('sqft_living15', 5),
     ('sqft_lot15', 6)]




```python
# dropping floors (the worst ranked) to see if it improves model. 

data_dr = data_df.drop(['floors'], axis=1)
```


```python
# no floors
outcome = 'price'
predictors1 = data_dr.drop(['price'], axis=1)
pred_sum1 = "+".join(predictors1.columns)
formula1 = outcome + "~" + pred_sum1
model1 = ols(formula= formula1, data=data_dr).fit()
model1.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th> <td>   0.656</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.656</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   4121.</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 01 Mar 2020</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>16:36:48</td>     <th>  Log-Likelihood:    </th> <td> -5260.7</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td> 21596</td>      <th>  AIC:               </th> <td>1.054e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 21585</td>      <th>  BIC:               </th> <td>1.063e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>    10</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
          <td></td>             <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>         <td>   11.7295</td> <td>    0.015</td> <td>  765.928</td> <td> 0.000</td> <td>   11.700</td> <td>   11.760</td>
</tr>
<tr>
  <th>waterfront[T.1.0]</th> <td>    0.5575</td> <td>    0.026</td> <td>   21.506</td> <td> 0.000</td> <td>    0.507</td> <td>    0.608</td>
</tr>
<tr>
  <th>bedrooms</th>          <td>   -0.4345</td> <td>    0.031</td> <td>  -13.853</td> <td> 0.000</td> <td>   -0.496</td> <td>   -0.373</td>
</tr>
<tr>
  <th>bathrooms</th>         <td>    0.6855</td> <td>    0.036</td> <td>   19.048</td> <td> 0.000</td> <td>    0.615</td> <td>    0.756</td>
</tr>
<tr>
  <th>sqft_living</th>       <td>    1.3036</td> <td>    0.040</td> <td>   32.330</td> <td> 0.000</td> <td>    1.225</td> <td>    1.383</td>
</tr>
<tr>
  <th>sqft_lot</th>          <td>   -0.1902</td> <td>    0.048</td> <td>   -3.928</td> <td> 0.000</td> <td>   -0.285</td> <td>   -0.095</td>
</tr>
<tr>
  <th>condition</th>         <td>    0.1570</td> <td>    0.014</td> <td>   11.206</td> <td> 0.000</td> <td>    0.130</td> <td>    0.184</td>
</tr>
<tr>
  <th>grade</th>             <td>    2.1125</td> <td>    0.031</td> <td>   67.984</td> <td> 0.000</td> <td>    2.052</td> <td>    2.173</td>
</tr>
<tr>
  <th>yr_built</th>          <td>   -0.6655</td> <td>    0.010</td> <td>  -63.435</td> <td> 0.000</td> <td>   -0.686</td> <td>   -0.645</td>
</tr>
<tr>
  <th>sqft_living15</th>     <td>    0.7643</td> <td>    0.030</td> <td>   25.805</td> <td> 0.000</td> <td>    0.706</td> <td>    0.822</td>
</tr>
<tr>
  <th>sqft_lot15</th>        <td>   -0.3904</td> <td>    0.048</td> <td>   -8.145</td> <td> 0.000</td> <td>   -0.484</td> <td>   -0.296</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>49.190</td> <th>  Durbin-Watson:     </th> <td>   1.962</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  56.684</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.063</td> <th>  Prob(JB):          </th> <td>4.91e-13</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 3.217</td> <th>  Cond. No.          </th> <td>    51.3</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
linreg = LinearRegression()

X = data_dr.drop(['price'], axis=1)
y = data_dr.price

cv_results1 = cross_val_score(linreg, X, y, cv=10, scoring="neg_mean_squared_error")
np.mean(cv_results1)
```




    -0.0962495949425783



Dropping that variable didn't improve my model (actually slightly hurt it).


```python
# seeing if model improves when the categorical variable 'waterfront' is removed.  It is mostly 0's.
data_dr_water = data_df.drop('waterfront', axis=1)

outcome = 'price'
predictors2 = data_dr_water.drop(['price'], axis=1)
pred_sum2 = "+".join(predictors2.columns)
formula2 = outcome + "~" + pred_sum2
model2 = ols(formula= formula2, data=data_dr_water).fit()
model2.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th> <td>   0.650</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.650</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   4013.</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 01 Mar 2020</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>16:42:54</td>     <th>  Log-Likelihood:    </th> <td> -5447.3</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td> 21596</td>      <th>  AIC:               </th> <td>1.092e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 21585</td>      <th>  BIC:               </th> <td>1.100e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>    10</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
        <td></td>           <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>     <td>   11.7157</td> <td>    0.015</td> <td>  758.434</td> <td> 0.000</td> <td>   11.685</td> <td>   11.746</td>
</tr>
<tr>
  <th>bedrooms</th>      <td>   -0.4667</td> <td>    0.032</td> <td>  -14.769</td> <td> 0.000</td> <td>   -0.529</td> <td>   -0.405</td>
</tr>
<tr>
  <th>bathrooms</th>     <td>    0.6622</td> <td>    0.037</td> <td>   17.991</td> <td> 0.000</td> <td>    0.590</td> <td>    0.734</td>
</tr>
<tr>
  <th>sqft_living</th>   <td>    1.2975</td> <td>    0.041</td> <td>   31.853</td> <td> 0.000</td> <td>    1.218</td> <td>    1.377</td>
</tr>
<tr>
  <th>sqft_lot</th>      <td>   -0.1571</td> <td>    0.049</td> <td>   -3.197</td> <td> 0.001</td> <td>   -0.253</td> <td>   -0.061</td>
</tr>
<tr>
  <th>floors</th>        <td>    0.1209</td> <td>    0.013</td> <td>    9.204</td> <td> 0.000</td> <td>    0.095</td> <td>    0.147</td>
</tr>
<tr>
  <th>condition</th>     <td>    0.1724</td> <td>    0.014</td> <td>   12.118</td> <td> 0.000</td> <td>    0.145</td> <td>    0.200</td>
</tr>
<tr>
  <th>grade</th>         <td>    2.0878</td> <td>    0.032</td> <td>   65.822</td> <td> 0.000</td> <td>    2.026</td> <td>    2.150</td>
</tr>
<tr>
  <th>yr_built</th>      <td>   -0.7056</td> <td>    0.011</td> <td>  -65.053</td> <td> 0.000</td> <td>   -0.727</td> <td>   -0.684</td>
</tr>
<tr>
  <th>sqft_living15</th> <td>    0.7740</td> <td>    0.030</td> <td>   25.906</td> <td> 0.000</td> <td>    0.715</td> <td>    0.833</td>
</tr>
<tr>
  <th>sqft_lot15</th>    <td>   -0.3259</td> <td>    0.048</td> <td>   -6.737</td> <td> 0.000</td> <td>   -0.421</td> <td>   -0.231</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>50.905</td> <th>  Durbin-Watson:     </th> <td>   1.967</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  64.257</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.015</td> <th>  Prob(JB):          </th> <td>1.11e-14</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 3.266</td> <th>  Cond. No.          </th> <td>    51.7</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
linreg = LinearRegression()

X = data_dr.drop(['price','waterfront'], axis=1)
y = data_dr.price

cv_results2 = cross_val_score(linreg, X, y, cv=10, scoring="neg_mean_squared_error")
np.mean(cv_results2)
```




    -0.09832069899904562



Removing waterfront hurts my model.  I will keep the original model and move on with examining lat, long, and zipcode.

## Categorical Variables

### Location, location, location

I have three independent variables left to examine; lat, long, and zipcode.  These all are geographical data indicating where the houses are located. I will look at each one to see how best to add to my model.


```python
# How many unique values?
kc_df.zipcode.nunique()
```




    70




```python
kc_df.lat.nunique()
```




    5033




```python
kc_df.long.nunique()
```




    751



### Does location affect price?


```python
# Scatter plot of long and lat color mapped to log-transformed price data.

kc_df.plot(kind="scatter", x="long", y="lat", alpha=0.4, figsize=(16,10),
    c=data_pred["price"], cmap="rainbow", colorbar=True,
    sharex=False)
plt.title("Location and Log-Transformed Price")
plt.show()

```


![png](student_files/student_80_0.png)


Obviously location affects price! You can see how the house prices are higher in certain areas.     


```python
# Looking at a hexbin plot for density of locations (just curious)
kc_df.plot.hexbin(x='long', y='lat', figsize=(12,8))
plt.title("Housing Density");
```


![png](student_files/student_82_0.png)


Moving on to look at zipcode


```python
kc_df['zipcode'] = kc_df.zipcode.astype('int')
```


```python
kc_df.plot(kind="scatter", x="long", y="lat", alpha=0.6, figsize=(12,8),
    c='zipcode', cmap="rainbow", colorbar=True,
    sharex=False)
plt.title("Zipcode Locations");
```


![png](student_files/student_85_0.png)


I don't see any way to easily group zipcodes. I will do some further exploration, but will probably end up one-hot encoding the feature into categories.  

The lat and long gave me a nice graph that showed the importance of location on price, but I don't see a way to easily use in my linear regression model. I am thinking using lat and long and zipcode would be redundant anyway, so I will use zipcode in my model.


```python
# looking at Zipcode
kc_df.zipcode.hist(bins = 70, figsize=(10,10), label='zipcode')
plt.legend()
plt.title('Zipcode');
```


![png](student_files/student_87_0.png)


Not normally distributed at all. Defininitely categorical.


```python
kc_df['zipcode'].value_counts()
```




    98103    601
    98038    589
    98115    583
    98052    574
    98117    553
            ... 
    98102    104
    98010    100
    98024     80
    98148     57
    98039     50
    Name: zipcode, Length: 70, dtype: int64




```python
# One-hot encoding 'zipcode' variable and adding to my model.

kc_df['zipcode'] = kc_df.zipcode.astype('str')
zip_dummy = pd.get_dummies(kc_df.zipcode, prefix = 'ZC')

final_df = pd.concat([data_df, zip_dummy], axis=1)
```


```python
# Checking data to see if everything is how I want it.
final_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 21596 entries, 0 to 21596
    Data columns (total 82 columns):
    bedrooms         21596 non-null float64
    bathrooms        21596 non-null float64
    sqft_living      21596 non-null float64
    sqft_lot         21596 non-null float64
    floors           21596 non-null float64
    condition        21596 non-null float64
    grade            21596 non-null float64
    yr_built         21596 non-null float64
    sqft_living15    21596 non-null float64
    sqft_lot15       21596 non-null float64
    price            21596 non-null float64
    waterfront       21596 non-null category
    ZC_98001         21596 non-null uint8
    ZC_98002         21596 non-null uint8
    ZC_98003         21596 non-null uint8
    ZC_98004         21596 non-null uint8
    ZC_98005         21596 non-null uint8
    ZC_98006         21596 non-null uint8
    ZC_98007         21596 non-null uint8
    ZC_98008         21596 non-null uint8
    ZC_98010         21596 non-null uint8
    ZC_98011         21596 non-null uint8
    ZC_98014         21596 non-null uint8
    ZC_98019         21596 non-null uint8
    ZC_98022         21596 non-null uint8
    ZC_98023         21596 non-null uint8
    ZC_98024         21596 non-null uint8
    ZC_98027         21596 non-null uint8
    ZC_98028         21596 non-null uint8
    ZC_98029         21596 non-null uint8
    ZC_98030         21596 non-null uint8
    ZC_98031         21596 non-null uint8
    ZC_98032         21596 non-null uint8
    ZC_98033         21596 non-null uint8
    ZC_98034         21596 non-null uint8
    ZC_98038         21596 non-null uint8
    ZC_98039         21596 non-null uint8
    ZC_98040         21596 non-null uint8
    ZC_98042         21596 non-null uint8
    ZC_98045         21596 non-null uint8
    ZC_98052         21596 non-null uint8
    ZC_98053         21596 non-null uint8
    ZC_98055         21596 non-null uint8
    ZC_98056         21596 non-null uint8
    ZC_98058         21596 non-null uint8
    ZC_98059         21596 non-null uint8
    ZC_98065         21596 non-null uint8
    ZC_98070         21596 non-null uint8
    ZC_98072         21596 non-null uint8
    ZC_98074         21596 non-null uint8
    ZC_98075         21596 non-null uint8
    ZC_98077         21596 non-null uint8
    ZC_98092         21596 non-null uint8
    ZC_98102         21596 non-null uint8
    ZC_98103         21596 non-null uint8
    ZC_98105         21596 non-null uint8
    ZC_98106         21596 non-null uint8
    ZC_98107         21596 non-null uint8
    ZC_98108         21596 non-null uint8
    ZC_98109         21596 non-null uint8
    ZC_98112         21596 non-null uint8
    ZC_98115         21596 non-null uint8
    ZC_98116         21596 non-null uint8
    ZC_98117         21596 non-null uint8
    ZC_98118         21596 non-null uint8
    ZC_98119         21596 non-null uint8
    ZC_98122         21596 non-null uint8
    ZC_98125         21596 non-null uint8
    ZC_98126         21596 non-null uint8
    ZC_98133         21596 non-null uint8
    ZC_98136         21596 non-null uint8
    ZC_98144         21596 non-null uint8
    ZC_98146         21596 non-null uint8
    ZC_98148         21596 non-null uint8
    ZC_98155         21596 non-null uint8
    ZC_98166         21596 non-null uint8
    ZC_98168         21596 non-null uint8
    ZC_98177         21596 non-null uint8
    ZC_98178         21596 non-null uint8
    ZC_98188         21596 non-null uint8
    ZC_98198         21596 non-null uint8
    ZC_98199         21596 non-null uint8
    dtypes: category(1), float64(11), uint8(70)
    memory usage: 4.1 MB



```python
# dropping one of the zipcode columns

final_df = final_df.drop('ZC_98004', axis =1)
```


```python
# Modeling dataset with multivariate linear regression.
outcome = 'price'
predictors_fin = final_df.drop(['price'], axis=1)
pred_sum_fin = "+".join(predictors_fin.columns)
formula_fin = outcome + "~" + pred_sum_fin
model_fin = ols(formula= formula_fin, data=final_df).fit()
model_fin.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th>  <td>   0.876</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.876</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   1900.</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 01 Mar 2020</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>16:45:18</td>     <th>  Log-Likelihood:    </th>  <td>  5749.7</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td> 21596</td>      <th>  AIC:               </th> <td>-1.134e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 21515</td>      <th>  BIC:               </th> <td>-1.069e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>    80</td>      <th>                     </th>      <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
          <td></td>             <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>         <td>   12.1556</td> <td>    0.016</td> <td>  775.067</td> <td> 0.000</td> <td>   12.125</td> <td>   12.186</td>
</tr>
<tr>
  <th>waterfront[T.1.0]</th> <td>    0.6583</td> <td>    0.016</td> <td>   41.414</td> <td> 0.000</td> <td>    0.627</td> <td>    0.689</td>
</tr>
<tr>
  <th>bedrooms</th>          <td>   -0.1885</td> <td>    0.019</td> <td>   -9.805</td> <td> 0.000</td> <td>   -0.226</td> <td>   -0.151</td>
</tr>
<tr>
  <th>bathrooms</th>         <td>    0.3335</td> <td>    0.022</td> <td>   15.025</td> <td> 0.000</td> <td>    0.290</td> <td>    0.377</td>
</tr>
<tr>
  <th>sqft_living</th>       <td>    1.3788</td> <td>    0.024</td> <td>   56.282</td> <td> 0.000</td> <td>    1.331</td> <td>    1.427</td>
</tr>
<tr>
  <th>sqft_lot</th>          <td>    0.6374</td> <td>    0.030</td> <td>   21.233</td> <td> 0.000</td> <td>    0.579</td> <td>    0.696</td>
</tr>
<tr>
  <th>floors</th>            <td>    0.0325</td> <td>    0.008</td> <td>    3.955</td> <td> 0.000</td> <td>    0.016</td> <td>    0.049</td>
</tr>
<tr>
  <th>condition</th>         <td>    0.1875</td> <td>    0.009</td> <td>   21.423</td> <td> 0.000</td> <td>    0.170</td> <td>    0.205</td>
</tr>
<tr>
  <th>grade</th>             <td>    1.0339</td> <td>    0.020</td> <td>   50.846</td> <td> 0.000</td> <td>    0.994</td> <td>    1.074</td>
</tr>
<tr>
  <th>yr_built</th>          <td>   -0.0830</td> <td>    0.008</td> <td>   -9.992</td> <td> 0.000</td> <td>   -0.099</td> <td>   -0.067</td>
</tr>
<tr>
  <th>sqft_living15</th>     <td>    0.5665</td> <td>    0.019</td> <td>   29.827</td> <td> 0.000</td> <td>    0.529</td> <td>    0.604</td>
</tr>
<tr>
  <th>sqft_lot15</th>        <td>   -0.1606</td> <td>    0.030</td> <td>   -5.417</td> <td> 0.000</td> <td>   -0.219</td> <td>   -0.102</td>
</tr>
<tr>
  <th>ZC_98001</th>          <td>   -1.1096</td> <td>    0.015</td> <td>  -76.305</td> <td> 0.000</td> <td>   -1.138</td> <td>   -1.081</td>
</tr>
<tr>
  <th>ZC_98002</th>          <td>   -1.1035</td> <td>    0.017</td> <td>  -64.573</td> <td> 0.000</td> <td>   -1.137</td> <td>   -1.070</td>
</tr>
<tr>
  <th>ZC_98003</th>          <td>   -1.0927</td> <td>    0.015</td> <td>  -71.045</td> <td> 0.000</td> <td>   -1.123</td> <td>   -1.063</td>
</tr>
<tr>
  <th>ZC_98005</th>          <td>   -0.4173</td> <td>    0.018</td> <td>  -23.519</td> <td> 0.000</td> <td>   -0.452</td> <td>   -0.383</td>
</tr>
<tr>
  <th>ZC_98006</th>          <td>   -0.4820</td> <td>    0.013</td> <td>  -36.050</td> <td> 0.000</td> <td>   -0.508</td> <td>   -0.456</td>
</tr>
<tr>
  <th>ZC_98007</th>          <td>   -0.4771</td> <td>    0.019</td> <td>  -25.294</td> <td> 0.000</td> <td>   -0.514</td> <td>   -0.440</td>
</tr>
<tr>
  <th>ZC_98008</th>          <td>   -0.4497</td> <td>    0.015</td> <td>  -29.402</td> <td> 0.000</td> <td>   -0.480</td> <td>   -0.420</td>
</tr>
<tr>
  <th>ZC_98010</th>          <td>   -0.8701</td> <td>    0.022</td> <td>  -40.265</td> <td> 0.000</td> <td>   -0.912</td> <td>   -0.828</td>
</tr>
<tr>
  <th>ZC_98011</th>          <td>   -0.6794</td> <td>    0.017</td> <td>  -39.961</td> <td> 0.000</td> <td>   -0.713</td> <td>   -0.646</td>
</tr>
<tr>
  <th>ZC_98014</th>          <td>   -0.8082</td> <td>    0.020</td> <td>  -40.051</td> <td> 0.000</td> <td>   -0.848</td> <td>   -0.769</td>
</tr>
<tr>
  <th>ZC_98019</th>          <td>   -0.7985</td> <td>    0.017</td> <td>  -46.039</td> <td> 0.000</td> <td>   -0.832</td> <td>   -0.764</td>
</tr>
<tr>
  <th>ZC_98022</th>          <td>   -1.0278</td> <td>    0.016</td> <td>  -62.777</td> <td> 0.000</td> <td>   -1.060</td> <td>   -0.996</td>
</tr>
<tr>
  <th>ZC_98023</th>          <td>   -1.1444</td> <td>    0.013</td> <td>  -84.799</td> <td> 0.000</td> <td>   -1.171</td> <td>   -1.118</td>
</tr>
<tr>
  <th>ZC_98024</th>          <td>   -0.6904</td> <td>    0.024</td> <td>  -29.229</td> <td> 0.000</td> <td>   -0.737</td> <td>   -0.644</td>
</tr>
<tr>
  <th>ZC_98027</th>          <td>   -0.6202</td> <td>    0.014</td> <td>  -44.335</td> <td> 0.000</td> <td>   -0.648</td> <td>   -0.593</td>
</tr>
<tr>
  <th>ZC_98028</th>          <td>   -0.7035</td> <td>    0.015</td> <td>  -45.945</td> <td> 0.000</td> <td>   -0.733</td> <td>   -0.673</td>
</tr>
<tr>
  <th>ZC_98029</th>          <td>   -0.5153</td> <td>    0.015</td> <td>  -34.582</td> <td> 0.000</td> <td>   -0.545</td> <td>   -0.486</td>
</tr>
<tr>
  <th>ZC_98030</th>          <td>   -1.0611</td> <td>    0.016</td> <td>  -67.117</td> <td> 0.000</td> <td>   -1.092</td> <td>   -1.030</td>
</tr>
<tr>
  <th>ZC_98031</th>          <td>   -1.0467</td> <td>    0.016</td> <td>  -67.396</td> <td> 0.000</td> <td>   -1.077</td> <td>   -1.016</td>
</tr>
<tr>
  <th>ZC_98032</th>          <td>   -1.1327</td> <td>    0.020</td> <td>  -57.270</td> <td> 0.000</td> <td>   -1.171</td> <td>   -1.094</td>
</tr>
<tr>
  <th>ZC_98033</th>          <td>   -0.3207</td> <td>    0.014</td> <td>  -23.236</td> <td> 0.000</td> <td>   -0.348</td> <td>   -0.294</td>
</tr>
<tr>
  <th>ZC_98034</th>          <td>   -0.5660</td> <td>    0.013</td> <td>  -42.614</td> <td> 0.000</td> <td>   -0.592</td> <td>   -0.540</td>
</tr>
<tr>
  <th>ZC_98038</th>          <td>   -0.9435</td> <td>    0.013</td> <td>  -71.255</td> <td> 0.000</td> <td>   -0.969</td> <td>   -0.917</td>
</tr>
<tr>
  <th>ZC_98039</th>          <td>    0.1698</td> <td>    0.028</td> <td>    5.996</td> <td> 0.000</td> <td>    0.114</td> <td>    0.225</td>
</tr>
<tr>
  <th>ZC_98040</th>          <td>   -0.2411</td> <td>    0.015</td> <td>  -15.818</td> <td> 0.000</td> <td>   -0.271</td> <td>   -0.211</td>
</tr>
<tr>
  <th>ZC_98042</th>          <td>   -1.0480</td> <td>    0.013</td> <td>  -78.388</td> <td> 0.000</td> <td>   -1.074</td> <td>   -1.022</td>
</tr>
<tr>
  <th>ZC_98045</th>          <td>   -0.7847</td> <td>    0.017</td> <td>  -47.180</td> <td> 0.000</td> <td>   -0.817</td> <td>   -0.752</td>
</tr>
<tr>
  <th>ZC_98052</th>          <td>   -0.4951</td> <td>    0.013</td> <td>  -37.861</td> <td> 0.000</td> <td>   -0.521</td> <td>   -0.469</td>
</tr>
<tr>
  <th>ZC_98053</th>          <td>   -0.5381</td> <td>    0.014</td> <td>  -37.952</td> <td> 0.000</td> <td>   -0.566</td> <td>   -0.510</td>
</tr>
<tr>
  <th>ZC_98055</th>          <td>   -0.9589</td> <td>    0.016</td> <td>  -61.434</td> <td> 0.000</td> <td>   -0.990</td> <td>   -0.928</td>
</tr>
<tr>
  <th>ZC_98056</th>          <td>   -0.7780</td> <td>    0.014</td> <td>  -55.091</td> <td> 0.000</td> <td>   -0.806</td> <td>   -0.750</td>
</tr>
<tr>
  <th>ZC_98058</th>          <td>   -0.9558</td> <td>    0.014</td> <td>  -69.500</td> <td> 0.000</td> <td>   -0.983</td> <td>   -0.929</td>
</tr>
<tr>
  <th>ZC_98059</th>          <td>   -0.7780</td> <td>    0.014</td> <td>  -56.945</td> <td> 0.000</td> <td>   -0.805</td> <td>   -0.751</td>
</tr>
<tr>
  <th>ZC_98065</th>          <td>   -0.6962</td> <td>    0.015</td> <td>  -46.067</td> <td> 0.000</td> <td>   -0.726</td> <td>   -0.667</td>
</tr>
<tr>
  <th>ZC_98070</th>          <td>   -0.7864</td> <td>    0.021</td> <td>  -37.660</td> <td> 0.000</td> <td>   -0.827</td> <td>   -0.745</td>
</tr>
<tr>
  <th>ZC_98072</th>          <td>   -0.6629</td> <td>    0.015</td> <td>  -42.806</td> <td> 0.000</td> <td>   -0.693</td> <td>   -0.633</td>
</tr>
<tr>
  <th>ZC_98074</th>          <td>   -0.5790</td> <td>    0.014</td> <td>  -42.117</td> <td> 0.000</td> <td>   -0.606</td> <td>   -0.552</td>
</tr>
<tr>
  <th>ZC_98075</th>          <td>   -0.5751</td> <td>    0.014</td> <td>  -39.906</td> <td> 0.000</td> <td>   -0.603</td> <td>   -0.547</td>
</tr>
<tr>
  <th>ZC_98077</th>          <td>   -0.7214</td> <td>    0.017</td> <td>  -42.178</td> <td> 0.000</td> <td>   -0.755</td> <td>   -0.688</td>
</tr>
<tr>
  <th>ZC_98092</th>          <td>   -1.0924</td> <td>    0.015</td> <td>  -74.832</td> <td> 0.000</td> <td>   -1.121</td> <td>   -1.064</td>
</tr>
<tr>
  <th>ZC_98102</th>          <td>   -0.1388</td> <td>    0.021</td> <td>   -6.482</td> <td> 0.000</td> <td>   -0.181</td> <td>   -0.097</td>
</tr>
<tr>
  <th>ZC_98103</th>          <td>   -0.2641</td> <td>    0.014</td> <td>  -19.545</td> <td> 0.000</td> <td>   -0.291</td> <td>   -0.238</td>
</tr>
<tr>
  <th>ZC_98105</th>          <td>   -0.1567</td> <td>    0.016</td> <td>   -9.527</td> <td> 0.000</td> <td>   -0.189</td> <td>   -0.124</td>
</tr>
<tr>
  <th>ZC_98106</th>          <td>   -0.7337</td> <td>    0.015</td> <td>  -49.121</td> <td> 0.000</td> <td>   -0.763</td> <td>   -0.704</td>
</tr>
<tr>
  <th>ZC_98107</th>          <td>   -0.2353</td> <td>    0.016</td> <td>  -14.759</td> <td> 0.000</td> <td>   -0.266</td> <td>   -0.204</td>
</tr>
<tr>
  <th>ZC_98108</th>          <td>   -0.7362</td> <td>    0.017</td> <td>  -42.266</td> <td> 0.000</td> <td>   -0.770</td> <td>   -0.702</td>
</tr>
<tr>
  <th>ZC_98109</th>          <td>   -0.0950</td> <td>    0.021</td> <td>   -4.523</td> <td> 0.000</td> <td>   -0.136</td> <td>   -0.054</td>
</tr>
<tr>
  <th>ZC_98112</th>          <td>   -0.0725</td> <td>    0.016</td> <td>   -4.592</td> <td> 0.000</td> <td>   -0.103</td> <td>   -0.042</td>
</tr>
<tr>
  <th>ZC_98115</th>          <td>   -0.2788</td> <td>    0.013</td> <td>  -20.967</td> <td> 0.000</td> <td>   -0.305</td> <td>   -0.253</td>
</tr>
<tr>
  <th>ZC_98116</th>          <td>   -0.3103</td> <td>    0.015</td> <td>  -20.784</td> <td> 0.000</td> <td>   -0.340</td> <td>   -0.281</td>
</tr>
<tr>
  <th>ZC_98117</th>          <td>   -0.2762</td> <td>    0.014</td> <td>  -20.441</td> <td> 0.000</td> <td>   -0.303</td> <td>   -0.250</td>
</tr>
<tr>
  <th>ZC_98118</th>          <td>   -0.6203</td> <td>    0.014</td> <td>  -45.474</td> <td> 0.000</td> <td>   -0.647</td> <td>   -0.594</td>
</tr>
<tr>
  <th>ZC_98119</th>          <td>   -0.1012</td> <td>    0.018</td> <td>   -5.730</td> <td> 0.000</td> <td>   -0.136</td> <td>   -0.067</td>
</tr>
<tr>
  <th>ZC_98122</th>          <td>   -0.2789</td> <td>    0.016</td> <td>  -17.879</td> <td> 0.000</td> <td>   -0.310</td> <td>   -0.248</td>
</tr>
<tr>
  <th>ZC_98125</th>          <td>   -0.5284</td> <td>    0.014</td> <td>  -37.362</td> <td> 0.000</td> <td>   -0.556</td> <td>   -0.501</td>
</tr>
<tr>
  <th>ZC_98126</th>          <td>   -0.5174</td> <td>    0.015</td> <td>  -35.092</td> <td> 0.000</td> <td>   -0.546</td> <td>   -0.488</td>
</tr>
<tr>
  <th>ZC_98133</th>          <td>   -0.6402</td> <td>    0.014</td> <td>  -46.879</td> <td> 0.000</td> <td>   -0.667</td> <td>   -0.613</td>
</tr>
<tr>
  <th>ZC_98136</th>          <td>   -0.3872</td> <td>    0.016</td> <td>  -24.552</td> <td> 0.000</td> <td>   -0.418</td> <td>   -0.356</td>
</tr>
<tr>
  <th>ZC_98144</th>          <td>   -0.4040</td> <td>    0.015</td> <td>  -27.193</td> <td> 0.000</td> <td>   -0.433</td> <td>   -0.375</td>
</tr>
<tr>
  <th>ZC_98146</th>          <td>   -0.7953</td> <td>    0.015</td> <td>  -51.629</td> <td> 0.000</td> <td>   -0.825</td> <td>   -0.765</td>
</tr>
<tr>
  <th>ZC_98148</th>          <td>   -0.9436</td> <td>    0.027</td> <td>  -35.110</td> <td> 0.000</td> <td>   -0.996</td> <td>   -0.891</td>
</tr>
<tr>
  <th>ZC_98155</th>          <td>   -0.6741</td> <td>    0.014</td> <td>  -48.689</td> <td> 0.000</td> <td>   -0.701</td> <td>   -0.647</td>
</tr>
<tr>
  <th>ZC_98166</th>          <td>   -0.7819</td> <td>    0.016</td> <td>  -49.471</td> <td> 0.000</td> <td>   -0.813</td> <td>   -0.751</td>
</tr>
<tr>
  <th>ZC_98168</th>          <td>   -1.0264</td> <td>    0.016</td> <td>  -65.208</td> <td> 0.000</td> <td>   -1.057</td> <td>   -0.996</td>
</tr>
<tr>
  <th>ZC_98177</th>          <td>   -0.4912</td> <td>    0.016</td> <td>  -31.329</td> <td> 0.000</td> <td>   -0.522</td> <td>   -0.460</td>
</tr>
<tr>
  <th>ZC_98178</th>          <td>   -0.9335</td> <td>    0.016</td> <td>  -59.218</td> <td> 0.000</td> <td>   -0.964</td> <td>   -0.903</td>
</tr>
<tr>
  <th>ZC_98188</th>          <td>   -1.0007</td> <td>    0.019</td> <td>  -52.055</td> <td> 0.000</td> <td>   -1.038</td> <td>   -0.963</td>
</tr>
<tr>
  <th>ZC_98198</th>          <td>   -1.0156</td> <td>    0.015</td> <td>  -65.759</td> <td> 0.000</td> <td>   -1.046</td> <td>   -0.985</td>
</tr>
<tr>
  <th>ZC_98199</th>          <td>   -0.2298</td> <td>    0.015</td> <td>  -15.348</td> <td> 0.000</td> <td>   -0.259</td> <td>   -0.200</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>1370.410</td> <th>  Durbin-Watson:     </th> <td>   1.997</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>6174.051</td>
</tr>
<tr>
  <th>Skew:</th>           <td>-0.107</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>       <td> 5.611</td>  <th>  Cond. No.          </th> <td>    116.</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



Adding zipcode to my model increased the adjusted R-squared from 0.657 to 0.876.  So zipcode can explain 21.9% of the 
variance in the price. That's a lot!  

The adjusted r-squared of 0.876 in my final model is pretty good. All the p-values are less than 0.05. The biggest issue I see is the negative coefficients for 'bedrooms' and 'yr_built'. I think that indicates interactions between features that may not be obvious. I am keeping them in, using the p-values as justification.

## Model Validation


```python
# K-folds cross validation of my final model using negative mean squared error
linreg = LinearRegression()

X = final_df.drop(['price'], axis=1)
y = final_df.price

cv_results3 = cross_val_score(linreg, X, y, cv=10, scoring="neg_mean_squared_error")
cv_results3
```




    array([-0.03428675, -0.0370451 , -0.03566962, -0.0357842 , -0.03316145,
           -0.0357037 , -0.03419218, -0.03669121, -0.03534284, -0.0309256 ])




```python
np.mean(cv_results3)
```




    -0.03488026597319434




```python
# Coefficient of variation of cross validation results
abs(np.std(cv_results3)/np.mean(cv_results3))*100
```




    4.942280141700788




```python
# Using r-squared
linreg = LinearRegression()

X = final_df.drop(['price'], axis=1)
y = final_df.price

cv_results4 = cross_val_score(linreg, X, y, cv=10)
cv_results4
```




    array([0.8759028 , 0.87644342, 0.86667069, 0.8741245 , 0.8675731 ,
           0.87361936, 0.87786505, 0.87663803, 0.87589277, 0.86835936])



Adding zipcode also improved my cross validation score. And the results showed little variation.


```python
# Train-test-split as another check on my model's performance.
y = final_df[["price"]]
X = final_df.drop(["price"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

linreg = LinearRegression()
linreg.fit(X_train, y_train)

y_hat_train = linreg.predict(X_train)
y_hat_test = linreg.predict(X_test)
train_mse = mean_squared_error(y_train, y_hat_train)
test_mse = mean_squared_error(y_test, y_hat_test)
print('Train Mean Squarred Error:', train_mse)
print('Test Mean Squarred Error:', test_mse)
```

    Train Mean Squarred Error: 0.03466335236355768
    Test Mean Squarred Error: 0.03337063914965873


The test MSE and train MSE are very similar, giving me confidence that the model isn't overfit.

# Interpret Results

### Final Model Summary

For the final model used 12 independent variables (80 if you count the 69 dummy variables used for zipcode separately)  and 21596 data points. The OLS regression model has an adjusted R-squared of 0.876. This gives me a fairly high level of confidence in my model to accurately predict housing prices. The remaining 0.124 could be influence of factors such as features that weren't included in the data set, sampling error, seasonal market fluctuations, or less tangeable factors like a seller's skills or bidding wars that bump up the price. The features that have the most effect on the sales price seem to be waterfront, zipcode, grade, sqft_living, and bathrooms.  

The p-values of all my independent variables are less than 0.05 which shows that they all have greater than 95% chance that the coefficient isn't zero. Another factor that gives me confidence in the performance of my model is the k-folds cross validation. I had little variation across the resulting negative mean squared errors. (Average of -0.035 and a CV of 4.9%) I also performed a train-test-split validation which produced similar results. (Train: 0.035 and Test: 0.033) This shows I haven't over fit the model.

#### Interpreting Coefficients
Here is a closer look at some of the coefficients (the dependent variable, 'price', is log-transformed): 


The min-max scaled and log-transformed independent variable 'sqft_living' has a coefficient of 1.3788. For a 1% increase/decrease in 'sqft_living' we expect about a 1.3788% increase/decrease in sales price with everything else remaining unchanged.

The categorical feature 'waterfront'has a coefficient of 0.6583. So if the property has a view of the water it increases the price about 93% with everything else remaining unchanged.

The min-max scaled feature 'grade' has a coefficient of 1.0339. If, for example, the grade is 4. Min-max scaled would make it 0.1. If the grade is 8, min-max scaled makes it 0.5. Going from a grade of 4 to a grade of 8 we would expect about a 36.4% increase in the sales price with everything else remaining unchanged.






```python

```


```python

```
