
# In Depth A/B Testing

## Introduction

In this lab, you'll explore a survey from Kaggle regarding budding data scientists. With this, you'll form some initial hypotheses, and test them using the tools you've acquired to date. 

## Objectives

You will be able to:
* Conduct statistical tests on a real world dataset

## Load the Dataset and Perform a Brief Exploration

The data is stored in a file called **multipleChoiceResponses_cleaned.csv**. Feel free to check out the original dataset referenced at the bottom of this lab, although this cleaned version will undoubtedly be easier to work with. Additionally, meta-data regarding the questions is stored in a file name **schema.csv**. Load in the data itself as a Pandas DataFrame, and take a moment to briefly get acquainted with it.

> Note: If you can't get the file to load properly, try chaging the encoding format as in `encoding='latin1'`


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
%matplotlib inline
```


```python
df = pd.read_csv('multipleChoiceResponses.csv', encoding='latin1')
df.head()
```

    /Users/matthew.mitchell/anaconda3/lib/python3.6/site-packages/IPython/core/interactiveshell.py:2785: DtypeWarning: Columns (31,83,86,87,98,99,109,116,123,124,127,129,130,164) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)





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
      <th>GenderSelect</th>
      <th>Country</th>
      <th>Age</th>
      <th>EmploymentStatus</th>
      <th>StudentStatus</th>
      <th>LearningDataScience</th>
      <th>CodeWriter</th>
      <th>CareerSwitcher</th>
      <th>CurrentJobTitleSelect</th>
      <th>TitleFit</th>
      <th>...</th>
      <th>JobFactorExperienceLevel</th>
      <th>JobFactorDepartment</th>
      <th>JobFactorTitle</th>
      <th>JobFactorCompanyFunding</th>
      <th>JobFactorImpact</th>
      <th>JobFactorRemote</th>
      <th>JobFactorIndustry</th>
      <th>JobFactorLeaderReputation</th>
      <th>JobFactorDiversity</th>
      <th>JobFactorPublishingOpportunity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Non-binary, genderqueer, or gender non-conforming</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Employed full-time</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Yes</td>
      <td>NaN</td>
      <td>DBA/Database Engineer</td>
      <td>Fine</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Female</td>
      <td>United States</td>
      <td>30.0</td>
      <td>Not employed, but looking for work</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Somewhat important</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Male</td>
      <td>Canada</td>
      <td>28.0</td>
      <td>Not employed, but looking for work</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>Very Important</td>
      <td>Very Important</td>
      <td>Very Important</td>
      <td>Very Important</td>
      <td>Very Important</td>
      <td>Very Important</td>
      <td>Very Important</td>
      <td>Very Important</td>
      <td>Very Important</td>
      <td>Very Important</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Male</td>
      <td>United States</td>
      <td>56.0</td>
      <td>Independent contractor, freelancer, or self-em...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Yes</td>
      <td>NaN</td>
      <td>Operations Research Practitioner</td>
      <td>Poorly</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Male</td>
      <td>Taiwan</td>
      <td>38.0</td>
      <td>Employed full-time</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Yes</td>
      <td>NaN</td>
      <td>Computer Scientist</td>
      <td>Fine</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 228 columns</p>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 16716 entries, 0 to 16715
    Columns: 228 entries, GenderSelect to JobFactorPublishingOpportunity
    dtypes: float64(13), object(215)
    memory usage: 29.1+ MB


## Wages and Education

You've been asked to determine whether education is impactful to salary. Develop a hypothesis test to compare the salaries of those with Master's degrees to those with Bachelor's degrees. Are the two statistically different according to your results?

> Note: The relevant features are stored in the 'FormalEducation' and 'AdjustedCompensation' features.


```python
import flatiron_stats as fs
```


```python
f1 = 'FormalEducation'
f2 = 'AdjustedCompensation'
subset = df[(~df[f1].isnull()) & (~df[f2].isnull())]
s1 = subset[subset[f1]=="Master's degree"][f2]
s2 = subset[subset[f1]=="Bachelor's degree"][f2]
print("Median Values: \ns1:{} \ns2:{}".format(round(s1.median(),2), round(s2.median(),2)))
print('Sample sizes: \ns1: {} \ns2: {}'.format(len(s1), len(s2)))
print("Welch's t-test p-value:", fs.p_value_welch_ttest(s1, s2))

print('\n\nRepeated Test with Ouliers Removed:')
outlier_threshold = 500000
s1 = subset[(subset[f1]=="Master's degree") & (subset[f2]<=outlier_threshold)][f2]
s2 = subset[(subset[f1]=="Bachelor's degree") & (subset[f2]<=outlier_threshold)][f2]
print('Sample sizes: \ns1: {} \ns2: {}'.format(len(s1), len(s2)))
print("Welch's t-test p-value with outliers removed:", fs.p_value_welch_ttest(s1, s2))

```

    Median Values: 
    s1:53539.72 
    s2:38345.78
    Sample sizes: 
    s1: 2006 
    s2: 1110
    Welch's t-test p-value: 0.3295681795245178
    
    
    Repeated Test with Ouliers Removed:
    Sample sizes: 
    s1: 2001 
    s2: 1106
    Welch's t-test p-value with outliers removed: 3.8278556524495855e-07


## Wages and Education II


```python
f1 = 'FormalEducation'
f2 = 'AdjustedCompensation'
subset = df[(~df[f1].isnull()) & (~df[f2].isnull())]
s1 = subset[subset[f1]=="Doctoral degree"][f2]
s2 = subset[subset[f1]=="Bachelor's degree"][f2]
print("Median Values: \ns1:{} \ns2:{}".format(round(s1.median(),2), round(s2.median(),2)))
print('Sample sizes: \ns1: {} \ns2: {}'.format(len(s1), len(s2)))
print("Welch's t-test p-value:", fs.p_value_welch_ttest(s1, s2))

print('\n\nRepeated Test with Ouliers Removed:')
outlier_threshold = 500000
s1 = subset[(subset[f1]=="Doctoral degree") & (subset[f2]<=outlier_threshold)][f2]
s2 = subset[(subset[f1]=="Bachelor's degree") & (subset[f2]<=outlier_threshold)][f2]
print('Sample sizes: \ns1: {} \ns2: {}'.format(len(s1), len(s2)))
print("Welch's t-test p-value with outliers removed:", fs.p_value_welch_ttest(s1, s2))
```

    Median Values: 
    s1:74258.9 
    s2:38345.78
    Sample sizes: 
    s1: 976 
    s2: 1110
    Welch's t-test p-value: 0.15681586933369485
    
    
    Repeated Test with Ouliers Removed:
    Sample sizes: 
    s1: 973 
    s2: 1106
    Welch's t-test p-value with outliers removed: 0.0


## Wages and Education III

Remember the multiple comparisons problem; rather then continuing on like this, perform an ANOVA test between the various 'FormalEducation' categories and their relation to 'AdjustedCompensation'.


```python
#Perform ANOVA here
```

## Additional Resources

Here's the original source where the data was taken from:  
    [Kaggle Machine Learning & Data Science Survey 2017](https://www.kaggle.com/kaggle/kaggle-survey-2017)

## Summary

In this lab, you practiced conducting actual hypothesis tests on actual data. From this, you saw how dependent results can be on the initial problem formulation, including preprocessing.
