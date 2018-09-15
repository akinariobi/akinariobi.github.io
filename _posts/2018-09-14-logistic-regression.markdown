---
layout: post
title:  "Classification using binary logistic regression"
date:   2018-09-14 16:57:51
---

Logistic Regression is an extension of simple linear regression  and it's used when the dependent variable (target) is categorical. There's three types of logistic regression: binary, multinomial and ordinal logistic regression but this article is only about binary version of this algorithm.


The logistic regression model is simply a non-linear transformation of the linear regression and its distribution is an S-shaped distribution function (cumulative density function) which is similar to the standard
normal distribution and constrains the estimated probabilities to lie between 0
and 1.

![s-shaped distribution](http://blog.datumbox.com/wp-content/uploads/2013/11/multinomial-logistic-regression.png)
[source][s-shaped]

#### how it works

This algorithm estimates the relationship between the dependent variable, what we want to predict, and the one or more independent variables or features, by estimating probabilities using logistic function.


#### practical cases

* You could use binomial logistic regression to understand whether exam performance can be predicted based on revision time, test anxiety and lecture attendance (i.e., where the dependent variable is "exam performance", measured on a dichotomous scale – "passed" or "failed" – and you have three independent variables: "revision time", "test anxiety" and "lecture attendance").

* ..or in order to determine if the email is spam (1) or not (0)

#### example

Let's build a logistic regression in Python using [Heart Disease][http://archive.ics.uci.edu/ml/datasets/Heart+Disease] data set from [UC Irvine Machine Learning Repository][data].

The raw input data from this data set is a text file (.data) with 14 columns and multiple rows in it.

{% highlight js %}
63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,0.0,6.0,0
67.0,1.0,4.0,160.0,286.0,0.0,2.0,108.0,1.0,1.5,2.0,3.0,3.0,2
67.0,1.0,4.0,120.0,229.0,0.0,2.0,129.0,1.0,2.6,2.0,2.0,7.0,1
37.0,1.0,3.0,130.0,250.0,0.0,0.0,187.0,0.0,3.5,3.0,0.0,3.0,0
41.0,0.0,2.0,130.0,204.0,0.0,2.0,172.0,0.0,1.4,1.0,0.0,3.0,0
56.0,1.0,2.0,120.0,236.0,0.0,0.0,178.0,0.0,0.8,1.0,0.0,3.0,0
62.0,0.0,4.0,140.0,268.0,0.0,2.0,160.0,0.0,3.6,3.0,2.0,3.0,3
57.0,0.0,4.0,120.0,354.0,0.0,0.0,163.0,1.0,0.6,1.0,0.0,3.0,0
63.0,1.0,4.0,130.0,254.0,0.0,2.0,147.0,0.0,1.4,2.0,1.0,7.0,2
53.0,1.0,4.0,140.0,203.0,1.0,2.0,155.0,1.0,3.1,3.0,0.0,7.0,1
...
{% endhighlight js %}


Each column corresponds to defined attribute:

```
Attribute Information:
   -- Only 14 used
      -- 1. (age)       
      -- 2. (sex)       
      -- 3. (cp: chest pain type
                -- Value 1: typical angina
                -- Value 2: atypical angina
                -- Value 3: non-anginal pain
                -- Value 4: asymptomatic)        
      -- 4. (trestbps: resting blood pressure (in mm Hg on admission to the
              hospital))  
      -- 5. (chol or serum cholestoral in mg/dl)      
      -- 6. (fbs: (fasting blood sugar > 120 mg/dl)  (1 = true; 0 =         false))       
      -- 7. (restecg: resting electrocardiographic results
                -- Value 0: normal
                -- Value 1: having ST-T wave abnormality (T wave inversions and/or ST
                            elevation or depression of > 0.05 mV)
                -- Value 2: showing probable or definite left ventricular hypertrophy
                            by Estes' criteria)   
      -- 8. (thalach: maximum heart rate achieved)   
      -- 9. (exang: exercise induced angina (1 = yes; 0 = no))     
      -- 10. (oldpeak: ST depression induced by exercise relative to rest)   
      -- 11. (slope: the slope of the peak exercise ST segment
                -- Value 1: upsloping
                -- Value 2: flat
                -- Value 3: downsloping)     
      -- 12. (ca: number of major vessels (0-3) colored by flourosopy)        
      -- 13. (thal: 3 = normal; 6 = fixed defect; 7 = reversable defect)      
      -- 14. (the predicted attribute) (num: diagnosis of heart disease (angiographic disease status)
                -- Value 0: < 50% diameter narrowing
                -- Value 1: > 50% diameter narrowing
                (in any major vessel: attributes 59 through 68 are vessels))
```
**predict variable**:

Num (y): has a patient *high* risk of heart disease? (1) -- yes, (0) -- no.

I converted .data to CSV in order to get a better visualization of each operation I'm doing.  

```python

# .data to csv converter

import csv

with open('processed.cleveland.data', 'r') as in_file:
    stripped = (line.strip() for line in in_file)
    lines = (line.split(",") for line in stripped if line)
    with open('data.csv', 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(('age', 'sex', 'chest pain type', 'blood pressure', 'serum cholestoral', 'fasting blood sugar', 'resting electrocardiographic results', 'max heart rate achieved', 'exercise induced angina', 'ST depression', 'nb of major vessels', 'thal', 'num (y)'))
        writer.writerows(lines)

```

First of all, we need to import all python dependencies. I'm using Jupyter Notebook for better representation.

```python
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
plt.rc("font", size=16)
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
```

![data image](../assets/img/classification-using-binary-logistic-regression/1.png)





[s-shaped]: http://blog.datumbox.com/wp-content/uploads/2013/11/multinomial-logistic-regression.png
[data]: http://archive.ics.uci.edu/ml/index.php
[data_source]: http://archive.ics.uci.edu/ml/datasets/Heart+Disease
