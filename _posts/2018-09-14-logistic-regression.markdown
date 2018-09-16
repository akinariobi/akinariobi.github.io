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

* ..or to determine if the email is spam (1) or not (0)

#### example

Let's build a logistic regression in Python using [Spambase Data Set][dataset] from [UC Irvine Machine Learning Repository][data].

The raw input data from this data set is a text file (.data) with 58 columns and multiple rows in it.

{% highlight js %}
0,0.64,0.64,0,0.32,0,0,0,0,0,0,0.64,0,0,0,0.32,0,1.29,1.93,0,0.96,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.778,0,0,3.756,61,278,1
0.21,0.28,0.5,0,0.14,0.28,0.21,0.07,0,0.94,0.21,0.79,0.65,0.21,0.14,0.14,0.07,0.28,3.47,0,1.59,0,0.43,0.43,0,0,0,0,0,0,0,0,0,0,0,0,0.07,0,0,0,0,0,0,0,0,0,0,0,0,0.132,0,0.372,0.18,0.048,5.114,101,1028,1
0.06,0,0.71,0,1.23,0.19,0.19,0.12,0.64,0.25,0.38,0.45,0.12,0,1.75,0.06,0.06,1.03,1.36,0.32,0.51,0,1.16,0.06,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.06,0,0,0.12,0,0.06,0.06,0,0,0.01,0.143,0,0.276,0.184,0.01,9.821,485,2259,1
0,0,0,0,0.63,0,0.31,0.63,0.31,0.63,0.31,0.31,0.31,0,0,0.31,0,0,3.18,0,0.31,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.137,0,0.137,0,0,3.537,40,191,1
0,0,0,0,0.63,0,0.31,0.63,0.31,0.63,0.31,0.31,0.31,0,0,0.31,0,0,3.18,0,0.31,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.135,0,0.135,0,0,3.537,40,191,1
0,0,0,0,1.85,0,0,1.85,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.223,0,0,0,0,3,15,54,1
0,0,0,0,1.92,0,0,0,0,0.64,0.96,1.28,0,0,0,0.96,0,0.32,3.85,0,0.64,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.054,0,0.164,0.054,0,1.671,4,112,1
0,0,0,0,1.88,0,0,1.88,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.206,0,0,0,0,2.45,11,49,1
0.15,0,0.46,0,0.61,0,0.3,0,0.92,0.76,0.76,0.92,0,0,0,0,0,0.15,1.23,3.53,2,0,0,0.15,0,0,0,0,0,0,0,0,0.15,0,0,0,0,0,0,0,0,0,0.3,0,0,0,0,0,0,0.271,0,0.181,0.203,0.022,9.744,445,1257,1
...
{% endhighlight js %}


Each column corresponds to defined attribute.

**Attribute Information**

{% highlight js %}

-- word_freq_make:         continuous.
-- word_freq_address:      continuous.
-- word_freq_all:          continuous.
-- word_freq_3d:           continuous.
-- word_freq_our:          continuous.
-- word_freq_over:         continuous.
-- word_freq_remove:       continuous.
-- word_freq_internet:     continuous.
-- word_freq_order:        continuous.
-- word_freq_mail:         continuous.
-- word_freq_receive:      continuous.
-- word_freq_will:         continuous.
-- word_freq_people:       continuous.
-- word_freq_report:       continuous.
-- word_freq_addresses:    continuous.
-- word_freq_free:         continuous.
-- word_freq_business:     continuous.
-- word_freq_email:        continuous.
-- word_freq_you:          continuous.
-- word_freq_credit:       continuous.
-- word_freq_your:         continuous.
-- word_freq_font:         continuous.
-- word_freq_000:          continuous.
-- word_freq_money:        continuous.
-- word_freq_hp:           continuous.
-- word_freq_hpl:          continuous.
-- word_freq_george:       continuous.
-- word_freq_650:          continuous.
-- word_freq_lab:          continuous.
-- word_freq_labs:         continuous.
-- word_freq_telnet:       continuous.
-- word_freq_857:          continuous.
-- word_freq_data:         continuous.
-- word_freq_415:          continuous.
-- word_freq_85:           continuous.
-- word_freq_technology:   continuous.
-- word_freq_1999:         continuous.
-- word_freq_parts:        continuous.
-- word_freq_pm:           continuous.
-- word_freq_direct:       continuous.
-- word_freq_cs:           continuous.
-- word_freq_meeting:      continuous.
-- word_freq_original:     continuous.
-- word_freq_project:      continuous.
-- word_freq_re:           continuous.
-- word_freq_edu:          continuous.
-- word_freq_table:        continuous.
-- word_freq_conference:   continuous.
-- char_freq_;:            continuous.
-- char_freq_(:            continuous.
-- char_freq_[:            continuous.
-- char_freq_!:            continuous.
-- char_freq_$:            continuous.
-- char_freq_#:            continuous.
-- capital_run_length_average: continuous.
-- capital_run_length_longest: continuous.
-- capital_run_length_total:   continuous.
-- spam_or_not: binary.

{% endhighlight js %}

**Predict Variable**

Spam: The given email is spam or not? (1) -- yes, (0) -- no.

Note that as mentioned in [documentation][datadoc] the "spam" concept is diverse: advertisements for products/web sites, make money fast schemes, chain letters, pornography...

{% highlight js %}
Our collection of spam e-mails came from our postmaster and individuals who had filed spam.
Our collection of non-spam e-mails came from filed work and personal e-mails,
and hence the word 'george' and the area code '650' are indicators of non-spam.
These are useful when constructing a personalized spam filter. One would either
have to blind such non-spam indicators or get a very wide collection of non-spam
to generate a general purpose spam filter.
{% endhighlight js %}

I converted .data to CSV to get a better visualization of each operation I'm doing.  

*Note: wf - word frequency, cf - character frequency*

{% highlight python %}

# .data to csv converter

import csv

with open('spambase.data', 'r') as in_file:
    stripped = (line.strip() for line in in_file)
    lines = (line.split(",") for line in stripped if line)
    with open('data.csv', 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(('(wf) make', '(wf) address', '(wf) all', '(wf) 3d', '(wf) our', '(wf) over', '(wf) remove', '(wf) internet', '(wf) order', '(wf) mail', '(wf) receive', '(wf) will', '(wf) people', '(wf) report', '(wf) addresses', '(wf) free', '(wf) business', '(wf) email', '(wf) you', '(wf) credit', '(wf) your', '(wf) font', '(wf) 000', '(wf) money', '(wf) hp', '(wf) hpl', '(wf) george', '(wf) 650', '(wf) lab', '(wf) labs', '(wf) telnet', '(wf) 857', '(wf) data', '(wf) 415', '(wf) 85', '(wf) technology', '(wf) 1999', '(wf) parts', '(wf) pm', '(wf) direct', '(wf) cs', '(wf) meeting', '(wf) original', '(wf) project', '(wf) re', '(wf) edu', '(wf) table', '(wf) conference', '(cf) ;', '(cf) )', '(cf) [', '(сf) !', 'cf_dollar', '(cf) #', 'capital_run_length_average','capital_run_length_longest','capital_run_length_total',  'spam (y)'))
        writer.writerows(lines)


{% endhighlight python %}

First of all, we need to import all Python dependencies that we're going to use for this algorithm's implementation. Henceforth, I'll be using Jupyter Notebook for better representation.

{% highlight python %}

import pandas as pd
import matplotlib.pyplot as plt
plt.rc("font", size=16)
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

{% endhighlight python %}


{% highlight python %}

data = pd.read_csv('data.csv', header=0)
data = data.dropna()
print(data.shape)

{% endhighlight python %}

![data image](https://akinariobi.github.io/assets/img/classification-using-binary-logistic-regression/1.png)
*fig. 1*

As displayed at fig. 1 there's 4601 rows and 58 columns in this file. Note that it's recommended to use higher amount of data than that mentioned in order to get more precise results.

Let's explore the data what we have.

{% highlight python %}

# get number of spam and non-spam emails
data['spam'].value_counts()

# get dependence between 'spam' variable and categorical attributes
data.groupby('spam').mean()

# plot a graph
sns.countplot(x='spam', data=data)
plt.show()

{% endhighlight python %}

![data image1](https://akinariobi.github.io/assets/img/classification-using-binary-logistic-regression/2.png)
![data image2](https://akinariobi.github.io/assets/img/classification-using-binary-logistic-regression/4.png)
![data image3](https://akinariobi.github.io/assets/img/classification-using-binary-logistic-regression/3.png)
*fig. 2*

As image above and [spambase documentations][datadoc] says there's 1813 (39.4%) spam and 2788 (60.6%) non-spam messages in this database so the ratio of spam to non-spam instances is 39:61.

Now we can calculate categorical means for other categorical variables such as frequency of some words and characters presented in emails to find all attributes that has a greatest impact on spam detection result.

![data image8](https://akinariobi.github.io/assets/img/classification-using-binary-logistic-regression/9.png)
![data image4](https://akinariobi.github.io/assets/img/classification-using-binary-logistic-regression/5.png)
![data image5](https://akinariobi.github.io/assets/img/classification-using-binary-logistic-regression/6.png)
![data image6](https://akinariobi.github.io/assets/img/classification-using-binary-logistic-regression/7.png)
![data image7](https://akinariobi.github.io/assets/img/classification-using-binary-logistic-regression/8.png)
*fig. 3*

I visually selected three parameters: **word_freq_credit**, **word_freq_order** and **word_freq_free** to analyze picked emails, so I've already checked (using matplotlib) that these attributes may be a good predictors of the outcome variable. In this article I use a data set consisted of 57 features but what if there's 557 different features? In this case, it becomes impossible to manually analyze that high amount of data. So, we can get out of such situation using Recursive Feature Elimination (hereinafter RFE) algorithm. RFE as its title suggests recursively removes features, builds a model using the remaining attributes and calculates model accuracy. RFE is able to work out the combination of attributes that contribute to the prediction on the target variable.


{% highlight python %}

# implementing RFE

from sklearn.cross_validation import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

X = data.loc[:, data.columns != 'spam']
SPAM = data.loc[:, data.columns == 'spam']
X_train, X_test, SPAM_train, SPAM_test = train_test_split(X, SPAM, test_size=0.4, random_state=0)
columns = X_train.columns
sample = SMOTE(random_state=0)
sample_data_X,sample_data_SPAM=sample.fit_sample(X_train,SPAM_train)
sample_data_X = pd.DataFrame(data=sample_data_X,columns=columns)
sample_data_SPAM = pd.DataFrame(data=sample_data_SPAM,columns=['spam'])

data_values=data.columns.values.tolist()
spam = ['spam']
X = [i for i in data_values if i not in y]
logreg = LogisticRegression()
rfe = RFE(logreg, 20)
rfe = rfe.fit(sample_data_X, sample_data_SPAM.values.ravel())

{% endhighlight python %}

{% highlight python %}

# RFE output

>>> rfe.ranking_
array([25, 26, 36,  1, 10, 18,  1, 17,  1, 30,  7, 27, 31, 24,  1,  1,  1,
       32, 29,  1, 21, 22,  1,  9,  1,  4,  1, 20,  1,  5, 12, 35,  1, 15,
        1,  3, 38, 19,  2, 16,  1,  1,  6,  1, 13,  1, 11,  1,  8, 28, 14,
       23,  1,  1, 33, 34, 37])

>>> rfe.support_
array([False, False, False,  True, False, False,  True, False,  True,
       False, False, False, False, False,  True,  True,  True, False,
       False,  True, False, False,  True, False,  True, False,  True,
       False,  True, False, False, False,  True, False,  True, False,
       False, False, False, False,  True,  True, False,  True, False,
        True, False,  True, False, False, False, False,  True,  True,
       False, False, False])

{% endhighlight python %}

Let's write few lines of code to display names of selected columns

{% highlight python %}

for i in range(len(rfe.support_)):
	res = list()
  # removing 'spam' column
	data_col = list(data.columns)[0:-1]
	if list(rfe.support_)[i] == True:
		print(data_col[i])


(wf) 3d
(wf) remove
(wf) order
(wf) addresses
(wf) free
(wf) business
(wf) credit
(wf) 000
(wf) hp
(wf) george
(wf) lab
(wf) data
(wf) 85
(wf) cs
(wf) meeting
(wf) project
(wf) edu
(wf) conference
cf_dollar
(cf) #

{% endhighlight python %}

It's time to update X and SPAM variables.

{% highlight python %}

# lol sorry George

features = ["(wf) 3d", "(wf) remove", "(wf) order", "(wf) addresses", "(wf) free", "(wf) business", "(wf) credit", "(wf) 000", "(wf) hp", "(wf) george", "(wf) lab", "(wf) data", "(wf) 85", "(wf) cs", "(wf) meeting","(wf) project", "(wf) edu", "(wf) conference", "cf_dollar", "(cf) #"]
X=sample_data_X[features]
SPAM=sample_data_SPAM['spam']

{% endhighlight python %}

Now we're ready to implement our logistic model *Logit* from statsmodels API.

{% highlight python %}

import statsmodels.api as sm

logit_model=sm.Logit(SPAM,X)
res=logit_model.fit()
print(res.summary2())

{% endhighlight python %}

{% highlight js %}

Results: Logit
==========================================================================
Model:                  Logit               Pseudo R-squared:    0.662    
Dependent Variable:     spam                AIC:                 1625.6112
Date:                   2018-09-16 23:49    BIC:                 1748.1357
No. Observations:       3382                Log-Likelihood:      -792.81  
Df Model:               19                  LL-Null:             -2344.2  
Df Residuals:           3362                LLR p-value:         0.0000   
Converged:              0.0000              Scale:               1.0000   
No. Iterations:         35.0000                                           
--------------------------------------------------------------------------
Coef.              Std.Err.     z             P>|z|     [0.025     0.975]  
--------------------------------------------------------------------------
(wf) 3d            3.1046     1.4984  2.0720 0.0383      0.1679     6.0414
(wf) remove        3.3628     0.4971  6.7652 0.0000      2.3885     4.3370
(wf) order         1.7641     0.3944  4.4734 0.0000      0.9912     2.5370
(wf) addresses     1.4784     0.8312  1.7787 0.0753     -0.1506     3.1075
(wf) free          1.1550     0.1779  6.4940 0.0000      0.8064     1.5036
(wf) business      1.0667     0.2501  4.2655 0.0000      0.5765     1.5568
(wf) credit        2.4684     0.8346  2.9576 0.0031      0.8327     4.1042
(wf) 000           3.4418     0.6699  5.1381 0.0000      2.1289     4.7547
(wf) hp           -2.4012     0.2715 -8.8426 0.0000     -2.9334    -1.8690
(wf) george      -19.2130     3.5327 -5.4385 0.0000    -26.1371   -12.2890
(wf) lab          -3.3957     1.7367 -1.9553 0.0506     -6.7996     0.0082
(wf) data         -2.0037     0.5618 -3.5663 0.0004     -3.1049    -0.9025
(wf) 85           -4.7539     1.4372 -3.3077 0.0009     -7.5708    -1.9370
(wf) cs         -314.8094 17685.6504 -0.0178 0.9858 -34978.0474 34348.4285
(wf) meeting      -2.5716     0.7676 -3.3501 0.0008     -4.0761    -1.0671
(wf) project      -1.7874     0.6351 -2.8142 0.0049     -3.0322    -0.5425
(wf) edu          -2.5245     0.3690 -6.8421 0.0000     -3.2476    -1.8013
(wf) conference   -4.1998     1.8731 -2.2422 0.0250     -7.8711    -0.5286
cf_dollar          5.7654     0.8386  6.8754 0.0000      4.1219     7.4089
(cf) #             1.2938     0.5791  2.2343 0.0255      0.1589     2.4288
==========================================================================
{% endhighlight js %}

#### optimization time

The *P* value is bigger than 0.05 for *(wf) cs*, *(wf) lab*, *(wf) adresses*, so let's remove them. Now take a look at standard error column (Std. Err.): George's standard deviation is too high so let's remove *(wf) George* from our list and let's also remove all features with standard deviation bigger than modulo 3.

**Desired Output:**

*Optimization terminated successfully.
Current function value: 0.382317
Iterations 10*

{% highlight js %}

Results: Logit
================================================================
Model:              Logit            Pseudo R-squared: 0.448    
Dependent Variable: spam             AIC:              2605.9947
Date:               2018-09-17 00:15 BIC:              2667.2569
No. Observations:   3382             Log-Likelihood:   -1293.0  
Df Model:           9                LL-Null:          -2344.2  
Df Residuals:       3372             LLR p-value:      0.0000   
Converged:          1.0000           Scale:            1.0000   
No. Iterations:     10.0000                                     
----------------------------------------------------------------
Coef.  Std.Err.    z     P>|z|   [0.025  0.975]
----------------------------------------------------------------
(wf) order       2.1720   0.2863   7.5853 0.0000  1.6108  2.7333
(wf) free        1.1694   0.1232   9.4903 0.0000  0.9279  1.4109
(wf) business    2.5323   0.2759   9.1771 0.0000  1.9915  3.0732
(wf) credit      4.9089   0.8423   5.8282 0.0000  3.2581  6.5597
(wf) hp         -3.0963   0.2739 -11.3050 0.0000 -3.6331 -2.5595
(wf) data       -2.5648   0.5032  -5.0973 0.0000 -3.5510 -1.5786
(wf) meeting    -3.3762   0.7573  -4.4582 0.0000 -4.8605 -1.8919
(wf) project    -2.3055   0.5391  -4.2766 0.0000 -3.3621 -1.2489
(wf) edu        -2.6597   0.3339  -7.9652 0.0000 -3.3142 -2.0052
(cf) #           2.7805   0.5914   4.7015 0.0000  1.6214  3.9397
================================================================       
{% endhighlight js %}

Looks a bit better now.

**Fitting Logistic Regression Model**

{% highlight python %}
# our data fits logistic regression model

>>> from sklearn.linear_model import LogisticRegression
>>> from sklearn import metrics
>>> X_train, X_test, Z_train, Z_test = train_test_split(X, Z, test_size=0.3, random_state=0)
>>> logreg = LogisticRegression()
>>> logreg.fit(X_train, Z_train)
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)

# cheking an accuracy of logistic regression classifier
>>> logreg.score(X_test, Z_test)
0.7763546798029557

{% endhighlight python %}

To conclude we can count the number of correct and incorrect predictions using confusion matrix from *sklearn.metrics* library.

{% highlight python %}

>>> from sklearn.metrics import confusion_matrix
>>> confusion_matrix = confusion_matrix(Z_test, spam_pred)
>>> confusion_matrix
array([[444,  71],
       [156, 344]])
>>> correct = confusion_matrix[0][0]+confusion_matrix[1][1]
>>> correct
788
>>> incorrect = confusion_matrix[0][1]+confusion_matrix[1][0]
>>> incorrect
227

{% endhighlight python %}

Thus, we've got 788 correct and 227 incorrect predictions and built a probabilistic model. Perfect!

[s-shaped]: http://blog.datumbox.com/wp-content/uploads/2013/11/multinomial-logistic-regression.png
[data]: http://archive.ics.uci.edu/ml/index.php
[dataset]: https://archive.ics.uci.edu/ml/datasets/spambase
[datadoc]: https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.DOCUMENTATION
