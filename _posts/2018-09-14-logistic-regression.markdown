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

Let's build a logistic regression in Python using [Spambase Data Set][dataset] data set from [UC Irvine Machine Learning Repository][data].

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

Spam (y): The given email is spam or not? (1) -- yes, (0) -- no.

Note that as mentioned in [documentation][datadoc] the "spam" concept is diverse: advertisements for products/web sites, make money fast schemes, chain letters, pornography...

{% highlight js %}
Our collection of spam e-mails came from our postmaster and individuals who had filed spam. Our collection of non-spam e-mails came from filed work and personal e-mails, and hence the word 'george' and the area code '650' are indicators of non-spam. These are useful when constructing a personalized spam filter. One would either have to blind such non-spam indicators or get a very wide collection of non-spam to generate a general purpose spam filter.
{% endhighlight js %}

I converted .data to CSV so as to get a better visualization of each operation I'm doing.  

*Note: wf - word frequency, cf - character frequency*

{% highlight python %}

# .data to csv converter

import csv

with open('spambase.data', 'r') as in_file:
    stripped = (line.strip() for line in in_file)
    lines = (line.split(",") for line in stripped if line)
    with open('data.csv', 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(('(wf) make', '(wf) address', '(wf) all', '(wf) 3d', '(wf) our', '(wf) over', '(wf) remove', '(wf) internet', '(wf) order', '(wf) mail', '(wf) receive', '(wf) will', '(wf) people', '(wf) report', '(wf) addresses', '(wf) free', '(wf) business', '(wf) email', '(wf) you', '(wf) credit', '(wf) your', '(wf) font', '(wf) 000', '(wf) money', '(wf) hp', '(wf) hpl', '(wf) george', '(wf) 650', '(wf) lab', '(wf) labs', '(wf) telnet', '(wf) 857', '(wf) data', '(wf) 415', '(wf) 85', '(wf) technology', '(wf) 1999', '(wf) parts', '(wf) pm', '(wf) direct', '(wf) cs', '(wf) meeting', '(wf) original', '(wf) project', '(wf) re', '(wf) edu', '(wf) table', '(wf) conference', '(cf) ;', '(cf) )', '(cf) [', '(сf) !', '(cf) $', '(cf) #', 'capital_run_length_average','capital_run_length_longest','capital_run_length_total',  'spam (y)'))
        writer.writerows(lines)


{% endhighlight python %}

First of all, we need to import all python dependencies. I'm also using Jupyter Notebook for better representation.

{% highlight python %}

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

{% endhighlight python %}

![data image](https://akinariobi.github.io/assets/img/classification-using-binary-logistic-regression/1.png)
*fig. 1*

As displayed at fig. 1 there's 4601 rows and 58 columns in this file. Note that it's recommended to use higher amount of data than that mentioned in order to get more precise results.

Let's explore the data what we have.

![data image1](https://akinariobi.github.io/assets/img/classification-using-binary-logistic-regression/2.png)
![data image2](https://akinariobi.github.io/assets/img/classification-using-binary-logistic-regression/4.png)
![data image3](https://akinariobi.github.io/assets/img/classification-using-binary-logistic-regression/3.png)
*fig. 2*

As image above and [spambase documentations][datadoc] says there's 1813 (39.4%) spam and 2788 (60.6%) non-spam messages in this database so the ratio of spam to non-spam instances is 39:61.

[s-shaped]: http://blog.datumbox.com/wp-content/uploads/2013/11/multinomial-logistic-regression.png
[data]: http://archive.ics.uci.edu/ml/index.php
[dataset]: https://archive.ics.uci.edu/ml/datasets/spambase
[datadoc]: https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.DOCUMENTATION
