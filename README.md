# classification-with-julia

We will be using the Bank Marketing dataset at https://archive.ics.uci.edu/ml/datasets/Bank+Marketing to predict if the client will subscribe a term deposit (variable y) ...

We will apply three different classifiers using **Julia** programming language.

The classifiers we will fit are; **Logistic Regression** from **Knet** package, and **Decision Tree** and **Random Forest** from **DecisionTree** package ...

``` bash
julia Classifiers_Script.jl

# Data Description
21×8 DataFrames.DataFrame

# │ Row │ variable       │ mean      │ min      │ median │ max       │ nunique │ nmissing │ eltype  │

# ├─────┼────────────────┼───────────┼──────────┼────────┼───────────┼─────────┼──────────┼─────────┤

# │ 1   │ age            │ 40.1136   │ 18       │ 38.0   │ 88        │         │ 0        │ Int64   │

# │ 2   │ job            │           │ admin.   │        │ unknown   │ 12      │ 0        │ String  │

# │ 3   │ marital        │           │ divorced │        │ unknown   │ 4       │ 0        │ String  │

# │ 4   │ education      │           │ basic.4y │        │ unknown   │ 8       │ 0        │ String  │

# │ 5   │ default        │           │ no       │        │ yes       │ 3       │ 0        │ String  │

# │ 6   │ housing        │           │ no       │        │ yes       │ 3       │ 0        │ String  │

# │ 7   │ loan           │           │ no       │        │ yes       │ 3       │ 0        │ String  │

# │ 8   │ contact        │           │ cellular │        │ telephone │ 2       │ 0        │ String  │

# │ 9   │ month          │           │ apr      │        │ sep       │ 10      │ 0        │ String  │

# │ 10  │ day_of_week    │           │ fri      │        │ wed       │ 5       │ 0        │ String  │

# │ 11  │ duration       │ 256.788   │ 0        │ 181.0  │ 3643      │         │ 0        │ Int64   │

# │ 12  │ campaign       │ 2.53727   │ 1        │ 2.0    │ 35        │         │ 0        │ Int64   │

# │ 13  │ pdays          │ 960.422   │ 0        │ 999.0  │ 999       │         │ 0        │ Int64   │

# │ 14  │ previous       │ 0.190337  │ 0        │ 0.0    │ 6         │         │ 0        │ Int64   │

# │ 15  │ poutcome       │           │ failure  │        │ success   │ 3       │ 0        │ String  │

# │ 16  │ emp_var_rate   │ 0.0849721 │ -3.4     │ 1.1    │ 1.4       │         │ 0        │ Float64 │

# │ 17  │ cons_price_idx │ 93.5797   │ 92.201   │ 93.749 │ 94.767    │         │ 0        │ Float64 │

# │ 18  │ cons_conf_idx  │ -40.4991  │ -50.8    │ -41.8  │ -26.9     │         │ 0        │ Float64 │

# │ 19  │ euribor3m      │ 3.62136   │ 0.635    │ 4.857  │ 5.045     │         │ 0        │ Float64 │

# │ 20  │ nr_employed    │ 5166.48   │ 4963.6   │ 5191.0 │ 5228.1    │         │ 0        │ Float64 │

# │ 21  │ y              │           │ no       │        │ yes       │ 2       │ 0        │ String  │

########## Knet Logistic Regression ##############

# Encoded data size
(4119, 66)

# Check that both data are of the same distribution
0.11087132 , 0.097087376

# size of train and test data
(65, 3707)(65, 412)

# accuracy before training; random accuracy
0.2354368932038835

# accuracy in train and test data
1.0
1.0

############### Decision Tree ####################

# Model Classes
String["0.0", "1.0"]

# Model accuracy on test data
1.0

######### Random Forest #########

# Random Forest using 5 random features, 10 trees, 0.7 portion of samples per tree, and a maximum tree depth of 6
# Model accuracy on test data
0.9757281553398058
```
