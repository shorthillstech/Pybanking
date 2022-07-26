## Bank Deposits Prediction Model

**Function** get_data():

<table border="2" cellpadding="2" cellspacing="2" width="100%">
    <tr><th width="20%">Input</th>
    <th width="20%">Type</th>
    <th>Description</th></tr>
    <tr><td style="text-align:center">training dataset</td>
    <td style="text-align:center">url<<i>string</i>></td>
    <td style="text-align:center">URL of a training CSV file</td></tr>
    <tr><td style="text-align:center">testing dataset</td>
    <td style="text-align:center">url<<i>string</i>></td>
    <td style="text-align:center">URL of a testing CSV file</td></tr>
</table>

**Returns** *pandas.dataframe*, *pandas.dataframe*

### Usage

```python
    data_train, data_test = get_data(train = 'https://raw.githubusercontent.com/../banking_dataset_train.csv', test = 'https://raw.githubusercontent.com/../banking_dataset_test.csv')
```   

<hr><br>

**Function** preprocess_inputs():

<table border="2" cellpadding="2" cellspacing="2" width="100%">
    <tr><th width="20%">Input</th>
    <th width="20%">Type</th>
    <th>Description</th></tr>
    <tr><td style="text-align:center">training dataset</td>
    <td style="text-align:center">*pandas.dataframe*</td>
    <td style="text-align:center">Model training dataset in dataframe</td></tr>
    <tr><td style="text-align:center">testing dataset</td>
    <td style="text-align:center">*pandas.dataframe*</td>
    <td style="text-align:center">Model testing dataset in dataframe</td></tr>
    <tr><td style="text-align:center">model_name</td>
    <td style="text-align:center"><i>string</i></td>
    <td style="text-align:center">Model name as a string <br><i>Default = "Logistic_Regression" | </i><br>
    "Support_Vector_Machine"<br>"Support_Vector_Machine_Optimized"<br>"Decision_Tree"<br>"Neural_Network"<br>
    "Random_Forest"</td></tr>
</table>

**Returns** *pandas.dataframe*, *pandas.dataframe*

### Usage

```python
    X, y = preprocess_inputs(data_train, data_test, "Random_Forest")
```   

<hr><br>

**Function** pretrained():

<table border="2" cellpadding="2" cellspacing="2" width="100%">
    <tr><th width="20%">Input</th>
    <th width="20%">Type</th>
    <th>Description</th></tr>
    <tr><td style="text-align:center">model_name</td>
    <td style="text-align:center"><i>string</i></td>
    <td style="text-align:center">Model name as a string <br><i>Default = "Logistic_Regression" | </i><br>
    "Support_Vector_Machine"<br>"Support_Vector_Machine_Optimized"<br>"Decision_Tree"<br>"Neural_Network"<br>
    "Random_Forest"</td></tr>
</table>

**Returns** model

### Usage

```python
    model = pretrained("Random_Forest")
```   

<hr><br>

**Function** train():

<table border="2" cellpadding="2" cellspacing="2" width="100%">
    <tr><th width="20%">Input</th>
    <th width="20%">Type</th>
    <th>Description</th></tr>
    <tr><td style="text-align:center">training dataset</td>
    <td style="text-align:center">*pandas.dataframe*</td>
    <td style="text-align:center">Model training dataset in dataframe</td></tr>
    <tr><td style="text-align:center">testing dataset</td>
    <td style="text-align:center">*pandas.dataframe*</td>
    <td style="text-align:center">Model testing dataset in dataframe</td></tr>
    <tr><td style="text-align:center">model_name</td>
    <td style="text-align:center"><i>string</i></td>
    <td style="text-align:center">Model name as a string <br><i>Default = "Logistic_Regression" | </i><br>
    "Support_Vector_Machine"<br>"Support_Vector_Machine_Optimized"<br>"Decision_Tree"<br>"Neural_Network"<br>
    "Random_Forest"</td></tr>
</table>

**Returns** model

### Usage

```python
    model = train(data_train, data_test, "Random_Forest")
```

<hr><br>

**Function** predict():

<table border="2" cellpadding="2" cellspacing="2" width="100%">
    <tr><th width="20%">Input</th>
    <th width="20%">Type</th>
    <th>Description</th></tr>
    <tr><td style="text-align:center">test dataset</td>
    <td style="text-align:center">*pandas.dataframe*</td>
    <td style="text-align:center">Model test dataset in dataframe</td></tr>
    <tr><td style="text-align:center">model</td>
    <td style="text-align:center">function()</td>
    <td style="text-align:center">Model function from pretrained / train</td></tr>
</table>

**Returns** <i>array</i>

### Usage

```python
    print(predict(X_test, model))
```

<hr><br>
