## Customer Churn Model

**Function** get_data():

<table border="2" cellpadding="2" cellspacing="2" width="100%">
    <tr><th width="20%">Arguement</th>
    <th width="20%">Type</th>
    <th>Description</th></tr>
    <tr><td style="text-align:center">url</td>
    <td style="text-align:center"><i>string</i></td>
    <td style="text-align:center">URL of a CSV file</td></tr>
</table>

**Returns** *pandas.dataframe*

### Usage

```python
    df = model_churn.get_data(url = 'https://raw.githubusercontent.com/../BankChurners.csv')
```   

<hr><br>

**Function** preprocess_inputs():

<table border="2" cellpadding="2" cellspacing="2" width="100%">
    <tr><th width="20%">Input</th>
    <th width="20%">Type</th>
    <th>Description</th></tr>
    <tr><td style="text-align:center">dataset</td>
    <td style="text-align:center">*pandas.dataframe*</td>
    <td style="text-align:center">Model dataset in dataframe</td></tr>
    <tr><td style="text-align:center">model_name</td>
    <td style="text-align:center"><i>string</i></td>
    <td style="text-align:center">Model name as a string <br><i>Default = "Logistic_Regression" | </i><br>
    "Support_Vector_Machine"<br>"Support_Vector_Machine_Optimized"<br>"Decision_Tree"<br>"Neural_Network"<br>
    "Random_Forest"</td></tr>
</table>

**Returns** *pandas.dataframe*, *pandas.dataframe*

### Usage

```python
    X, y = preprocess_inputs(df, "Neural_Network")
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
    model = pretrained("Neural_Network")
```   

<hr><br>

**Function** train():

<table border="2" cellpadding="2" cellspacing="2" width="100%">
    <tr><th width="20%">Input</th>
    <th width="20%">Type</th>
    <th>Description</th></tr>
    <tr><td style="text-align:center">dataset</td>
    <td style="text-align:center">*pandas.dataframe*</td>
    <td style="text-align:center">New training dataset in dataframe</td></tr>
    <tr><td style="text-align:center">model_name</td>
    <td style="text-align:center"><i>string</i></td>
    <td style="text-align:center">Model name as a string <br><i>Default = "Logistic_Regression" | </i><br>
    "Support_Vector_Machine"<br>"Support_Vector_Machine_Optimized"<br>"Decision_Tree"<br>"Neural_Network"<br>
    "Random_Forest"</td></tr>
</table>

**Returns** model

### Usage

```python
    model = train(df, "Neural_Network")
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
