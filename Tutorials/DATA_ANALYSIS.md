# Exploratory Data Analysis

### Import & Setup

```python
    script_dir = os.path.dirname( __file__ )
    mymodule_dir = os.path.join( script_dir, '..', 'EDA' )
    sys.path.append( mymodule_dir )
    from data_analysis import Analysis
    eda = Analysis()
```

**Function** analysis():

<table border="2" cellpadding="2" cellspacing="2" width="100%">
    <tr><th width="20%">Input</th>
    <th width="20%">Type</th>
    <th>Description</th></tr>
    <tr><td style="text-align:center">dataset</td>
    <td style="text-align:center"><i>pandas.dataframe</i></td>
    <td style="text-align:center">Model dataset in dataframe</td></tr>
    <tr><td style="text-align:center">EDA_type</td>
    <td style="text-align:center"><i>string</i></td>
    <td style="text-align:center">name of EDA required<br><i>Default = "dataprep" | </i><br>
    "profiling"<br>"sweetviz"</td></tr>
</table>

**Returns** <i>eda.EDA_type_analysis() --> HTML</i>

### Usage

```python
    analysis(df, "sweetviz")
```   

Imports Analysis().dataprep_analysis from /pybanking/src/pybanking/EDA/data_analysis.py

<hr><br>

## DATAPREP

**Function** dataprep_analysis():

<table border="2" cellpadding="2" cellspacing="2" width="100%">
    <tr><th width="20%">Input</th>
    <th width="20%">Type</th>
    <th>Description</th></tr>
    <tr><td style="text-align:center">dataset</td>
    <td style="text-align:center"><i>pandas.dataframe</i></td>
    <td style="text-align:center">Model dataset in dataframe</td></tr>
</table>

**Returns** <i>HTML file in browser</i>

### Usage

```python
    da.dataprep_analysis(df)
```   
<a href="pybanking/gitbook_data/DataPrep Report.html">Dataprep output report HTML</a>

<a href="pybanking/gitbook_data/DataPrep Plot.html">Dataprep output plot HTML</a>

<a href="https://docs.dataprep.ai/user_guide/eda/introduction.html#userguide-eda">Further Documentation on Dataprep</a>

<hr><br>

## SWEETVIZ

**Function** sweetviz_analysis():

<table border="2" cellpadding="2" cellspacing="2" width="100%">
    <tr><th width="20%">Input</th>
    <th width="20%">Type</th>
    <th>Description</th></tr>
    <tr><td style="text-align:center">dataset</td>
    <td style="text-align:center"><i>pandas.dataframe</i></td>
    <td style="text-align:center">Model dataset in dataframe</td></tr>
</table>

**Returns** <i>HTML file in browser</i>

### Usage

```python
    da.sweetviz_analysis(df)
```   

<a href="pybanking/gitbook_data/SWEETVIZ_REPORT.html">Sweetviz output HTML</a>

<a href="https://github.com/fbdesignpro/sweetviz">Further Documentation on Sweetviz</a>

<hr><br>

## PANDAS PROFILING

**Function** pandas_analysis():

<table border="2" cellpadding="2" cellspacing="2" width="100%">
    <tr><th width="20%">Input</th>
    <th width="20%">Type</th>
    <th>Description</th></tr>
    <tr><td style="text-align:center">dataset</td>
    <td style="text-align:center"><i>pandas.dataframe</i></td>
    <td style="text-align:center">Model dataset in dataframe</td></tr>
</table>

**Returns** <i>HTML file in browser</i>

### Usage

```python
    da.pandas_analysis(df)
```   

<a href="pybanking/gitbook_data/Pandas Profiling Report.html">Pandas profiling output HTML</a>

<a href="https://github.com/ydataai/pandas-profiling">Further Documentation on Pandas Profiling</a>

<hr><br>
