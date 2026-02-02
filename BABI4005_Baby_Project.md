# About the Dataset

This dataset was found on Kaggle: https://www.kaggle.com/datasets/bismasajjad/womens-representation-in-global-stem-education

This dataset provides information from top countries (China, Canada, the US, Germany, India, and Australia) on Female Enrollment and Graduation in four major STEM Fields (Biology, Mathematics, Computer Science, and Engineering). 

It covers Graduation Rate, Enrollment Rate, Gender Gap Index, Year, Country and STEM Field

## Import Packages 


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
```

## Loading in the Data


```python
## Loading the customer data
## set the dataframe

df = pd.read_csv("women_in_stem.csv")

## Top 5 rows of dataframe
df.head()
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
      <th>Country</th>
      <th>Year</th>
      <th>Female Enrollment (%)</th>
      <th>Female Graduation Rate (%)</th>
      <th>STEM Fields</th>
      <th>Gender Gap Index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>China</td>
      <td>2018</td>
      <td>20.4</td>
      <td>43.2</td>
      <td>Engineering</td>
      <td>0.52</td>
    </tr>
    <tr>
      <th>1</th>
      <td>China</td>
      <td>2005</td>
      <td>35.6</td>
      <td>29.3</td>
      <td>Mathematics</td>
      <td>0.98</td>
    </tr>
    <tr>
      <th>2</th>
      <td>China</td>
      <td>2005</td>
      <td>53.7</td>
      <td>32.4</td>
      <td>Biology</td>
      <td>0.60</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Germany</td>
      <td>2007</td>
      <td>65.0</td>
      <td>63.6</td>
      <td>Mathematics</td>
      <td>0.69</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Canada</td>
      <td>2010</td>
      <td>54.4</td>
      <td>28.8</td>
      <td>Engineering</td>
      <td>0.74</td>
    </tr>
  </tbody>
</table>
</div>



## Checking dataset structure

#### Column and row count


```python
df.shape
```




    (500, 6)



- This dataset contains 500 rows, 6 columns

#### Column Check


```python
df.columns
```




    Index(['Country', 'Year', 'Female Enrollment (%)',
           'Female Graduation Rate (%)', 'STEM Fields', 'Gender Gap Index'],
          dtype='object')



- The columns include data on Country, Year, Female Enrollment (%), Female Graduation Rate (%), STEM Fields, Gender Gap Index

#### Datatype Check

- Checking the data types of each column
- This is also a validity check; ensuring that columns have the correct datatype based off the info recorded


```python
df.dtypes
```




    Country                        object
    Year                            int64
    Female Enrollment (%)         float64
    Female Graduation Rate (%)    float64
    STEM Fields                    object
    Gender Gap Index              float64
    dtype: object



#### Dataset characteristics


```python
df.info ## more look into the data
```




    <bound method DataFrame.info of        Country  Year  Female Enrollment (%)  Female Graduation Rate (%)  \
    0        China  2018                   20.4                        43.2   
    1        China  2005                   35.6                        29.3   
    2        China  2005                   53.7                        32.4   
    3      Germany  2007                   65.0                        63.6   
    4       Canada  2010                   54.4                        28.8   
    ..         ...   ...                    ...                         ...   
    495  Australia  2016                   37.2                        50.1   
    496      India  2010                   59.1                        61.2   
    497    Germany  2010                   34.7                        34.2   
    498  Australia  2011                   61.2                        39.9   
    499      India  2008                   27.6                        49.1   
    
              STEM Fields  Gender Gap Index  
    0         Engineering              0.52  
    1         Mathematics              0.98  
    2             Biology              0.60  
    3         Mathematics              0.69  
    4         Engineering              0.74  
    ..                ...               ...  
    495           Biology              0.91  
    496       Engineering              0.66  
    497       Engineering              0.92  
    498       Mathematics              0.85  
    499  Computer Science              0.54  
    
    [500 rows x 6 columns]>




```python
df.describe() ## only includes numeric columns
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
      <th>Year</th>
      <th>Female Enrollment (%)</th>
      <th>Female Graduation Rate (%)</th>
      <th>Gender Gap Index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>500.000000</td>
      <td>500.000000</td>
      <td>500.000000</td>
      <td>500.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2011.418000</td>
      <td>43.939800</td>
      <td>36.715200</td>
      <td>0.745980</td>
    </tr>
    <tr>
      <th>std</th>
      <td>7.187112</td>
      <td>14.316864</td>
      <td>15.964231</td>
      <td>0.138183</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2000.000000</td>
      <td>20.100000</td>
      <td>10.100000</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2005.000000</td>
      <td>31.575000</td>
      <td>23.150000</td>
      <td>0.630000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2011.000000</td>
      <td>43.500000</td>
      <td>35.900000</td>
      <td>0.750000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2018.000000</td>
      <td>56.500000</td>
      <td>50.425000</td>
      <td>0.860000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2023.000000</td>
      <td>69.500000</td>
      <td>64.900000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



## Data Quality Checks

#### Completeness Check
- Missing value (null) check


```python
## Checking for NaN's --> summing for each column
df.isna().sum()
```




    Country                       0
    Year                          0
    Female Enrollment (%)         0
    Female Graduation Rate (%)    0
    STEM Fields                   0
    Gender Gap Index              0
    dtype: int64



- This dataset shows no null/missing values across all columns and rows
- Completeness doesnt seem to be an issue; However, this doesnt conclude the dataset is extending across all data quality dimensions 

#### Uniqueness Check
- Checking if values are unique (and/or unique when supposed to be)


```python
df.duplicated().sum() ## duplicate check
```




    np.int64(0)



- No duplicated rows in this data

#### Validity Check

- Checking percent rows if there are numbers outside the range of 0-100%
- Checking index column values are between 0-1


```python
## Defining/naming the percent columns
percent_cols = ["Female Enrollment (%)", "Female Graduation Rate (%)"]

for cols in percent_cols:
    out_of_range = df[(df[c] < 0)] | df[(df[c] > 100)] ## checks the data within a specific range
    print(cols, "Out of range rows:", len(out_of_range)) ## prints nicely the info given (provides context)
```

    Female Enrollment (%) Out of range rows: 0
    Female Graduation Rate (%) Out of range rows: 0
    

- No invalid values recorded in this data set
- percent values lower than 0% and higher than 100%


```python
## Checking Gender gap index 

gender_gap_out = df[(df["Gender Gap Index"] < 0) | (df["Gender Gap Index"] > 1)] ## setting requirements (0-1)
print("Gender Gap Index rows out of range:", len(gender_gap_out)) ## printing out rows out of range with context
```

    Gender Gap Index rows out of range: 0
    

- No invalid values inputted in this column 
- Invalid values would include (if present) a gender gap index value outside the range 0-1

#### Consistency Check
- Checking if Female Graduation and Female enrollment are related
- This is a consistency check because I'd assume that enrollment is related with graduation (with exceptions of the assumption that not everyone who enrolls, graduates)
- Checking if theres a reasonable gap


```python
grad_minus_enroll = (df["Female Graduation Rate (%)"] - df["Female Enrollment (%)"])
grad_minus_enroll.describe() ## Describes stats of grad minus enroll 
```




    count    500.000000
    mean      -7.224600
    std       21.270204
    min      -55.100000
    25%      -23.000000
    50%       -7.850000
    75%        7.750000
    max       43.000000
    dtype: float64



##### What this implies
- On average (mean) graduation rate is around 7% lower than enrollment 
- This is normal, as not everyone who enrolls graduates
- 25% of rows have a difference less than or equal to -23%
- This gap is larger, but could imply dropout rates higher in some cases
- The min (greatest negative difference) is at -55.10%
- the max (greatest positive difference) is at 43%
- This could be seen as odd and open for interpretation (could be a difference in how data was collected in a case) 

## Data Summaries

### Describing stats for the whole data set (numeric & non-numeric)


```python
df.describe(include = "all").round(2) ## describe function including numberic and non-numeric columns 
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
      <th>Country</th>
      <th>Year</th>
      <th>Female Enrollment (%)</th>
      <th>Female Graduation Rate (%)</th>
      <th>STEM Fields</th>
      <th>Gender Gap Index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>500</td>
      <td>500.00</td>
      <td>500.00</td>
      <td>500.00</td>
      <td>500</td>
      <td>500.00</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>top</th>
      <td>China</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Mathematics</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>88</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>137</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>NaN</td>
      <td>2011.42</td>
      <td>43.94</td>
      <td>36.72</td>
      <td>NaN</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>std</th>
      <td>NaN</td>
      <td>7.19</td>
      <td>14.32</td>
      <td>15.96</td>
      <td>NaN</td>
      <td>0.14</td>
    </tr>
    <tr>
      <th>min</th>
      <td>NaN</td>
      <td>2000.00</td>
      <td>20.10</td>
      <td>10.10</td>
      <td>NaN</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>NaN</td>
      <td>2005.00</td>
      <td>31.58</td>
      <td>23.15</td>
      <td>NaN</td>
      <td>0.63</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>NaN</td>
      <td>2011.00</td>
      <td>43.50</td>
      <td>35.90</td>
      <td>NaN</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>NaN</td>
      <td>2018.00</td>
      <td>56.50</td>
      <td>50.42</td>
      <td>NaN</td>
      <td>0.86</td>
    </tr>
    <tr>
      <th>max</th>
      <td>NaN</td>
      <td>2023.00</td>
      <td>69.50</td>
      <td>64.90</td>
      <td>NaN</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>



#### Some info to take from this:
- Data includes years 2000-2023, 2011 being the average
- China (top) appears the most at 88 times (freq)
- Mathematics appears the most at 137
- 500 rows of data
- 6 countries, 4 STEM field 

### Enrollment % by Year


```python
df.groupby("Year")["Female Enrollment (%)"].mean() ## calculating average enrollment by year 
```




    Year
    2000    46.563636
    2001    42.673684
    2002    44.142308
    2003    44.939130
    2004    45.962500
    2005    44.640000
    2006    43.286364
    2007    43.633333
    2008    37.168750
    2009    44.783333
    2010    47.318182
    2011    41.562500
    2012    41.940000
    2013    46.246667
    2014    43.294444
    2015    43.595000
    2016    39.369565
    2017    47.466667
    2018    47.275000
    2019    47.513636
    2020    44.894118
    2021    44.887500
    2022    42.644828
    2023    40.627586
    Name: Female Enrollment (%), dtype: float64



#### Some info to take from this:
- Female enrollment averages by year
- Enrollment averages tend to stay in the high 30's and low to mid 40's
- Enrollment is higher from 2017-2019, almost reaching 48%

### Enrollment by STEM Fields


```python
df.groupby("STEM Fields")["Female Enrollment (%)"].mean() ## average enrollment by STEM field
```




    STEM Fields
    Biology             42.929412
    Computer Science    43.082143
    Engineering         44.011364
    Mathematics         45.449635
    Name: Female Enrollment (%), dtype: float64



#### Info to take from this:
- STEM fields involve Mathematics, Computer science, Biology, Engineering
- Biology has the lowest average
- Mathematics has the highest average
- all 4 range in the low to mid 40's

### Enrollment & Graduation Rate by STEM Field


```python
df.groupby("STEM Fields")[["Female Enrollment (%)", "Female Graduation Rate (%)"]].mean().sort_values("Female Enrollment (%)", ascending = False) 
## Sorts female enrollment high-low 
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
      <th>Female Enrollment (%)</th>
      <th>Female Graduation Rate (%)</th>
    </tr>
    <tr>
      <th>STEM Fields</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Mathematics</th>
      <td>45.449635</td>
      <td>35.448175</td>
    </tr>
    <tr>
      <th>Engineering</th>
      <td>44.011364</td>
      <td>38.183333</td>
    </tr>
    <tr>
      <th>Computer Science</th>
      <td>43.082143</td>
      <td>37.461607</td>
    </tr>
    <tr>
      <th>Biology</th>
      <td>42.929412</td>
      <td>35.842857</td>
    </tr>
  </tbody>
</table>
</div>



#### Info to take:
- Mathematics has the highest enrollment rate but the lowest graduation rate, indicating a larger gap
- Engineering has a slightly lower enrollment rate than mathematics but the highest graduation rate
- Computer science has moderate rates in both enrollment and graduation
- Biology has the lowest enrollment rate and a graduation rate of 35.8%
- There are differences across fields but nothing extreme

### Enrollment & Graduation by Country


```python
## grouping by country
df.groupby("Country")[[
    "Female Enrollment (%)",
    "Female Graduation Rate (%)"]].mean().sort_values("Female Enrollment (%)", ascending = False) ## sorting enrollment high-low
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
      <th>Female Enrollment (%)</th>
      <th>Female Graduation Rate (%)</th>
    </tr>
    <tr>
      <th>Country</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>USA</th>
      <td>45.715854</td>
      <td>35.995122</td>
    </tr>
    <tr>
      <th>Canada</th>
      <td>44.955814</td>
      <td>38.072093</td>
    </tr>
    <tr>
      <th>Australia</th>
      <td>44.035632</td>
      <td>37.231034</td>
    </tr>
    <tr>
      <th>China</th>
      <td>43.375000</td>
      <td>37.431818</td>
    </tr>
    <tr>
      <th>India</th>
      <td>42.784146</td>
      <td>35.902439</td>
    </tr>
    <tr>
      <th>Germany</th>
      <td>42.648000</td>
      <td>35.396000</td>
    </tr>
  </tbody>
</table>
</div>



#### Info to take:
- USA has the highest enrollment rate but one of the lowest graduation rates
- Germany and India have the lowest enrollment and graduation rates
- Graduation rates across all countries are lower than the corresponding enrollment rates

#### AI Discolsure
- AI disclosure: In the above codes, I asked ChatGPT for what needs to be added to the code to change the sort of the values displayed (sort_values("Female Enrollment (%)", ascending = False)
- For a cleaner look showing which countries are highest to lowest in Female Enrollment (%)

## Graphs & Visuals

### Bar Graph

#### Average Female Enrollment by STEM Field


```python
enrollment_by_field = df.groupby("STEM Fields")["Female Enrollment (%)"].mean() ## grouping enrollment by field

enrollment_by_field.plot(kind="bar") ## bar chart

## naming and labelling chart
plt.title("Average Female Enrollment by STEM Field")
plt.ylabel("Female Enrollment (%)")
plt.xlabel("STEM Field")

## layout/structuring
plt.xticks(rotation=45) ## rotates the x-axis labels by 45 --> cleaner look
plt.tight_layout() ## adjusting spacing

## display
plt.show()
```


    
![png](output_55_0.png)
    


#### Insights:
- Enrollment is similar across fields (no field has a significantly larger enrollment %)
- Mathematics has the largest enrollment %, indicating higher participation
- Biology has the lowest enrollment rate, but not significantly smaller

### Scatter Plot

#### Scatter plot showing Female Enrollment vs Graduation Rate by Stem field
- scatter plot is coloured, colours represent the STEM fields 
- X axis (female enrollment), y axis (female graduation)


```python
plt.figure(figsize=(8, 6))

## this uses seaborn instead of just matplotlib --> adding colour to show potential relationship
sns.scatterplot(x="Female Enrollment (%)", y="Female Graduation Rate (%)", hue="STEM Fields", data=df, palette="Set2")

## labelling and naming 
plt.title("Female Enrollment vs Graduation Rate by STEM Field")
plt.xlabel("Female Enrollment (%)")
plt.ylabel("Female Graduation Rate (%)")
plt.grid(True)

## shows legend
plt.legend(title="STEM Fields")

## Display
plt.show()
```


    
![png](output_59_0.png)
    


### Insights
- Fields overlap heavily, no field has a clear pattern of enrollment vs graduation
- High enrollment does not consistently lead to higher graduation rates
- Retention patterns spread across all STEM Fields, not significant in one
- Some cases show high enrollment with low graduation rate, this could indicate retention challenges in some cases
- This graph has colour so widely spread, no indication that STEM Field alone explain differences in rates
- This may mean Country or other factors can play in

### Scatter Plot without colour
- x axis (Female Enrollment), y-axis (Female Graduation)


```python
plt.figure(figsize=(8, 6))

## scatter plot 
plt.scatter(df["Female Enrollment (%)"],
    df["Female Graduation Rate (%)"], color = 'green')

## labels and naming
plt.xlabel("Female Enrollment (%)")
plt.ylabel("Female Graduation Rate (%)")
plt.title("Female Enrollment vs Graduation Rate")

## display
plt.show()
```


    
![png](output_62_0.png)
    


### Insights
- Similarly to the scatter with colour, dots are spread throughout with no strong indication that high enrollment does not necessarily lead to higher graduation rates
- Factors beyond enrollment can effect graduation rate
- This distribution can indicate that completion varies widely with country, field and years
- visualizes gaps that were observed in the consistency check between enrollment and graduation rates

### Scatter Plot with Line


```python
## Choosing variables
x = df["Female Enrollment (%)"]
y = df["Female Graduation Rate (%)"]

## using numpy, slope, intercept
## creating a trend line
m, b = np.polyfit(x, y, 1)

## structure
plt.figure(figsize=(8, 6))

## defining labels, names and caracteristics of graph
plt.scatter(x, y, color = 'Green')
plt.plot(x, m*x + b) #3 slope-intercept formula
plt.xlabel("Female Enrollment (%)")
plt.ylabel("Female Graduation Rate (%)")
plt.title("Female Enrollment vs Graduation Rate (with Trend)")

## display
plt.show()
```


    
![png](output_65_0.png)
    


### Insights
- This graph is the same as the scatter plot above (without line)
- Insights remain the same
- The line indicates no strong linear pattern in enrollment vs graduation relationships

### Trend
#### Female Enrollment by Year


```python
## Line graph to show enrollment trend over time

trend = df.groupby("Year")["Female Enrollment (%)"].mean() ## averaging enrollment % and grouping by year

plt.figure()
plt.plot(trend) ## line plot

## Titles and labels
plt.title("Female Enrollment on Average Over Time")
plt.xlabel("Year")
plt.ylabel("Female Enrollment (%)")

## Display
plt.show()
```


    
![png](output_68_0.png)
    


### Insights:
- Average enrollment fluctuates over time, instead of steadily 
- There is a dip in between 2005-2010, indicating lower enrollment rates in this time
- Enrollment rate recovers and reaches peaks from 2015-2020
- After 2020 it starts to decrease slightly

## Additional Analysis

#### Correlation between Grad Rate, Enroll Rate, Gender Gap Index


```python
df[["Female Enrollment (%)", "Female Graduation Rate (%)", "Gender Gap Index"]].corr() 
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
      <th>Female Enrollment (%)</th>
      <th>Female Graduation Rate (%)</th>
      <th>Gender Gap Index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Female Enrollment (%)</th>
      <td>1.000000</td>
      <td>0.016205</td>
      <td>-0.080639</td>
    </tr>
    <tr>
      <th>Female Graduation Rate (%)</th>
      <td>0.016205</td>
      <td>1.000000</td>
      <td>0.027454</td>
    </tr>
    <tr>
      <th>Gender Gap Index</th>
      <td>-0.080639</td>
      <td>0.027454</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Insights
- Enrollment amd graduation rate show very close to 0, showing no linear relationship with each other
- Circles back to how high enrollment does not equal higher graduation
- Gender gap Index and Enrollment have a slight negative relationship, meaning that gender gap index does not strongly relate to enrollment rates
- Overall correlations across these 3 topics are very small (close to 0), meaning no variable strongly impacts the others
