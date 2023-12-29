# -*- coding: utf-8 -*-
"""Complaint Data YTD.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1APuPbtu3yK8vOVOjvJeQr9l3KnSz5May
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
import plotly.express as px

"""Data Reading: From google colab"""

from google.colab import drive
drive.mount('/content/drive')
location ='/content/drive/MyDrive/Colab Notebooks/big-data-Colab/NYPD_Complaint_Data_Current__Year_To_Date_.csv'

#For Dixita
from google.colab import drive
drive.mount('/content/drive')
location ='/content/drive/MyDrive/Final_Project/NYPD_Complaint_Data_Current__Year_To_Date_.csv'

df = pd.read_csv(location)
print(df)

"""Data Structure"""

print(len(df))

df.info()

df.columns

df.dtypes

df.isna().sum()

df.head()

"""Data Cleaning

"""

#Remove unnecessary columns
column_name=['HADEVELOPT','HOUSING_PSA','JURISDICTION_CODE','JURIS_DESC','KY_CD','PARKS_NM','STATION_NAME']
df.drop(columns=column_name, inplace=True)
df

#Feature engineering
#For categorical features, convert all string values to numerical values. For Nan and (null) values, populate as '0'
df = df.replace("(null)", np.NaN)
df = df.fillna(value = 0)
df

#Type Conversion
df['CMPLNT_FR_DT'] = pd.to_datetime(df['CMPLNT_FR_DT'], errors='coerce')

# Remove rows containing NaN values
df = df.dropna(subset=['BORO_NM'])

"""EDA"""

df.OFNS_DESC.value_counts().sort_values(ascending=False).iloc[0:10].plot(kind='bar', color='green', title='Top 10 crimes happening in NYC')

df.LAW_CAT_CD.value_counts().plot(kind='pie', title='Level of offense',autopct='%1.1f%%', startangle=130)

df = df[df['BORO_NM'] != 0]
df.BORO_NM.value_counts().plot(kind='bar', color='brown', title='Number of crimes happening in a Borough')

# Group the data by LAW_CAT_CD and CRM_ATPT_CPTD_CD and count occurrences
grouped_data = df.groupby(['LAW_CAT_CD', 'CRM_ATPT_CPTD_CD']).size().unstack()

# Plot the bar chart
ax = grouped_data.plot(kind='bar', figsize=(10, 6), width=0.7, align='center')

# Customize the chart
ax.set_xlabel('LAW_CAT_CD')
ax.set_ylabel('Count')
ax.set_title('Completed and Attempted Crimes by Law Category')
ax.legend(title='CRM_ATPT_CPTD_CD', loc='upper right')

# Show the chart
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()

# Group the data by LAW_CAT_CD and LOCATION and count occurrences
df_filtered = df[(df['LAW_CAT_CD'] != 0) & (df['LOC_OF_OCCUR_DESC'] != 0)]
grouped_data = df.groupby(['LAW_CAT_CD', 'LOC_OF_OCCUR_DESC']).size().unstack()

# Plot the bar chart
ax = grouped_data.plot(kind='bar', figsize=(10, 6), width=0.7, align='center')

# Customize the chart
ax.set_xlabel('LAW_CAT_CD')
ax.set_ylabel('Count')
ax.set_title('Crimes by Law Category and Location')
ax.legend(title='LOCATION', loc='upper right')

# Show the chart
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()

"""Suspect Analysis"""

# Group the data by SUSP_AGE_GROUP and count occurrences
wanted_age_groups=['<18','18-24','25-44','45-64','65+','UNKNOWN']
dfformat = df[df['SUSP_AGE_GROUP'].isin(wanted_age_groups)]
age_group_counts = dfformat['SUSP_AGE_GROUP'].value_counts().sort_index()

# Create a bar graph
plt.figure(figsize=(10, 6))
age_group_counts.plot(kind='bar', color='skyblue', edgecolor='black')

# Customize the chart
plt.xlabel('Suspect Age Group')
plt.ylabel('Count')
plt.title('Distribution of Suspect Age Groups')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

# Show the chart
plt.tight_layout()
plt.show()

# Group the data by SUSP_SEX and count occurrences
sex_counts = df['SUSP_SEX'].value_counts()

# Create a pie chart
plt.figure(figsize=(5,5))
plt.pie(sex_counts, labels=sex_counts.index, autopct='%1.1f%%', startangle=140)

# Customize the chart
plt.title('Distribution of Suspect Sexes')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

# Show the chart
plt.tight_layout()
plt.show()

# Group the data by SUSP_AGE_GROUP and count occurrences
wanted_age_groups=['WHITE HISPANIC','BLACK','WHITE','BLACK HISPANIC','ASIAN / PACIFIC ISLANDER', 'AMERICAN INDIAN/ALASKAN NATIVE','UNKNOWN']
dfformat = df[df['SUSP_RACE'].isin(wanted_age_groups)]
age_group_counts = dfformat['SUSP_RACE'].value_counts().sort_index()

# Create a bar graph
plt.figure(figsize=(10, 6))
age_group_counts.plot(kind='bar', color='skyblue', edgecolor='black')

# Customize the chart
plt.xlabel('Suspect Race')
plt.ylabel('Count')
plt.title('Distribution of Suspect Race')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

# Show the chart
plt.tight_layout()
plt.show()

"""Victim Analysis"""

# Group the data by VIC_AGE_GROUP and count occurrences
wanted_age_groups=['<18','18-24','25-44','45-64','65+','UNKNOWN']
dfformat = df[df['VIC_AGE_GROUP'].isin(wanted_age_groups)]
age_group_counts = dfformat['VIC_AGE_GROUP'].value_counts().sort_index()

# Create a bar graph
plt.figure(figsize=(10, 6))
age_group_counts.plot(kind='bar', color='skyblue', edgecolor='black')

# Customize the chart
plt.xlabel('Victim Age Group')
plt.ylabel('Count')
plt.title('Distribution of Victim Age Groups')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

# Show the chart
plt.tight_layout()
plt.show()

# Group the data by VIC_SEX and count occurrences
sex_counts = df['VIC_SEX'].value_counts()

# Create a pie chart
plt.figure(figsize=(5,5))
plt.pie(sex_counts, labels=sex_counts.index, autopct='%1.1f%%', startangle=140)

# Customize the chart
plt.title('Distribution of Victim Sexes')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

# Show the chart
plt.tight_layout()
plt.show()


# 'F': Female
# 'M': Male
# 'E': Either or Unknown (used when the gender of the victim is not specified or when it could be either male or female)
# 'L': LGBTQ+ or Lesbian (used to specify the victim's sexual orientation or gender identity)
# 'D': Non-binary or Other Gender Identity (used to specify a gender identity other than male or female)

# Group the data by SUSP_AGE_GROUP and count occurrences
wanted_age_groups=['WHITE HISPANIC','BLACK','WHITE','BLACK HISPANIC','ASIAN / PACIFIC ISLANDER', 'AMERICAN INDIAN/ALASKAN NATIVE','UNKNOWN']
dfformat = df[df['VIC_RACE'].isin(wanted_age_groups)]
age_group_counts = dfformat['VIC_RACE'].value_counts().sort_index()

# Create a bar graph
plt.figure(figsize=(10, 6))
age_group_counts.plot(kind='bar', color='skyblue', edgecolor='black')

# Customize the chart
plt.xlabel('Victim Race')
plt.ylabel('Count')
plt.title('Distribution of Victim Race')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

# Show the chart
plt.tight_layout()
plt.show()

df['VIC_SEX'].value_counts().iloc[:3].plot(kind="bar",  color = '#FF0000', title = 'Sex Crime Victims by Gender', rot=0)

#victims by gender percentage
vic_sex_per = df['VIC_SEX'].value_counts() / df['VIC_SEX'].shape[0] *100
print(vic_sex_per)

df['SUSP_SEX'].value_counts().iloc[:3].plot(kind="bar",  color = '#FF0000', title = 'Sex Crime Suspects by Gender', rot=0)

#victims by gender percentage
susp_sex_per = df['SUSP_SEX'].value_counts() / df['SUSP_SEX'].shape[0] *100
print(susp_sex_per)

column1 = 'SUSP_SEX'
column2 = 'VIC_SEX'
df = df[(df[column1] != 0) & (df[column2] != 0)]
df.reset_index(drop=True, inplace=True)

susp_race_column = 'SUSP_SEX'
vic_race_column = 'VIC_SEX'

# Create a cross-tabulation (contingency table) of the two variables
cross_table = pd.crosstab(df[susp_race_column], df[vic_race_column])

# Plot the heatmap
plt.figure(figsize=(10, 5))
sns.heatmap(cross_table, cmap='viridis', annot=True, fmt='d', cbar=True)
plt.title(f'Heatmap of {susp_race_column} vs {vic_race_column}')
plt.xlabel(vic_race_column)
plt.ylabel(susp_race_column)
plt.show()

"""Time Analysis"""

df['CMPLNT_YEAR'] = df['CMPLNT_FR_DT'].dt.year
df['CMPLNT_YEAR'].value_counts().sort_index()

#Year 2023 only can't derive solid conclusion from just one year but temprature does affect number of crimes

df['CMPLNT_MONTH'] = df['CMPLNT_FR_DT'].dt.month
df['CMPLNT_MONTH'].value_counts().sort_index().plot(kind="bar", title = "Total Crime Events by Month")

#Year 2023 only can't derive solid conclusion from just one year but temprature does affect number of crimes

df['CMPLNT_HOUR'] = df['CMPLNT_FR_TM'].str.split(':').str[0]
df['CMPLNT_HOUR'].value_counts().sort_index().plot(kind="bar", title = "Total Crime Events by Hour")

#Graph says crimes happens mostly during nights

borough_crime_counts = df['CMPLNT_NUM'].groupby(df['BORO_NM']).count()

borough_crime_counts_df = borough_crime_counts.reset_index()
borough_crime_counts_df.columns = ['BORO_NM', 'Crime_Count']
borough_crime_counts_df = borough_crime_counts_df.replace(0, np.NaN)
borough_crime_counts_df = borough_crime_counts_df.dropna()

#FIPS - Federal Information Processing Standards
fips_value = [36005, 36047, 36061, 36081, 36085]
borough_crime_counts_df['fips'] = fips_value

print(borough_crime_counts_df)

r = requests.get('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json')
counties = json.loads(r.text)
target_states = ['36'] #36 is for New York state
counties['features'] = [f for f in counties['features'] if f['properties']['STATE'] in target_states]

fig = px.choropleth(borough_crime_counts_df, geojson=counties, locations='fips', color='Crime_Count',
                    color_continuous_scale='Viridis',
                    scope='usa',
                    labels={'Crime Count': 'Crime_Count'}
                    )
fig.update_geos(center={'lon': -74, 'lat': 43}, projection_scale=6)
fig.update_layout(margin={'r': 0, 't': 0, 'l': 0, 'b': 0})
fig.show()

# Assuming df is your DataFrame and 'column_name' is the name of the column
unique_values = df['LAW_CAT_CD'].unique()
print(unique_values)

# Assume 'df' is your DataFrame and 'categorical_column' is the column you want to one-hot encode
df_encoded = pd.get_dummies(df, columns=['LAW_CAT_CD'])

# Display the resulting DataFrame
print(df_encoded)

# Assume 'df' is your DataFrame and 'categorical_column' is the column you want to one-hot encode
df = pd.get_dummies(df, columns=['OFNS_DESC'])

# Display the resulting DataFrame
print(df)

# Group by 'BORO_NM' and 'OFNS_DESC', count occurrences, and reset the index
crime_counts = df.groupby(['BORO_NM']).size().reset_index(name='COUNT')

# Find the index of the maximum count for each borough
idx = crime_counts.groupby(['BORO_NM'])['COUNT'].transform(max) == crime_counts['COUNT']

# Filter the rows with maximum counts for each borough
most_common_crime_by_borough = crime_counts[idx]

most_common_crime_by_borough

column1 = 'SUSP_RACE'
column2 = 'VIC_RACE'
df = df[(df[column1] != 0) & (df[column2] != 0)]
df.reset_index(drop=True, inplace=True)

susp_race_column = 'SUSP_RACE'
vic_race_column = 'VIC_RACE'

# Create a cross-tabulation (contingency table) of the two variables
cross_table = pd.crosstab(df[susp_race_column], df[vic_race_column])

# Plot the heatmap
plt.figure(figsize=(10, 5))
sns.heatmap(cross_table, cmap='viridis', annot=True, fmt='d', cbar=True)
plt.title(f'Heatmap of {susp_race_column} vs {vic_race_column}')
plt.xlabel(vic_race_column)
plt.ylabel(susp_race_column)
plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Assuming 'OFNS_DESC' is a categorical target variable
target_column = 'LAW_CAT_CD'
feature_column = 'SUSP_RACE'

# Separate features and target variable
X = df[feature_column]
y = df[target_column]

# Handle missing values (if any)
X = X.dropna()

# Convert categorical variables into dummy/indicator variables (one-hot encoding)
X = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100)

# Fit the model to the training data
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Display classification report
print('\nClassification Report:')
print(classification_report(y_test, y_pred))

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

y_DT = df['LAW_CAT_CD']

selected_features = ['SUSP_RACE', 'SUSP_SEX', 'SUSP_AGE_GROUP']
X_DT = df[selected_features]

X_DT = pd.get_dummies(X_DT)

# Split the data into training and testing sets
X_train_DT, X_test_DT, y_train_DT, y_test_DT = train_test_split(X_DT, y_DT, test_size=0.3, random_state=1)

# Create a Decision Tree classifier
decision_tree_classifier = DecisionTreeClassifier()

# Train the model
decision_tree_classifier.fit(X_train_DT, y_train_DT)

# Make predictions on the test set
y_pred_dt = decision_tree_classifier.predict(X_test_DT)

# Evaluate the model
accuracy_dt = accuracy_score(y_test_DT, y_pred_dt)
print(f"Decision Tree Accuracy: {accuracy_dt:.2f}")

from sklearn.model_selection import GridSearchCV

# Define the hyperparameters to tune
param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create a Decision Tree classifier
decision_tree_classifier = DecisionTreeClassifier()

# Perform grid search
grid_search = GridSearchCV(decision_tree_classifier, param_grid, cv=5)
grid_search.fit(X_train_DT, y_train_DT)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Create a Decision Tree classifier
decision_tree_classifier = DecisionTreeClassifier()

# Train the model
decision_tree_classifier.fit(X_train_DT, y_train_DT)

# Make predictions on the test set
y_pred_dt = decision_tree_classifier.predict(X_test_DT)

# Evaluate the model
accuracy_dt = accuracy_score(y_test_DT, y_pred_dt)
print(f"Decision Tree Accuracy: {accuracy_dt:.2f}")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Select relevant features and target variable
selected_features = ['SUSP_RACE', 'SUSP_SEX', 'SUSP_AGE_GROUP']
features = df[selected_features]

target = df['LAW_CAT_CD']  # Replace with your actual target variable name

features = pd.get_dummies(features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Standardize features (optional but recommended for logistic regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a logistic regression model
logistic_model = LogisticRegression()

# Train the model on the training set
logistic_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = logistic_model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Display classification report and confusion matrix
print('\nClassification Report:')
print(classification_report(y_test, y_pred))

print('\nConfusion Matrix:')
print(confusion_matrix(y_test, y_pred))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Assuming df is your DataFrame with NYPD complaint data

# Select relevant features (excluding the target variable 'SUSP_AGE_GROUP')
selected_features = ['VIC_AGE_GROUP', 'VIC_RACE', 'VIC_SEX']

# Drop rows with missing values in the selected features
df = df[selected_features + ['SUSP_AGE_GROUP']].dropna()

# Encode categorical variables
df_encoded = pd.get_dummies(df, columns=['VIC_AGE_GROUP', 'VIC_RACE', 'VIC_SEX'])

# Split data into features (X) and target variable (y)
X = df_encoded.drop('SUSP_AGE_GROUP', axis=1)
y = df_encoded['SUSP_AGE_GROUP']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Display classification report
print('\nClassification Report:')
print(classification_report(y_test, y_pred))

from wordcloud import WordCloud
import matplotlib.pyplot as plt

exclude_words = ['RELATED OFFENSES']

# Generate word cloud data, excluding specified words
text = ' '.join(df['OFNS_DESC'].dropna())
for word in exclude_words:
    text = text.replace(word, '')

# Create and generate a word cloud image
wordcloud = WordCloud(width=800, height=400, background_color='black', max_words=150, colormap='viridis').generate(text)

# Display the generated image:
plt.figure(figsize=(12, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Offense Description in NYPD Complaint Data')
plt.show()

location1='/content/drive/MyDrive/Colab Notebooks/big-data-Colab/income_tax_data.csv'
df1= pd.read_csv(location1)
print(df1)

"""Compare NYC yearly income rate againt number of complaints Boro Wise"""

|import pandas as pd
import matplotlib.pyplot as plt

# Extract year from the 'CMPLNT_FR_DT' column in NYPD complaint data
df['Year'] = pd.to_datetime(df['CMPLNT_FR_DT'], errors='coerce').dt.year

# Drop rows with NaT values and filter for the years between 1999 and 2021
df = df.dropna(subset=['Year'])
df = df[(df['Year'] >= 1999) & (df['Year'] <= 2021)]

# Group by year and count the number of complaints
complaints_per_year = df.groupby('Year').size()

complaints_per_year_county = df.groupby(['Year', 'BORO_NM']).size().unstack()

df1['Avg Income'] = df1['NY AGI of All Returns (in thousands) *'] / df1['Number of All Returns']

income_tax_ratio_county = df1.groupby(['Tax Year', 'County'])['Avg Income'].mean().unstack()

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

# Plot the number of complaints per year and county
complaints_per_year_county.plot(ax=axes[0], marker='o')
axes[0].set_xlabel('Year')
axes[0].set_ylabel('Number of Complaints')
axes[0].set_title('Number of Complaints Over Years (County-wise)')

# Plot the mean ratio of 'NY AGI of All Returns' by 'Number of All Returns' per year and county
income_tax_ratio_county.plot(ax=axes[1], marker='s')
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Avg Income (NY AGI / Number of Returns)')
axes[1].set_title('Mean Ratio Over Years (County-wise)')

# Title and legend
plt.suptitle('County-wise Comparison Over Years (1999-2021)')
plt.legend(loc='upper left')

# Adjust layout for better visualization
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Show the plot
plt.show()

import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from google.colab import drive
drive.mount('/content/drive')
location ='drive/MyDrive/Final_Project/NYC_Data_Crime_Factors.csv'

df = pd.read_csv(location)
df.sort_values('Boro_NM')
print(df)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(df['Boro_NM'], df['Complaint_Count'], marker='o', label='Complaint Count', color='skyblue', linestyle='-')
plt.title("Complaint count")
plt.xlabel('Borough Name')
plt.ylabel('Values')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(df['Boro_NM'], df['Median_Income'], marker='o', label='Median Income', color='skyblue', linestyle='-')
plt.title("Median Income")
plt.xlabel('Borough Name')
plt.ylabel('Values')
plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(df['Boro_NM'], df['Complaint_Count'], marker='o', label='Complaint Count', color='skyblue', linestyle='-')
plt.title("Complaint count")
plt.xlabel('Borough Name')
plt.ylabel('Values')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(df['Boro_NM'], df['Poverty_Rate'], marker='o', label='Poverty_Rate', color='skyblue', linestyle='-')
plt.title("Poverty_Rate")
plt.xlabel('Borough Name')
plt.ylabel('Values')
plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(df['Boro_NM'], df['Complaint_Count'], marker='o', label='Complaint Count', color='skyblue', linestyle='-')
plt.title("Complaint count")
plt.xlabel('Borough Name')
plt.ylabel('Values')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(df['Boro_NM'], df['Foreign_Popluation'], marker='o', label='Foreign_Popluation', color='skyblue', linestyle='-')
plt.title("Foreign_Popluation")
plt.xlabel('Borough Name')
plt.ylabel('Values')
plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(df['Boro_NM'], df['Complaint_Count'], marker='o', label='Complaint Count', color='skyblue', linestyle='-')
plt.title("Complaint count")
plt.xlabel('Borough Name')
plt.ylabel('Values')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(df['Boro_NM'], df['Education'], marker='o', label='Education', color='skyblue', linestyle='-')
plt.title("Education")
plt.xlabel('Borough Name')
plt.ylabel('Values')
plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(df['Boro_NM'], df['Complaint_Count'], marker='o', label='Complaint Count', color='skyblue', linestyle='-')
plt.title("Complaint count")
plt.xlabel('Borough Name')
plt.ylabel('Values')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(df['Boro_NM'], df['School_Enrol'], marker='o', label='School_Enrol', color='skyblue', linestyle='-')
plt.title("School_Enrol")
plt.xlabel('Borough Name')
plt.ylabel('Values')
plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(df['Boro_NM'], df['Complaint_Count'], marker='o', label='Complaint Count', color='skyblue', linestyle='-')
plt.title("Complaint count")
plt.xlabel('Borough Name')
plt.ylabel('Values')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(df['Boro_NM'], df['Class_of_worker'], marker='o', label='Class_of_worker', color='skyblue', linestyle='-')
plt.title("Class_of_worker")
plt.xlabel('Borough Name')
plt.ylabel('Values')
plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(df['Boro_NM'], df['Complaint_Count'], marker='o', label='Complaint Count', color='skyblue', linestyle='-')
plt.title("Complaint count")
plt.xlabel('Borough Name')
plt.ylabel('Values')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(df['Boro_NM'], df['Employment_Rate'], marker='o', label='Employment_Rate', color='skyblue', linestyle='-')
plt.title("Employment_Rate")
plt.xlabel('Borough Name')
plt.ylabel('Values')
plt.legend()

plt.tight_layout()
plt.show()