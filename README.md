# NYPD_Data_Analysis

Objective 

● Analyze the spatial and temporal patterns of crime in New York City.

● Identify the key factors influencing crime, such as demographics, socio-economic factors, 
and environmental factors.

● Develop predictive models to forecast crime patterns and evaluate their accuracy.

● Provide actionable insights to the NYPD to support evidence-based decision-making in 
crime prevention strategies.

Data Sets 
1) NYPD Complaint Data Historic
Source: NYC Open Data
2) Personal Income Tax Filers
Source: NYC Open Data
3) Crime Factors
Source: Census Bureau data
4) NYPD Arrest Data Historic
Source: NYC Open Data
5) Crime and Incarceration by State
Kaggle

Data Cleaning 

● Deleting columns that have assigned values for the NYPD record maintenance

● Converting categorical columns to numeric values.

● Handling missing values

● Converting date columns to date type.

Profiling Criminals: Machine Learning 
1) NYPD Complain Dataset 
2) Build models to profile potential suspects
3) Feature Selection: 
Offense Description, Borough, Victim age group, Victim race, Victim Sex
4) Targets:
Suspect Race, Suspect Age Group, Suspect Sex
5) Libraries: pandas, sklearn, tensorflow

Machine Learning Algorithms Used

● Logistic Regression - Ideal for capturing linear relationships 

● Random Forests - Effective in handling categorical data with good 
interpretability 

● Gradient Boosting - Effective in improving accuracy in unbalanced and diverse 
datasets 

● Neural Network - Ability to capture complex and non-linear relationships

Data Processing 

● Removing rows with null values

● Feature Selection 

● Label Encoding 

● Data Splitting: 80-20 

● Feature Normalization

● One-Hot Encoding

Machine Learning Outcome 

● Gradient Boosting: Provided best accuracy 

● Neural Network: Didn’t give the best accuracy (categorical data, optimization)

● Can make predictions for some crimes, but not for all 

● Suspects can’t be profiled just based on these factors and NYPD needs to 
collect more data if they want to profile potential suspects

What could be the factors influencing crimes?

● According to FBI(Federal Bureau of Investigation) reports, “Historically, the causes and 
origins of crime have been the subjects of investigation by many disciplines. Some factors 
that are known to affect the volume and type of crime occurring from place to place are”

● To assess criminality and law enforcement’s response from jurisdiction to jurisdiction, one 
must consider many variables, some of which, while having significant impact on crime, are 
not readily measurable or applicable among all locales.

Preparing the data for comparisons

● Data are gathered from US Census Bureau which is US Federal statistical system agency 
which is reliable source of data for American community and economy.

● They release data every year from their latest surveys like Census 2022 and American 
Community Survey(ACS). 

● For our purpose we collected Median income, Poverty rate, Employment rate, Foreign 
people ratio, Education rate, School enrollment count, Count of worker class for each 
borough to identify any relation between crime and these factors.

Complaint Vs. Arrest
One city may report more crime than a comparable one, not because there is more crime, but 
rather because its law enforcement agency, through proactive efforts, identifies more offenses.
Attitudes of the citizens toward crime and their crime reporting practices, especially concerning 
minor offenses, also have an impact on the volume of crimes known to police.

Conclusion
According to FBI, everyone should be cautioned against comparing statistical data of 
individual reporting units from cities, counties, metropolitan areas, states, or colleges or 
universities solely on the basis of their population coverage or student enrollment. Until 
data users examine all the variables that affect crime in a town, city, county, state, region, or 
other jurisdiction, they can make no meaningful comparisons.
