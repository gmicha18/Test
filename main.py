import pandas as pd
from sklearn.preprocessing import StandardScaler, KMeans

#There are invalid dates, duplicates, 

#import csv 
covid_file = pd.read_csv('MD_COVID_19_dataset.csv')

#convert dates and change invalid dates to NaT
covid_file['DATE'] = pd.to_datetime(covid_file['DATE'], errors='coerce', format='mixed')

#Remove the NaT dates
covid_file.dropna(subset=['DATE'], inplace=True)

#Remove Duplicates - base it off date and facility_type
covid_file.drop_duplicates(subset=['DATE','Facility_Type'], keep='last', inplace=True)

#split who and what 
covid_file[['Group','Facility']] = covid_file['Facility_Type'].str.split(': ',expand=True,n=1)

#remove old facility_type column
covid_file = covid_file.drop(columns=['Facility_Type'])

#pivot identifier columns
id_vars = ['OBJECTID', 'DATE', 'Group', 'Facility']

# unpivot county coloumns
county_vars = [col for col in covid_file.columns if col not in id_vars]

# Use melt() to transform the data
covid_long = covid_file.melt(
    id_vars=id_vars,
    value_vars=county_vars,
    var_name='County',
    value_name='Cases'
)


#Clean the Cases column to ensure it is a numeric integer
covid_long['Cases'] = pd.to_numeric(covid_long['Cases'], errors='coerce')
covid_long.dropna(subset=['Cases'], inplace=True)
covid_long['Cases'] = covid_long['Cases'].astype(int)

#Create Month and Week columns for monitoring
covid_long['Month'] = covid_long['DATE'].dt.month
covid_long['Week'] = covid_long['DATE'].dt.isocalendar().week

#Normalize the Cases column using StandardScaler
scaler = StandardScaler()
covid_long['Cases_Normalized'] = scaler.fit_transform(covid_long[['Cases']])

#K-Means - Cluster based on 
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
covid_long['Risk_Level'] = kmeans.fit_predict(covid_long['Cases_Normalized'])

#output.csv file
covid_long.to_csv('output.csv', index=False)
