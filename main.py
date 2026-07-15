import pandas as pd
from sklearn.preproccessing import StandardScaler

#import csv 
covid_file = pd.read_csv('MD_COVID_19_dataset.csv')

#convert dates and change invalid dates to NaT
covid_file['DATE'] = pd.to_datetime(covid_file['DATE'], errors='coerce', format='mixed')

#Remove the NaT dates
covid_file.dropna(subset=['DATE'], inplace=True)

#split who and what which is Person & Facility group
covid_file[['Group','Facility_Group']] = covid_file['Facility_Type'].str.split(': ',expand=True,n=1)

#remove old facility_type column
covid_file = df.drop(columns=['Facility_Type'])

# Define the columns that identify a row (everything exs)
id_vars = ['OBJECTID', 'DATE', 'Group', 'Facility_Group']

# Get the list of county columns to unpivot
county_vars = [col for col in covid_file.columns if col not in id_vars]

# Use melt() to transform the data
df_long = covid_file.melt(
    id_vars=id_vars,
    value_vars=county_vars,
    var_name='County',
    value_name='Cases'
)


# Clean the 'Cases' column to ensure it is a numeric integer
df_long['Cases'] = pd.to_numeric(df_long['Cases'], errors='coerce')
df_long.dropna(subset=['Cases'], inplace=True)
df_long['Cases'] = df_long['Cases'].astype(int)

# Create 'Month' and 'Week' columns for monitoring
df_long['Month'] = df_long['DATE'].dt.month
df_long['Week'] = df_long['DATE'].dt.isocalendar().week

# Normalize the 'Cases' column using StandardScaler
scaler = StandardScaler()
df_long['Cases_Normalized'] = scaler.fit_transform(df_long[['Cases']])

# Save the final cleaned data to 'output.csv'
df_long.to_csv('output.csv', index=False)

# Add a print statement to confirm completion
print("Script finished. Cleaned data saved to 'output.csv'.")