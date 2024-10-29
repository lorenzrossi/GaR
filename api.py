import requests
import pandas as pd
import numpy as np
import eurostat
import json

dataset_code = "sts_inpr_m"
format = "json"

    # Define specific filters for EU27, 2004-2023, and industry production index (IPI with 2015=100)
specs = {
    'geo': 'EU27_2020',  # Geo code for EU27 (as of 2020)
    'unit': 'I15',       # Index, base year 2015 = 100
    's_adj': 'SA',       # Seasonally adjusted data
    'nace_r2': 'B+C+D',    # Industrial production (sections B-D in NACE classification)
    'time': '2004-2023'  # Time range 2004 to 2023
}

# Base URL with dataset code and format (JSON)
url = f'https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/{dataset_code}?format={format}&lang=en'

def download_eurostat_industrial_production(base_url, filters):
    """
    Downloads the monthly industrial production data of EU27 from 2004 to 2023.
    Data is indexed with the base year 2015 = 100.
    """
    # Eurostat API base URL for the industrial production index dataset (sts_inpr_m)
    
    # Define specific filters for EU27, 2004-2023, and industry production index (IPI with 2015=100)
    
    # Append filters to the URL
    filter_query = '&'.join([f'{key}={value}' for key, value in filters.items()])
    full_url = f"{base_url}&{filter_query}"
    
    # Send request to Eurostat API
    response = requests.get(full_url)
    
    # Check if the request was successful
    if response.status_code != 200:
        raise Exception(f"Failed to retrieve data: {response.status_code} - {response.text}")
    
    # Parse the JSON response
    data = response.json()

    # Debug: print the keys in the response
    print("Response keys:", data.keys())

    ### Normalize the JSON structure into a pandas DataFrame for analysis
    #series_data = data['dataSets'][0]['series']
    #dimensions = data['structure']['dimensions']['series']
##
    ### Extract the time periods
    #time_periods = [dimension['id'] for dimension in data['structure']['dimensions']['observation'][0]['values']]
##
    ## Create DataFrame to store the extracted data
    #rows = []
    #for series_key, series_value in data.items():
    #    obs = series_value['observations']
    #    # Split the series key to map dimensions (e.g., geo, unit, s_adj)
    #    keys = series_key.split(':')
    #    for obs_key, obs_value in obs.items():
    #        time_index = int(obs_key)
    #        time_period = time_periods[time_index]
    #        rows.append({
    #            'geo': keys[0],
    #            'unit': keys[1],
    #            's_adj': keys[2],
    #            'nace_r2': keys[3],
    #            'time': time_period,
    #            'value': obs_value[0]
    #        })
##
    ## Create DataFrame from the list of rows
    #df = pd.DataFrame(rows)
##
    ## Filter DataFrame for EU27 and seasonally adjusted data
    #df_filtered = df[(df['geo'] == 'EU27_2020') & (df['s_adj'] == 'SA')]
#
    ## Convert the time column to a datetime object for better handling
    #df_filtered['time'] = pd.to_datetime(df_filtered['time'], format='%Y-%m')
#
    ## Sort the DataFrame by time
    #df_filtered = df_filtered.sort_values(by='time')
#
    ## Display the first few rows
    #print(df_filtered.head())
#
    ## Save to a CSV file
    #df_filtered.to_csv('eu27_industrial_production_2004_2023.csv', index=False)
    #print("Data saved to eu27_industrial_production_2004_2023.csv")

# Call the function to download and save the data
    return data

# Call the function to download and save the data

raws = download_eurostat_industrial_production(base_url=url, filters=specs)

# Simplified filter settings to check the response structure
specs2 = {
    'geo': 'EU27_2020',
    'unit': 'I15',
    's_adj': 'SA',
    'nace_r2': 'B',  # Testing with just 'B' (Mining)
    'time': '2004'
}

# Define the URL with the dataset code and format
url2 = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/sts_inpr_m?format=json&lang=en"

def inspect_eurostat_response(base_url, filters):
    # Construct the query with filters
    filter_query = '&'.join([f'{key}={value}' for key, value in filters.items()])
    full_url = f"{base_url}&{filter_query}"
    
    # Send the request
    response = requests.get(full_url)
    if response.status_code != 200:
        print(f"Request failed with status {response.status_code}: {response.text}")
        return None

    # Parse the JSON response
    data = response.json()
    
    # Print out the dimension details for inspection
    if 'dimension' in data:
        print("Dimension information:", data['dimension'])
    else:
        print("Unexpected response structure:", data.keys())
    
    return data

# Call the function to inspect the response
raw_data = inspect_eurostat_response(url2, specs2)