# load packages
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from faker import Faker
import pycountry_convert as pc
import json

# generate object for making fake data for UK
faker = Faker(['en_GB'])

# define path of data file
PATH = './Data/customer_information.csv'

# reading the CSV file
df = pd.read_csv(PATH)

# explore
df.head()
df.shape

# set seed
random.seed(0)

# generate sample id
sample_ID = np.random.randint(10000, 99999, size=1000)
sample_ID = sample_ID.tolist()
for i in range(1000):
    sample_ID[i] = '000' + str(sample_ID[i])

df["sid"] = sample_ID

# dataset with sid and sensitive PII
df_s = df[['sid', 'given_name', 'surname', 'phone_number', 'national_insurance_number', 'bank_account_number']]

# dataset without sensitive PII
df_ns = df.drop(columns=['given_name', 'surname', 'phone_number', 'national_insurance_number', 'bank_account_number'])

############### current country: all in UK --> remove ###############
df_ns = df_ns.drop(columns = 'current_country')

############ gender --> code ################
df_ns['gender'] = np.where(df_ns['gender'] == "M", 1, 0)
# add coding information to a dictionary
imp_info = {'gender': {'male': 1, 'female': 0}}

########### blood_group --> code ############
df_ns['blood_group'].unique()
# code
bg_code = {'B+': 1, 'O-': 2, 'O+': 3, 'A-': 4, 'A+': 5, 'AB+': 6, 'B-': 7, 'AB-': 8}
df_ns['blood_group'] = df_ns['blood_group'].replace(bg_code)
# add coding info into dictionary
imp_info['blood_group'] = bg_code

########### education_level --> code ##########
df_ns['education_level'].unique()
# code
el_code = {'phD': 1, 'primary': 2, 'bachelor': 3, 'secondary': 4, 'other': 5, 'masters': 6}
df_ns['education_level'] = df_ns['education_level'].replace(el_code)
# add coding info into dictionary
imp_info['education_level'] = el_code

########### country of birth --> continent ##############
df_ns['country_of_birth'].describe()
cb_count = df_ns.groupby(['country_of_birth']).size().reset_index(name='count')
cb_count
# define functional for converting country to continent
def country_to_continent(country_name):
    if country_name in ['Korea', 'Palestinian Territory', 'Timor-Leste']:
        return 'Asia'
    elif country_name in ['Saint Barthelemy','United States Minor Outlying Islands']:
        return 'North America'
    elif country_name in ['Saint Helena', 'Reunion', 'Western Sahara', 'Libyan Arab Jamahiriya', "Cote d'Ivoire"]:
        return 'Africa'
    elif country_name in ['Antarctica (the territory South of 60 deg S)','Bouvet Island (Bouvetoya)']:
        return 'Antarctica'
    elif country_name == 'Svalbard & Jan Mayen Islands':
        return 'the Arctic Ocean'
    elif country_name == 'Pitcairn Islands':
        return 'Oceania'
    elif country_name in ['Slovakia (Slovak Republic)', 'Holy See (Vatican City State)']:
        return 'Europe'
    elif country_name == 'British Indian Ocean Territory (Chagos Archipelago)':
        return 'Indian Ocean'
    elif country_name == 'Netherlands Antilles':
        return 'South America'
    else:
        country_alpha2 = pc.country_name_to_country_alpha2(country_name)
        country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
        country_continent_name = pc.convert_continent_code_to_continent_name(country_continent_code)
        return country_continent_name
# convert
df_ns['continent_of_birth'] = df_ns['country_of_birth'].apply(country_to_continent)
# check numbers in each continent
n = df_ns.groupby(['continent_of_birth']).size().reset_index(name='count')
print(n)

############ country_names --> code ##############
# unique country_names
unique_country_names = df_ns['country_of_birth'].unique()
n = unique_country_names.size
# code country of birth using random numbers
country_codes = np.random.randint(100, 999, size = n)
country_dict = dict(zip(unique_country_names, country_codes.tolist()))
# replace country names with codes
df_ns['country_of_birth'] = df_ns['country_of_birth'].replace(country_dict)
# add coding info into dictionary
imp_info['country_of_birth'] = country_dict

################ continent_names --> code ##############
# unique contient_names
unique_continent_names = df_ns['continent_of_birth'].unique()
n = unique_continent_names.size
# code place of birth using random numbers
continent_codes = np.random.randint(10, 99, size = n)
continent_dict = dict(zip(unique_continent_names, continent_codes.tolist()))
# replace continent names with codes
df_ns['continent_of_birth'] = df_ns['continent_of_birth'].replace(continent_dict)
# add coding info into dictionary
imp_info['continent_of_birth'] = country_dict

################ postcode --> banding --> replace with fake postcode ################
# define function for finding the index of the first digit in a string
def find_first_digit(s):
    for i, c in enumerate(s):
        if c.isdigit():
            return i
            break
# keep characters before the first digit
df_ns['postcode'] = df['postcode'].apply(lambda x: x[:find_first_digit(x)])
# unique postcodes
unique_postcode = df_ns['postcode'].unique()
# generate fake postcodes for each unique postcode
replace_postcode = {}
for i in unique_postcode:
    replace_postcode[i] = faker.postcode()
replace_postcode
# replace the column in data set
df_ns['postcode'] = df_ns['postcode'].replace(replace_postcode)
# store coding info in dictionary
imp_info['postcode'] = replace_postcode

############# birthdate --> age --> standardise ###########
# convert to age
birthyear = pd.to_datetime(df['birthdate']).dt.year
df_ns['age'] = 2022 - birthyear
df_ns = df_ns.drop(columns = ['birthdate'])
# define function for standardisation
def std(x):
    mean = x.mean()
    sd = x.std()
    x_s = (x-mean)/sd
    out = [x_s, {'mean': mean, 'sd': sd}]
    return out
# standardise age
tmp = std(df_ns['age'])
df_ns['age'] = tmp[0]
# store mean/sd info in dictionary
imp_info['age'] = tmp[1]

################ weight --> standardise ###############
# standardise weight
tmp = std(df_ns['weight'])
df_ns['weight'] = tmp[0]
# store mean/sd info in dictionary
imp_info['weight'] = tmp[1]

################ height --> standardise ####################
# standardise height
tmp = std(df_ns['height'])
df_ns['height'] = tmp[0]
# store mean/sd info in dictionary
imp_info['height'] = tmp[1]

################# avg_n_drinks_per_week --> standardise ###################
# standardise avg_n_drinks_per_week
tmp = std(df_ns['avg_n_drinks_per_week'])
df_ns['avg_n_drinks_per_week'] = tmp[0]
# store mean/sd info in dictionary
imp_info['avg_n_drinks_per_week'] = tmp[1]

##################### avg_n_cigret_per_week --> standardise #################
# standardise avg_n_cigret_per_week
tmp = std(df_ns['avg_n_cigret_per_week'])
df_ns['avg_n_cigret_per_week'] = tmp[0]
# store mean/sd info in dictionary
imp_info['avg_n_cigret_per_week'] = tmp[1]

#################### n_countries_visited --> standardise ###############
# standardise n_countries_visited
tmp = std(df_ns['n_countries_visited'])
df_ns['n_countries_visited'] = tmp[0]
# store mean/sd info in dictionary
imp_info['n_countries_visited'] = tmp[1]

############### calculate k-anonimity ##################
# consider cc_staus only
groups = df_ns.groupby(['cc_status']).size().reset_index(name='count')
k = groups['count'].min()
print(k) # 47-anonymity
# consider cc_status and gender
groups = df_ns.groupby(['cc_status', 'gender']).size().reset_index(name='count')
k = groups['count'].min()
print(k) # 20-anonymity

############# save CSVs ############
# sensitive PII file
df_s.to_csv('Data/sensitive_info.csv', index = False)
# file for researchers
df_ns.to_csv('Data/researchers_dataset.csv', index = False)
# dictionary
with open('Data/res_data_coding.json', 'w') as fp:
    json.dump(imp_info, fp, indent = 4)
