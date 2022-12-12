# load packages
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import pycountry_convert as pc
import json

# define path of data file
PATH = 'CDM_CW2_G2/Data/customer_information.csv'

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
# remove post code and country of birth
df_ns = df_ns.drop(columns = ['postcode', 'country_of_birth'])

########## generate dictionary ###########
imp_info = {'instructions': {'Categorical variables': 'Coding information can be found in this file',
                            'Continuous variables': 'Standardised: Mean and standard deviations can be found in this file to reverse to original values',
                            'Password': 'Dataset is password protected, password can be found in the password.txt file'}}

############ gender --> code ################
df_ns['gender'] = np.where(df_ns['gender'] == "M", 1, 0)
# add coding information to a dictionary
imp_info['gender'] = {'male': 1, 'female': 0}

########### education_level --> banding ##########
df_ns['education_level'] = df_ns['education_level'].replace({'bachelor': 'undergraduate',
                                                             'master': 'graduate',
                                                             'phD': 'graduate'})
# code
el_code = {'graduate': 1, 'primary': 2, 'undergraduate': 3, 'secondary': 4, 'other': 5}
df_ns['education_level'] = df_ns['education_level'].replace(el_code)
# add coding info into dictionary
imp_info['education_level'] = el_code

########### blood_group --> code ############
df_ns['blood_group'].unique()
# code
bg_code = {'B+': 1, 'O-': 2, 'O+': 3, 'A-': 4, 'A+': 5, 'AB+': 6, 'B-': 7, 'AB-': 8}
df_ns['blood_group'] = df_ns['blood_group'].replace(bg_code)
# add coding info into dictionary
imp_info['blood_group'] = bg_code

############# birthdate --> age --> banding ###########
# convert to age
birthyear = pd.to_datetime(df['birthdate']).dt.year
df_ns['age'] = 2022 - birthyear
df_ns = df_ns.drop(columns = ['birthdate'])
# by quartile
df_ns['age'] = pd.qcut(df_ns['age'], 5, labels = ['18-28', '29-39', '40-48', '48-59', '60+'])

############ standardisation
# define function for standardisation
def std(x):
    mean = x.mean()
    sd = x.std()
    x_s = (x-mean)/sd
    out = [x_s, {'mean': mean, 'sd': sd}]
    return out

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
groups = df_ns.groupby(['gender', 'age', 'education_level']).size().reset_index(name='count')
u_groups = groups.loc[groups['count'] == 1]
u_groups.shape
######
k = groups['count'].min()
print(k) # 2-anonymity

######
groups = df_ns.groupby(['gender', 'age']).size().reset_index(name='count')
u_groups = groups.loc[groups['count'] == 1]
u_groups.shape
######
k = groups['count'].min()
print(k) # 87-anonymity

############# save CSVs ############
# direct identifiers file
df_s.to_csv('CDM_CW2_G2/Supporting_material/direct_identifiers.csv', index = False)
# file for researchers
df_ns_reorder = df_ns[['sid', 'gender','age', 'education_level', 'cc_status', 'weight', 'height', 'blood_group', 
                       'avg_n_drinks_per_week','avg_n_cigret_per_week','n_countries_visited']]
df_ns_reorder.to_csv('CDM_CW2_G2/Anonymised_data/Imperial_researchers/researchers_dataset.csv', index = False)
# dictionary
with open('CDM_CW2_G2/Anonymised_data/Imperial_researchers/coding.json', 'w') as fp:
    json.dump(imp_info, fp, indent = 4)

########### password for file ###########
password = str(np.random.randint(10000, 99999, size=1))
with open('CDM_CW2_G2/Anonymised_data/Imperial_researchers/password.txt', 'w') as f:
    f.write(password)