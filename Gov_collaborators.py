import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import pycountry_convert as pc

# define path
PATH = './Data/customer_information.csv'

# reading the CSV file
df = pd.read_csv(PATH)
 
# explore
df.head()
df.shape

# set seed
random.seed(0)

############### only keep geographic and education characteristics ###############
df_ns = df[['country_of_birth', 'postcode', 'cc_status', 'education_level', 'n_countries_visited']]

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
# drop country of birth column
df_ns = df_ns.drop(columns = 'country_of_birth')
# check numbers in each continent
a = df_ns.groupby(['continent_of_birth']).size().reset_index(name='count')
a.loc[a['count'] < 30]
# combine Antarctica, Indian Ocean, the Arctic Ocean and South America --> smallest categry contains at least 50 people OR drop the records
df_ns['continent_of_birth'] = df_ns['continent_of_birth'].replace({'Antarctica': 'Other continents',
                                                                   'Indian Ocean': 'Other continents',
                                                                   'the Arctic Ocean': 'Other continents',
                                                                   'South America': 'Other continents'})
# check numbers in each continent
a = df_ns.groupby(['continent_of_birth']).size().reset_index(name='count')
a

############################### postcode --> banding ##################################
####### keep outbound characters only
# define function for finding the index of the first digit in a string
def find_first_digit(s):
    for i, c in enumerate(s):
        if c.isdigit():
            return i
            break
# keep characters before the first digit
df_ns['postcode'] = df['postcode'].apply(lambda x: x[:find_first_digit(x)])
# check number in each category
post_count = df_ns.groupby(['postcode']).size().reset_index(name='count')
post_count
# get dictionary for convert to UK country
postcode_country = pd.read_csv('Data/postcode_country.csv')
post_to_country = dict(zip(postcode_country['Postcode area'], postcode_country['Country']))
# convert to country
df_ns['UK_country'] = df_ns['postcode'].replace(post_to_country)
# drop postcode column
df_ns = df_ns.drop(columns = 'postcode')
# check number of individuals in each category
a = df_ns.groupby(['UK_country']).size().reset_index(name='count')
a.loc[a['count'] < 30]
# combine Channel Islands. Isle of Man, Northern Ireland and Wales
df_ns['UK_country'] = df_ns['UK_country'].replace({'Channel Islands': 'Overseas territories',
                                                   'Isle of Man': 'Overseas territories'})
# check number of individuals in each category
a = df_ns.groupby(['UK_country']).size().reset_index(name='count')
a

############### education level ################
a = df_ns.groupby(['education_level']).size().reset_index(name='count')
a
### combine some categories
df_ns['education_level'] = df_ns['education_level'].replace({'primary': 'School',
                                                             'secondary': 'School',
                                                             'masters': 'Postgraduate',
                                                             'phD': 'Postgraduate'})

############### calculate k-anonimity ##################
df_ns.describe()

# df_ns.groupby(['cc_status', 'UK_country', 'continent_of_birth', 'education level']).size().reset_index(name='count')
# 2 * 3 * 6 * 4 = 144
a = df_ns.groupby(['UK_country', 'continent_of_birth', 'education_level']).size().reset_index(name='count')
b = a.loc[a['count']==1]
a.shape
b.shape
# remove the 20 individuals?
# sample with probability?
# partial sample?

# save CSVs
# sensitive file: same as the sensitive_info file for researchers
# file for government collaborators

# df.to_csv(PATH)

