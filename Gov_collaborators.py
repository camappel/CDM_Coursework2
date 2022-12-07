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

# generate sample id
sample_ID = np.random.randint(10000, 99999, size=1000)
sample_ID = sample_ID.tolist()
for i in range(1000):
    sample_ID[i] = '000' + str(sample_ID[i])

df["sid"] = sample_ID

############### only keep geographic and education characteristics ###############
df_ns = df[['sid', 'country_of_birth', 'postcode', 'cc_status', 'education_level', 'n_countries_visited']]

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
    elif country_name in ['Antarctica (the territory South of 60 deg S)']:
        return 'Antarctica'
    elif country_name == 'Pitcairn Islands':
        return 'Oceania'
    elif country_name in ['Slovakia (Slovak Republic)', 'Holy See (Vatican City State)', 'British Indian Ocean Territory (Chagos Archipelago)', 'Bouvet Island (Bouvetoya)', 'Svalbard & Jan Mayen Islands']:
        return 'Europe'
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
                                                             'phD': 'Postgraduate',
                                                             'bachelor': 'Undergraduate',
                                                             'other': 'Other'})

############## n_countries_visited ################
# explore: uniform distribution
df_ns['n_countries_visited'].describe()
hist_1 = df_ns['n_countries_visited'].hist(bins=10)
hist_1.plot()
plt.show()
# binning
bins = [0, 10, 20, 30, 40, 50]
labels = ['(0, 10]', '(10, 20]', '(20, 30]', '(30, 40]', '(40, 50]']
df_ns['n_countries_visited'] = pd.cut(df_ns['n_countries_visited'], bins = bins, labels = labels, include_lowest = False)
# check number in each category
df_ns.groupby(['n_countries_visited']).size().reset_index(name='count')

############### calculate k-anonimity ##################
df_ns.describe()

# df_ns.groupby(['cc_status', 'UK_country', 'continent_of_birth', 'education level']).size().reset_index(name='count')
# 2 * 3 * 6 * 4 = 144
a = df_ns.groupby(['UK_country', 'continent_of_birth', 'education_level']).size().reset_index(name='count')
b = a.loc[a['count']==1]
a.shape
b.shape
# get sid of the 27 individuals
df_unique = pd.merge(b, df_ns,  how='left', 
                  left_on=['UK_country', 'continent_of_birth', 'education_level'], 
                  right_on = ['UK_country', 'continent_of_birth', 'education_level'])
df_unique[['sid', 'count']]
# remove 
df_ns = df_ns[df_ns['sid'].isin(df_unique['sid']) == False]

########## remove sid column ##############
df_ns = df_ns.drop(columns = 'sid')

# 2-anonymity
a = df_ns.groupby(['UK_country', 'continent_of_birth', 'education_level']).size().reset_index(name='count')
a['count'].min()

# save CSVs
# sensitive file: same as the sensitive_info file for researchers
# file for government collaborators
df_ns.to_csv('Data/gov_dataset.csv', index = False)

