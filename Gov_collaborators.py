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

# dataset with sid and sensitive PII
df_s = df[['sid', 'given_name', 'surname', 'phone_number', 'national_insurance_number', 'bank_account_number']]

# dataset without sensitive PII
df_ns = df.drop(columns=['given_name', 'surname', 'phone_number', 'national_insurance_number', 'bank_account_number'])

############### current country: all in UK --> remove ###############
df_ns = df_ns.drop(columns = 'current_country')

############# birthdate --> age --> banding ###########
# convert to age
birthyear = pd.to_datetime(df['birthdate']).dt.year
df_ns['age'] = 2022 - birthyear
df_ns = df_ns.drop(columns = ['birthdate'])
# explore: uniform distribution
df_ns['age'].describe()
hist_1 = df_ns['age'].hist(bins=10)
hist_1.plot()
plt.show()
# binning
df_ns['age'] = pd.qcut(df_ns['age'], q=[0, .25, .5, .75, 1])
# check number in each category
df_ns.groupby(['age']).size().reset_index(name='count')

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
df_ns['place_of_birth'] = df_ns['country_of_birth'].apply(country_to_continent)
# drop country of birth column
df_ns = df_ns.drop(columns = 'country_of_birth')
# check numbers in each continent
a = df_ns.groupby(['place_of_birth']).size().reset_index(name='count')
#a.loc[a['count']<10]
#a.loc[a['place_of_birth']=='South America']

################ postcode --> banding ################
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
# get the postcode with only 1 person
post_agr = post_count['postcode'].loc[post_count['count'] == 1]
# group these postcodes together
df_ns['postcode'] = np.where(df_ns['postcode'].isin(post_agr), '000', df_ns['postcode'])
# check number in each category
post_count = df_ns.groupby(['postcode']).size().reset_index(name='count')
post_count

################ weight --> banding ###############
# explore: uniform distribution
df_ns['weight'].describe()
hist_1 = df_ns['weight'].hist(bins=10)
hist_1.plot()
plt.show()
# binning
df_ns['weight'] = pd.qcut(df_ns['weight'], q=[0, .25, .5, .75, 1])
# check number in each category
df_ns.groupby(['weight']).size().reset_index(name='count')

################ height --> banding ####################
# explore: uniform distribution
df_ns['height'].describe()
hist_1 = df_ns['height'].hist(bins=10)
hist_1.plot()
plt.show()
# binning
df_ns['height'] = pd.qcut(df_ns['height'], q=[0, .25, .5, .75, 1])
# check number in each category
df_ns.groupby(['height']).size().reset_index(name='count')

################# avg_n_drinks_per_week --> banding ###################
# explore: uniform distribution
df_ns['avg_n_drinks_per_week'].describe()
hist_1 = df_ns['avg_n_drinks_per_week'].hist(bins=10)
hist_1.plot()
plt.show()
# binning
df_ns['avg_n_drinks_per_week'] = pd.qcut(df_ns['avg_n_drinks_per_week'], q=[0, .25, .5, .75, 1])
# check number in each category
df_ns.groupby(['avg_n_drinks_per_week']).size().reset_index(name='count')

##################### avg_n_cigret_per_week--> banding #################
# explore: uniform distribution
df_ns['avg_n_cigret_per_week'].describe()
hist_1 = df_ns['avg_n_cigret_per_week'].hist(bins=10)
hist_1.plot()
plt.show()
# binning
df_ns['avg_n_cigret_per_week'] = pd.qcut(df_ns['avg_n_cigret_per_week'], q=[0, .25, .5, .75, 1])
# check number in each category
df_ns.groupby(['avg_n_cigret_per_week']).size().reset_index(name='count')

#################### n_countries_visited --> banding ###############
# explore: uniform distribution
df_ns['n_countries_visited'].describe()
hist_1 = df_ns['n_countries_visited'].hist(bins=10)
hist_1.plot()
plt.show()
# binning
df_ns['n_countries_visited'] = pd.qcut(df_ns['n_countries_visited'], q=[0, .25, .5, .75, 1])
# check number in each category
df_ns.groupby(['n_countries_visited']).size().reset_index(name='count')

############### calculate k-anonimity ##################
df_ns.describe()

# df_ns.groupby(['gender', 'age', 'postcode', 'weight', 'height', 'avg_n_drinks_per_week', 'avg_n_cigret_per_week','n_countries_visited']).size().reset_index(name='count')
a = df_ns.groupby(['age', 'postcode']).size().reset_index(name='count')
a.loc[a['count']>0]

# save CSVs
# sensitive file
# file for researchers
# df.to_csv(PATH)

