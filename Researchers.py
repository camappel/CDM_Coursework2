import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

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

############ gender --> code ################
df_ns['gender'] = np.where(df_ns['gender'] == "Male", 1, 0)

# dictionary with info on coding
coding = {'gender': [('male', 1), ('female', 0)]}

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
bins = [18, 30, 40, 50, 60, 70]
df_ns['age'] = pd.cut(df_ns['age'], bins = bins, include_lowest = False)
# check number in each category
df_ns.groupby(['age']).size().reset_index(name='count')

########### code country of birth: hash function? ##############
df_ns['country_of_birth'].describe()


############### current country: all in UK --> remove ###############
df_ns = df_ns.drop(columns = 'current_country')

################ postcode --> banding ################
##### keep outbound characters and digits
# keep characters before the space
df_ns['postcode'] = df['postcode'].apply(lambda x: x[:x.find(' ')])

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
bins = [30, 40, 50, 60, 70, 80, 90, 100]
df_ns['weight'] = pd.cut(df_ns['weight'], bins=bins, include_lowest = False)
# check number in each category
df_ns.groupby(['weight']).size().reset_index(name='count')

################ height --> banding ####################
# explore: uniform distribution
df_ns['height'].describe()
hist_1 = df_ns['height'].hist(bins=10)
hist_1.plot()
plt.show()
# binning
bins = [1.3, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
labels = ['<= 1.5', '(1.5, 1.6]', '(1.6, 1.7]', '(1.7, 1.8]', '(1.8, 1.9]', '> 1.9']
df_ns['height'] = pd.cut(df_ns['height'], bins=bins, labels = labels, include_lowest = False)
# check number in each category
df_ns.groupby(['height']).size().reset_index(name='count')

### BMI?

################# avg_n_drinks_per_week --> banding ###################
# explore: uniform distribution
df_ns['avg_n_drinks_per_week'].describe()
hist_1 = df_ns['avg_n_drinks_per_week'].hist(bins=10)
hist_1.plot()
plt.show()
# binning
bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
labels = ['[0, 1]', '(1, 2]', '(2, 3]', '(3, 4]', '(4, 5]', '(5, 6]', '(6, 7]', '(7, 8]', '(8, 9]', '(9, 10]']
df_ns['avg_n_drinks_per_week'] = pd.cut(df_ns['avg_n_drinks_per_week'], bins = bins, labels = labels, include_lowest = True)
# check number in each category
df_ns.groupby(['avg_n_drinks_per_week']).size().reset_index(name='count')

##################### avg_n_cigret_per_week--> banding #################
# explore: uniform distribution
df_ns['avg_n_cigret_per_week'].describe()
hist_1 = df_ns['avg_n_cigret_per_week'].hist(bins=10)
hist_1.plot()
plt.show()
# binning
bins = [0, 100, 200, 300, 400, 500]
labels = ['<= 100', '(100, 200]', '(200, 300]', '(300, 400]', '(400, 500]']
df_ns['avg_n_cigret_per_week'] = pd.cut(df_ns['avg_n_cigret_per_week'], bins = bins, labels = labels, include_lowest = False)
# check number in each category
df_ns.groupby(['avg_n_cigret_per_week']).size().reset_index(name='count')

#################### n_countries_visited --> banding ###############
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

# df_ns.groupby(['gender', 'age', 'postcode', 'weight', 'height', 'avg_n_drinks_per_week', 'avg_n_cigret_per_week','n_countries_visited']).size().reset_index(name='count')
a = df_ns.groupby(['age', 'postcode']).size().reset_index(name='count')
a.loc[a['count']>0]

# save CSVs
# sensitive file
# file for researchers
df.to_csv(PATH)
