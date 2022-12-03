import pandas as pd
import numpy as np

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

# code gender
df_ns['gender'] = np.where(df_ns['gender'] == "Male", 1, 0)

# dataset with info on coding
coding = {'gender': [('male', 1), ('female', 0)]}

# birthday to age
birthyear = pd.to_datetime(df['birthdate']).dt.year
df_ns['age'] = 2022 - birthyear
df_ns = df_ns.drop(columns = ['birthdate'])

# code country of birth: hash function?
df_ns['country_of_birth'].describe()


# current country: all in UK
df_ns = df_ns.drop(columns = 'current_country')

# postcode --> keeping outbound digits only
df_ns['postcode'] = df_ns['postcode'].apply(lambda x: x[:x.find(' ')])

# weight

# height

# avg_n_drinks_per_week

# avg_n_cigret_per_week

# n_countries_visited

# calculate k-anonimity
df_ns.describe()
df_ns['n_countries_visited'].describe()
df_ns['avg_n_cigret_per_week'].describe()
df_ns['age'].describe()

df_ns.groupby(['n_countries_visited']).size().reset_index(name='count')
df_ns.loc[df['n_countries_visited'] == 50]
# save CSVs
# sensitive file
# file for researchers
df.to_csv(PATH)
