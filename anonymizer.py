import pandas

# define path
PATH = './Data/customer_information.csv'

# reading the CSV file
df = pandas.read_csv(PATH)
 
# displaying the contents of the CSV file
print(df)

# save CSV
df.to_csv(PATH)