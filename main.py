import pandas as pd
from characteristics import *
from preps import *
df = pd.read_csv('sii6df.csv')
nan_check(df)
n_df = cat_features(df)
create_metric(n_df)
print(n_df)
X = n_df.drop('success', axis = 1)
y = n_df['success']

X_rand = get_rand_frame(X)
#print(df.dtypes)
