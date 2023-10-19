import pandas as pd
from characteristics import *
from preps import *
from dec_tree import *
from sklearn.metrics import accuracy_score

df = pd.read_csv('sii6df.csv')
nan_check(df)
n_df = cat_features(df)
create_metric(n_df)
print(n_df)
X = n_df.drop('success', axis = 1)
y = n_df['success']

X_rand = get_rand_frame(X)

X_train, X_test, y_train, y_test = splitter(X_rand, y, test_size=0.2, random_state=42)

dtc = Decision_Tree(max_depht=4, min_samples=2)
d1_y_train = y_train.values.reshape(1, y_train.shape[0])[0]
dtc.fit(X_train.values, y_train.values.reshape(-1, 1))
y_pred = dtc.predict(X_test.values)
print(accuracy_score(y_test.values.reshape(-1, 1), y_pred))

#print(df.dtypes)
