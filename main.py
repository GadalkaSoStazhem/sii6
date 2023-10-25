from dec_tree import Not_Binary_Decision_Tree
from characteristics import metrics
from preps import *
from graphics import draw_plots


df = pd.read_csv('sii6df.csv')
nan_check(df)
n_df = cat_features(df)
create_metric(n_df)
#print(n_df.head(5))
X = n_df.drop('success', axis = 1)
y = n_df['success']

X_rand = get_rand_frame(X)

X_train, X_test, y_train, y_test = splitter(X_rand, y, test_size=0.2, random_state=41)
d1_y_train = y_train.values.reshape(-1, 1)
dtc = Not_Binary_Decision_Tree()
dtc.fit(X_train.values, d1_y_train)
y_pr = dtc.predict(X_test.values)
#print(conf_matrix(y_test.values, y_pr))
metrics(y_test.values, y_pr)

y_t_arr = np.array([y[0] for y in y_test.values])
draw_plots(y_t_arr, np.array(y_pr))

