import numpy as np
import pandas

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sql2v_explain import sql_explainer
from sklearn.model_selection import cross_val_score, KFold

q_path = '/home/zxh/join-order-benchmark/'
NAME_DATASET = 'job_trainData_sql.txt'
NUM_SQL_FEATURE = 38
NUM_DB_METRIC = 16

# regression
def baseline_model(num_feature = NUM_SQL_FEATURE, num_output = NUM_DB_METRIC):
    # create model
    model = Sequential()
    model.add(Dense(num_feature * 2, input_dim=num_feature, kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_feature, kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_output, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model


class SqlParser:
    # DML: select delete insert update      1 2 3 4

    def __init__(self, benchmark='job', sql = '10a', cur_op='oltp_read_write.lua', num_event=1000, p_r_range=0.6, p_u_index=0.2, p_i=0.1, p_d=0.1):

        if benchmark == 'job':

            self.SQL_NAMES = [sql]

            ########### Convert: sql statement => feature vector
            explainer = sql_explainer(sql)
            self.sql_vector = np.array(explainer.explain())

            print("### SQL_VECTOR")
            print(self.sql_vector)
            ################################################################################################################################

            ######### Prepare training data
            fs = open(NAME_DATASET, 'r')
            df = pandas.read_csv(fs, sep=' ', header=None)
            lt_sql = df.values

            sql_X = lt_sql[:, 0:NUM_SQL_FEATURE]  # op_type   events  table_size
            sql_Y = lt_sql[:, NUM_SQL_FEATURE:]
            print("### Format (input, output):")
            print(sql_X[0])
            print(sql_Y[0])

            sc_X = StandardScaler()
            self.X_train = sc_X.fit_transform(sql_X)
            # X_test = X_train[50:]
            # X_train = X_train[:50]

            sc_Y = StandardScaler()
            self.Y_train = sc_Y.fit_transform(sql_Y)
            # Y_test = Y_train[20:]                   # 2:8
            # Y_train = Y_train[:50]
            ################################################################################################################################

    def train(self):
        ######### Train model
        seed = 7
        np.random.seed(seed)
        # estimators = []
        # estimators.append(('standardize', StandardScaler()))
        # estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=50, verbose=0)))
        self.estimator = KerasRegressor(build_fn=baseline_model, epochs=1000, batch_size=50, verbose=1)  # epochs
        kfold = KFold(n_splits=10, random_state=seed)       # n_splites packs, with 1 packet as the test set
        results = cross_val_score(self.estimator, self.X_train, self.Y_train, cv=kfold)

        self.estimator.fit(self.X_train, self.Y_train)
        ################################################################################################################################

    def estimate(self):
        # Estimate the accurancy of this trained model
        pass


    def predict_sql_resource(self):

        return self.estimator.predict(
            self.sql_vector)

    def update(self):
        pass


parser = SqlParser()
parser.train()

