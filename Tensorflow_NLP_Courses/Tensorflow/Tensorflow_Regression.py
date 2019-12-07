import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

housing = pd.read_csv('/home/umairshah/cal_housing_clean.csv')
# print(housing.head())

# print(housing.info())

# print(housing.describe())
print(housing.columns)

X = housing.drop('medianHouseValue', axis = 1)
y = housing['medianHouseValue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)

scaler = MinMaxScaler()
scaler.fit(X_train)

X_train = pd.DataFrame(data = scaler.transform(X_train), columns = X_train.columns,
                       index = X_train.index)

X_test = pd.DataFrame(data = scaler.transform(X_test), columns = X_test.columns,
                      index = X_test.index)

# print(X_test.head())

# Creating feature columns
feat_cols = []

for col in X_train.columns:
    col = tf.feature_column.numeric_column(col)
    feat_cols.append(col)

print(feat_cols)


# Creating input function

input_func = tf.estimator.inputs.pandas_input_fn(x = X_train, y = y_train,
                                                 batch_size = 10, num_epochs = 10,
                                                 shuffle = True)
# print(input_func)

# Creating the Model
model = tf.estimator.LinearRegressor(feature_columns = feat_cols, optimizer = 'Adam')

# Training the model
model.train(input_fn = input_func, steps = 100)

# Predicting the model
# First we'll create prediction input function
prediction_input_func = tf.estimator.inputs.pandas_input_fn(x = X_test, batch_size = 10,
                                                            num_epochs = 1, shuffle = False)

# Predictons
pred_gen = model.predict(input_fn = prediction_input_func)
predictions = list(pred_gen)

# print(predictions)
print('\n')
print(len(predictions))

# Extracting the key values from dictionary
final_pred = []

for pred in predictions:
    final_pred.append(pred['predictions'])

print(len(final_pred))

# Calculating the Error!
RMSE = mean_squared_error(y_test, final_pred)

print(RMSE)
print(housing.describe())
