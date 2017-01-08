'''
Created on Jan 8, 2017

@author: wuchunlei
'''

import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import recurrent
from keras.layers.core import Dense
from keras.engine.training import slice_X
plt.switch_backend('agg')

ROLLING = 6
OUTPUT_COUNT = 2
HIDDEN_DIM = 512


def transform(x, xmin, xmax):
    return (x - xmin) / (xmax - xmin)


def inverse_transform(x, xmin, xmax):
    return (xmax - xmin) * x + xmin


# fix random seed for reproducibility
print('------------------------------------------------------------------')
np.random.seed(2017)
RNN = recurrent.LSTM


def read_df():
    file_name = "D:\\Documents\\GitHub\\learnML\\plotfuture\\future.xlsx"
    output_df = pd.read_excel(file_name, sheetname='Sheet1', parse_cols="A:E")
    return output_df


print('read data from excel.')

source_df = read_df()
# drop Boll nan
source_df = source_df.drop(source_df[source_df['date'].isnull() == True].index)
# date_format = "%Y/%m/%d"
# source_df['date'] = pd.to_datetime(source_df['date'], errors='coerce', format=dateformat)
# reindex
source_df['date'] = range(len(source_df['date']))
# normalization
str_predict = 'upBoll'
fun_args = (source_df[str_predict].min(), source_df[str_predict].max())
new_df = pd.DataFrame(source_df[str_predict].copy(), index=source_df['date'], columns=[str_predict])
new_df[str_predict] = new_df[str_predict].apply(transform, args=fun_args)


def separate_data(datasets, rolling, ncount):
    end_date = len(datasets[str_predict]) - rolling - ncount
    docX, docY, index_date = [], [], []
    for i in range(end_date):
        train_df = datasets[str_predict][i:(i + rolling)]
        result_df = datasets[str_predict][(i + rolling):(i + rolling + ncount)]
        docX.append(train_df.as_matrix())
        docY.append(result_df.as_matrix())
        index_date.append(i + rolling - 1)
    alsX = np.array(docX)
    alsY = np.array(docY)
    return alsX, alsY, index_date


print('separate data.')
X_data, y_data, index_date = separate_data(new_df, ROLLING, OUTPUT_COUNT)

# split data
split_at = len(X_data) - math.floor(len(X_data) / 10)
(X_train, x_test) = (slice_X(X_data, 0, split_at), slice_X(X_data, split_at))
(Y_train, y_test) = (slice_X(y_data, 0, split_at), slice_X(y_data, split_at))
# reshape
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
print(X_train.shape)
print(y_test.shape)
# model
print('model compile.')
model = Sequential()
model.add(RNN(HIDDEN_DIM, input_dim=ROLLING))
model.add(Dense(OUTPUT_COUNT))
# model compile
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

# train the model
BATCH_SIZE = 128
NB_EPOCH = 20
model.fit(X_train, Y_train, batch_size=BATCH_SIZE,
          nb_epoch=NB_EPOCH, verbose=1,
          validation_data=(x_test, y_test))

train_predict = model.predict(X_train)
test_predict = model.predict(x_test)

# plot the datasets
df = pd.DataFrame(X_data[:, -1].copy(), index=index_date, columns=[str_predict])
df[str_predict] = df[str_predict].apply(inverse_transform, args=fun_args)
sts_df = df.copy()
n_count = 0
all_predict = np.append(train_predict, test_predict, axis=0)
while n_count < OUTPUT_COUNT:
    col_2 = str_predict + '_' + str(n_count)
    df[col_2] = all_predict[:, n_count]
    df[col_2] = df[col_2].apply(inverse_transform, args=fun_args)
    n_count += 1

fig = plt.figure(dpi=300)
fig.set_size_inches(100, 10)
file_name = "D:\\Documents\\GitHub\\learnML\\plotfuture\\result.png"
ax = fig.add_subplot(1, 1, 1)
ax.plot(source_df.index, source_df['current'], color='b', linewidth=1.5)
ax.plot(source_df.index, source_df['downBoll'], color='c', linewidth=1.5)
ax.plot(source_df.index, source_df['aveBoll'], color='g', linewidth=1.5)
ax.plot(source_df.index, source_df[str_predict], color='r', linewidth=1.5)
for i in range(len(df.index)):
    plt_data = df.ix[df.index[i]].tolist()
    ax.plot(source_df.index[ROLLING-1+i:ROLLING-1+i+len(plt_data)].tolist(),
            plt_data, color='k', alpha=0.6, linewidth=1.0)
fig.savefig(file_name)
fig.clf()
plt.close()
print('------------------------------------------------------------------')
