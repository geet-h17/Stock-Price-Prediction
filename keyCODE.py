
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')


import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('Reliance.csv')
df

df.index = pd.to_datetime(df['Date'])
df


df = df.drop(['Date'], axis='columns')
df

df['Open-Close'] = df.Open - df.Close
df['High-Low'] = df.High - df.Low


X = df[['Open-Close', 'High-Low']]
X.head()

y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
y
split_percentage = 0.8
split = int(split_percentage*len(df))


X_train = X[:split]
y_train = y[:split]


X_test = X[split:]
y_test = y[split:]

cls = SVC().fit(X_train, y_train)


df['Predicted_Signal'] = cls.predict(X)

df['Return'] = df.Close.pct_change()

df['Strategy_Return'] = df.Return *df.Predicted_Signal.shift(1)

df['Cum_Ret'] = df['Return'].cumsum()
df

df['Cum_Strategy'] = df['Strategy_Return'].cumsum()
df

import matplotlib.pyplot as plt


plt.plot(df['Cum_Ret'],color='red')
plt.plot(df['Cum_Strategy'],color='blue')

plt.show()

