#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 16:17:30 2020

@author: abdurrahim
"""

#googleın 2012-2016 araso stock price verisi ile eğitiyoruz
#2017nin ocak ayının verisi ile test ediyoruz sonra ileriki bir tarih için tahmin
#de bulunuyoruz

#part 1 -data preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd #dataset iyi yönetmek için

#import training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

#feature scaling
#rnn için normalisation önerilir
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1)) #0, 1 arasıdna normalize ediyoruz
training_set_scaled = sc.fit_transform(training_set)

#creating a data structure with 60 timesteps and 1 output
#timesteps = t anı için önceki t anında kaç veri kullanılacak biz 60 seçtik
x_train = []
y_train = []

#xtrain 1er kaydırarak 60lı olarak verisetini dolduruyoruz yani 60x1197lik tablo
#ytrain ise sadece 60. elemanlar dolduruluyor çünkü 60. sıradaki eleman için
#önceki 60a bakarak ilişki kuruyoruz yani inputlar şimdiki ve önceki bilgilerin
#birleştirilmesi ile output üretir bu outputta bi sonraki zamanın inputudur
#böylece bellek sistemi kurulmuş olur farklı zamanlarda belleğe alınan veriler
#arasında korelasyon vardır bunun için veriyi böyle düzenleriz ki rnn ilişkiyi 
#çözebilsin

for i in range(60, 1258): #0dan 1257ye kadar i'yi arttır
    x_train.append(training_set_scaled[i-60:i, 0]) #xtraini 0dan 60a kadar doldur
    y_train.append(training_set_scaled[i, 0])   #ytraini sadece 60. elemanları doldur
    

x_train, y_train = np.array(x_train), np.array(y_train)

#reshaping
#kerastaki rnn yapısına verebilmek için 3d tensor haline getiriyoruz
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#part 2 -building rnn
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#initialising the rnn
regressor = Sequential()

#4tane lstm layer oluyor daha robust bir model için
#first lstm layer and some dropout regularisation(avoid overfitting)
"""number of units = lstmde kaç hücre(memory units) olacak relevant number seçecez
return sequences = katmanlar birbirlerine bağlanacak mı
"""
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
regressor.add(Dropout(0.2)) #nöronların %20si ignore edilecek modelin kalitesini arttırmak için

#adding a second lstm layer and some dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2)) #nöronların %20si ignore edilecek modelin kalitesini arttırmak için

#adding a third lstm layer and some dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2)) #nöronların %20si ignore edilecek modelin kalitesini arttırmak için

#adding a fourth lstm layer and some dropout regularisation
#return_sequences olmayacak çünkü son katman
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2)) #nöronların %20si ignore edilecek modelin kalitesini arttırmak için

#output layer
regressor.add(Dense(units = 1))

#compiling the rnn
#rmsprop rnn için iyidir ama adamda çok genel iyi ve güçlü bir optimizerdır
#regresyon yaptığımız için mse kullanmak daha mantıklı
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#fitting the rnn to the training set
#normalde 100 epoch çok uzun sürdüğü için 25 yaptım
regressor.fit(x_train, y_train, epochs = 25, batch_size = 32)

#part 3 - making the predictions and visualising the results
#getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

#getting the predicted stock price of 2017
#tahmin için o andan 60 gün öncesinin verisine sahip olmamız lazım çünkü modeli böyle eğittik
#13. video çok önemli hata yapmamak için
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

x_test = []
#test verisi 20 gün onun için böyle
for i in range(60, 80): #0dan 1257ye kadar i'yi arttır
    x_test.append(inputs[i-60:i, 0]) #xtraini 0dan 60a kadar doldur
    
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
predicted_stock_price = regressor.predict(x_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price) #scaling geri alınıyor

#visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

"""
EVALUATING
Hi guys,

as seen in the practical lectures, the RNN we built was a regressor. Indeed, we were dealing with Regression because we were trying to predict a continuous outcome (the Google Stock Price). For Regression, the way to evaluate the model performance is with a metric called RMSE (Root Mean Squared Error). It is calculated as the root of the mean of the squared differences between the predictions and the real values.

However for our specific Stock Price Prediction problem, evaluating the model with the RMSE does not make much sense, since we are more interested in the directions taken by our predictions, rather than the closeness of their values to the real stock price. We want to check if our predictions follow the same directions as the real stock price and we don’t really care whether our predictions are close the real stock price. The predictions could indeed be close but often taking the opposite direction from the real stock price.

Nevertheless if you are interested in the code that computes the RMSE for our Stock Price Prediction problem, please find it just below:

    import math
    from sklearn.metrics import mean_squared_error
    rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))

Then consider dividing this RMSE by the range of the Google Stock Price values of January 2017 (that is around 800) to get a relative error, as opposed to an absolute error. It is more relevant since for example if you get an RMSE of 50, then this error would be very big if the stock price values ranged around 100, but it would be very small if the stock price values ranged around 10000.

Enjoy Deep Learning!

IMPROVING
Hi guys,

here are different ways to improve the RNN model:

    Getting more training data: we trained our model on the past 5 years of the Google Stock Price but it would be even better to train it on the past 10 years.
    Increasing the number of timesteps: the model remembered the stock prices from the 60 previous financial days to predict the stock price of the next day. That’s because we chose a number of 60 timesteps (3 months). You could try to increase the number of timesteps, by choosing for example 120 timesteps (6 months).
    Adding some other indicators: if you have the financial instinct that the stock price of some other companies might be correlated to the one of Google, you could add this other stock price as a new indicator in the training data.
    Adding more LSTM layers: we built a RNN with four LSTM layers but you could try with even more.
    Adding more neurones in the LSTM layers: we highlighted the fact that we needed a high number of neurones in the LSTM layers to respond better to the complexity of the problem and we chose to include 50 neurones in each of our 4 LSTM layers. You could try an architecture with even more neurones in each of the 4 (or more) LSTM layers.

Enjoy Deep Learning!

TUNING
Hi guys,

you can do some Parameter Tuning on the RNN model we implemented.

Remember, this time we are dealing with a Regression problem because we predict a continuous outcome (the Google Stock Price).

Parameter Tuning for Regression is the same as Parameter Tuning for Classification which you learned in Part 1 - Artificial Neural Networks, the only difference is that you have to replace:

scoring = 'accuracy'  

by:

scoring = 'neg_mean_squared_error' 

in the GridSearchCV class parameters.

Enjoy Deep Learning!
"""