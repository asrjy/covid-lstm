import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
import requests
from bs4 import BeautifulSoup
from datetime import date


today = date.today()
confirmed_url = "https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
deaths_url = "https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
recovered_url = "https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"


d1 = today.strftime("%m/%d/%y").replace(' 0', ' ')
if(d1[0] == '0'):
    d1 = d1[1:]

def get_confirmed():
    con_casesnew = pd.read_csv(confirmed_url)
    con_casesnew = con_casesnew.sum(axis=0)
    con_casesnew.drop(["Country/Region","Lat","Long"], axis=0, inplace=True)
    con_casesnew = pd.DataFrame(con_casesnew)
    con_casesnew.rename(columns={0: "Infected"}, inplace=True)
    return con_casesnew
get_confirmed()



def get_deaths():
    deaths_new = pd.read_csv(deaths_url)
    deaths_new = deaths_new.sum(axis=0)
    deaths_new.drop(["Country/Region","Lat","Long"], axis=0, inplace=True)
    deaths_new = pd.DataFrame(deaths_new)
    deaths_new.rename(columns={0: "Deaths"}, inplace=True)
    return deaths_new
get_deaths()

def get_recovered():
    rec_new = pd.read_csv(recovered_url)
    rec_new = rec_new.sum(axis=0)
    rec_new.drop(["Country/Region","Lat","Long"], axis=0, inplace=True)
    rec_new = pd.DataFrame(rec_new)
    rec_new.rename(columns={0: "Recovered"}, inplace=True)
    
    return rec_new
get_recovered()

def get_newvals():
    url_wminfo = "https://www.worldometers.info/coronavirus/"
    page = requests.get(url_wminfo)
    soup = BeautifulSoup(page.content, 'html.parser')
    result = soup.find_all("div", {"id":"maincounter-wrap"})
    numbers = []
    for r in result:
        numbers.append(int(r.text.split(':')[-1].replace(',','').replace(' ','')))
    return numbers
    
numbers = get_newvals()

df = get_confirmed()

close_data = df['Infected'].values
close_data = close_data.reshape((-1,1))

split_percent = 0.80
split = int(split_percent*len(close_data))

close_train = close_data[:split]
close_test = close_data[split:]

cclose = close_data

look_back = 2

train_generator = TimeseriesGenerator(cclose, cclose, length=look_back, batch_size=20)     
test_generator = TimeseriesGenerator(close_test, close_test, length=look_back, batch_size=1)

from keras.models import Sequential
from keras.layers import LSTM, Dense

model = Sequential()
model.add(
    LSTM(10,
        activation='relu',
        input_shape=(look_back,1))
)
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

num_epochs = 100
model.fit_generator(train_generator, epochs=num_epochs, verbose=1)

prediction = model.predict_generator(test_generator)

cclose = cclose.reshape((-1))
close_train = close_train.reshape((-1))
close_test = close_test.reshape((-1))
prediction = prediction.reshape((-1))



close_data = close_data.reshape((-1))

def predict(num_prediction, model):
    prediction_list = close_data[-look_back:]
    
    for _ in range(num_prediction):
        x = prediction_list[-look_back:]
        x = x.reshape((1, look_back, 1))
        out = model.predict(x)[0][0]
        prediction_list = np.append(prediction_list, out)
    prediction_list = prediction_list[look_back-1:]
        
    return prediction_list
num_prediction = 1
forecast = predict(num_prediction, model)
forecast = numbers[0]

from matplotlib.pyplot import figure
figure(num=None, figsize=(12, 9), dpi=80, facecolor='w', edgecolor='k')
plt.plot(df, label='confirmed')
plt.legend(loc="upper left")

df_temp = df
df_temp.loc[d1] = forecast

from matplotlib.pyplot import figure
figure(num=None, figsize=(12,9), dpi=80, facecolor='w', edgecolor='k')
plt.plot(df_temp[-7:], range(7), linestyle='--', marker='o', color='b', label='confirmed')
plt.legend(loc="upper left")

df_rec = get_recovered()

close_data_rec = df_rec['Recovered'].values
close_data_rec = close_data.reshape((-1,1))

split_percent = 0.80
split = int(split_percent*len(close_data_rec))

close_train_rec = close_data_rec[:split]
close_test_rec = close_data_rec[split:]

cclose_rec = close_data_rec

look_back = 2

train_generator_rec = TimeseriesGenerator(cclose_rec, cclose_rec, length=look_back, batch_size=20)     
test_generator_rec = TimeseriesGenerator(close_test_rec, close_test_rec, length=look_back, batch_size=1)



prediction_rec = model.predict_generator(test_generator_rec)

cclose_rec = cclose_rec.reshape((-1))
close_train_rec = close_train_rec.reshape((-1))
close_test_rec = close_test_rec.reshape((-1))
prediction_rec = prediction_rec.reshape((-1))



close_data_rec = close_data_rec.reshape((-1))

def predict(num_prediction, model):
    prediction_list = close_data_rec[-look_back:]
    
    for _ in range(num_prediction):
        x = prediction_list[-look_back:]
        x = x.reshape((1, look_back, 1))
        out = model.predict(x)[0][0]
        prediction_list = np.append(prediction_list, out)
    prediction_list = prediction_list[look_back-1:]
        
    return prediction_list
num_prediction = 1
forecast_rec = predict(num_prediction, model)
forecast_rec = numbers[2]

from matplotlib.pyplot import figure
figure(num=None, figsize=(12, 9), dpi=80, facecolor='w', edgecolor='k')
plt.plot(df_rec, color='g', label='recoveries')
plt.legend(loc="upper left")

df_temp_rec = df_rec
df_temp_rec.loc[d1] = forecast_rec

from matplotlib.pyplot import figure
figure(num=None, figsize=(12,9), dpi=80, facecolor='w', edgecolor='k')
plt.plot(df_temp_rec[-7:], range(7), linestyle='--', marker='o', color='g', label='recoveries')
plt.legend(loc="upper left")

df_dea = get_deaths()



close_data_dea = df_dea['Deaths'].values
close_data_dea = close_data_dea.reshape((-1,1))

split_percent = 0.80
split = int(split_percent*len(close_data_dea))

close_train_dea = close_data_dea[:split]
close_test_dea = close_data_dea[split:]

cclose_dea = close_data_dea

look_back = 2

train_generator_dea = TimeseriesGenerator(cclose_dea, cclose_dea, length=look_back, batch_size=20)     
test_generator_dea = TimeseriesGenerator(close_test_dea, close_test_dea, length=look_back, batch_size=1)

prediction_dea = model.predict_generator(test_generator_dea)

cclose_dea = cclose_dea.reshape((-1))
close_train_dea = close_train_dea.reshape((-1))
close_test_dea = close_test_dea.reshape((-1))
prediction_dea = prediction_dea.reshape((-1))

close_data_dea = close_data_dea.reshape((-1))

def predict(num_prediction, model):
    prediction_list = close_data_dea[-look_back:]
    
    for _ in range(num_prediction):
        x = prediction_list[-look_back:]
        x = x.reshape((1, look_back, 1))
        out = model.predict(x)[0][0]
        prediction_list = np.append(prediction_list, out)
    prediction_list = prediction_list[look_back-1:]
        
    return prediction_list
num_prediction = 1
forecast_dea = predict(num_prediction, model)
forecast_dea = numbers[1]

from matplotlib.pyplot import figure
figure(num=None, figsize=(12, 9), dpi=80, facecolor='w', edgecolor='k')
plt.plot(df_dea, color='r', label='deaths')
plt.legend(loc="upper left")

df_temp_dea = df_dea
df_temp_dea.loc[d1] = forecast_dea

from matplotlib.pyplot import figure
figure(num=None, figsize=(12,9), dpi=80, facecolor='w', edgecolor='k')
plt.plot(df_temp_dea[-7:], range(7), linestyle='--', marker='o', color='r', label='deaths')
plt.legend(loc="upper left")

