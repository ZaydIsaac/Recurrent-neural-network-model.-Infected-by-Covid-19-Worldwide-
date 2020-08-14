# -*- coding: utf-8 -*-

""" Many of the variables are written in Spanish, the comments will help yo
    u to see their translation. """

import matplotlib.pyplot as plt; import datetime
import tensorflow as tf; import numpy as np; import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler; import pandas as pd

def lotes(data_train, length):
    start = np.random.randint(0, len(data_train) - length )
    lote_y = np.array(data_train[start:start+length+1]).reshape(1,length+1)             # Helper function to split data in batches 
    return lote_y[:,:-1].reshape(-1, length, 1), lote_y[:,1:].reshape(-1, length, 1)

add = []; index = []

s = datetime.datetime.strptime('01-22-2020',"%m-%d-%Y")
e = datetime.datetime.strptime('08-20-2020',"%m-%d-%Y")             # A date range is created to read the records from the database.
r = [ s+datetime.timedelta(days=i) for i in range((e-s).days) ]
rang = [ i.strftime("%m-%d-%Y") for i in r ]; del(r)

for i in rang:
    try:
        dic=" PLACE THE ADDRESS OF THE FILE YOU DOWNLOADED IN THE REPOSITORY HERE /csse_covid_19_data/csse_covid_19_daily_reports/{}.csv".format(i)
        D= pd.read_csv(dic, delimiter=','); D['Confirmed'] = D['Confirmed'].replace(np.nan, 0)
        a = sum(D['Confirmed']) ; b = D['Last_Update'][0]
        add.append(a); index.append(b)
    except:
        break

Data = pd.DataFrame()
Data['infect'] = add
Data.index = index
Data.index = pd.to_datetime(Data.index) 

size_train = 10
normalizer = MinMaxScaler()
Data_train= normalizer.fit_transform(Data.head(len(Data)-size_train))
Data_test= normalizer.transform(Data.tail(size_train))
Data_infect = Data.tail(size_train)

inputs= 1
steps= size_train
nodes= 200
outputs= 1
learning_rate= 0.0001
epochs= 13000
x= tf.placeholder( tf.float32, [None, steps, inputs])
y= tf.placeholder( tf.float32, [None, steps, outputs])
capa= tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.BasicLSTMCell(num_units=nodes, activation= tf.nn.tanh), output_size=outputs)

salidas, estados= tf.nn.dynamic_rnn(capa, x, dtype= tf.float32)    # salidas = outputs ; estados = state
fun_error= tf.reduce_mean(tf.square(salidas-y))
optimizador = tf.train.AdamOptimizer(learning_rate= learning_rate)    # optimizador = optimizer
train= optimizador.minimize(fun_error)
init= tf.global_variables_initializer() 
saver= tf.train.Saver()     

with tf.Session() as sesion:
    sesion.run(init)
    for i in range(epochs):
        lote_x, lote_y = lotes(Data_train, steps)
        sesion.run(train, feed_dict={x: lote_x, y:lote_y})
        if i%100==0:
            error= fun_error.eval(feed_dict={x: lote_x, y:lote_y})
            print(i,"\t Loss: " ,error)
            saver.save(sesion, ".\modelo")
            
with tf.Session() as sesion:
	saver.restore(sesion, ".\modelo")
	entrenamiento_seed = list(Data_train[-steps:])
	for i in range(steps):
		lote_x = np.array(entrenamiento_seed[-steps:]).reshape(1,steps,1)
		prediccion_y= sesion.run(salidas, feed_dict={x:lote_x})
		entrenamiento_seed.append(prediccion_y[0,-1,0])
        
resultados = normalizer.inverse_transform(np.array(entrenamiento_seed[steps:]).reshape(steps,1))   # resultados = results
Data_infect['Predict'] = resultados

plt.plot_date(Data_infect.index, Data_infect['Predict'], '-'); plt.plot_date(Data_infect.index, Data_infect['infect'], '.')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
plt.gcf().autofmt_xdate()

