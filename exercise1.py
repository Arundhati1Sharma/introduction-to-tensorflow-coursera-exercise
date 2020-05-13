import tensorflow as tf
import numpy as np
from tensorflow import keras
def house_model(y_new):
   xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0], dtype=float)
   ys = np.array([100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 450.0, 500.0, 550.0,600.0, 650.0,700.0], dtype=float)

   model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
   model.compile(optimizer='sgd',loss='mean_squared_error')
   model.fit(xs,ys,epochs=500)

   return (model.predict(y_new)[0]+1) //100
prediction = house_model([7.0])
print(prediction)
