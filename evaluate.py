from tensorflow import keras
from matplotlib import pyplot as plt
import numpy as np

#loads the model
model = keras.models.load_model('my_model.h5')

#x is the x value. y_ is the y value for the linear function. y is the output of the model
x = np.arange(-10,10,0.01)[:,np.newaxis]
y_ = 2*x+1
y = model.predict(x)
#plots y_ and y.
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x,y_,label="desired output")
plt.plot(x,y,label="output")
plt.legend(loc = 'upper right')
plt.title('validation')
plt.savefig('eval.png', pad_inches=0.1) #saves the plot in png format. pad_inches is used to prevent the output png from having excess padding
plt.show() #display the plot