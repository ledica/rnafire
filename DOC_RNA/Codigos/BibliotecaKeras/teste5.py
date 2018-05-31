# Numeric Python Library.
import numpy
# Python Data Analysis Library.
import pandas
# Scikit-learn Machine Learning Python Library modules.
#   Preprocessing utilities.
from sklearn import preprocessing
#   Cross-validation utilities.
from sklearn import cross_validation
# Python graphical library
from matplotlib import pyplot
 
# Keras perceptron neuron layer implementation.
from keras.layers import Dense
# Keras Dropout layer implementation.
from keras.layers import Dropout
# Keras Activation Function layer implementation.
from keras.layers import Activation
# Keras Model object.
from keras.models import Sequential
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


import copy 

 

# load pima indians dataset
dataset = np.loadtxt('data2.csv', delimiter=',')

# split into input and output variables
X = dataset[:,1:12]
y = dataset[:,12]


X_ori=  copy.copy(X)
Y_ori=  copy.copy(y)
def normalizaX(X):
	for j in range(X.shape[1]):
		maximo=max(X[:,j])
		minimo=min(X[:,j])
		for i in range(X.shape[0]):
			X[i,j]=(X[i,j]-minimo)/(maximo-minimo)
		print(max(X[:,j]),min(X[:,j]))
	return X
maximo_y=max(y[:])
minimo_y=min(y[:])
def normalizaY(Y):
	maximo=max(y[:])
	minimo=min(y[:])
	print("Y", maximo,minimo)
	for i in range(len(y)):
		y[i]=(y[i]-minimo)/(maximo-minimo)
	print(max(y[:]),min(y[:]))
	return Y

X=normalizaX(X)
y=normalizaY(y)
print(X.shape,y.shape)
# Data Scaling from 0 to 1, X and y originally have very different scales.
#"X_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
#y_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
#"X_scaled = (X_scaler.fit_transform(X.reshape(-1,1)))
#y_scaled = (y_scaler.fit_transform(y.reshape(-1,1)))
 
#print(X_scaled.shape,y_scaled.shape)
# Preparing test and train data: 60% training, 40% testing.
X_train, X_test, y_train, y_test = cross_validation.train_test_split( \
    X, y, test_size=0.40, random_state=3)


#(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.33, random_state=seed)

# New sequential network structure.
model = Sequential()
 
# Input layer with dimension 1 and hidden layer i with 128 neurons. 
model.add(Dense(11, input_dim=11, activation='relu'))
# Dropout of 20% of the neurons and activation layer.
model.add(Dropout(.2))
model.add(Activation("linear"))
# Hidden layer j with 64 neurons plus activation layer.
model.add(Dense(1000, activation='relu'))
model.add(Activation("sigmoid"))
# Hidden layer k with 64 neurons.
model.add(Dense(64, activation='relu'))
model.add(Activation("sigmoid"))
# Output Layer.
model.add(Dense(1))
 
# Model is derived and compiled using mean square error as loss
# function, accuracy as metric and gradient descent optimizer.
model.compile(loss='mse', optimizer='adam', metrics=['msle','mse', 'mae', 'mape', 'cosine'])
 
# Training model with train data. Fixed random seed:
numpy.random.seed(3)
history=model.fit(X, y, nb_epoch=1000, batch_size=2, verbose=2)

# evaluate the model
scores = model.evaluate(X, y)
print("Accuracy: %.2f%%" % (scores[1]*100))


# Calculate predictions
PredValSet = model.predict(X)

# Save predictions
numpy.savetxt("valresults.csv", PredValSet, delimiter=",")


#Plot actual vs predition for validation set
ValResults = numpy.genfromtxt("valresults.csv", delimiter=",")
plt.plot(y,ValResults,'ro')
plt.show()

#Compute R-Square value for validation set
ValR2Value = r2_score(y,ValResults)
print("Validation Set R-Square=",ValR2Value)


# plot metrics
plt.plot(history.history['mean_squared_logarithmic_error'])
plt.title("erro logarítmico médio")
plt.show()
plt.savefig('img3.png')
#plt.plot(history.history['rmse'])
#plt.title("erro quadrado medio")
#plt.show()
#plt.savefig('img4.png')
plt.plot(history.history['mean_squared_error'])
plt.title("erro quadrado medio")
plt.show()
plt.savefig('img5.png')
plt.plot(history.history['mean_absolute_error'])
plt.title("erro absoluto medio")
plt.savefig('img6.png')
plt.show()
plt.plot(history.history['mean_absolute_percentage_error'])
plt.title("erro percentual absoluto medio")
plt.show()
plt.savefig('img7.png')
plt.plot(history.history['cosine_proximity'])
plt.title("proximidade cosseno")
plt.show()
plt.savefig('img8.png')

#plt.plot(history.history['accuracy'])
#plt.show()


X=normalizaX(X_ori)
predicted = model.predict(X)
# Plot in blue color the predicted adata and in green color the
# actual data to verify visually the accuracy of the model.


for i in range(len(y)):
	y[i]=(y[i]*(maximo_y-minimo_y))+minimo_y
	predicted[i,0]=(predicted[i,0]*(maximo_y-minimo_y))+minimo_y

#print(maximo_y,minimo_y)
#print("Y", max(y[:]),min(y[:]))
#print("S", max(predicted[:,0]),min(predicted[:,0]))

#print(predicted.shape,y.shape)

pyplot.plot(predicted, color="red",label="simulado")
pyplot.plot(y, color="black",label="observacao",alpha=.5)

pyplot.legend(loc='upper left')
pyplot.show()
pyplot.savefig('img9.png')

mes_x = dataset[:,0:1]


pyplot.scatter(mes_x[:,0],predicted[:,0], color='red',label="simulado")
pyplot.scatter(mes_x[:,0], y,  color='black',label="observacao",alpha=.5)
pyplot.xticks(np.arange(12), ('JAN', 'FEV', 'MAR', 'ABR', 'MAI', 'JUN', 'JUL', 'AGO','SET','OUT','NOV','DEZ'))
pyplot.legend(loc='upper left')  
pyplot.show()
pyplot.savefig('img10.png')



def plotResultado(mes,modis,rna):
    import matplotlib.pyplot as plt
    import numpy as np

    #data_a = [[1,2,5], [5,7,2,2,5], [7,2,5]]
    #data_b = [[6,4,2], [1,2,5,3,2], [2,3,5,1]]

    data_a=[[],[],[],[],[],[],[],[],[],[],[],[]]
    data_b=[[],[],[],[],[],[],[],[],[],[],[],[]]

    for l in range(0,len(rna)):
    	
    	data_a[int(mes[l])-1].append(modis[l])
    	data_b[int(mes[l])-1].append(rna[l])

    ticks = ['JAN', 'FEV', 'MAR', 'ABR', 'MAI', 'JUN', 'JUL', 'AGO','SET','OUT','NOV','DEZ']

    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color='black')


    plt.figure()

    #bpl = plt.boxplot(data_a, positions=np.array(xrange(len(data_a)))*2.0-0.4, sym='', widths=0.6,showmeans=True,patch_artist=True)
    #bpr = plt.boxplot(data_b, positions=np.array(xrange(len(data_b)))*2.0+0.4, sym='', widths=0.6,showmeans=True,patch_artist=True)
    bpl = plt.boxplot(data_a, positions=np.array(range(len(data_a)))*2.0-0.4, sym='', widths=0.6,patch_artist=True)
    bpr = plt.boxplot(data_b, positions=np.array(range(len(data_b)))*2.0+0.4, sym='', widths=0.6,patch_artist=True)
   
    set_box_color(bpl, '#2C7BB6') # colors are from http://colorbrewer2.org/
    set_box_color(bpr, '#D7191C')

    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c='#2C7BB6', label='observacao')
    plt.plot([], c='#D7191C', label='simulado')
    plt.legend(loc='upper left')
    plt.title("Ocorrencia de fogo na Serra do Espinhaco entre 2002 e 2017")
    plt.ylabel("Numero de deteccoes de fogo")
    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    plt.xlim(-2, len(ticks)*2)
    #plt.ylim(0, 8)
    plt.tight_layout()
    plt.show()
    plt.savefig('sboxcompare.png')



plotResultado(mes_x,y,predicted[:,0])


