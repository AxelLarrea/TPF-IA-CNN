# Alumno: Axel Larrea

import numpy as np
import pandas as pd
import os
import time


def getDataSet(csvName):
    """ Función para leer un archivo csv.
        csvName: nombre del archivo .csv.
    """
    current_path = os.getcwd()
    current_path = current_path + "/"
    csv_url = current_path + csvName
    data = pd.read_csv(csv_url)
    return data

def getSampleOfData(dataset, sample_quant):
    """ Función para obtener una muestra aleatoria del archivo .csv leído.
        sample_quant: cantidad de ejemplos a tomar.
    """
    data_sample = dataset.sample(n=sample_quant)
    return data_sample

# def softmax(Z):
#     """ Función de activación que se aplica a cada dato de la matriz Z """
#     exp = np.exp(Z - np.max(Z)) #El np.max(Z) es para evitar el overflow
#     return exp / exp.sum(axis=0)

def sigmoid(x):
    """ Función de activación que se aplica a cada dato de la matriz x. """
    return 1/(1+np.exp(-x))

def get_accuracy(predictions, data_test):
    """ Función para obtener promedio de aciertos.
        predictions: clases obtenidas de las predicciones.
        data_test: clases obtenidas de la columna existente Cluster del dataset
    """
    return np.sum(predictions == data_test)/data_test.size

def normalization(vec):
    """ Función para normalizar los datos del vector vec.
        vec: array correspondiente a una columna de datos de la matriz.
    """
    vec = vec/np.sqrt(np.dot(vec,vec.T))
    return vec

def normalization_all(mat):
    """ Función para normalizar todos los datos de la matriz mat.
        mat: matriz de datos.
    """
    M_all=[]
    for i in range(len(mat)):
        K=normalization(mat[i])
        M_all.append(K)
    return M_all

def dataset_normalization(mat, columns):
    """ Función para normalizar todos los datos de la matriz mat.
        mat: matriz de datos(dataset).
        columns: columnas de datos a tomar del .csv.
    """
    M_all=[]
    for i in columns:
        K=normalization(mat[i])
        M_all.append(K)
    return M_all

def load_data(quantity=0):
    """ Función principal para cargar los datos.
        quantity: cantidad de datos a cargar.
    """
    data = getDataSet("datosIA.csv")
    if (quantity != 0):
        data_sample = getSampleOfData(data, quantity)
        dataN = dataset_normalization(data_sample, ['x', 'y', 'z', 'h'])
    else:
        data_sample = data
        dataN = dataset_normalization(data_sample, ['x', 'y', 'z', 'h'])
    return dataN, data_sample

def data_formater(data):
    """ Función que le da el formato final al conjunto de datos, para luego utilizarlo en la generación del .csv.
        data: dataset.
    """
    data = data.drop(['testID', 'Instance_number', 'idNode', 'Cluster'], axis=1)
    data = data.reset_index(drop=True)
    data = data.assign(cluster=[0]*len(data))
    return data

def data_formater_cluster(data):
    """ Función que le da el formato final al conjunto de datos, para luego utilizarlo en la generación del .csv.
        data: dataset.
    """
    data = data.drop(['testID', 'Instance_number', 'idNode', 'x', 'y', 'z', 'h'], axis=1)
    data = data.reset_index(drop=True)
    data = data['Cluster']
    return data

def generateCSV(predictions, mainData):
    """ Función que generará el archivo .csv de salida.
        predictions: predicciones(clases) obtenidas luego de que los datos se procesen.
        mainData: dataset con el formato necesario para la generación del archivo .csv.
    """
    for i in range(len(predictions)):
        mainData.at[i, 'cluster'] = predictions[i]
    return mainData



class competitive_network(object):
    """ Objeto cnn """
    def __init__(self, x_dim, output_num, a):
        """ x_dim: cantidad de columnas.
            output_num: cantidad de clases en las que se clasificará.
            a: learning rate(velocidad de aprendizaje).
        """
        W = np.random.rand(output_num, x_dim)
        self.W = normalization_all(W)
        self.a = a
        
    def forward_propagation(self, x):
        """ Función para la propagación hacia adelante, toma los registros(filas) de la matriz principal(dataset), los cuales se multiplican(producto punto) con cada 
            peso correspondiente a la matriz W, luego pasa los datos por la función de activación de la neurona
            para devolver un dato de salida.
            x: vector de datos correspondiente a un registro(fila).
        """
        z_layer = np.dot(self.W, x.T)
        a_layer = sigmoid(z_layer)
        argmax = np.argmax(a_layer)
        return argmax
    
    def back_propagation(self, argmax, x):
        """ Función para la propagación hacia atras, esta actualiza los pesos de la matriz de pesos correspondientes a las neuronas.
            argmax: es un dato de salida proveniente de forward propagation.
            x: vector correspondiente a un registro.
        """
        self.W[argmax] = self.a * (x - self.W[argmax])
        self.W[argmax] = normalization(self.W[argmax])
        self.a-=self.decay
    
    def train(self, X, num_iter):
        """ Función para entrenar las neuronas
            X: matriz de datos a utilizar para entrenar las neuronas
            num_iter: numero de iteraciones
        """

        X = np.array(X)
        self.decay = self.a / num_iter
        for item in range(num_iter):
            for i in range(X.shape[0]):
                argmax=self.forward_propagation(X[i])
                self.back_propagation(argmax, X[i])
            
    def prediction(self, X_test):
        """ Función que predice la clase a la que pertencerá cada registro(fila) del documento.
            X_test: dataset utilizado en el entrenamiento de las neuronas
        """
        sample_num = np.shape(X_test)[0]
        predict_results = []
        for i in range(sample_num):
            predict_result = self.forward_propagation(X_test[i])
            predict_results.append(predict_result)
        return predict_results
        


if __name__ == '__main__':
    start = time.time()
    print('-----------------------1.Carga de datos------------------------')
    data, data_sample = load_data(1000)                                                     # Cantidad de datos a cargar (introducir numero en load_data())
    dataMat = np.mat(data).T

    print('--------------------2.Ajuste de paramétros------------------')
    num_iter = 1000                                                                      # Numero de iteraciones
    x_dim = np.shape(dataMat)[1]
    output_num = int(input('Ingresar cantidad de clases en las que se clasificara: '))  # Cantidad de clases
    lr = 0.15                                                                           # Learning rate

    print('----------------------3.Entrenamiento del modelo-----------------------')
    cnn = competitive_network(x_dim, output_num, lr)
    cnn.train(dataMat,num_iter)

    print('----------------------4.Prediccion------------------------')
    predict_results = cnn.prediction(dataMat)

    print('-------------------5.Generando .csv de salida----------------------')

    data_test = data_formater_cluster(data_sample)
    data_sample = data_formater(data_sample)
    csv_output = generateCSV(predict_results, data_sample)
    csv_output.to_csv('salida.csv', header=True, index=False)

    print('Registros procesados: ', len(data_sample))
    print('Columnas procesadas: ', np.shape(dataMat)[1])
    print('Acierto: ', str(get_accuracy(predict_results, data_test)*100) + '%')
    total_time = time.time() - start
    print('Tiempo de ejecución: ', round(total_time, 2), 'segundos')