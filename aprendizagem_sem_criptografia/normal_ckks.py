import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler




data = pd.read_csv("creditcard.csv")
matrix = data.to_numpy()



def theta_function(X, y, a, l, x_alternativo, y_alternativo):
  theta = np.zeros((X.shape[1], 1))
  c1, c2 = None, None
  lista1 = []
  lista2 = []
  for i in range(10**3):
    h0 = hy_function(theta[0,0], X[:, [0]])
    b = np.reshape(X[:, 0], (1, X.shape[0]))
    c = h0 - y
    theta[0,0] = theta[0,0] - (a/X.shape[0]) * (b @ c)
    h = h_function(theta[1:, [0]], X[:, 1:])
    d = X[:, 1:].T
    e = h - y
    f = l * np.reshape(theta[1:, 0], (theta.shape[0] -1, 1))
    g = (d @ e  + f)
    theta[1:, :] = theta[1:, :] - ((a/X.shape[0]) * g)
    c1 = c_function(y_alternativo, h_function(theta, x_alternativo), l)[0][0]
    c2 = c_function(y, h_function(theta, X), l)[0][0]
    lista1.append(c1)
    lista2.append(c2)
  return theta, lista1, lista2



def h_function(theta, X):
  g = X @ theta
  h = 1 / (1 + np.exp(-g))
  return h



def hy_function(theta, x):
  g = theta * x
  h = 1 / (1 + np.exp(-g))
  return h



def c_function(y, h, l):
  m = np.size(y)
  t3 = l * (h.T @ h)
  t1 = y.T @ np.log(h + 0.000001)
  t2 = (1 - y.T) @ np.log(1.000001 - h) 
  j0 = - ( t1 + t2 ) / m  + t3 / (2 * m)
  return j0



matrixAux0 = matrix[matrix[:, 30] == 0]
matrixAux1 = matrix[matrix[:, 30] == 1]

matrixAux = np.zeros((3690,31))
matrixAux[:3321] = matrixAux0[:3321]
matrixAux[3321:] = matrixAux1[:369]
X = matrixAux[:, :30]
y = matrixAux[:, [30]]

matrixAuxTest = np.zeros((1230, 31))
matrixAuxTest[:1107] = matrixAux0[3321:4428]
matrixAuxTest[1107:] = matrixAux1[369:492]

xTeste = matrixAuxTest[:, :30]
yTeste = matrixAuxTest[:, [30]]


# matrixAux0 = matrix[matrix[:,30] == 0] # 284315 x 31
# matrixAux1 = matrix[matrix[:,30] == 1] # 492 x 31
# matrixAux = np.zeros((440,31))
# matrixAux[:400] = matrixAux0[:400]
# matrixAux[400:] = matrixAux1[:40]
# X = matrix[:150000, :30]
# y = matrix[:150000, [30]]

# matrixAuxTest = np.zeros((99, 31))
# matrixAuxTest[:90] = matrixAux0[400:490]
# matrixAuxTest[90:] = matrixAux1[400:409]

# xTeste = matrix[150000:, :30]
# yTeste = matrix[150000:, [30]]



scaler = StandardScaler()
X = scaler.fit_transform(X)
xTeste = scaler.transform(xTeste)



print("treinando")

t, lista1, lista2 = theta_function(X, y, 0.01, 0.001, xTeste, yTeste)



print("hipotese")

hipotese = h_function(t, xTeste)
variavel = hipotese > 0.5
print(classification_report(yTeste, variavel))


fig, ax = plt.subplots()
indices_lista = [x for x in range(10**3)]
plt.rcParams["font.size"] = "10"
plt.rcParams["font.weight"] = "bold"
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
  label.set_fontsize(12)
  label.set_fontweight("bold")
plt.plot(indices_lista, lista1, label="custo parcial do conjunto de teste")
plt.plot(indices_lista, lista2, label="custo parcial do conjunto de treinamento")
plt.xlabel("repetições do treinamento", fontweight="bold", fontsize=12)
plt.ylabel("custo", fontweight="bold", fontsize=12)
plt.title("Gráfico dos custos parciais - alpha = 0,01; lambda = 0,001", fontweight="bold")
plt.legend()
plt.grid()
fig.savefig("custos.eps", format="eps")
plt.show()