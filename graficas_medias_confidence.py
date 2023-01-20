import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

path = "C:/Users/pablo/Desktop/CodigoTFM/disentangled_experiments-main/disentangled_experiments-main/trained_models/VAE/val_loss_32.csv"
datos = pd.read_csv(path, sep = ',')
path2 = "C:/Users/pablo/Desktop/CodigoTFM/disentangled_experiments-main/disentangled_experiments-main/trained_models/VAE/loss_9.csv"
datos2= pd.read_csv(path2, sep = ',')

y1 = datos['1'] 
y2 = datos['2']
y3 = datos['3']
y4 = datos['4']
y5 = datos['5']
y6 = datos['6']
y7 = datos['7']
y8 = datos['8']
y9 = datos['9']
y10 = datos['10']
y_average = datos['media']
y_desv = datos['DESV']
y_mas = datos['mediamas']
y_menos = datos['mediamenos']

# y12 = datos2['1'] 
# y22= datos2['2']
# y32 = datos2['3']
# y42 = datos2['4']
# y52 = datos2['5']
# y62 = datos2['6']
# y72 = datos2['7']
# y82 = datos2['8']
# y92 = datos2['9']
# y102 = datos2['10']
# y_average2 = datos2['media']
# y_desv2 = datos2['DESV']
# y_mas2 = datos2['mediamas']
# y_menos2 = datos2['mediamenos']

x = np.linspace(0,y1.shape[-1], y1.shape[-1]) 

plt.figure(figsize=(20, 10))
plt.axis([0, 50, 0, 150])

plt.scatter(x,y1,  color='blue', linewidths=0.05)
plt.scatter(x,y2,  color='blue', linewidths=0.05)
plt.scatter(x,y3,  color='blue', linewidths=0.05)
plt.scatter(x,y4,  color='blue', linewidths=0.05)
plt.scatter(x,y5,  color='blue', linewidths=0.05)
plt.scatter(x,y6, color='blue', linewidths=0.05)
plt.scatter(x,y7, color='blue', linewidths=0.05)
plt.scatter(x,y8, color='blue', linewidths=0.05)
plt.scatter(x,y9, color='blue', linewidths=0.05)
plt.scatter(x,y10, color='blue', linewidths=0.05)

plt.plot(y_average, label='Average Val Loss', color = 'blue')
#plt.plot(y_average2, label='Average Loss', color = 'red')
#plt.scatter(x,t_average, label='Running average')
plt.ylabel('Wmse')
plt.xlabel('Epochs')
plt.grid(linestyle=':')
# plot our confidence band
plt.fill_between(x, y_mas, y_menos, alpha=0.2, color='tab:orange')
# plt.fill_between(x, y_mas2, y_menos2, alpha=0.2, color='tab:green')
plt.legend(loc='upper right')
plt.show()


