import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(path):
    data=pd.read_csv(path)
    normal=data[data["Class"]==0].drop("Class",axis=1)
    fraud=data[data["Class"]==1].drop("Class",axis=1)

    return normal,fraud

def scale_data(normal, fraud):
    scaler=StandardScaler()
    x_normal=scaler.fit_transform(normal)
    x_fraud=scaler.transform(fraud)

    return x_normal,x_fraud,scaler

def reconstruction_error(x,recon):
    return np.mean((x-recon)**2,axis=1)