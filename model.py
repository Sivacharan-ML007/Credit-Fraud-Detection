import tensorflow as tf
from sklearn.model_selection import train_test_split
from utils import load_data, scale_data 

DATA_PATH = "data/creditcard.csv"
MODEL_PATH = "model/autoencoder.h5"


# ===================== LOAD =====================
normal,fraud=load_data(DATA_PATH)
x_normal,x_fraud,scaler=scale_data(normal, fraud)

x_train,x_test = train_test_split(
    x_normal, test_size=0.2, random_state=42
)

input_dim = x_train.shape[1]

reg = tf.keras.regularizers.l1_l2(l1=0.001,l2=0.001)

# ===================== MODEL =====================
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32,activation='relu',
                          kernel_regularizer=reg,
                          input_shape=(input_dim,)),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(16,activation='relu',
                          kernel_regularizer=reg),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(8,activation='relu'),

    tf.keras.layers.Dense(16,activation='relu'),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(32,activation='relu'),
    tf.keras.layers.Dense(input_dim,activation='linear')
])


# ===================== TRAIN =====================
model.compile(optimizer='adam', loss='mse')

model.fit(
    x_train, x_train,
    epochs=15,
    batch_size=32,
    validation_data=(x_test,x_test),
    shuffle=True
)

# ===================== SAVE =====================
model.save(MODEL_PATH)

print(f"\nModel saved at: {MODEL_PATH}")