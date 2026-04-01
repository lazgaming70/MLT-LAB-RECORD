print("727723EUIT223 - SHOBAN CHIDDARTH")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

np.random.seed(42)
normal_data = np.random.normal(0, 1, (1000, 10))
fraud_data = np.random.normal(4, 1, (50, 10))
X = np.vstack([normal_data, fraud_data])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.title("PCA Feature Reduction")
plt.show()

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
plt.title("t-SNE Visualization")
plt.show()

input_dim = X_scaled.shape[1]
input_layer = Input(shape=(input_dim,))

encoded = Dense(6, activation="relu")(input_layer)
encoded = Dense(3, activation="relu")(encoded)
decoded = Dense(6, activation="relu")(encoded)
decoded = Dense(input_dim, activation="sigmoid")(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(
    X_scaled,
    X_scaled,
    epochs=20,
    batch_size=32,
    shuffle=True
)

reconstructed = autoencoder.predict(X_scaled)
mse = np.mean(np.power(X_scaled - reconstructed, 2), axis=1)

threshold = np.mean(mse) + 2 * np.std(mse)
anomalies = mse > threshold

print("Number of detected anomalies:", np.sum(anomalies))