import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.utils import plot_model
from matplotlib import pyplot as plt

# Generate synthetic data
num_samples = 1000
num_snps = 100

# Define feature names based on your data (replace these with your actual feature names)
feature_names = [f'feature_{i}' for i in range(num_snps)]

synthetic_data = np.random.randint(0, 3, size=(num_samples, num_snps), dtype='int8')
synthetic_df = pd.DataFrame(data=synthetic_data, columns=feature_names)

# Generate class labels (0 or 1)
synthetic_df['class_label'] = np.random.randint(0, 2, size=num_samples)

X_train, X_test = train_test_split(synthetic_df, test_size=0.2, random_state=42)

# Define the latent dimension
latent_dim = int(X_train.shape[1] / 2)  # Ensure that latent_dim is an integer

# Build the Variational Autoencoder (VAE) model with class label as input
input_dim = X_train.shape[1]

# Encoder
encoder = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(input_dim + 1,)),  # Add 1 for class label
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2 * latent_dim)  # Two times latent_dim for mean and log variance
])

# Decoder
decoder = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(latent_dim,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(input_dim, activation='sigmoid')
])

# VAE Model with class label as input
class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mean, log_var):
        epsilon = tf.random.normal(shape=tf.shape(mean))
        return mean + tf.exp(0.5 * log_var) * epsilon

    def call(self, inputs):
        # Concatenate the feature columns and class label
        concatenated_inputs = tf.concat([inputs[0], tf.cast(inputs[1], dtype=tf.float32)], axis=1)
        z_mean, z_log_var = tf.split(self.encoder(concatenated_inputs), num_or_size_splits=2, axis=1)
        z = self.reparameterize(z_mean, z_log_var)
        reconstructed = self.decoder(z)
        return reconstructed

vae = VAE(encoder, decoder)

# Loss function for VAE with class label
def vae_loss(x, x_reconstructed):
    z_mean, z_log_var = tf.split(vae.encoder(x), num_or_size_splits=2, axis=1)
    reconstruction_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(x, x_reconstructed))
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    return reconstruction_loss + kl_loss

# Compile the VAE model
vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), loss=lambda x, x_reconstructed: vae_loss(x, x_reconstructed))

# Train the VAE with class label
history = vae.fit(
    x=[X_train.drop('class_label', axis=1), X_train['class_label']],
    y=X_train.drop('class_label', axis=1),
    epochs=150,
    batch_size=32,
    validation_data=(
        [X_test.drop('class_label', axis=1), X_test['class_label']],
        X_test.drop('class_label', axis=1)
    )
)

# Rest of your code...

# Extract encoder weights
encoder_weights = vae.encoder.get_weights()

# Analyze encoder weights to identify representative features
feature_importance = []

# Combine the absolute weights across all latent dimensions
for i in range(latent_dim):
    # Get weights for the i-th latent dimension
    weights = encoder_weights[0][:, i]  # Weights for the i-th latent dimension
    # Calculate the absolute sum of weights for each feature
    absolute_weights = np.abs(weights)
    feature_importance.append(absolute_weights)

# Combine feature importance across all latent dimensions
combined_importance = np.sum(feature_importance, axis=0)

# Get the names of the original features
original_feature_names = synthetic_df.columns[:-1]  # Exclude the class label

# Print the most representative features across all latent dimensions
sorted_indices = np.argsort(combined_importance)[::-1]
top_features = [original_feature_names[j] for j in sorted_indices]
print("Most Representative Features Across All Latent Dimensions:")
print(top_features)

# Plot training history
plt.plot(pd.DataFrame(history.history))
plt.ylabel("loss")
plt.xlabel("epochs")
plt.show()
plt.savefig("loss_and_epoch.png")

# Save model architecture to a file
plot_model(vae, show_shapes=True, to_file='vae_model.png')
