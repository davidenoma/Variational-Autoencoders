import tensorflow as tf


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class ConditionalVAE(tf.keras.Model):
    def __init__(self, num_classes, num_hidden):
        super(ConditionalVAE, self).__init__()
        self.num_hidden = num_hidden

        # Encoder layers
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(num_samples, num_snps)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
        ])

        # Latent variable tf.keras.layers
        self.mu = tf.keras.layers.Dense(num_hidden)
        self.log_var = tf.keras.layers.Dense(num_hidden)

        # Decoder tf.keras.layers
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(num_hidden + num_classes,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(num_snps, activation='sigmoid'),  # Assuming binary data, adjust activation if needed
        ])

        # Class label projector
        self.label_projector = tf.keras.Sequential([
            tf.keras.layers.Dense(num_hidden, activation='relu'),
        ])

    def condition_on_label(self, z, y):
        projected_label = self.label_projector(y)
        combined = tf.concat([z, projected_label], axis=-1)
        return combined

    def reparameterize(self, mu, log_var):
        epsilon = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(0.5 * log_var) * epsilon

    def call(self, inputs):
        x, y = inputs
        # Pass the input through the encoder
        encoded = self.encoder(x)
        # Compute the mean and log variance vectors
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        # Reparameterize the latent variable
        z = self.reparameterize(mu, log_var)
        # Pass the latent variable through the decoder
        decoded = self.decoder(self.condition_on_label(z, y))
        # Return the encoded output, decoded output, mean, and log variance
        return encoded, decoded, mu, log_var

    def sample(self, num_samples, y):
        # Generate random noise
        z = tf.random.normal((num_samples, self.num_hidden))
        # Pass the noise through the decoder to generate samples
        samples = self.decoder(self.condition_on_label(z, y))
        # Return the generated samples
        return samples

# Generate synthetic data with two class labels (0 and 1)
num_samples = 1000
num_snps = 100

# Define feature names based on your data (replace these with your actual feature names)
feature_names = [f'feature_{i}' for i in range(num_snps)]

# Assign class labels (0 or 1)
class_labels = np.random.choice([0, 1], size=num_samples)

# Generate synthetic data
synthetic_data = np.random.randint(0, 3, size=(num_samples, num_snps), dtype='int8')

# Combine data and labels into a DataFrame
synthetic_df = pd.DataFrame(data=synthetic_data, columns=feature_names)
synthetic_df['label'] = class_labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    synthetic_df.drop('label', axis=1), synthetic_df['label'], test_size=0.2, random_state=42
)

# Convert labels to one-hot encoding for TensorFlow model
y_train = tf.one_hot(y_train, depth=2)
y_test = tf.one_hot(y_test, depth=2)

# Define the VAE architecture
num_classes = 2  # Two class labels
num_hidden = 64  # Adjust as needed

cvae = ConditionalVAE(num_classes, num_hidden)

# Define the legacy optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False)

# Define the loss function
def loss_function(recon_x, x, mu, logvar):
    BCE = tf.reduce_sum(tf.keras.losses.binary_crossentropy(x, recon_x))
    KLD = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar))
    return BCE + KLD

# Compile the model
cvae.compile(optimizer=optimizer, loss=loss_function)

# Train the model
epochs = 10
batch_size = 64
cvae.fit([X_train.values, y_train.numpy()], X_train.values, epochs=epochs, batch_size=batch_size)

# After training, you can use the model to generate samples
num_samples_to_generate = 10
random_labels = tf.one_hot([0] * num_samples_to_generate, depth=num_classes)
generated_samples = cvae.sample(num_samples_to_generate, random_labels)
