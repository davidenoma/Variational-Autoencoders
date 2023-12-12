import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from models.betavae_ import BetaVAE

class CVAE(BetaVAE):
  def __init__(self, latent_dim, input_dims=(28, 28, 1), kernel_size=(3, 3), strides=(2, 2), num_components=10, prefix='tcvae'):
    super(CVAE, self).__init__(latent_dim, input_dims=input_dims, kernel_size=kernel_size, strides=strides, prefix=prefix)
    self.num_classes = num_components

    self.cond_encoder = tf.keras.Sequential([
        tf.keras.layers.Dense(512),
        tf.keras.layers.Dense(256),
        tf.keras.layers.Dense(2 * self.latent_dim),
    ])

  def forward(self, batch, apply_sigmoid=False):
    mean_z, logvar_z = self.encode(batch)
    z_sample = self.reparameterize(mean_z, logvar_z)
    x_pred = self.decode({'z': z_sample, 'y': batch['y']}, apply_sigmoid=apply_sigmoid)

    return mean_z, logvar_z, z_sample, x_pred

  def encode(self, batch):
    params_z = self.encoder(batch['x'])
    mean_z_u, logvar_z_u = tf.split(
      self.cond_encoder(tf.concat([params_z, batch['y']], axis=1)),
      num_or_size_splits=2, axis=-1
    )
    return mean_z_u, logvar_z_u

  def decode(self, batch, apply_sigmoid=False):
    logits = self.decoder(tf.concat([batch['z'], batch['y']], axis=1))
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs
    return logits

  def generate(self, eps=None, num_generated_images=15, **kwargs):
    if eps is None:
      eps = tf.random.normal(shape=(num_generated_images, self.latent_dim), dtype=tf.float32)

    num_samples = eps.shape[0]

    if 'y' not in kwargs and 'target' not in kwargs:
      cond = np.zeros((num_samples, self.num_classes))
      target = 0
      for i in range(num_samples):
        cond[i, target] = 1.0
        target += 1
        if target >= self.num_classes:
            target = 0
      cond = tf.convert_to_tensor(cond, dtype=tf.float32)
    elif 'target' in kwargs:
      cond = np.zeros((num_samples, self.num_classes))
      target = kwargs['target']
      for i in range(num_samples):
        cond[i, target] = 1.0
    else:
      cond = kwargs['y'][0:num_samples]
    
    return self.decode({'z': eps, 'y': cond}, apply_sigmoid=True)



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

# Instantiate the CVAE model
latent_dim = 10  # You can choose the desired latent dimension
cvae_model = CVAE(latent_dim, num_components=num_classes)

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Adjust the learning rate as needed
cvae_model.compile(optimizer, loss='binary_crossentropy')  # Adjust the loss function as needed
cvae_model.train_step(batch=32,optimizers=optimizer)
# Train the model
batch_size = 32  # You can adjust the batch size as needed
epochs = 10  # You can adjust the number of training epochs
cvae_model.fit(X_train, {'x': X_train, 'y': y_train}, batch_size=batch_size, epochs=epochs)

# Generate synthetic data using the trained model
generated_samples = cvae_model.generate(num_generated_images=15)



