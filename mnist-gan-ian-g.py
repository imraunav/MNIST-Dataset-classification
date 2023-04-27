import tensorflow as tf
from tensorflow.keras import layers, Model, Input, losses, callbacks
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
import numpy as np

def make_generator():
    noise = Input(shape=(100,)) #nvis in goodfeli 
    h0 = layers.Dense(units=1200, activation='relu', name="h0")(noise)
    h1 = layers.Dense(units=1200, activation='relu', name="h1")(h0)
    y = layers.Dense(units=784, activation='sigmoid', name="y")(h1)
    gen_img = layers.Reshape((28,28,1),name="Gen-img")(y)
    return Model(inputs=[noise,],
                 outputs=[gen_img,],
                 name="Generator")

def make_discriminator():
    # don't have maxout activation fucntions so using relu instead
    img = Input(shape=(28,28,1))
    flat_img = layers.Flatten()(img)
    h0 = layers.Dense(units=240, activation='relu', name="h0")(flat_img)
    h1 = layers.Dense(units=240, activation='relu', name="h1")(h0)
    h1 = layers.Dropout(0.8)(h1)
    y = layers.Dense(units=1, activation='sigmoid', name="y")(h1)
    return Model(inputs=[img,],
                 outputs=[y,],
                 name="Discriminator")

class myGan(Model):
    def __init__(self, generator, discriminator, batch_size=100, latent_dim=1):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size=batch_size
        self.latent_dim = latent_dim

    def compile(self, generator_optimizer, discriminator_optimizer, generator_loss, discriminator_loss):
        super().compile()
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss

    def train_step(self, real_images):

        dloss_fn = self.discriminator_loss
        gloss_fn = self.generator_loss
        d_optimizer = self.discriminator_optimizer
        g_optimizer = self.generator_optimizer

        # Random sample from latent dim
        random_latent_vector = tf.random.uniform(shape=(self.batch_size, self.latent_dim))

        # Decode and generate image
        generated_images = self.generator(random_latent_vector)

        all_images = tf.concat([real_images, generated_images], axis=0)
        # Assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.ones((self.batch_size, 1)), tf.zeros((tf.shape(real_images)[0], 1))], axis=0
        )

        # some trick?
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        with tf.GradientTape() as tape:
            predictions = self.discriminator(all_images, training=True)
            d_loss = dloss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
        
        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(self.batch_size, self.latent_dim))
        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((self.batch_size, 1))
        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            generated_images = self.generator(random_latent_vector, training=True)
            predictions = self.discriminator(generated_images, training=False)
            g_loss = gloss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        return {"Discriminator loss":d_loss, "Generator loss":g_loss}
        
def main():
    generator = make_generator()
    generator.summary()
    discriminator = make_discriminator()
    discriminator.summary()

    my_gan = myGan(generator, discriminator)

    generator_loss = losses.BinaryCrossentropy()
    discriminator_loss = losses.BinaryCrossentropy()
    generator_optimizer = Adam(learning_rate=1e-4)
    discriminator_optimizer = Adam(learning_rate=1e-3)


    my_gan.compile(generator_loss=generator_loss,
                discriminator_loss=discriminator_loss,
                generator_optimizer=generator_optimizer,
                discriminator_optimizer=discriminator_optimizer)
    (x_train, _), (x_test, _) = mnist.load_data()
    all_digits = np.concatenate([x_train, x_test])
    all_digits = all_digits.astype("float32") / 255.0
    all_digits = np.reshape(all_digits, (-1, 28, 28, 1))

    batch_size=100
    dataset = tf.data.Dataset.from_tensor_slices(all_digits)
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

    set_callbacks = [callbacks.CSVLogger(filename="MNIST-trianing-log.csv", separator=",", append=True),
                callbacks.TensorBoard('./logs', update_freq=1),
                callbacks.BackupAndRestore(backup_dir="./tmp/backup"),
                callbacks.ModelCheckpoint(
                filepath="./checkpoints/model_best.hdf5",
                save_weights_only=True,
                monitor='Generator loss',
                mode='min',
                save_best_only=True)]

    hist = my_gan.fit(dataset, epochs=5000, batch_size=batch_size, callbacks=set_callbacks)

if __name__ == "__main__":
    main()