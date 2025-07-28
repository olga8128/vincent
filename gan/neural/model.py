import tensorflow as tf
from tqdm import tqdm
import os

class Model:
    GEN_NAME="_generator"
    DIS_NAME="_discriminator"
    def __init__(self,target_size,save_filename=None,generator_name=None,discriminator_name=None):
        self.target_size = target_size
        self.save_filename = save_filename
        self.discriminator = self.build_discriminator()
        if generator_name:
            self.generator = tf.keras.models.load_model(generator_name)
        else:
            self.generator = self.build_generator()
            print("🍓 Creando generador")
        if discriminator_name:
            self.discriminator = tf.keras.models.load_model(discriminator_name)
        else:
            self.discriminator = self.build_discriminator()
            print("🥝 Creando discriminador")

    # --- Discriminador PatchGAN ---
    def build_discriminator(self):
        inp = tf.keras.Input(shape=(self.target_size, self.target_size, 3))
        tar = tf.keras.Input(shape=(self.target_size, self.target_size, 3))
        x = tf.keras.layers.Concatenate()([inp, tar])

        x = tf.keras.layers.Conv2D(64, 4, strides=2, padding='same')(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Conv2D(128, 4, strides=2, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Conv2D(256, 4, strides=2, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Conv2D(1, 4, padding='same')(x)

        return tf.keras.Model([inp, tar], x, name="Discriminator")

    # --- Generador (U-Net) ---
    def build_generator(self):
        inputs = tf.keras.Input(shape=(self.target_size, self.target_size, 3))

        def downsample(x, filters):
            x = tf.keras.layers.Conv2D(filters, 4, strides=2, padding="same")(x)
            x = tf.keras.layers.BatchNormalization()(x)
            return tf.keras.layers.LeakyReLU()(x)

        def upsample(x, skip, filters):
            x = tf.keras.layers.Conv2DTranspose(filters, 4, strides=2, padding="same")(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            return tf.keras.layers.Concatenate()([x, skip])

        d1 = downsample(inputs, 64)
        d2 = downsample(d1, 128)
        d3 = downsample(d2, 256)
        d4 = downsample(d3, 512)
        d5 = downsample(d4, 512)
        d6 = downsample(d5, 512)

        b = tf.keras.layers.Conv2D(512, 4, strides=2, padding="same")(d6)
        b = tf.keras.layers.ReLU()(b)

        u1 = upsample(b, d6, 512)
        u2 = upsample(u1, d5, 512)
        u3 = upsample(u2, d4, 512)
        u4 = upsample(u3, d3, 256)
        u5 = upsample(u4, d2, 128)
        u6 = upsample(u5, d1, 64)

        outputs = tf.keras.layers.Conv2DTranspose(3, 4, strides=2, padding="same", activation="tanh")(u6)

        return tf.keras.Model(inputs, outputs, name="Generator")

    # --- Pérdidas ---

    def generator_loss(self,disc_generated_output, gen_output, target, loss_obj):
        gan_loss = loss_obj(tf.ones_like(disc_generated_output), disc_generated_output)
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        return gan_loss + 100 * l1_loss

    def discriminator_loss(self,disc_real_output, disc_generated_output, loss_obj):
        real_loss = loss_obj(tf.ones_like(disc_real_output), disc_real_output)
        fake_loss = loss_obj(tf.zeros_like(disc_generated_output), disc_generated_output)
        return real_loss + fake_loss

    # --- Entrenamiento ---
    @tf.function
    def train_step(self,input_image, target, generator, discriminator, gen_optimizer, disc_optimizer):
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
        loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = generator(input_image, training=True)

            disc_real_output = discriminator([input_image, target], training=True)
            disc_generated_output = discriminator([input_image, gen_output], training=True)

            gen_loss = self.generator_loss(disc_generated_output, gen_output, target, loss_obj)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output, loss_obj)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        gen_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        disc_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        return gen_loss, disc_loss

    def train(self,dataset,epochs):
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.0002,  
            decay_steps=100000,           
            decay_rate=0.96,             
            staircase=True              
        )
        gen_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.5)
        disc_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.5)

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs }")
            for step, (inp, tar) in enumerate(tqdm(dataset.dataset)):
                gen_loss, disc_loss = self.train_step(inp, tar, self.generator, self.discriminator, gen_optimizer, disc_optimizer)

            print(f"Gen loss: {gen_loss.numpy():.4f} | Disc loss: {disc_loss.numpy():.4f}")
            self.save_generator_model()

    def save_generator_model(self):
        dis_name=self.save_filename.replace(self.GEN_NAME,self.DIS_NAME)
        self.generator.save(self.save_filename)
        self.discriminator.save(dis_name)
        print(f"✅ Generador guardado en {self.save_filename}")
        print(f"✅ Discriminador guardado en {dis_name}")

