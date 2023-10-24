import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.models import Model
from tensorflow.keras.layers import Input, Dense, TimeDistributed, Flatten, Concatenate, LSTM, MaxPool1D, Reshape, Conv2DTranspose, BatchNormalization, UpSampling1D, Cropping1D, Cropping2D, Conv1D
from tensorflow.keras import backend as K
import wandb
from wandb.keras import WandbCallback
import pickle
import variables

# Load variables
vars = variables.load_variables_from_json()
window_duration = vars["window_duration"]  # TODO: Set this based on ... something (there'll have to be an outer layer running the music through the various window durations / models)
latent_dim = vars["latent_dim"]
time_resolution = vars["time_resolution"]
model_cutoff = vars["model_cutoff"]

architecture_ID = "R-F"
wav_file_name = "wav-file-name"
checkpoint_path = f'C:/Users/MrLin/Documents/Experiments/Deep Audio Embedding/{wav_file_name}/saved models/last-{architecture_ID}'

# load spectrogram and infer variables
# Load S_mag_crop from the pickled object
with open(f"C:/Users/MrLin/Documents/Experiments/Deep Audio Embedding/{wav_file_name}/S_mag_crop.pkl", "rb") as f:
    S_mag = pickle.load(f)

# Infer T, img_width, and img_height
T = S_mag.shape[1]
img_width, img_height = S_mag.shape[-2:]
img_size = (img_width, img_height)

wandb.login()
id = wandb.util.generate_id()
print(f"Wandb ID: {id}")
wandb.init(entity='your-wandb', project='Deep Audio Embedding', group='Vivaldi', name=architecture_ID, id=id, resume="allow",
           config=vars["wandb_config"])
config = wandb.config

S_mag = (S_mag-np.min(S_mag))/(np.max(S_mag) - np.min(S_mag))

# At time of writing, no variables were being changed, but this is where this should go...
variables.write_variables_to_json(vars)


# build generator of subimages of Spectrogram, sliding across time
def inputs_generator(t_start, t_end, S, dt_, img_height, img_width):
    t = t_start
    while t <= t_end:
        clip_tensor = []
        top_frames = []

        for j in range(t - dt_*(l - 1), t + 1, dt_):
            sub_img = np.transpose(S[:img_height, j:j+img_width])
            clip_tensor.append(sub_img.astype(np.float32))  # pixel values are normalized to [0,1]
        clip_tensor = np.array(clip_tensor)

        for j in range(t, t + 2*dt_, dt_):
            sub_img = np.transpose(S[:img_height, j:j + img_width])
            top_frames.append(sub_img.astype(np.float32))
        top_frames = np.array(top_frames)

        yield ((clip_tensor, top_frames), 0)  # 0 represents the target output of the model. the metric in .compile is computed using this. the addloss layer outputs the loss itself just for convienience.
        t += dt_


l = config.l
clip_tensor_shape = (l,) + img_size  # (l tensor slices, time-steps, freq)
top_frames_shape = (2,) + img_size  # (2 tensor slices, time-steps, freq)

batch_size = config.batch_size
epochs = config.epochs

t_start = int((l - 1))  # first frame index in the valid range
t_end = int(len(T) - img_width - 1)  # last frame index in the valid range, assuming images start at t=0 and go to t=T-1
samples = np.floor(t_end - t_start + 1)
steps_per_epoch_tr = int(np.floor(samples / batch_size)) - 1 # number of batches

ds_train = tf.data.Dataset.from_generator(
    inputs_generator,
    args=[t_start, t_end, S_mag, 1, img_height, img_width],
    output_types=((tf.float32, tf.float32), tf.float32),
    output_shapes=(((l, img_width, img_height), (2, img_width, img_height)), ()))

ds_train = ds_train.shuffle(int(samples * 0.7),
                            # turn off shuffle if you are training on a subset of the data for hyperparameter tuning and plan to visualize performance on the first steps_per_epoch_tuning batches of the TRAINING set
                            reshuffle_each_iteration=False)  # the argument into shuffle is the buffer size. this can be smaller than the number of samples, especially when using larger datasets
ds_train = ds_train.batch(batch_size, drop_remainder=True)  # insufficient data error was thrown without adding the .repeat(). I added the +1 dataset at the end for good measure
ds_train = ds_train.repeat(epochs + 1)


# Short-Window Encoder
def short_window_encoder(img_size):
    layers_to_add = len(img_size) // config.layer_scaling_threshold

    base_filters = config.encoder_base_filters_short
    initial_filter_size = 3

    encoder_input = Input(shape=img_size, name='encoder_input')
    x = encoder_input
    for i in range(3 + layers_to_add):
        filter_size = initial_filter_size
        x = Conv1D(base_filters * (2 ** min(i, 2)), filter_size, strides=1, activation='relu', padding='same',
                   kernel_initializer='RandomNormal', bias_initializer='zeros')(x)
        x = MaxPool1D(pool_size=2, padding='valid', data_format='channels_last')(x)

    x = Flatten(data_format='channels_last')(x)
    z_encoder = Dense(latent_dim, activation='linear', name='z_encoder',
                      kernel_initializer='RandomNormal', bias_initializer='zeros')(x)

    return Model(encoder_input, z_encoder, name='ShortWindow_Encoder')


# Long-Window Encoder
def long_window_encoder(img_size):
    layers_to_add = len(img_size) // (config.layer_scaling_threshold // 2)  # potentially more layers for long-window model

    base_filters = config.encoder_base_filters_long
    initial_filter_size = 4  # larger initial filter for capturing broader patterns

    encoder_input = Input(shape=img_size, name='encoder_input')
    x = encoder_input
    for i in range(4 + layers_to_add):  # start with a deeper architecture
        filter_size = initial_filter_size if i == 0 else 3
        x = Conv1D(base_filters * (2 ** min(i, 2)), filter_size, strides=1, activation='relu', padding='same',
                   kernel_initializer='RandomNormal', bias_initializer='zeros')(x)
        x = MaxPool1D(pool_size=2, padding='valid', data_format='channels_last')(x)

    x = Flatten(data_format='channels_last')(x)
    z_encoder = Dense(latent_dim, activation='linear', name='z_encoder',
                      kernel_initializer='RandomNormal', bias_initializer='zeros')(x)

    return Model(encoder_input, z_encoder, name='LongWindow_Encoder')


# Decoder for Short-Window Model
def short_window_decoder(window_duration):
    width_factor = len(img_size) // config.layer_scaling_threshold
    layers_to_add = min(config.decoder_max_layers, int(width_factor))  # Number of layers, capped at config.decoder_max_layers

    decoder_input = Input(shape=(latent_dim,), name='decoder_input')
    x = Dense(config.decoder_base_filters * latent_dim, activation='relu', use_bias=False, kernel_initializer='RandomNormal')(
        decoder_input)
    x = Reshape((latent_dim, config.decoder_base_filters))(x)

    for i in range(layers_to_add):
        x = UpSampling1D(2)(x)
        x = Conv1D(config.decoder_base_filters // (2 ** i), 3, strides=1, activation='relu', padding='same',
                   kernel_initializer='RandomNormal', bias_initializer='zeros')(x)
        x = BatchNormalization(axis=-1)(x)

    z_decoded = Conv1D(img_height, 1, strides=1, activation='sigmoid', padding='same',
                       kernel_initializer='RandomNormal', bias_initializer='zeros')(x)
    z_decoded_crop = Cropping1D(1)(z_decoded)

    return Model(decoder_input, z_decoded_crop, name='ShortWindow_Decoder')


# Decoder for Long-Window Model
def long_window_decoder(window_duration):
    width_factor = len(img_size) // config.layer_scaling_threshold
    layers_to_add = min(config.decoder_max_layers, int(1.5 * width_factor))  # Add more layers for long-window model

    decoder_input = Input(shape=(latent_dim,), name='decoder_input')
    x = Dense(config.decoder_base_filters * 4 * (latent_dim // 3), activation='relu', use_bias=False,
              kernel_initializer='RandomNormal')(decoder_input)
    x = Reshape((4, config.decoder_base_filters * (latent_dim // 3)))(x)

    for i in range(layers_to_add):
        x = UpSampling1D(2)(x)
        x = Conv1D(config.decoder_base_filters // (2 ** i), 4 if i == 0 else 3, strides=1, activation='relu', padding='same',
                   kernel_initializer='RandomNormal', bias_initializer='zeros')(x)
        x = BatchNormalization(axis=-1)(x)

    z_decoded = Conv1D(img_height, 1, strides=1, activation='sigmoid', padding='same',
                       kernel_initializer='RandomNormal', bias_initializer='zeros')(x)
    z_decoded_crop = Cropping1D(1)(z_decoded)

    return Model(decoder_input, z_decoded_crop, name='LongWindow_Decoder')


# Decide which model to use based on window_duration
if window_duration < model_cutoff:
    Encoder = short_window_encoder(img_size)
    Decoder = short_window_decoder(window_duration)
else:
    Encoder = long_window_encoder(img_size)
    Decoder = long_window_decoder(window_duration)

# Input branch 1: prediction from frame sequence
input_clip_tensor = Input(shape=clip_tensor_shape, name='input_clip_tensor')
z = TimeDistributed(Encoder)(input_clip_tensor)
w = LSTM(16, activation="tanh", return_sequences=False, dropout=0.1, name='LSTM_0')(z)
vhat = Dense(latent_dim, activation='tanh', name='vhat')(w)

# Input branch 2: compute target
input_top_frames = Input(shape=top_frames_shape, name='input_top_frames')
z_top_frames = TimeDistributed(Encoder)(input_top_frames)
z_i = z_top_frames[:, 0, :]
z_i_plus_1 = z_top_frames[:, 1, :]
v = z_i_plus_1 - z_i

# Output branch 0: Decoder, reconstruct top image
x_i = input_top_frames[:, 0, :, :]
x_i_hat = Decoder(z_i)


# Output branch 2: compute loss
class Add_model_loss(keras.layers.Layer):

    def compute_loss(self, inputs): # later this may also include the reconstruction loss from decoder. loss-weight parameters can be used to weight the impact of the reconstruction and forcasting loss
        v_, vhat_, x_i_, x_i_hat_, z_i_, z_ = inputs
        off_center_loss = K.mean(K.square(z_i_), axis=1)
        reconstruction_loss = keras.metrics.binary_crossentropy(K.flatten(x_i_), K.flatten(x_i_hat_))  # The binary cross entropy averaged over all pixels should have a value around ~ 0.5-2 when model untrained
        # forcasting_loss = K.mean(K.square(vhat_ - v_))  # The value of this is harder to know in advance since it depends on to what spatial scale the encoder maps the images. Use negative cosine similarity instead
        forcasting_loss = tf.keras.losses.CosineSimilarity(axis=-1)(vhat_, v_)  #shape:(batch size,) in [-1, 1] where 1 indicates diametrically opposed. I believe the batch axis is still 0

        # Get curvature loss
        VSegment = K.l2_normalize(z_[:, 1:, :] - z_[:, 0:-1, :], axis=-1)  # normalized velocity vectors in frame sequence. shape (batch, timesteps, features)
        neg_cosine_theta = -K.sum(VSegment[:, 0:-1, :]*VSegment[:, 1:, :], axis=-1)  # take dot product of each pair of consecutive normalized velocity vectors= cos(theta)--> *-1 makes opposing vectors have a value -cos(theta)=1 ie high loss
        curvature_loss = K.mean(neg_cosine_theta, axis=1)  # shape:(batch size,).  mean (across time) of -cos similarity between each consecutive pair of velocity vectors. encourages embeddings with lower curvature

        loss = config.alpha*forcasting_loss + config.beta*reconstruction_loss + config.gamma*off_center_loss + config.delta*curvature_loss
        return loss, off_center_loss, reconstruction_loss, forcasting_loss, curvature_loss

    def call(self, layer_inputs):
        if not isinstance(layer_inputs, list):
            ValueError('Input must be list of [v, vhat, x_i, x_i_hat, z_i, z_]')
        loss, off_center_loss, reconstruction_loss, forcasting_loss, curvature_loss = self.compute_loss(layer_inputs)
        self.add_loss(loss)
        self.add_metric(off_center_loss, name='off_center_loss')
        self.add_metric(reconstruction_loss, name='reconstruction_loss')
        self.add_metric(forcasting_loss, name='forecasting_loss')
        self.add_metric(curvature_loss, name='curvature_loss')
        return loss  # we dont need to return vhat again since this layer wont be used during inference. it doesnt matter what is returned here as long as its a scalar

output = Add_model_loss()([v, vhat, x_i, x_i_hat, z_i, z])

# ---- Load model ----  if you are resuming training. (It takes a long time..) In that case, Do NOT recompile model or you will lose the optimizer state.------------
# model = keras.models.load_model(r"C:\Users\MrLin\Documents\Experiments\Deep Video Embedding\saved models\last-no-forecast")
# model.summary()

# Create new model for training-----------
model = Model([input_clip_tensor, input_top_frames], output)
model.compile(loss=None,  # compute loss internally
              optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False))

tf.keras.utils.plot_model(
    model, to_file='model.png', show_shapes=True,
    show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96)

# Make checkpoint callback
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=False,
    monitor='loss',  # the loss, weighted by class weights
    # change to 'loss' instead of 'val_loss' if you're tuning hyperparams and training on a small subset of the data
    mode='min',
    save_best_only=False,
    save_freq='epoch')  # change to false if you're tuning hyperparams and training on a small subset of the data

model.fit(ds_train,
          steps_per_epoch=steps_per_epoch_tr,
          # 5607,  # steps_per_epoch needs to equal exactly the number of batches in the Dataset generator
          epochs=config.epochs,
          verbose=2,

          callbacks=[WandbCallback(), model_checkpoint_callback])  #

# embed
exec(open("Embed_audio_data_callable.py").read())
