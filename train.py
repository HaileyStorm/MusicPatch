import random

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
import os
import sys

# Load variables
vars = variables.load_variables_from_json()
window_durations = vars["window_durations"]
latent_dim = vars["latent_dim"]
model_cutoff = vars["model_cutoff"]
layer_scaling_threshold = vars["layer_scaling_threshold"]
encoder_base_filters_short = vars["encoder_base_filters_short"]
encoder_base_filters_long = vars["encoder_base_filters_long"]
decoder_base_filters = vars["decoder_base_filters"]
decoder_max_layers = vars["decoder_max_layers"]
config = None

wandb.login()
#id = wandb.util.generate_id()
#print(f"Wandb ID: {id}")


# build generator of subimages of Spectrogram, sliding across time
def inputs_generator(t_start, t_end, S, dt_):
    img_width, img_height = S.shape[-2:]
    t = t_start
    while t <= t_end:
        clip_tensor = []
        top_frames = []

        for j in range(t - dt_*(config["l"] - 1), t + 1, dt_):
            sub_img = np.transpose(S[:img_height, j:j+img_width])
            clip_tensor.append(sub_img.astype(np.float32))  # pixel values are normalized to [0,1]
        clip_tensor = np.array(clip_tensor)

        for j in range(t, t + 2*dt_, dt_):
            sub_img = np.transpose(S[:img_height, j:j + img_width])
            top_frames.append(sub_img.astype(np.float32))
        top_frames = np.array(top_frames)

        yield ((clip_tensor, top_frames), 0)  # 0 represents the target output of the model. the metric in .compile is computed using this. the addloss layer outputs the loss itself just for convienience.
        t += dt_


# Function to create the dataset
def create_dataset(len_T, S_mag):
    img_width, img_height = S_mag.shape[-2:]
    t_start = int((config["l"] - 1))  # first frame index in the valid range
    t_end = int(len_T - img_width - 1)  # last frame index in the valid range, assuming images start at t=0 and go to t=T-1
    samples = np.floor(t_end - t_start + 1)
    steps_per_epoch_tr = int(np.floor(samples / config["batch_size"])) - 1 # number of batches

    # Normalize
    S_mag = (S_mag - np.min(S_mag)) / (np.max(S_mag) - np.min(S_mag))

    ds_train = tf.data.Dataset.from_generator(
        inputs_generator,
        args=[t_start, t_end, S_mag, 1],
        output_types=((tf.float32, tf.float32), tf.float32),
        output_shapes=(((config["l"], img_width, img_height), (2, img_width, img_height)), ()))

    ds_train = ds_train.shuffle(int(samples * 0.7),
                                # turn off shuffle if you are training on a subset of the data for hyperparameter tuning and plan to visualize performance on the first steps_per_epoch_tuning batches of the TRAINING set
                                reshuffle_each_iteration=False)  # the argument into shuffle is the buffer size. this can be smaller than the number of samples, especially when using larger datasets
    ds_train = ds_train.batch(config["batch_size"], drop_remainder=True)  # insufficient data error was thrown without adding the .repeat(). I added the +1 dataset at the end for good measure
    ds_train = ds_train.repeat(config["epochs"] + 1)
    return ds_train, steps_per_epoch_tr


def create_aggregated_dataset(window_duration, spectrograms_directory):
    # Initialize list to store all clip_tensors and top_frames
    all_clip_tensors = []
    all_top_frames = []

    # Collect all matching spectrogram files for the given window_duration
    matching_files = [f for f in os.listdir(spectrograms_directory) if f.endswith(f"_{window_duration}s.pkl")]
    img_width, img_height = 0, 0

    for file in matching_files:
        with open(os.path.join(spectrograms_directory, file), "rb") as f:
            S_mag = pickle.load(f)

        # Normalize
        S_mag = (S_mag - np.min(S_mag)) / (np.max(S_mag) - np.min(S_mag))

        # Infer T, img_width, and img_height
        T = S_mag.shape[1]
        img_width, img_height = S_mag.shape[-2:]
        print((img_width, img_height))

        # Slide window across the spectrogram and generate clip_tensors and top_frames
        for t in range(T - img_width - 2):  # Adjusted to allow room for top_frames
            clip_tensor = S_mag[:, t:t + img_width]
            top_frames = S_mag[:, t + img_width:t + img_width + 2]
            all_clip_tensors.append(clip_tensor)
            all_top_frames.append(top_frames)

    # Shuffle the data
    combined = list(zip(all_clip_tensors, all_top_frames))
    random.shuffle(combined)
    all_clip_tensors[:], all_top_frames[:] = zip(*combined)

    # Convert lists to TensorFlow dataset
    ds_train = tf.data.Dataset.from_tensor_slices((all_clip_tensors, all_top_frames))

    # Adjust the structure of the dataset to match what Keras expects
    ds_train = ds_train.map(lambda clip_tensor, top_frames: ((clip_tensor, top_frames), 0))

    ds_train = ds_train.batch(config["batch_size"], drop_remainder=True)
    ds_train = ds_train.repeat(config["epochs"] + 1)

    return ds_train, len(all_clip_tensors) // config["batch_size"], (img_width, img_height)



# Short-Window Encoder
def short_window_encoder(img_size):
    layers_to_add = img_size[0] // layer_scaling_threshold

    base_filters = encoder_base_filters_short
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
    layers_to_add = img_size[0] // (layer_scaling_threshold // 2)  # potentially more layers for long-window model

    base_filters = encoder_base_filters_long
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
def short_window_decoder(img_size):
    width_factor = img_size[0] // layer_scaling_threshold
    layers_to_add = min(decoder_max_layers, int(width_factor))  # Number of layers, capped at decoder_max_layers

    decoder_input = Input(shape=(latent_dim,), name='decoder_input')
    x = Dense(decoder_base_filters * latent_dim, activation='relu', use_bias=False, kernel_initializer='RandomNormal')(
        decoder_input)
    x = Reshape((latent_dim, decoder_base_filters))(x)

    for i in range(layers_to_add):
        x = UpSampling1D(2)(x)
        x = Conv1D(decoder_base_filters // (2 ** i), 3, strides=1, activation='relu', padding='same',
                   kernel_initializer='RandomNormal', bias_initializer='zeros')(x)
        x = BatchNormalization(axis=-1)(x)

    z_decoded = Conv1D(img_size[1], 1, strides=1, activation='sigmoid', padding='same',
                       kernel_initializer='RandomNormal', bias_initializer='zeros')(x)
    z_decoded_crop = Cropping1D(1)(z_decoded)

    return Model(decoder_input, z_decoded_crop, name='ShortWindow_Decoder')


# Decoder for Long-Window Model
def long_window_decoder(img_size):
    width_factor = img_size[0] // layer_scaling_threshold
    layers_to_add = min(decoder_max_layers, int(1.5 * width_factor))  # Add more layers for long-window model

    decoder_input = Input(shape=(latent_dim,), name='decoder_input')
    x = Dense(decoder_base_filters * 4 * (latent_dim // 3), activation='relu', use_bias=False,
              kernel_initializer='RandomNormal')(decoder_input)
    x = Reshape((4, decoder_base_filters * (latent_dim // 3)))(x)

    for i in range(layers_to_add):
        x = UpSampling1D(2)(x)
        x = Conv1D(decoder_base_filters // (2 ** i), 4 if i == 0 else 3, strides=1, activation='relu', padding='same',
                   kernel_initializer='RandomNormal', bias_initializer='zeros')(x)
        x = BatchNormalization(axis=-1)(x)

    z_decoded = Conv1D(img_size[1], 1, strides=1, activation='sigmoid', padding='same',
                       kernel_initializer='RandomNormal', bias_initializer='zeros')(x)
    z_decoded_crop = Cropping1D(1)(z_decoded)

    return Model(decoder_input, z_decoded_crop, name='LongWindow_Decoder')


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

        loss = config["alpha"]*forcasting_loss + config["beta"]*reconstruction_loss + config["gamma"]*off_center_loss + config["delta"]*curvature_loss
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


def process_single_duration(duration, spectrograms_folder, config, is_tuning=False):
    if is_tuning:
        # Load S_mag_crop from the pickled object
        spectrogram_file = f"{spectrograms_folder}/{duration}s.pkl"
        with open(spectrogram_file, "rb") as f:
            S_mag = pickle.load(f)

        # Infer T, img_width, and img_height
        T = S_mag.shape[1]
        img_width, img_height = S_mag.shape[-2:]
        img_size = (img_width, img_height)

        # Normalize
        S_mag = (S_mag - np.min(S_mag)) / (np.max(S_mag) - np.min(S_mag))

        # Create dataset
        ds_train, steps_per_epoch_tr = create_dataset(len(T), S_mag)
    else:
        ds_train, steps_per_epoch_tr, img_size = create_aggregated_dataset(duration, spectrograms_directory)

    # Decide which model to use based on window_duration
    if duration < model_cutoff:
        Encoder = short_window_encoder(img_size)
        Decoder = short_window_decoder(img_size)
    else:
        Encoder = long_window_encoder(img_size)
        Decoder = long_window_decoder(img_size)

    if is_tuning:
        base_model_path = f'BaseModels/model_{duration}s.tf'
        if os.path.exists(base_model_path):
            model = keras.models.load_model(base_model_path)
        else:
            raise ValueError(f"Base model for duration {duration} not found!")
    else:
        # Input branch 1: prediction from frame sequence
        clip_tensor_shape = (config["l"],) + img_size  # (l tensor slices, time-steps, freq)
        top_frames_shape = (2,) + img_size  # (2 tensor slices, time-steps, freq)
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

        output = Add_model_loss()([v, vhat, x_i, x_i_hat, z_i, z])

        model = Model([input_clip_tensor, input_top_frames], output)
        model.compile(loss=None,  # compute loss internally
                      optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False))

    # Make checkpoint callback
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(spectrograms_folder, "checkpoints"),
        save_weights_only=False,
        monitor='loss',  # the loss, weighted by class weights
        # change to 'loss' instead of 'val_loss' if you're tuning hyperparams and training on a small subset of the data
        mode='min',
        save_best_only=False,
        save_freq='epoch')  # change to false if you're tuning hyperparams and training on a small subset of the data

    # Fit the model
    model.fit(ds_train,
              steps_per_epoch=steps_per_epoch_tr,
              epochs=config["epochs"],
              verbose=2,
              callbacks=[WandbCallback(), model_checkpoint_callback])

    # Save the model
    if is_tuning:
        model_folder = os.path.join(spectrograms_folder, f"model_{duration}s.tf")
    else:
        model_folder = os.path.join("BaseModels", f"model_{duration}s.tf")
    model.save(model_folder)


# Determine whether to process a single song or all songs in the directory
if len(sys.argv) > 1:
    # Single song specified via command line argument
    song_arg = sys.argv[1]

    if song_arg.endswith('.wav'):
        spectrograms_folder = os.path.join(os.path.dirname(song_arg), os.path.basename(song_arg).split('.')[0])
        if not os.path.exists(spectrograms_folder):
            raise ValueError(f"Spectrograms folder {spectrograms_folder} does not exist.")
    else:
        spectrograms_folder = song_arg

    # Set wandb config
    config = vars["wandb_config_tune"]

    # Loop over each window duration and fine-tune the corresponding model
    for duration in window_durations:
        wandb.init(project='Deep Audio Embedding', group='Vivaldi', name=f"FineTune_{duration}", config=config)
        process_single_duration(duration, spectrograms_folder, config, True)

else:
    # No song specified, process all songs in the 'Audio/train/spectrograms' directory
    spectrograms_directory = 'Audio/train/spectrograms'
    songs = [f for f in os.listdir(spectrograms_directory)]

    # Set wandb config
    config = vars["wandb_config_base"]

    # Loop over each window duration and train the base model
    for duration in window_durations:
        wandb.init(project='Deep Audio Embedding', group='Vivaldi', name=f"BaseModel_{duration}", config=config)
        process_single_duration(duration, spectrograms_directory, config, False)

# Save variables
variables.write_variables_to_json(vars)

# embed
#exec(open("Embed_audio_data_callable.py").read())
