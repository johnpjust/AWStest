import tensorflow as tf
from tensorflow.keras import layers, Input, Model
import numpy as np
import os
import datetime

""" kNN doesn't scale well due to being essentially a brute force comparison algorithm.  Validate results by observing classification error.  
The concept being that a model with less classification error will find nearest neighbors better as well."""

class parser_:
    pass

args = parser_()
args.device = '/gpu:0'  # '/gpu:0'
args.learning_rate = np.float32(1e-2)
args.clip_norm = 0.1
args.batch_size = 200 ## 6/50,
args.epochs = 5000
args.patience = 20
args.load = ''
args.tensorboard = r'D:\AlmacoEarCounts\Tensorboard'
args.early_stopping = 500
args.manualSeed = None
args.manualSeedw = None
args.p_val = 0.2
args.num_frames = 6

args.path = os.path.join(args.tensorboard,
                         'frames{}_{}'.format(args.num_frames,
                             str(datetime.datetime.now())[:-7].replace(' ', '-').replace(':', '-')))

if not args.load:
    print('Creating directory experiment..')
    pathlib.Path(args.path).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.path, 'args.json'), 'w') as f:
        json.dump(str(args.__dict__), f, indent=4, sort_keys=True)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

with tf.device('/gpu:0'):
    encoder_input = Input(shape=(28, 28, 1), name='original_img')
    x = layers.Conv2D(16, 3, activation='relu')(encoder_input)
    x = layers.Conv2D(32, 3, activation='relu')(x)
    x = layers.MaxPooling2D(3)(x)
    x = layers.Conv2D(32, 3, activation='relu')(x)
    x = layers.Conv2D(16, 3, activation='relu')(x)
    encoder_output = layers.GlobalMaxPooling2D()(x)

    encoder = Model(encoder_input, encoder_output, name='encoder')
    encoder.summary()

    decoder_input = Input(shape=(16,), name='encoded_img')
    x = layers.Reshape((4, 4, 1))(decoder_input)
    x = layers.Conv2DTranspose(16, 3, activation='relu')(x)
    x = layers.Conv2DTranspose(32, 3, activation='relu')(x)
    x = layers.UpSampling2D(3)(x)
    x = layers.Conv2DTranspose(16, 3, activation='relu')(x)
    decoder_output = layers.Conv2DTranspose(1, 3, activation='relu')(x)

    decoder = Model(decoder_input, decoder_output, name='decoder')
    decoder.summary()

    autoencoder_input = Input(shape=(28, 28, 1), name='img')
    encoded_img = encoder(autoencoder_input)
    decoded_img = decoder(encoded_img)
    autoencoder = Model(autoencoder_input, decoded_img, name='autoencoder')
    autoencoder.summary()

actfun = tf.nn.relu
with tf.device('/gpu:0'):
    inputs = tf.keras.Input(shape=(108, 192, 3*args.num_frames), name='img') ## (108, 192, 3)
    x = layers.Conv2D(32, 7, activation=actfun, strides=2)(inputs)
    block_output = layers.MaxPooling2D(3, strides=2)(x)

    x = layers.Conv2D(32, 1, activation=actfun, padding='same')(block_output)
    x = layers.Conv2D(32, 3, activation=None, padding='same')(x)
    x = layers.add([x, block_output])
    x = layers.Conv2D(32, 1, activation=actfun)(x)
    block_output = layers.AveragePooling2D(pool_size=2, strides=2)(x)

    x = layers.Conv2D(32, 1, activation=actfun, padding='same')(block_output)
    x = layers.Conv2D(32, 3, activation=None, padding='same')(x)
    x = layers.add([x, block_output])
    x = layers.Conv2D(32, 1, activation=actfun)(x)
    x = layers.AveragePooling2D(2, strides=2)

    x = layers.Conv2D(32, 1, activation=actfun)(block_output)
    x = layers.Conv2D(32, 3, activation=None)(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(32, activation=actfun)(x)
    x = layers.Dense(1)(x)
    counts = tf.keras.activations.softplus(x)
    model = tf.keras.Model(inputs, counts, name='toy_resnet')
    model.summary()



root = None
args.start_epoch = 0

print('Creating optimizer..')
with tf.device(args.device):
    optimizer = tf.optimizers.Adam()
root = tf.train.Checkpoint(optimizer=optimizer,
                           model=model,
                           optimizer_step=tf.compat.v1.train.get_global_step())

if args.load:
    load_model(args, root, load_start_epoch=True)

print('Creating scheduler..')
# use baseline to avoid saving early on
scheduler = EarlyStopping(model=model, patience=args.patience, args=args, root=root)

with open(os.path.join(args.path, 'modelsummary.txt'), 'w') as f:
    with redirect_stdout(f):
        model.summary()

with tf.device(args.device):
    train(model, optimizer, scheduler, data_loader_train, data_loader_val, data_loader_test, args, empty_logs_train, empty_logs_val, empty_logs_test)
