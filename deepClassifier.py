import numpy as np
from glob import glob
import os
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from skimage import io

def train(model, optimizer, scheduler, data_loader_train, data_loader_val, data_loader_test, args, empty_logs_train, empty_logs_val, empty_logs_test):
    epoch = args.start_epoch
    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):

        # print("train")
        # for ind in range(len(data_loader_train)):
        for x_mb, y_mb in data_loader_train:
            # x_mb = tf.signal.frame(data_loader_train[ind][0], args.num_frames, 1, axis=0)
            # x_mb = np.array(list(more_itertools.windowed(data_loader_train[ind][0], n=args.num_frames, step=1)))
            # x_mb = windowed(data_loader_train[ind][0], n=args.num_frames, step=1)
            # y_mb = data_loader_train[ind][1]
            for i_ in range(2):
                if i_ == 0:
                    x_mb = windowed(data_loader_train[ind][0], n=args.num_frames, step=1)
                    y_mb = data_loader_train[ind][1]
                elif i_ == 1:
                    if np.random.choice(a=[False, True]):
                        x_mb = windowed(data_loader_train[ind][0][empty_logs_train[ind] == 0], n=args.num_frames, step=1)
                        y_mb = 0
                    else:
                        x_mb = np.repeat(data_loader_train[0][0], args.num_frames, axis=-1)
                        y_mb = 0
                    # x_mb = windowed(data_loader_train[ind][0][empty_logs_train[ind] == 0], n=args.num_frames, step=1)
                    # y_mb = 0
                # else:
                #     x_mb = np.repeat(data_loader_train[0][0], args.num_frames, axis=-1)
                #     y_mb = 0

                count = 0
                grads = [np.zeros_like(x) for x in model.trainable_variables]
                for x_ in batch(x_mb, args.batch_size):
                    with tf.GradientTape() as tape:
                        count_ = tf.reduce_sum(model(x_, training=True))
                    count += count_
                    grads_ = tape.gradient(count_, model.trainable_variables)
                    grads = [x1 + x2 for x1, x2 in zip(grads, grads_)]
                grads = [None if grad is None else tf.clip_by_norm(grad, clip_norm=args.clip_norm) for grad in grads]
                loss = count-y_mb
                ## scale gradients by log length or number of batches (else longer logs will be weighted unduly)
                # globalstep = optimizer.apply_gradients(zip([2*loss*x/np.float32(x_mb.shape[0]) for x in grads], model.trainable_variables))
                globalstep = optimizer.apply_gradients(zip([2*loss*x for x in grads], model.trainable_variables))

                if i_ == 0:
                    tf.summary.scalar('loss/train', loss**2, globalstep)

        ## potentially update batch norm variables manuallu
        ## variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='batch_normalization')

        # train_loss = tf.reduce_mean(train_loss)
        validation_loss = []
        for x_mb, y_mb in data_loader_val:
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y_mb, model(x_, training=False))).numpy()
            validation_loss.append(loss)
        validation_loss = tf.reduce_mean(validation_loss)
        # print("validation loss:  " + str(validation_loss))

        test_loss=[]
        for x_mb, y_mb in data_loader_test:
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y_mb, model(x_, training=False))).numpy()
            test_loss.append(loss)

        test_loss = tf.reduce_mean(test_loss)

        # print("test loss:  " + str(test_loss))

        stop = scheduler.on_epoch_end(epoch=epoch, monitor=validation_loss)

        #### tensorboard
        # tf.summary.scalar('loss/train', train_loss, tf.compat.v1.train.get_global_step())
        tf.summary.scalar('loss/validation', validation_loss, globalstep)
        tf.summary.scalar('loss/test', test_loss, globalstep) ##tf.compat.v1.train.get_global_step()

        if stop:
            break

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
from scipy import ndimage, stats

########### import and preprocess data
# files = glob(r'C:\Users\mlworker\Downloads\geological_similarity\geological_similarity\**\*.jpg', recursive=True)
# data = np.array([io.imread(x) for x in files])
# labels = np.array([os.path.basename(os.path.dirname(x)) for x in files])
# np.save(r'D:\pycharm_projects\AWSgeo\data.npy', data)
# np.save(r'D:\pycharm_projects\AWSgeo\labels.npy', labels)
# bgs = []
# # img_mode = stats.mode(data, axis=0)
# bg = np.median(data, axis=0)
# # bg = img_mode.mode.squeeze()
# # bg = ndimage.median_filter(bg, (3,3,3))
# np.save(r'D:\pycharm_projects\AWSgeo\bg.npy', bg)

data = np.load(r'D:\pycharm_projects\AWSgeo\data.npy')/255.0
labels = np.load(r'D:\pycharm_projects\AWSgeo\labels.npy')
bg = np.load(r'D:\pycharm_projects\AWSgeo\bg.npy')/255.0
data = data - bg
data = data.astype(np.float32)
labels = np.unique(labels, return_inverse=True)[1].astype(np.float32)

data_split_ind = np.random.permutation(len(data))
data_loader_train = data[data_split_ind[:int((1-2*args.p_val)*len(data_split_ind))]]
train_labels = labels[data_split_ind[:int((1-2*args.p_val)*len(data_split_ind))]]
data_loader_val = data[data_split_ind[int((1-2*args.p_val)*len(data_split_ind)):int((1-args.p_val)*len(data_split_ind))]]
val_labels = labels[data_split_ind[int((1-2*args.p_val)*len(data_split_ind)):int((1-args.p_val)*len(data_split_ind))]]
data_loader_test = data[data_split_ind[int((1-args.p_val)*len(data_split_ind)):]]
test_labels = labels[data_split_ind[int((1-args.p_val)*len(data_split_ind)):]]

train_dataset = tf.data.Dataset.from_tensor_slices((data_loader_train, train_labels))
train_dataset = train_dataset.shuffle(buffer_size=data_loader_train.shape[0].numpy()).batch(args.batch_size)

train_dataset = tf.data.Dataset.from_tensor_slices((data_loader_train, train_labels))
train_dataset = train_dataset.batch(2*args.batch_size)

test_dataset = tf.data.Dataset.from_tensor_slices((data_loader_test, test_labels))
test_dataset = test_dataset.batch(2*args.batch_size)


########### setup GPU
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

######### Create Model
actfun = tf.nn.relu
with tf.device(args.device):
    inputs = tf.keras.Input(shape=(108, 192, 3), name='img') ## (108, 192, 3)
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
    logits = layers.Dense(6)(x)
    model = tf.keras.Model(inputs, logits, name='toy_resnet')
    model.summary()


writer = tf.summary.create_file_writer(args.path)
writer.set_as_default()
tf.compat.v1.train.get_or_create_global_step()

# global_step = tf.compat.v1.train.get_global_step()
# global_step.assign(0)

root = None
args.start_epoch = 0

print('Creating optimizer..')
with tf.device(device):
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
