import numpy as np
from glob import glob
import os
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from skimage import io
from lr_scheduler import *
from contextlib import redirect_stdout
import datetime
from sklearn import metrics
import pandas as pd

def train(model, optimizer, scheduler, data_loader_train, data_loader_val, data_loader_test, args):
    epoch = args.start_epoch
    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):

        for x_mb, y_mb in data_loader_train:
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y_mb, model(x_mb, training=True)))
            grads = tape.gradient(loss, model.trainable_variables)
            grads = [None if grad is None else tf.clip_by_norm(grad, clip_norm=args.clip_norm) for grad in grads]
            globalstep = optimizer.apply_gradients(zip(grads, model.trainable_variables))
            tf.summary.scalar('loss/train', loss, globalstep)

        ## potentially update batch norm variables manuallu
        ## variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='batch_normalization')

        validation_loss = []
        for x_mb, y_mb in data_loader_val:
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y_mb, model(x_mb, training=False))).numpy()
            validation_loss.append(loss)
        validation_loss = tf.reduce_mean(validation_loss)
        # print("validation loss:  " + str(validation_loss))

        test_loss=[]
        for x_mb, y_mb in data_loader_test:
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y_mb, model(x_mb, training=False))).numpy()
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

def load_model(args, root, load_start_epoch=False):
    # def f():
    print('Loading model..')
    root.restore(tf.train.latest_checkpoint(args.load or args.path))

class parser_:
    pass

args = parser_()
args.device = '/gpu:0'  # '/gpu:0'
args.learning_rate = np.float32(1e-2)
args.clip_norm = 0.1
args.batch_size = 500 ## 6/50,
args.epochs = 5000
args.patience = 20
args.load = ''
args.tensorboard = r'D:\pycharm_projects\AWSgeo\Tensorboard'
args.early_stopping = 500
args.manualSeed = None
args.manualSeedw = None
args.p_val = 0.2


args.path = os.path.join(args.tensorboard,
                         'model_{}'.format(str(datetime.datetime.now())[:-7].replace(' ', '-').replace(':', '-')))

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
# label_names = np.unique(labels, return_inverse=True)[0]
labels = np.unique(labels, return_inverse=True)[1]

data_split_ind = np.random.permutation(len(data))
data_loader_train = data[data_split_ind[:int((1-2*args.p_val)*len(data_split_ind))]]
train_labels = labels[data_split_ind[:int((1-2*args.p_val)*len(data_split_ind))]]
data_loader_val = data[data_split_ind[int((1-2*args.p_val)*len(data_split_ind)):int((1-args.p_val)*len(data_split_ind))]]
val_labels = labels[data_split_ind[int((1-2*args.p_val)*len(data_split_ind)):int((1-args.p_val)*len(data_split_ind))]]
data_loader_test = data[data_split_ind[int((1-args.p_val)*len(data_split_ind)):]]
test_labels = labels[data_split_ind[int((1-args.p_val)*len(data_split_ind)):]]

train_dataset = tf.data.Dataset.from_tensor_slices((data_loader_train, train_labels))
train_dataset = train_dataset.shuffle(buffer_size=data_loader_train.shape[0]).batch(args.batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices((data_loader_val, val_labels))
val_dataset = val_dataset.batch(2*args.batch_size)

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
    inputs = tf.keras.Input(shape=data.shape[1:], name='img') ## (108, 192, 3)
    x = layers.Conv2D(32, 7, activation=actfun, strides=2)(inputs)
    block_output = layers.MaxPooling2D(3, strides=2)(x)

    x = layers.Conv2D(32, 1, activation=actfun, padding='same')(block_output)
    x = layers.Conv2D(32, 3, activation=None, padding='same')(x)
    x = layers.add([x, block_output])
    x = layers.Conv2D(32, 1, activation=actfun)(x)
    # block_output = layers.AveragePooling2D(pool_size=2, strides=2)(x)

    # x = layers.Conv2D(32, 1, activation=actfun, padding='same')(block_output)
    # x = layers.Conv2D(32, 3, activation=None, padding='same')(x)
    # x = layers.add([x, block_output])
    # x = layers.Conv2D(32, 1, activation=actfun)(x)
    # block_output = layers.AveragePooling2D(2, strides=2)(x)

    # x = layers.Conv2D(32, 1, activation=actfun)(block_output)
    # x = layers.Conv2D(32, 3, activation=None)(x)
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
    train(model, optimizer, scheduler, train_dataset, val_dataset, test_dataset, args)


################# check classification accuracy ##############################
test_x_mb = []
test_y_mb = []
for x_mb, y_mb in test_dataset:
    test_x_mb.append(tf.nn.softmax(model(x_mb, training=False)).numpy())
    test_y_mb.append(y_mb)
test_x_mb = np.argmax(np.concatenate(test_x_mb), axis=1)
test_y_mb = np.concatenate(test_y_mb)

#### confusion matrix
conf_mat = metrics.confusion_matrix(test_y_mb.astype(np.float32), test_x_mb) ## tree methods tend to have higher false negative rates than ANN
cols = ['andesite', 'gneiss', 'marble', 'quartzite', 'rhyolite', 'schist']
df = pd.DataFrame(conf_mat/np.expand_dims(np.sum(conf_mat, axis=1), axis=1), columns=cols, index=cols)
print(df)

########################## get logit embeddings for kNN search ########################
test_x_mb = []
test_y_mb = []
for x_mb, y_mb in test_dataset:
    test_x_mb.append(model(x_mb, training=False).numpy())
    test_y_mb.append(y_mb)
test_x_mb = np.concatenate(test_x_mb)
test_y_mb = np.concatenate(test_y_mb)
np.savetxt(r'D:\pycharm_projects\AWSgeo\Tensorboard\model_2019-12-17-00-38-33\test_logits.csv', test_x_mb, delimiter=',')
np.savetxt(r'D:\pycharm_projects\AWSgeo\Tensorboard\model_2019-12-17-00-38-33\test_labels.csv', test_y_mb, delimiter=',')

val_x_mb = []
val_y_mb = []
for x_mb, y_mb in val_dataset:
    val_x_mb.append(model(x_mb, training=False).numpy())
    val_y_mb.append(y_mb)
val_x_mb = np.concatenate(val_x_mb)
val_y_mb = np.concatenate(val_y_mb)
np.savetxt(r'D:\pycharm_projects\AWSgeo\Tensorboard\model_2019-12-17-00-38-33\val_logits.csv', val_x_mb, delimiter=',')
np.savetxt(r'D:\pycharm_projects\AWSgeo\Tensorboard\model_2019-12-17-00-38-33\val_labels.csv', val_y_mb, delimiter=',')

train_x_mb = []
train_y_mb = []
for x_mb, y_mb in train_dataset:
    train_x_mb.append(model(x_mb, training=False).numpy())
    train_y_mb.append(y_mb)
train_x_mb = np.concatenate(train_x_mb)
train_y_mb = np.concatenate(train_y_mb)
np.savetxt(r'D:\pycharm_projects\AWSgeo\Tensorboard\model_2019-12-17-00-38-33\train_logits.csv', train_x_mb, delimiter=',')
np.savetxt(r'D:\pycharm_projects\AWSgeo\Tensorboard\model_2019-12-17-00-38-33\train_labels.csv', train_y_mb, delimiter=',')

dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(2*args.batch_size)

data_x_mb = []
data_y_mb = []
for x_mb, y_mb in dataset:
    data_x_mb.append(model(x_mb, training=False).numpy())
    data_y_mb.append(y_mb)
data_x_mb = np.concatenate(data_x_mb)
data_y_mb = np.concatenate(data_y_mb)
np.save(r'D:\pycharm_projects\AWSgeo\Tensorboard\model_2019-12-17-00-38-33\data_logits.npy', data_x_mb)
np.save(r'D:\pycharm_projects\AWSgeo\Tensorboard\model_2019-12-17-00-38-33\data_labels.npy', data_y_mb)

#### tensorboard --logdir=D:\pycharm_projects\AWSgeo\Tensorboard

########### C:\Program Files\NVIDIA Corporation\NVSMI
######### nvidia-smi  -l 2