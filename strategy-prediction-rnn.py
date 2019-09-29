###
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

# Data Dimension
from utils import load_data

num_input = 42  # MNIST data input (image shape: 28x28)
timesteps = 20  # Timesteps
n_classes = 3  # Number of classes, one class per digit
data_num = 10000


# Helper Functions to load data
def load_data(mode='train'):
    """
     Function to load training data
    :param mode:train or test
    :return data and the corresponding labels
    """
    l = pd.read_csv(".\\input\\t_l.csv")
    d = pd.read_csv(".\\input\\t_s.csv")

    la = l.values.tolist()  # 200000*2
    da = d.values.tolist()  # 200000*8

    label = np.array(la)  # 200000*2
    data = np.array(da)  # 200000*8

    label = np.reshape(label, (data_num, timesteps, n_classes))  # 10000*30
    data = np.reshape(data, (data_num, timesteps, num_input))  # 10000*120

    train_num = int(data_num * 3 / 5)
    validation_num = train_num + int(data_num * 1 / 5)

    if mode == 'train':
        x_train = data[0:train_num]
        y_train = label[0:train_num]
        x_valid = data[train_num:validation_num]
        y_valid = label[train_num:validation_num]
        return x_train, y_train, x_valid, y_valid
    else:
        x_test = data[validation_num:]
        y_test = label[validation_num:]
        return x_test, y_test


def get_next_batch(x, y, start, end):
    x_batch = x[start:end]
    y_batch = y[start:end]
    return x_batch, y_batch


x_train, y_train, x_valid, y_valid = load_data(mode='train')
print("Size of:")
print("- Training-set:\t\t{}".format(len(x_train)))
print("- Validation-set:\t\t{}".format(len(x_valid)))

# Hyperparameters
learning_rate = 0.01  # The opetimization initial learning rate
epoches = 20  # Total number of training epoches
batch_size = 10  # Training batch size
display_freq = 100  # Frequency of displaying the training results

# Network configurations
num_hidden_units = 128  # Number of hidden units of the RNN
attention_size = 128


# Network Helper Functions

# weight and bais wrappers
def weight_variable(shape):
    """
    Create a weight variable with appropriate initialization
    :param name:weight name
    :param shape:weight shape
    :return: initialization weight variable
    """
    W = tf.get_variable('W',
                            dtype=tf.float32,
                            shape=shape,
                            initializer=tf.truncated_normal_initializer(stddev=0.01))
    return W


def bias_variable(shape):
    """
    Create a bais variable with appropriate initialization
    :param name:bias name
    :param shape:bias shape
    :return: initialization bias variable
    """
    b = tf.get_variable('b', dtype=tf.float32, initializer=tf.constant(0., shape=shape, dtype=tf.float32))
    return b


def attention_variables(attention_size,hidden_size):
    """
        Create variables needed for attention vector
        :param output_logits_size: D
        :param attention_size:
        :param hidden_size:
        :return:
    """
    # Trainable parameters
    # init s0 vector, [B, A]
    batch_init_state = tf.get_variable('s',
                           dtype=tf.float32,
                           shape=[batch_size, hidden_size],
                           initializer=tf.truncated_normal_initializer(stddev=0.01))
    Ua = tf.get_variable('Ua',
                         dtype=tf.float32,
                         shape=[hidden_size, attention_size],
                         initializer=tf.truncated_normal_initializer(stddev=0.01))
    Wa = tf.get_variable('Wa',
                         dtype=tf.float32,
                         shape=[hidden_size, attention_size],
                         initializer=tf.truncated_normal_initializer(stddev=0.01))
    va = tf.get_variable('va',
                         dtype=tf.float32,
                         shape=[attention_size],
                         initializer=tf.truncated_normal_initializer(stddev=0.01))
    Ur = tf.get_variable('Ur',
                         dtype=tf.float32,
                         shape=[hidden_size, hidden_size],
                         initializer=tf.truncated_normal_initializer(stddev=0.01))
    Cr = tf.get_variable('Cr',
                         dtype=tf.float32,
                         shape=[attention_size, hidden_size],
                         initializer=tf.truncated_normal_initializer(stddev=0.01))
    Uz = tf.get_variable('Uz',
                         dtype=tf.float32,
                         shape=[hidden_size, hidden_size],
                         initializer=tf.truncated_normal_initializer(stddev=0.01))
    Cz = tf.get_variable('Cz',
                         dtype=tf.float32,
                         shape=[attention_size, hidden_size],
                         initializer=tf.truncated_normal_initializer(stddev=0.01))
    U = tf.get_variable('U',
                        dtype=tf.float32,
                        shape=[hidden_size, hidden_size],
                        initializer=tf.truncated_normal_initializer(stddev=0.01))
    C = tf.get_variable('C',
                        dtype=tf.float32,
                        shape=[attention_size, hidden_size],
                        initializer=tf.truncated_normal_initializer(stddev=0.01))
    return batch_init_state, Ua, Wa, va, Ur, Cr, Uz, Cz, U, C


# Building a RNN network
def BiRNN(x, timesteps, num_hidden):
    # Prepare data shape to match 'rnn' function requirement
    # Current data input shape: (batch_size,timesteps,n_inputs)
    # Required shape: 'timesteps' tensors list of shape (batch_size,num_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, num_input)
    x = tf.unstack(x, timesteps, 1)

    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get BiRNN cell output

    output_logits, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
    # T list [batch_size, fw+bw length]
    return output_logits


def get_context_vector(previous_state, attention_size, Uh, Wa, va):
    # previous_state (B, hidden_num)
    # Uh (B,T,A)
    # Wa (hidden_num, A)
    # score.shape (B,T)
    # va.shape (attention_size)
    Ws = tf.tensordot(previous_state, Wa, axes=1)    # (B, A)
    # (B,A) ->(B,1,A)
    Ws = Ws[:, None, :]
    va = va[:, None]
    # (B, T, 1)
    score = tf.matmul(tf.tanh(Uh+Ws), va)
    score = tf.squeeze(score, axis=2)
    # (B, T)
    alphas = tf.nn.softmax(score, axis=1, name='alphas')  # (B,T) shape
    # (B, T, 1)
    alphas = alphas[:, :, None]
    # (B, T, A)
    alphas = tf.broadcast_to(alphas, [batch_size, timesteps, attention_size])
    context_vector = tf.multiply(Uh, alphas)
    # (B, A)
    context_vector = tf.reduce_sum(context_vector, axis=1)
    return context_vector


def get_new_state(batch_state, context_vector, Ur, Cr, Uz, Cz, U, C):
    # tf.tensordot(a, b, axes=1) == tf.matmul(a,b)
    # (B,2n) (2n,2n)
    UrS = tf.tensordot(batch_state, Ur, axes=1, name='UrS')
    # (B, A) (A, 2n)
    CrC = tf.tensordot(context_vector, Cr, axes=1, name='CrC')
    r = tf.sigmoid(UrS+CrC)

    # (B, 2n) (2n,2n)
    UzS = tf.tensordot(batch_state, Uz, axes=1, name='UzS')
    # (B, A) (A, 2n)
    CzC = tf.tensordot(context_vector, Cz, axes=1, name='CzC')
    z = tf.sigmoid(UzS+CzC)

    # (B, 2n) dot (B, 2n)
    rs = tf.multiply(r, batch_state)
    # (B, 2n) (2n,2n)
    Urs = tf.tensordot(rs, U, axes=1, name='Urs')
    # (B, A) (A, 2n)
    Cc = tf.tensordot(context_vector, C, axes=1, name='Cc')
    # tmp ~si
    s_ = tf.tanh(Urs+Cc)

    # (B, 2n)
    s = tf.multiply((1-z), batch_state) + tf.multiply(z, s_)
    return s


def attention_block(rnn_output_logits, attention_size):
    """
    Attention mechanism layer which reduces Bi-RNN outputs with Attention vector.
    rational: Bahdanau Attention: eij = vTa tanh (Wa si−1 + Ua hj)
                                    shape:
                                        1*hidden_num * (hidden_num*attention_size * attention_size*1 +
                                        hidden_num*2attention_size * 2attention_size*1)
    Args:
        rnn_output_logits: The Attention inputs.
            Matches outputs of Bi-RNN layer (not final state):
                In case of Bidirectional RNN, this must be a tuple (outputs_fw, outputs_bw) containing the forward and
                the backward RNN outputs `Tensor`.
                    If time_major == False (default),
                        outputs_fw is a `Tensor` shaped:
                        `[batch_size, max_time, cell_fw.output_size]`
                        and outputs_bw is a `Tensor` shaped:
                        `[batch_size, max_time, cell_bw.output_size]`.
        attention_size: Linear size of the Attention weights.(also called context vector size, ci)
                        attention_size is equal to 2*hidden_layer_num in the paper
                        hj - R2n
    Returns:
        The Attention output `Tensor`.
        In case of Bidirectional RNN, this will be a `Tensor` shaped:
            `[batch_size, timesteps, Attention_size]`.
    """

    # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
    # [batch_size,timesteps,D], D=2n
    rnn_output_logits = tf.convert_to_tensor(rnn_output_logits)       # [T, B, D]
    rnn_output_logits = tf.transpose(rnn_output_logits, [1, 0, 2])  # (T,B,D) => (B,T,D)
    hidden_size = 2*num_hidden_units
    batch_init_state, Ua, Wa, va, Ur, Cr, Uz, Cz, U, C= attention_variables(attention_size, hidden_size)

    # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
    # the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
    # U*hj does not involve i, so it can be compute first
    Uh = tf.tensordot(rnn_output_logits, Ua, axes=1, name='Uh')
    state_list=[]
    batch_state = batch_init_state  # value? variable itself?
    i = 0
    while i < timesteps:
        context_vector = get_context_vector(batch_state, attention_size, Uh, Wa, va)
        new_state = get_new_state(batch_state,context_vector,Ur, Cr, Uz, Cz, U, C)
        state_list.append(new_state)
        # update states
        batch_state = new_state
        i = i + 1
    # [T, B, 2n]
    states = tf.convert_to_tensor(state_list)
    # [T. B, 2n] => [B, T, 2n]
    states = tf.transpose(states, perm=[1, 0, 2])
    return states


# create the network graph

# Placeholders for inputs (x) and outputs(y)
x = tf.placeholder(tf.float32, shape=[None, timesteps, num_input], name='X')
y = tf.placeholder(tf.float32, shape=[None, timesteps, n_classes], name='Y')
# 2 Dense layers, activation='relu'
def predict_strategy(states):
    # input shape (B, T, 2n)
    dense1 = tf.layers.dense(inputs=states, units=128,
                             activation=tf.nn.sigmoid, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    dense2 = tf.layers.dense(inputs=dense1, units=n_classes,
                             activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    # (B, T, n_classes)
    return dense2

# create weight matrix initialized randomely from N~(0, 0.01)
# W = weight_variable(shape=[2 * num_hidden_units, n_classes])

# create bias vector initialized as zero
# b = bias_variable(shape=[n_classes])
# num_hidden_units = 4
rnn_output_logits = BiRNN(x, timesteps, num_hidden_units)    # T [B, D]
states = tf.convert_to_tensor(rnn_output_logits)       # [T, B, D]
states = tf.transpose(rnn_output_logits, [1, 0, 2])  # (T,B,D) => (B,T,D)
# state = predict_strategy(rnn_output_logits)
# states = attention_block(rnn_output_logits, attention_size)
pred = predict_strategy(states)






# outputs=np.array(output_logits)
# y_pred = tf.nn.softmax(output_logits)

# Model predictions
# cls_prediction = tf.argmax(output_logits, axis=1, name='predictions')
def HybridLoss(pred, labels, weights, biases):
    y_pred = tf.concat(pred, axis=0)
    y_labels = tf.concat(tf.unstack(labels, timesteps, 1), axis=0)
    # outputs:(list)    (?,4)
    # y:(list)          (?,4)
    y_pred_strategy = y_pred[0:, 0:3]
    y_pred_timeout = y_pred[0:, 3:]

    label_strategy = y_labels[0:, 0:3]
    label_timeout = y_labels[0:, 3:]

    loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label_strategy,
                                                                      logits=y_pred_strategy)) + 0.045 * tf.nn.l2_loss(
        weights) + 0.045 * tf.nn.l2_loss(biases)

    accurate_strategy_pred = tf.equal(tf.arg_max(y_pred_strategy, 1), tf.arg_max(label_strategy, 1),
                                      name='accurate_strategy_pred')
    accu_strategy_pred = tf.cast(accurate_strategy_pred, tf.float32)

    timeout_distance = tf.square(label_timeout - y_pred_timeout)
    loss1 = tf.reduce_mean(tf.multiply(accu_strategy_pred, timeout_distance) + (1 - accu_strategy_pred) * 2)
    # loss2 = 0.045*tf.nn.l2_loss(weights) + 0.045*tf.nn.l2_loss(biases)

    # loss2 = tf.reduce_mean(tf.convert_to_tensor(tf.where(accurate_strategy_pred,timeout_distance,max_timeout_loss_value)))/2

    loss = loss1 + loss2

    return loss


def cosine(q, a):
    normalize_q = tf.nn.l2_normalize(q, 0)
    normalize_a = tf.nn.l2_normalize(a, 0)
    cos_similarity = tf.reduce_sum(tf.multiply(normalize_q, normalize_a))
    return cos_similarity


def Accuracy(outputs, labels):
    y_pred = tf.concat(outputs, axis=0)
    y_labels = tf.concat(tf.unstack(labels, timesteps, 1), axis=0)

    y_pred_strategy = y_pred[0:, 0:3]
    y_pred_timeout = y_pred[0:, 3:]

    label_strategy = y_labels[0:, 0:3]
    label_timeout = y_labels[0:, 3:]

    accurate_strategy_pred = tf.equal(tf.arg_max(y_pred_strategy, 1), tf.arg_max(label_strategy, 1),
                                      name='accurate_strategy_pred')
    accurate_strategy_pred = tf.cast(accurate_strategy_pred, tf.float32)

    timeout_distance = tf.square(label_timeout - y_pred_timeout)
    i = tf.multiply(accurate_strategy_pred, timeout_distance)
    error = tf.multiply(accurate_strategy_pred, timeout_distance) + (1 - accurate_strategy_pred) * 2

    # normalize_y_pred = tf.nn.l2_normalize(y_pred, 0)
    # normalize_y_labels = tf.nn.l2_normalize(y_labels, 0)
    # accu=tf.losses.cosine_distance(normalize_y_labels, normalize_y_pred,axis=0)

    # accu=[]
    # for i in range(len(y_pred)):
    #     # accu.append((cosine(y_pred[i],y_labels[i])).eval(session=sess))
    #     i
    return accurate_strategy_pred, error

# pred (B, T, n_classes) => (B*T,n_classes)
pred = tf.concat(pred, axis=0)
# label (B, T, n_classes)
labels = y
labels = tf.concat(labels, axis=0)

# cls_prediction = tf.arg_max(pred, axis=1, name='predictions')
# Define the loss function, optimizer, and accuracy
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=pred), name='loss')
# loss = tf.losses.mean_squared_error(y_true, y_pred)
# loss = HybridLoss(outputs, y, W, b)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam-op').minimize(loss)
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1), name='correct_pred')
# correct_prediction, timeout_error = Accuracy(outputs, y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
# error = tf.reduce_mean(tf.cast(timeout_error, tf.float32), name='error')

# Creating the op for initializing all variables
init = tf.global_variables_initializer()

sess = tf.InteractiveSession()
sess.run(init)
global_step = 0
# Number of training iterations in each epoch
num_tr_iter = int(len(y_train) / batch_size)
num_va_iter = int(len(y_valid) / batch_size)
training_epoch = []
validation_loss = []
validation_accu = []
# validation_timeout_error = []
training_loss = []
training_accu = []
# training_timeout_error = []
for epoch in range(epoches):
    training_epoch.append(epoch)
    print('Training epoch: {}'.format(epoch + 1))
    iter_loss = []
    iter_accu = []
    # iter_erro = []
    for iteration in range(num_tr_iter):
        global_step += 1
        start = iteration * batch_size
        end = (iteration + 1) * batch_size
        x_batch, y_batch = get_next_batch(x_train, y_train, start, end)
        x_batch = np.reshape(x_batch, (batch_size, timesteps, num_input))
        y_batch = np.reshape(y_batch, (batch_size, timesteps, n_classes))
        # Run optimization op (backprop)
        feed_dict_batch = {x: x_batch, y: y_batch}

        if iteration % display_freq == 0:
            # Calculate and display the batch loss and accuracy
            loss_batch, acc_batch = sess.run([loss, accuracy], feed_dict=feed_dict_batch)

            iter_loss.append(loss_batch)
            iter_accu.append(acc_batch)
            # iter_erro.append(erro_batch)
            print("iter {0:3d}:\t Loss={1:.2f},\tTraining Accuracy={2:.01%}".
                  format(iteration, loss_batch, acc_batch))
    # count loss for training data
    iter_average_loss = sum(iter_loss) / len(iter_loss)
    iter_average_accu = sum(iter_accu) / len(iter_accu)
    # iter_average_erro = sum(iter_erro) / len(iter_erro)
    training_loss.append(iter_average_loss)
    training_accu.append(iter_average_accu)
    # training_timeout_error.append(iter_average_erro)

    # Run validation after every epoch
    iter_loss = []
    iter_accu = []
    # iter_erro = []
    for iteration in range(num_va_iter):
        global_step += 1
        start = iteration * batch_size
        end = (iteration + 1) * batch_size
        x_batch, y_batch = get_next_batch(x_valid, y_valid, start, end)
        x_batch = np.reshape(x_batch, (batch_size, timesteps, num_input))
        y_batch = np.reshape(y_batch, (batch_size, timesteps, n_classes))
        # Run optimization op (backprop)
        feed_dict_valid = {x: x_batch, y: y_batch}
        loss_valid, acc_valid = sess.run([loss, accuracy], feed_dict=feed_dict_valid)
        iter_loss.append(loss_valid)
        iter_accu.append(acc_valid)

    # count loss for validation data
    iter_valid_average_loss = sum(iter_loss) / len(iter_loss)
    iter_valid_average_accu = sum(iter_accu) / len(iter_accu)

    validation_loss.append(iter_average_loss)
    validation_accu.append(iter_average_accu)

    # validation_timeout_error.append(error_valid)
    print('---------------------------------------------------------')
    print("Epoch: {0}, validation loss: {1:.2f}, validation accuracy: {2:.01%}".
          format(epoch + 1, iter_valid_average_loss, iter_valid_average_accu))
    print('---------------------------------------------------------')

# Test
x_test, y_test = load_data(mode='test')
# Run test
num_te_iter = int(len(y_test) / batch_size)
iter_loss = []
iter_accu = []
# iter_erro = []
for iteration in range(num_te_iter):
    global_step += 1
    start = iteration * batch_size
    end = (iteration + 1) * batch_size
    x_batch, y_batch = get_next_batch(x_test, y_test, start, end)
    x_batch = np.reshape(x_batch, (batch_size, timesteps, num_input))
    y_batch = np.reshape(y_batch, (batch_size, timesteps, n_classes))
    # Run optimization op (backprop)
    feed_dict_test = {x: x_batch, y: y_batch}
    loss_test, acc_test = sess.run([loss, accuracy], feed_dict=feed_dict_test)
    iter_loss.append(loss_test)
    iter_accu.append(acc_test)

# count loss for validation data
iter_test_average_loss = sum(iter_loss) / len(iter_loss)
iter_test_average_accu = sum(iter_accu) / len(iter_accu)

print('---------------------------------------------------------')
print("Test loss: {0:.2f}, test accuracy: {1:.01%}".format(iter_test_average_loss, iter_test_average_accu))
print('---------------------------------------------------------')

# print result
DATETIME_NOW = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# scatter plot
plt.plot(training_epoch, validation_loss, 'r--')
plt.plot(training_epoch, training_loss, 'b-')
plt.legend(['Validation Loss', 'Training Loss'])

# change axes ranges

# add title
plt.title('Relationship Between Epochs and Losses')

# add x and y labels
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('result_strategy_loss_{}.png'.format(DATETIME_NOW))

plt.cla()

plt.plot(training_epoch, validation_accu, 'r--')
plt.plot(training_epoch, training_accu, 'b-')
plt.legend(['Validation_accu', 'Training_accu'])
plt.title('Relationship between Epochs and Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.savefig('result_strategy_accuracy_{}.png'.format(DATETIME_NOW))

# plt.cla()
#
# # plt.plot(training_epoch,validation_timeout_error,'r--')
# plt.plot(training_epoch, training_timeout_error, 'b-')
# plt.legend(['Training_error'])
# plt.title('Relationship between Epochs and Timeout Error')
# plt.xlabel('Epoch')
# plt.ylabel('Timeout Error')
#
# plt.savefig('result_tiemout_error_{}.png'.format(DATETIME_NOW))
