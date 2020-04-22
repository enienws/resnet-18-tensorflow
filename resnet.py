from collections import namedtuple

import tensorflow as tf
import numpy as np

import utils

WEIGHT_DECAY_KEY = 'WEIGHT_DECAY'

HParams = namedtuple('HParams',
                    'batch_size, num_gpus, weight_decay')

class ResNet(object):
    def __init__(self, hp, images, labels, global_step, name=None, reuse_weights=False):
        self._hp = hp # Hyperparameters
        self._images = images # Input images
        self._labels = labels # Input labels
        self._global_step = global_step
        self._name = name
        self._reuse_weights = reuse_weights
        self.lr = tf.placeholder(tf.float32, name="lr")
        self.is_train = tf.placeholder(tf.bool, name="is_train")
        self._counted_scope = []
        self._flops = 0
        self._weights = 0


    def build_tower(self, images, labels):
        print('Building model')
        image_width = 256
        image_height = 256
        num_samples = 4

        # filters = [128, 128, 256, 512, 1024]
        filters = [64, 64, 128, 256, 256]
        kernels = [7, 3, 3, 3, 3]
        strides = [2, 0, 2, 1, 1]

        images = tf.reshape(images, (-1, image_height, image_width, 1))

        # conv1
        print('\tBuilding unit: conv1')
        with tf.variable_scope('conv1'):
            x = self._conv(images, kernels[0], filters[0], strides[0])
            x = self._bn(x)
            x = self._relu(x)
            x = tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')

        # conv2_x
        x = self._residual_block(x, name='conv2_1')
        x = self._residual_block(x, name='conv2_2')

        # conv3_x
        x = self._residual_block_first(x, filters[2], strides[2], name='conv3_1')
        x = self._residual_block(x, name='conv3_2')

        # conv4_x
        x = self._residual_block_first(x, filters[3], strides[3], name='conv4_1')
        x = self._residual_block(x, name='conv4_2')

        # conv5_x
        x = self._residual_block_first(x, filters[4], strides[4], name='conv5_1')
        x = self._residual_block(x, name='conv5_2')

        # # add spatial information
        # vertical_tensor, horizontal_tensor = self._get_spatial_tensors(32, tf.shape(x))
        # x = tf.concat([x, vertical_tensor, horizontal_tensor], -1)

        orig_shape = tf.shape(x, name="Shape3D")
        # [N*T,H',W',C'] -> [N,T,H',W',C']
        x = tf.reshape(x, tf.concat([[-1], [num_samples], orig_shape[1:]], 0, name="Concat3D"), name="Reshape3D")

        # 3D Conv Layer1
        with tf.variable_scope('3d_conv_1_1'):
            x = self._conv_3d(x, 256, [1, 3, 3], [1, 1, 1, 1, 1])
            x = self._batch_norm_3d(x)
            x = self._relu_3d(x)
        with tf.variable_scope('3d_conv_1_2'):
            x = self._conv_3d(x, 256, [3, 1, 1], [1, 1, 1, 1, 1])
            x = self._batch_norm_3d(x)
            x = self._relu_3d(x)

        # # 3D Conv Layer2
        with tf.variable_scope('3d_conv_2_1'):
            x = self._conv_3d(x, 256, [1, 3, 3], [1, 1, 2, 2, 1])
            x = self._batch_norm_3d(x)
            x = self._relu_3d(x)
        with tf.variable_scope('3d_conv_2_2'):
            x = self._conv_3d(x, 256, [3, 1, 1], [1, 1, 1, 1, 1])
            x = self._batch_norm_3d(x)
            x = self._relu_3d(x)

        # # 3D Conv Layer3
        with tf.variable_scope('3d_conv_3_1'):
            x = self._conv_3d(x, 256, [1, 3, 3], [1, 1, 4, 4, 1])
            x = self._batch_norm_3d(x)
            x = self._relu_3d(x)
        with tf.variable_scope('3d_conv_3_2'):
            x = self._conv_3d(x, 256, [3, 1, 1], [1, 1, 1, 1, 1])
            x = self._batch_norm_3d(x)
            x = self._relu_3d(x)

        # 3D Conv Layer4
        with tf.variable_scope('3d_conv_4_1'):
            x = self._conv_3d(x, 256, [1, 3, 3], [1, 1, 8, 8, 1])
            x = self._batch_norm_3d(x)
            x = self._relu_3d(x)
        with tf.variable_scope('3d_conv_4_2'):
            x = self._conv_3d(x, 256, [3, 1, 1], [1, 1, 1, 1, 1])
            x = self._batch_norm_3d(x)
            x = self._relu_3d(x)

        # 3D Conv Layer5
        with tf.variable_scope('3d_conv_5_1'):
            x = self._conv_3d(x, 256, [1, 3, 3], [1, 1, 16, 16, 1])
            x = self._batch_norm_3d(x)
            x = self._relu_3d(x)
        with tf.variable_scope('3d_conv_5_2'):
            x = self._conv_3d(x, 256, [3, 1, 1], [1, 1, 1, 1, 1])
            x = self._batch_norm_3d(x)
            x = self._relu_3d(x)

        # 3D Conv Final
        with tf.variable_scope('features'):
            x = self._conv_3d(x, 64, [1, 1, 1], [1, 1, 1, 1, 1])

        out_features = x

        return out_features


    def build_model(self):
        width = 32
        height = 32
        channels = 64
        num_samples = 4
        num_reference = 3
        num_labels = 16

        # Split images and labels into (num_gpus) groups
        # images = tf.split(self._images, num_or_size_splits=self._hp.num_gpus, axis=0)
        # labels = tf.split(self._labels, num_or_size_splits=self._hp.num_gpus, axis=0)

        # Build towers for each GPU
        # self._logits_list = []
        self._preds_list = []
        self._loss_list = []
        self._acc_list = []

        for i in range(self._hp.num_gpus):
            with tf.device('/GPU:%d' % i), tf.variable_scope(tf.get_variable_scope()):
                with tf.name_scope('tower_%d' % i) as scope:
                    print('Build a tower: %s' % scope)
                    if self._reuse_weights or i > 0:
                        tf.get_variable_scope().reuse_variables()

                    # preds, loss  = self.build_tower(self._images, self._labels)
                    out_features = self.build_tower(self._images[i], self._labels[i])
                    # self._logits_list.append(logits)
                    # self._preds_list.append(preds)
                    # self._loss_list.append(loss)
                    # self._acc_list.append(acc)

            with tf.device('/CPU:0'):
                splited_features = tf.split(out_features, num_or_size_splits=num_samples, axis=1)
                splited_labels = tf.split(self._labels[i], num_or_size_splits=num_samples, axis=1)
                tf.logging.info("out_features_stacked: %s, labels: %s", out_features.get_shape(),
                                self._labels[i].get_shape())

                reference_features = tf.stack(splited_features[:num_reference], axis=1)
                reference_labels = tf.stack(splited_labels[:num_reference], axis=1)
                target_features = tf.stack(splited_features[num_reference:], axis=1)
                target_labels = tf.stack(splited_labels[num_reference:], axis=1)

                with tf.name_scope('similarity_matrix') as name_scope:
                    ref = tf.transpose(tf.reshape(reference_features, [-1, width * height * num_reference, channels]),
                                       perm=[0, 2, 1])
                    tar = tf.reshape(target_features, [-1, width * height, channels])

                    innerproduct = tf.matmul(tar, ref)
                    # softmax_axis = 2 if predict_backward else 1
                    # Why using temprature??
                    #TODO :: work on temperature
                    # temperature = 1.0 if self.is_train else 0.5
                    # look to formula 2 in paper.
                    similarity = tf.nn.softmax(innerproduct / 1, axis=1)
                    _, h, w = similarity.shape

                with tf.name_scope('prediction') as name_scope:
                    ref = tf.reshape(reference_labels, (-1, width * height * num_reference))
                    dense_reference_labels = tf.one_hot(ref, num_labels)

                    prediction = tf.matmul(similarity, dense_reference_labels)
                    logits = tf.reshape(prediction, [-1, height, width, num_labels])
                    target_labels = tf.reshape(target_labels, [-1, height, width, 1])
                    tf.logging.info('image after unit %s: %s', name_scope, prediction.get_shape())

                reshaped_logits = tf.reshape(logits, (-1, num_labels))
                reshaped_target_labels = tf.reshape(target_labels, (-1,))

                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=reshaped_logits, labels=reshaped_target_labels)
                loss = tf.reduce_mean(loss)  # take the mean across the batches.
                self._loss_list.append(loss)
                accuracy = tf.metrics.accuracy(reshaped_target_labels, tf.reshape(tf.argmax(input=logits, axis=-1), (-1,)))
                self._acc_list.append(accuracy)

                # tf.summary.scalar((self._name + "/" if self._name else "") + "cross_entr_loss", self.loss)
                # tf.summary.scalar((self._name + "/" if self._name else "") + "accuracy", self.accuracy[1])

        # Merge losses, accuracies of all GPUs
        with tf.device('/CPU:0'):
            self.loss = tf.reduce_mean(self._loss_list, name="cross_entropy")
            tf.summary.scalar((self._name+"/" if self._name else "") + "cross_entropy", self.loss)
            self.acc = tf.reduce_mean(self._acc_list, name="accuracy")
            tf.summary.scalar((self._name+"/" if self._name else "") + "accuracy", self.acc)


    def build_train_op(self):
        # Learning rate
        tf.summary.scalar((self._name+"/" if self._name else "") + 'learing_rate', self.lr)

        opt = tf.train.AdamOptimizer(self.lr)
        self._grads_and_vars_list = []

        # Computer gradients for each GPU
        for i in range(self._hp.num_gpus):
            with tf.device('/GPU:%d' % i), tf.variable_scope(tf.get_variable_scope()):
                with tf.name_scope('tower_%d' % i) as scope:
                    print('Compute gradients of tower: %s' % scope)
                    if self._reuse_weights or i > 0:
                        tf.get_variable_scope().reuse_variables()

                    # Add l2 loss
                    costs = [tf.nn.l2_loss(var) for var in tf.get_collection(utils.WEIGHT_DECAY_KEY)]
                    l2_loss = tf.multiply(self._hp.weight_decay, tf.add_n(costs))
                    total_loss = self._loss_list[i] + l2_loss
                    # total_loss = self._loss_list[i]
                    # total_loss = self._loss_list + l2_loss

                    # Compute gradients of total loss
                    grads_and_vars = opt.compute_gradients(total_loss, tf.trainable_variables())

                    # Append gradients and vars
                    self._grads_and_vars_list.append(grads_and_vars)

        # Merge gradients
        print('Average gradients')
        with tf.device('/CPU:0'):
            grads_and_vars = self._average_gradients(self._grads_and_vars_list)

            # Finetuning
            # if self._hp.finetune:
            #     for idx, (grad, var) in enumerate(grads_and_vars):
            #         if "unit3" in var.op.name or \
            #             "unit_last" in var.op.name or \
            #             "/q" in var.op.name or \
            #             "logits" in var.op.name:
            #             print('\tScale up learning rate of % s by 10.0' % var.op.name)
            #             grad = 10.0 * grad
            #             grads_and_vars[idx] = (grad,var)

            # Apply gradient
            apply_grad_op = opt.apply_gradients(grads_and_vars, global_step=self._global_step)

            # Batch normalization moving average update
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            self.train_op = tf.group(*(update_ops+[apply_grad_op]))


    def _residual_block_first(self, x, out_channel, strides, name="unit"):
        in_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name) as scope:
            print('\tBuilding residual unit: %s' % scope.name)

            # Shortcut connection
            if in_channel == out_channel:
                if strides == 1:
                    shortcut = tf.identity(x)
                else:
                    shortcut = tf.nn.max_pool(x, [1, strides, strides, 1], [1, strides, strides, 1], 'VALID')
            else:
                shortcut = self._conv(x, 1, out_channel, strides, name='shortcut')
            # Residual
            x = self._conv(x, 3, out_channel, strides, name='conv_1')
            x = self._bn(x, name='bn_1')
            x = self._relu(x, name='relu_1')
            x = self._conv(x, 3, out_channel, 1, name='conv_2')
            x = self._bn(x, name='bn_2')
            # Merge
            x = x + shortcut
            x = self._relu(x, name='relu_2')
        return x


    def _residual_block(self, x, input_q=None, output_q=None, name="unit"):
        num_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name) as scope:
            print('\tBuilding residual unit: %s' % scope.name)
            # Shortcut connection
            shortcut = x
            # Residual
            x = self._conv(x, 3, num_channel, 1, input_q=input_q, output_q=output_q, name='conv_1')
            x = self._bn(x, name='bn_1')
            x = self._relu(x, name='relu_1')
            x = self._conv(x, 3, num_channel, 1, input_q=output_q, output_q=output_q, name='conv_2')
            x = self._bn(x, name='bn_2')

            x = x + shortcut
            x = self._relu(x, name='relu_2')
        return x


    def _average_gradients(self, tower_grads):
        """Calculate the average gradient for each shared variable across all towers.

        Note that this function provides a synchronization point across all towers.

        Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
           List of pairs of (gradient, variable) where the gradient has been averaged
           across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # If no gradient for a variable, exclude it from output
            if grad_and_vars[0][0] is None:
                continue

            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)

        return average_grads

    def _get_spatial_tensors(self, dimension, orig_shape):
        # dimension 32
        # orig_shape: batch(?), 32, 32, 256
        horizontal_line = np.zeros((dimension, 1))

        incr = 0
        for j, i in zip(range(0, dimension), range(-int(dimension / 2), int(dimension / 2))):
            if i == 0:
                incr = 1
            horizontal_line[j] = 2 / float(dimension) * (i + incr)

        horizontal = horizontal_line
        for i in range(dimension - 1):
            horizontal = np.concatenate((horizontal, horizontal_line), axis=1)

        vertical = horizontal
        horizontal = horizontal.transpose()

        vertical_tensor = tf.reshape(tf.convert_to_tensor(vertical, dtype=tf.float32), (dimension, dimension, 1))
        horizontal_tensor = tf.reshape(tf.convert_to_tensor(horizontal, dtype=tf.float32), (dimension, dimension, 1))

        vertical_tensor_br = tf.broadcast_to(vertical_tensor, tf.concat([orig_shape[:3], [1]], 0))
        horizontal_tensor_br = tf.broadcast_to(horizontal_tensor, tf.concat([orig_shape[:3], [1]], 0))

        return vertical_tensor_br, horizontal_tensor_br

    #3D Convolutions
    def _conv_3d(self, x, num_of_filters, kernel_size, dilation_rate, name = 'conv'):
        # in_shape = x.get_shape()
        # with tf.variable_scope(name):
        #     with tf.device('/CPU:0'):
        #         kernel = tf.get_variable('kernel', [kernel_size[0], kernel_size[1], kernel_size[2], in_shape[4], num_of_filters],
        #                          tf.float32, initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / kernel_size[1] / kernel_size[2] / num_of_filters)))
        #
        #     if kernel not in tf.get_collection(WEIGHT_DECAY_KEY):
        #         tf.add_to_collection(WEIGHT_DECAY_KEY, kernel)
        #
        #     x = tf.nn.conv3d(x,
        #                      filter=kernel,
        #                      strides=[1,1,1,1,1],
        #                      padding='SAME',
        #                      data_format='NDHWC',
        #                      dilations=dilation_rate
        #                          )
        # return x
        dilation_rate_ = (dilation_rate[1], dilation_rate[2], dilation_rate[3])
        x = tf.layers.conv3d(x, filters=num_of_filters,
                             kernel_size=kernel_size, padding='SAME',
                             data_format="channels_last",
                             kernel_initializer=tf.initializers.random_normal(mean=0.0,
                                                                              stddev=np.sqrt(2.0 / kernel_size[1] / kernel_size[2] / num_of_filters)
                                                                              ),
                             dilation_rate=dilation_rate_,
                             activation=None,
                             name=name
                             )
        kernel = tf.get_collection(tf.GraphKeys.VARIABLES, tf.get_variable_scope().name + '/conv')[0]
        if kernel not in tf.get_collection(WEIGHT_DECAY_KEY):
            tf.add_to_collection(WEIGHT_DECAY_KEY, kernel)

        return x

    def _batch_norm_3d(self, x, name='bn'):
        x = tf.contrib.layers.batch_norm(
            x,
            #decay=self._batch_norm_decay,
            center=True,
            scale=True,
            #epsilon=self._batch_norm_epsilon,
            is_training=self.is_train,
            fused=True,
            data_format='NHWC'
            # name=name
        )
        return x

    def _relu_3d(self, x, name='relu'):
        x = tf.nn.relu(x, name=name)
        return x

    # Helper functions(counts FLOPs and number of weights)
    def _conv(self, x, filter_size, out_channel, stride, pad="SAME", input_q=None, output_q=None, name="conv"):
        b, h, w, in_channel = x.get_shape().as_list()
        x = utils._conv(x, filter_size, out_channel, stride, pad, input_q, output_q, name)
        f = 2 * (h/stride) * (w/stride) * in_channel * out_channel * filter_size * filter_size
        w = in_channel * out_channel * filter_size * filter_size
        scope_name = tf.get_variable_scope().name + "/" + name
        self._add_flops_weights(scope_name, f, w)
        return x

    # def _fc(self, x, out_dim, input_q=None, output_q=None, name="fc"):
    #     b, in_dim = x.get_shape().as_list()
    #     x = utils._fc(x, out_dim, input_q, output_q, name)
    #     f = 2 * (in_dim + 1) * out_dim
    #     w = (in_dim + 1) * out_dim
    #     scope_name = tf.get_variable_scope().name + "/" + name
    #     self._add_flops_weights(scope_name, f, w)
    #     return x

    def _bn(self, x, name="bn"):
        x = utils._bn(x, self.is_train, self._global_step, name)
        # f = 8 * self._get_data_size(x)
        # w = 4 * x.get_shape().as_list()[-1]
        # scope_name = tf.get_variable_scope().name + "/" + name
        # self._add_flops_weights(scope_name, f, w)
        return x

    def _relu(self, x, name="relu"):
        x = utils._relu(x, 0.0, name)
        # f = self._get_data_size(x)
        # scope_name = tf.get_variable_scope().name + "/" + name
        # self._add_flops_weights(scope_name, f, 0)
        return x

    def _get_data_size(self, x):
        return np.prod(x.get_shape().as_list()[1:])

    def _add_flops_weights(self, scope_name, f, w):
        if scope_name not in self._counted_scope:
            self._flops += f
            self._weights += w
            self._counted_scope.append(scope_name)
