import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell


# 其实是全连接层
def prenet(inputs, is_training, layer_sizes, scope=None):
    x = inputs
    drop_rate = 0.5 if is_training else 0.0
    with tf.variable_scope(scope or 'prenet'):
        for i, size in enumerate(layer_sizes):
            dense = tf.layers.dense(x, units=size, activation=tf.nn.relu, name='dense_%d' % (i + 1))
            x = tf.layers.dropout(dense, rate=drop_rate, training=is_training, name='dropout_%d' % (i + 1))
    return x


def encoder_cbhg(inputs, input_lengths, is_training, output_size):
    input_channels = inputs.get_shape()[2]  # 128
    return cbhg(
        inputs,
        input_lengths,
        is_training,
        scope='encoder_cbhg',
        K=16,
        projections=[128, input_channels],  # [128, 128]
        output_size=output_size)


def post_cbhg(inputs, input_dim, is_training, output_size):
    return cbhg(
        inputs,
        None,
        is_training,
        scope='post_cbhg',
        K=8,
        projections=[256, input_dim],  # [256, 80]
        output_size=output_size)


def cbhg(inputs, input_lengths, is_training, scope, K, projections, output_size):
    """
    :param inputs: [B, T, pre_128 or post_80]
    :param input_lengths: 输入文本长度
    :param is_training:
    :param scope:
    :param K: 卷积核最大值
    :param projections: 投影层维度
    :param output_size: 输出维度
    :return:
    """
    with tf.variable_scope(scope):
        with tf.variable_scope('conv_bank'):
            # 卷积层: [B, T, K*128]
            conv_outputs = tf.concat(
                [conv1d(inputs, k, 128, tf.nn.relu, is_training, 'conv1d_%d' % k) for k in range(1, K + 1)],
                axis=-1
            )

        # 时间上相邻两个取最大值
        # Maxpooling: [B, T, K*128]
        maxpool_output = tf.layers.max_pooling1d(
            conv_outputs,
            pool_size=2,
            strides=1,
            padding='same')

        # Two projection layers:
        # [B, T, pre_128 or post_256]
        proj1_output = conv1d(maxpool_output, 3, projections[0], tf.nn.relu, is_training, 'proj_1')
        # [B, T, pre_128 or post_80]
        proj2_output = conv1d(proj1_output, 3, projections[1], None, is_training, 'proj_2')

        # Residual connection:
        highway_input = proj2_output + inputs

        half_size = output_size // 2
        assert half_size * 2 == output_size, 'encoder and postnet sizes must be even.'

        # Handle dimensionality mismatch:
        if highway_input.shape[2] != half_size:
            highway_input = tf.layers.dense(highway_input, half_size)

        # 4-layer HighwayNet: [B, T, depth//2]
        for i in range(4):
            highway_input = highwaynet(highway_input, 'highway_%d' % (i + 1), half_size)
        rnn_input = highway_input

        # 双向RNN: [B, T, input_dim]
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            GRUCell(half_size),
            GRUCell(half_size),
            rnn_input,
            sequence_length=input_lengths,  # 输入序列长度
            dtype=tf.float32)
        # [B, T, output_size]
        return tf.concat(outputs, axis=2)  # Concat forward and backward


def highwaynet(inputs, scope, depth):
    with tf.variable_scope(scope):
        H = tf.layers.dense(
            inputs,
            units=depth,
            activation=tf.nn.relu,
            name='H')
        T = tf.layers.dense(
            inputs,
            units=depth,
            activation=tf.nn.sigmoid,
            name='T',
            bias_initializer=tf.constant_initializer(-1.0))
        return H * T + inputs * (1.0 - T)


def conv1d(inputs, kernel_size, channels, activation, is_training, scope):
    with tf.variable_scope(scope):
        conv1d_output = tf.layers.conv1d(
            inputs,
            filters=channels,
            kernel_size=kernel_size,
            activation=activation,
            padding='same')
        return tf.layers.batch_normalization(conv1d_output, training=is_training)
