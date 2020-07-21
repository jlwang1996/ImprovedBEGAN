import tensorflow as tf
from tensorflow.contrib import layers
#conv2d(inputs, # 输入的张量
　　 filters, # 卷积过滤器的数量
    kernel_size, # 卷积窗口的大小
    strides=(1, 1), # 卷积步长
    padding='valid', # 可选，默认为 valid，padding 的模式，有 valid 和 same 两种，大小写不区分。
    data_format='channels_last', # 可选，默认 channels_last，分为 channels_last 和 channels_first 两种模式，代表了输入数据的维度类型，如果是 channels_last，那么输入数据的 shape 为 (batch, height, width, channels)，如果是 channels_first，那么输入数据的 shape 为 (batch, channels, height, width)
    dilation_rate=(1, 1),# 可选，默认为 (1, 1)，卷积的扩张率，如当扩张率为 2 时，卷积核内部就会有边距，3×3 的卷积核就会变成 5×5。
    activation=None, # 可选，默认为 None，如果为 None 则是线性激活。
    use_bias=True, # 可选，默认为 True，是否使用偏置。
    kernel_initializer=None, # 可选，默认为 None，即权重的初始化方法，如果为 None，则使用默认的 Xavier 初始化方法。
    bias_initializer=<tensorflow.python.ops.init_ops.Zeros object at 0x000002596A1FD898>, # 可选，默认为零值初始化，即偏置的初始化方法。
    kernel_regularizer=None,# 可选，默认为 None，施加在权重上的正则项。
    bias_regularizer=None, # 可选，默认为 None，施加在偏置上的正则项。
    activity_regularizer=None, # 可选，默认为 None，施加在输出上的正则项。
    kernel_constraint=None, # 可选，默认为 None，施加在权重上的约束项。
    bias_constraint=None, # 可选，默认为 None，施加在偏置上的约束项。
    trainable=True, # 可选，默认为 True，布尔类型，如果为 True，则将变量添加到 GraphKeys.TRAINABLE_VARIABLES 中。
    name=None, # 可选，默认为 None，卷积层的名称。
    reuse=None) # 可选，默认为 None，布尔类型，如果为 True，那么如果 name 相同时，会重复利用。
    input 和tf.nn.conv2d 一样，必须是4维张量，input.shape=[batch, in_height, in_width, in_channels]

def decoder(name, z, num_units, num_repeats, is_training, unit=8, carry=None, batch_norm=False, reuse=False):

    with tf.variable_scope(name, reuse=reuse):
        batch_norm_fn = tf.layers.batch_normalization if batch_norm else None
        normalize_params = {"momentum": 0.9, "training": is_training, "scale": True, "fused": True} if batch_norm else None
        initializer = tf.random_normal_initializer(0., 0.02)
        num_output = unit*unit*num_units
        img = layers.fully_connected(z, num_output, weights_initializer=initializer, activation_fn=None, normalizer_fn=batch_norm_fn, normalizer_params=normalize_params)
        img = inj = tf.reshape(img, [-1, unit, unit, num_units])

        for i in range(num_repeats):
            img = in_x = layers.conv2d(img, num_units, 3, 1, weights_initializer=initializer, activation_fn=tf.nn.elu)
            img = layers.conv2d(img, num_units, 3, 1, weights_initializer=initializer, biases_initializer=None, activation_fn=tf.nn.elu, normalizer_fn=batch_norm_fn, normalizer_params=normalize_params)

            if carry is not None:
                img = carry*in_x+(1-carry)*img

            if i < num_repeats-1:
                img_shape = img.get_shape()
                inj = tf.image.resize_nearest_neighbor(inj, (2*int(img_shape[1]), 2*int(img_shape[2])))
                img = tf.image.resize_nearest_neighbor(img, (2*int(img_shape[1]), 2*int(img_shape[2])))
                img = tf.concat([img, inj], axis=-1)

        out = layers.conv2d(img, 3, 3, 1, weights_initializer=initializer, activation_fn=None)

    return out


def encoder(name, img, num_z, num_units, num_repeats, is_training, unit=8, carry=None, batch_norm=False, reuse=False):

    with tf.variable_scope(name, reuse=reuse):
        batch_norm_fn = tf.layers.batch_normalization if batch_norm else None
        normalize_params = {"momentum": 0.9, "training": is_training, "scale": True, "fused": True} if batch_norm else None
        initializer = tf.random_normal_initializer(0., 0.02)
        img = layers.conv2d(img, num_units, 3, 1, weights_initializer=initializer, biases_initializer=None, activation_fn=tf.nn.elu, normalizer_fn=batch_norm_fn, normalizer_params=normalize_params)

        for i in range(num_repeats):
            img = in_x = layers.conv2d(img, num_units*(i+1), 3, 1, weights_initializer=initializer, activation_fn=tf.nn.elu)
            img = layers.conv2d(img, num_units*(i+1), 3, 1, weights_initializer=initializer, biases_initializer=None, activation_fn=tf.nn.elu, normalizer_fn=batch_norm_fn, normalizer_params=normalize_params)

            if carry is not None:
                img = carry*in_x+(1-carry)*img

            if i < num_repeats-1:
                img = layers.conv2d(img, num_units*(i+1), 3, 2, weights_initializer=initializer, activation_fn=tf.nn.elu)

        img = tf.reshape(img, [-1, unit*unit*num_units*num_repeats])
        z = layers.fully_connected(img, num_z, weights_initializer=initializer, activation_fn=None)

    return z


class ImprovedBEGAN:

    def __init__(self, name, img_size, num_z, num_units, num_repeats, batch_norm=False):
        self._name = name
        self._img_size = img_size
        self._num_z = num_z
        self._num_units = num_units
        self._num_repeats = num_repeats
        self._batch_norm = batch_norm

        self.z_in = None
        self.training = None
        self.carry = None
        self.g = None

        self.gamma = None
        self.img = None
        self.inj = None
        self.g_lr = None
        self.d_lr = None
        self.lambda_k = None
        self.lambda_noise = None
        self.global_step = None
        self.z_enc = None
        self.d_real = None
        self.d_fake = None
        self.t_vars = None
        self.g_vars = None
        self.d_vars = None
        self.d_loss_real = None
        self.d_loss_fake = None
        self.d_loss_rec = None
        self.g_loss = None
        self.d_loss = None
        self.train_op = None
        self.measure = None

    @property
    def name(self):
        return self._name

    def build_model(self):

        with tf.variable_scope(self.name):
            self.z_in = tf.placeholder(tf.float32, [self._img_size[0], self._num_z])

            self.training = tf.placeholder(tf.bool, shape=())
            self.carry = tf.placeholder(tf.float32, shape=())

            self.g = decoder("generator", self.z_in, self._num_units, self._num_repeats, self.training, self.carry, batch_norm=self._batch_norm)

    def build_train_op(self):

        with tf.variable_scope(self.name):
            self.gamma = tf.placeholder(tf.float32, shape=())
            self.img = tf.placeholder(tf.float32, self._img_size)
            self.inj = tf.placeholder(tf.float32, self._img_size)
            self.g_lr = tf.placeholder(tf.float32, shape=())
            self.d_lr = tf.placeholder(tf.float32, shape=())
            self.lambda_k = tf.placeholder(tf.float32, shape=())
            self.lambda_noise = tf.placeholder(tf.float32, shape=())
            noised_img = self.img+self.inj
            k = tf.Variable(0., trainable=False)

            self.global_step = tf.Variable(0, trainable=False)
            global_step_update_op = tf.assign_add(self.global_step, 1)

            self.z_enc = encoder("encoder", self.img, self._num_z, self._num_units, self._num_repeats, self.training, self.carry, batch_norm=self._batch_norm)
            z_enc_fake = encoder("encoder", self.g, self._num_z, self._num_units, self._num_repeats, self.training, self.carry, batch_norm=self._batch_norm, reuse=True)
            z_rec = encoder("encoder", noised_img, self._num_z, self._num_units, self._num_repeats, self.training, self.carry, batch_norm=self._batch_norm, reuse=True)
            self.d_real = decoder("decoder", self.z_enc, self._num_units, self._num_repeats, self.training, self.carry, batch_norm=self._batch_norm)
            self.d_fake = decoder("decoder", z_enc_fake, self._num_units, self._num_repeats, self.training, self.carry, batch_norm=self._batch_norm, reuse=True)
            d_rec = decoder("decoder", z_rec, self._num_units, self._num_repeats, self.training, self.carry, batch_norm=self._batch_norm, reuse=True)

            self.t_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
            self.g_vars = [var for var in self.t_vars if "generator" in var.name]
            self.d_vars = [var for var in self.t_vars if "encoder" in var.name or "decoder" in var.name]

            self.d_loss_real = tf.reduce_mean(tf.abs(self.d_real-self.img))
            self.d_loss_fake = tf.reduce_mean(tf.abs(self.d_fake-self.g))
            self.d_loss_rec = tf.reduce_mean(tf.square(d_rec-self.img))

            self.g_loss = self.d_loss_fake
            self.d_loss = self.d_loss_real-k*self.d_loss_fake+self.lambda_noise*self.d_loss_rec
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            g_update_ops = [update_op for update_op in update_ops if "generator" in update_op.name]
            d_update_ops = [update_op for update_op in update_ops if "encoder" in update_op.name or "decoder" in update_op.name]

            with tf.control_dependencies(g_update_ops):
                g_train_op = tf.train.AdamOptimizer(self.g_lr, 0.5).minimize(self.g_loss, var_list=self.g_vars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

            with tf.control_dependencies(d_update_ops):
                d_train_op = tf.train.AdamOptimizer(self.d_lr, 0.5).minimize(self.d_loss, var_list=self.d_vars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

            balance = self.gamma*self.d_loss_real-self.d_loss_fake
            self.measure = self.d_loss_real+tf.abs(balance)

            assign = tf.clip_by_value(k+self.lambda_k*balance, 0, 1)
            k_update_op = tf.assign(k, assign)
            self.train_op = [g_train_op, d_train_op, k_update_op, global_step_update_op]

            tf.summary.scalar("measure", self.measure)
            tf.summary.scalar("d_loss_real", self.d_loss_real)
            tf.summary.scalar("d_loss_fake", self.d_loss_fake)
            tf.summary.scalar("d_loss_rec", self.d_loss_rec)
            tf.summary.scalar("g_loss", self.g_loss)
            tf.summary.scalar("d_loss", self.d_loss)
            tf.summary.scalar("k", k)


