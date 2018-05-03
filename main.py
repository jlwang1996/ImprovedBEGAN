import numpy as np
from models import *
from dataloader import *


num_train = 300000
save_period = 500
batch_size = 20
img_check = 200
img_size = (64, 64)
num_z = 64
num_units = 128
num_repeats = 4
summary_period = 25
write_meta_graph = True
data_queue = loader("./CelebA/", batch_size=batch_size, img_size=img_size)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess, coord)
    model = ImprovedBEGAN("ImprovedBEGAN", img_size, num_z, num_units, num_repeats)

    model.build_model()
    model.build_train_op()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    batches = []

    summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter("./logs/"+model.name, sess.graph)

    for i in range(num_train):
        batch = sess.run(data_queue)
        batch = norm_img(batch)

        global_step = sess.run(model.global_step)
        l_rate = max(5E-5*0.95**(global_step//2000), 1E-7)

        inj = np.random.normal(size=[batch_size, *img_size, 3])

        feed_dict = {model.img: batch, model.inj: inj,
                     model.z_in: np.random.uniform(-1., 1., [batch_size, num_z]),
                     model.carry: max(1-i/16000., 0),
                     model.g_lr: l_rate, model.d_lr: l_rate,
                     model.gamma: 0.5, model.lambda_k: 0.001, model.lambda_noise: 2,
                     model.training: True}

        sess.run(model.train_op, feed_dict=feed_dict)

        if write_meta_graph:
            write_meta_graph = False

        if i % summary_period == 0:
            summary_str = sess.run(summary, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step)
            summary_writer.flush()

