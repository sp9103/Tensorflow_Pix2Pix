import tensorflow as tf

EPS = 1e-12


# Class for batch normalization node
class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name,
                                            reuse=tf.AUTO_REUSE     # if tensorflow vesrion < 1.4, delete this line
                                            )

# Class for instance normalization node
class instance_norm(object):
    def __init__(self, name="instance_norm"):
        with tf.variable_scope(name):
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.instance_norm(x,
                                            scope=self.name,
                                            reuse=tf.AUTO_REUSE     # if tensorflow vesrion < 1.4, delete this line
                                            )

# leaky relu function
def lrelu(X, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * X + f2 * tf.abs(X)


class Pix2Pix:
    # Network Parameters
    def __init__(self, sess, batch_size):
        self.learning_rate = 0.0002

        self.sess = sess
        self.batch_size = batch_size
        self.keep_prob = 0.5
        self.image_shape = [472, 472, 3]
        self.l1_weight = 100.0

        '''channels'''
        # Gen_Encoding
        self.ch_G0 = 4
        self.ch_G1 = 64
        self.ch_G2 = 128
        self.ch_G3 = 256
        self.ch_G4 = 512
        self.ch_G5 = 512
        self.ch_G6 = 512
        self.ch_G7 = 512
        self.ch_G8 = 512
        # Gen_Decoding
        self.ch_G9 = 512
        self.ch_G10 = 512
        self.ch_G11 = 512
        self.ch_G12 = 512
        self.ch_G13 = 256
        self.ch_G14 = 128
        self.ch_G15 = 64
        self.ch_G16 = 3
        # Discrim
        self.ch_D0 = 3
        self.ch_D1 = 64
        self.ch_D2 = 128
        self.ch_D3 = 256
        self.ch_D4 = 512
        self.ch_D5 = 1

        '''parameters'''
        # Gen_encoding
        self.G_W1 = tf.Variable(tf.truncated_normal([4, 4, self.ch_G0, self.ch_G1], stddev=0.02), name="G_W1")

        self.G_W2 = tf.Variable(tf.truncated_normal([4, 4, self.ch_G1, self.ch_G2], stddev=0.02), name='G_W2')
        self.G_in2 = instance_norm(name="G_in2")

        self.G_W3 = tf.Variable(tf.truncated_normal([4, 4, self.ch_G2, self.ch_G3], stddev=0.02), name='G_W3')
        self.G_in3 = instance_norm(name="G_in3")

        self.G_W4 = tf.Variable(tf.truncated_normal([4, 4, self.ch_G3, self.ch_G4], stddev=0.02), name='G_W4')
        self.G_in4 = instance_norm(name="G_in4")

        self.G_W5 = tf.Variable(tf.truncated_normal([4, 4, self.ch_G4, self.ch_G5], stddev=0.02), name='G_W5')
        self.G_in5 = instance_norm(name="G_in5")

        self.G_W6 = tf.Variable(tf.truncated_normal([4, 4, self.ch_G5, self.ch_G6], stddev=0.02), name='G_W6')
        self.G_in6 = instance_norm(name="G_in6")

        self.G_W7 = tf.Variable(tf.truncated_normal([4, 4, self.ch_G6, self.ch_G7], stddev=0.02), name='G_W7')
        self.G_in7 = instance_norm(name="G_in7")

        self.G_W8 = tf.Variable(tf.truncated_normal([4, 4, self.ch_G7, self.ch_G8], stddev=0.02), name='G_W8')
        self.G_in8 = instance_norm(name="G_in8")

        # Gen_Decoding
        self.G_W9 = tf.Variable(tf.truncated_normal([4, 4, self.ch_G9, self.ch_G8], stddev=0.02), name='G_W9')
        self.G_in9 = instance_norm(name="G_in9")

        self.G_W10 = tf.Variable(tf.truncated_normal([4, 4, self.ch_G10, self.ch_G9 + self.ch_G7], stddev=0.02), name='G_W10')
        self.G_in10 = instance_norm(name="G_in10")

        self.G_W11 = tf.Variable(tf.truncated_normal([4, 4, self.ch_G11, self.ch_G10 + self.ch_G6], stddev=0.02), name='G_W11')
        self.G_in11 = instance_norm(name="G_in11")

        self.G_W12 = tf.Variable(tf.truncated_normal([4, 4, self.ch_G12, self.ch_G11 + self.ch_G5], stddev=0.02), name='G_W12')
        self.G_in12 = instance_norm(name="G_in12")

        self.G_W13 = tf.Variable(tf.truncated_normal([4, 4, self.ch_G13, self.ch_G12 + self.ch_G4], stddev=0.02), name='G_W13')
        self.G_in13 = instance_norm(name="G_in13")

        self.G_W14 = tf.Variable(tf.truncated_normal([4, 4, self.ch_G14, self.ch_G13 + self.ch_G3], stddev=0.02), name='G_W14')
        self.G_in14 = instance_norm(name="G_in14")

        self.G_W15 = tf.Variable(tf.truncated_normal([4, 4, self.ch_G15, self.ch_G14 + self.ch_G2], stddev=0.02), name='G_W15')
        self.G_in15 = instance_norm(name="G_in15")

        self.G_W16 = tf.Variable(tf.truncated_normal([4, 4, self.ch_G16, self.ch_G15 + self.ch_G1], stddev=0.02), name='G_W16')

        # Discrim
        self.D_W1 = tf.Variable(tf.truncated_normal([4, 4, self.ch_D0, self.ch_D1], stddev=0.02), name='D_W1')
        self.D_in1 = instance_norm(name="D_in1")

        self.D_W2 = tf.Variable(tf.truncated_normal([4, 4, self.ch_D1, self.ch_D2], stddev=0.02), name='D_W2')
        self.D_in2 = instance_norm(name="D_in2")

        self.D_W3 = tf.Variable(tf.truncated_normal([4, 4, self.ch_D2, self.ch_D3], stddev=0.02), name='D_W3')
        self.D_in3 = instance_norm(name="D_in3")

        self.D_W4 = tf.Variable(tf.truncated_normal([4, 4, self.ch_D3, self.ch_D4], stddev=0.02), name='D_W4')
        self.D_in4 = instance_norm(name="D_in4")

        self.D_W5 = tf.Variable(tf.truncated_normal([4, 4, self.ch_D4, self.ch_D5], stddev=0.02), name='D_W5')

        self.gen_params = [
            self.G_W1,
            self.G_W2,
            self.G_W3,
            self.G_W4,
            self.G_W5,
            self.G_W6,
            self.G_W7,
            self.G_W8,
            self.G_W9,
            self.G_W10,
            self.G_W11,
            self.G_W12,
            self.G_W13,
            self.G_W14,
            self.G_W15,
            self.G_W16
        ]

        self.discrim_params = [
            self.D_W1,
            self.D_W2,
            self.D_W3,
            self.D_W4,
            self.D_W5
        ]

        self._build_model()

    # Build the Network
    def _build_model(self):
        self.input_img = tf.placeholder(tf.float32, [self.batch_size] + self.image_shape)
        self.target_img = tf.placeholder(tf.float32, [self.batch_size] + self.image_shape)
        self.seg_img = tf.placeholder(tf.float32, [self.batch_size] + self.image_shape[:-1] + [1])

        concat_img = tf.concat([self.input_img, self.seg_img], axis=3)

        input_img_resized = tf.image.resize_images(concat_img, [256, 256])
        target_img_resized = tf.image.resize_images(self.target_img, [256, 256])

        gen_img = self.generate(input_img_resized)

        d_real = self.discriminate(target_img_resized)
        d_fake = self.discriminate(gen_img)

        self.D_loss = tf.reduce_mean(-(tf.log(d_real + EPS) + tf.log(1 - d_fake + EPS)))

        # G_loss_GAN = tf.reduce_mean(-tf.log(d_fake + EPS))
        self.G_loss_GAN = tf.reduce_mean(-tf.log(d_fake + EPS))
        # G_loss_L1 = tf.reduce_mean(tf.abs(self.target_img - gen_img))
        # self.G_loss_L1 = tf.reduce_mean(tf.abs(self.target_img - gen_img))
        # self.G_loss = G_loss_GAN + G_loss_L1 * self.l1_weight
        self.G_loss = self.G_loss_GAN

        self.train_op_discrim = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5).minimize(self.D_loss, var_list=self.discrim_params)
        self.train_op_gen = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5).minimize(self.G_loss, var_list=self.gen_params)

    def generate(self, input_img):
        h1 = h1_ = tf.nn.conv2d(input_img, self.G_W1, strides=[1, 2, 2, 1], padding='SAME')  # [?,256,256,4] -> [?,128,128,64]
        h1 = lrelu(h1)

        h2 = tf.nn.conv2d(h1, self.G_W2, strides=[1, 2, 2, 1], padding='SAME')  # [?,128,128,64] -> [?,64,64,128]
        h2 = h2_ = self.G_in2(h2)
        h2 = lrelu(h2)

        h3 = tf.nn.conv2d(h2, self.G_W3, strides=[1, 2, 2, 1], padding='SAME')  # [?,64,64,128] -> [?,32,32,256]
        h3 = h3_ = self.G_in3(h3)
        h3 = lrelu(h3)

        h4 = tf.nn.conv2d(h3, self.G_W4, strides=[1, 2, 2, 1], padding='SAME')  # [?,32,32,256] -> [?,16,16,512]
        h4 = h4_ = self.G_in4(h4)
        h4 = lrelu(h4)

        h5 = tf.nn.conv2d(h4, self.G_W5, strides=[1, 2, 2, 1], padding='SAME')  # [?,16,16,512] -> [?,8,8,512]
        h5 = h5_ = self.G_in5(h5)
        h5 = lrelu(h5)

        h6 = tf.nn.conv2d(h5, self.G_W6, strides=[1, 2, 2, 1], padding='SAME')  # [?,8,8,512] -> [?,4,4,512]
        h6 = h6_ = self.G_in6(h6)
        h6 = lrelu(h6)

        h7 = tf.nn.conv2d(h6, self.G_W7, strides=[1, 2, 2, 1], padding='SAME')  # [?,4,4,512] -> [?,2,2,512]
        h7 = h7_ = self.G_in7(h7)
        h7 = lrelu(h7)

        h8 = tf.nn.conv2d(h7, self.G_W8, strides=[1, 2, 2, 1], padding='SAME')  # [?,2,2,512] -> [?,1,1,512]
        h8 = self.G_in8(h8)
        h8 = tf.nn.relu(h8)

        h9 = tf.nn.conv2d_transpose(h8, self.G_W9, output_shape=[self.batch_size, 2, 2, self.ch_G9], strides=[1, 2, 2, 1])  # [?,1,1,512] -> [?,2,2,512]
        h9 = tf.nn.dropout(self.G_in9(h9), keep_prob=self.keep_prob)
        h9 = tf.nn.relu(h9)
        h9 = tf.concat([h9, h7_], axis=3)

        h10 = tf.nn.conv2d_transpose(h9, self.G_W10, output_shape=[self.batch_size, 4, 4, self.ch_G10], strides=[1, 2, 2, 1])  # [?,2,2,512+512] -> [?,4,4,512]
        h10 = tf.nn.dropout(self.G_in10(h10), keep_prob=self.keep_prob)
        h10 = tf.nn.relu(h10)
        h10 = tf.concat([h10, h6_], axis=3)

        h11 = tf.nn.conv2d_transpose(h10, self.G_W11, output_shape=[self.batch_size, 8, 8, self.ch_G11], strides=[1, 2, 2, 1])  # [?,4,4,512+512] -> [?,8,8,512]
        h11 = tf.nn.dropout(self.G_in11(h11), keep_prob=self.keep_prob)
        h11 = tf.nn.relu(h11)
        h11 = tf.concat([h11, h5_], axis=3)

        h12 = tf.nn.conv2d_transpose(h11, self.G_W12, output_shape=[self.batch_size, 16, 16, self.ch_G12], strides=[1, 2, 2, 1])  # [?,8,8,512+512] -> [?,16,16,512]
        h12 = self.G_in12(h12)
        h12 = tf.nn.relu(h12)
        h12 = tf.concat([h12, h4_], axis=3)

        h13 = tf.nn.conv2d_transpose(h12, self.G_W13, output_shape=[self.batch_size, 32, 32, self.ch_G13], strides=[1, 2, 2, 1])  # [?,16,16,512+512] -> [?,32,32,256]
        h13 = self.G_in13(h13)
        h13 = tf.nn.relu(h13)
        h13 = tf.concat([h13, h3_], axis=3)

        h14 = tf.nn.conv2d_transpose(h13, self.G_W14, output_shape=[self.batch_size, 64, 64, self.ch_G14], strides=[1, 2, 2, 1])  # [?,32,32,256+256] -> [?,64,64,128]
        h14 = self.G_in14(h14)
        h14 = tf.nn.relu(h14)
        h14 = tf.concat([h14, h2_], axis=3)

        h15 = tf.nn.conv2d_transpose(h14, self.G_W15, output_shape=[self.batch_size, 128, 128, self.ch_G15], strides=[1, 2, 2, 1])  # [?,64,64,128+128] -> [?,128,128,64]
        h15 = self.G_in15(h15)
        h15 = tf.nn.relu(h15)
        h15 = tf.concat([h15, h1_], axis=3)

        h16 = tf.nn.conv2d_transpose(h15, self.G_W16, output_shape=[self.batch_size, 256, 256, self.ch_G16], strides=[1, 2, 2, 1])  # [?,128,128,64+64] -> [?,256,256,3]
        h16 = tf.nn.tanh(h16)

        return h16

    def discriminate(self, src):
        h1 = tf.nn.conv2d(src, self.D_W1, strides=[1, 2, 2, 1], padding='SAME')  # [?,256,256,6] -> [?,128,128,64]
        h1 = self.D_in1(h1)
        h1 = lrelu(h1)

        h2 = tf.nn.conv2d(h1, self.D_W2, strides=[1, 2, 2, 1], padding='SAME')  # [?,128,128,64] -> [?,64,64,128]
        h2 = self.D_in2(h2)
        h2 = lrelu(h2)

        h3 = tf.nn.conv2d(h2, self.D_W3, strides=[1, 2, 2, 1], padding='SAME')  # [?,64,64,128] -> [?,32,32,256]
        h3 = self.D_in3(h3)
        h3 = lrelu(h3)

        h4 = tf.pad(h3, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')  # [?,32,32,256] -> [?,31,31,512]
        h4 = tf.nn.conv2d(h4, self.D_W4, strides=[1, 1, 1, 1], padding='VALID')
        h4 = self.D_in4(h4)
        h4 = lrelu(h4)

        h5 = tf.pad(h4, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')  # [?,31,31,256] -> [?,30,30,1]
        h5 = tf.nn.conv2d(h5, self.D_W5, strides=[1, 1, 1, 1], padding='VALID')
        h5 = tf.nn.sigmoid(h5)

        return h5

    # Method for generating the fake images
    def sample_generator(self, input_image, input_seg, batch_size=1):
        input_img = tf.placeholder(tf.float32, [batch_size] + self.image_shape)
        input_segimg = tf.placeholder(tf.float32, [batch_size] + self.image_shape[:-1] + [1])
        concat_img = tf.concat([input_img, input_segimg], axis=3)

        input_img_resized = tf.image.resize_images(concat_img, [256, 256])

        h1 = h1_ = tf.nn.conv2d(input_img_resized, self.G_W1, strides=[1, 2, 2, 1], padding='SAME')  # [?,256,256,3] -> [?,128,128,64]
        h1 = lrelu(h1)

        h2 = tf.nn.conv2d(h1, self.G_W2, strides=[1, 2, 2, 1], padding='SAME')  # [?,128,128,64] -> [?,64,64,128]
        h2 = h2_ = self.G_in2(h2)
        h2 = lrelu(h2)

        h3 = tf.nn.conv2d(h2, self.G_W3, strides=[1, 2, 2, 1], padding='SAME')  # [?,64,64,128] -> [?,32,32,256]
        h3 = h3_ = self.G_in3(h3)
        h3 = lrelu(h3)

        h4 = tf.nn.conv2d(h3, self.G_W4, strides=[1, 2, 2, 1], padding='SAME')  # [?,32,32,256] -> [?,16,16,512]
        h4 = h4_ = self.G_in4(h4)
        h4 = lrelu(h4)

        h5 = tf.nn.conv2d(h4, self.G_W5, strides=[1, 2, 2, 1], padding='SAME')  # [?,16,16,512] -> [?,8,8,512]
        h5 = h5_ = self.G_in5(h5)
        h5 = lrelu(h5)

        h6 = tf.nn.conv2d(h5, self.G_W6, strides=[1, 2, 2, 1], padding='SAME')  # [?,8,8,512] -> [?,4,4,512]
        h6 = h6_ = self.G_in6(h6)
        h6 = lrelu(h6)

        h7 = tf.nn.conv2d(h6, self.G_W7, strides=[1, 2, 2, 1], padding='SAME')  # [?,4,4,512] -> [?,2,2,512]
        h7 = h7_ = self.G_in7(h7)
        h7 = lrelu(h7)

        h8 = tf.nn.conv2d(h7, self.G_W8, strides=[1, 2, 2, 1], padding='SAME')  # [?,2,2,512] -> [?,1,1,512]
        h8 = self.G_in8(h8)
        h8 = tf.nn.relu(h8)

        h9 = tf.nn.conv2d_transpose(h8, self.G_W9, output_shape=[batch_size, 2, 2, self.ch_G9], strides=[1, 2, 2, 1])  # [?,1,1,512] -> [?,2,2,512]
        h9 = tf.nn.dropout(self.G_in9(h9), keep_prob=self.keep_prob)
        h9 = tf.nn.relu(h9)
        h9 = tf.concat([h9, h7_], axis=3)

        h10 = tf.nn.conv2d_transpose(h9, self.G_W10, output_shape=[batch_size, 4, 4, self.ch_G10], strides=[1, 2, 2, 1])  # [?,2,2,512+512] -> [?,4,4,512]
        h10 = tf.nn.dropout(self.G_in10(h10), keep_prob=self.keep_prob)
        h10 = tf.nn.relu(h10)
        h10 = tf.concat([h10, h6_], axis=3)

        h11 = tf.nn.conv2d_transpose(h10, self.G_W11, output_shape=[batch_size, 8, 8, self.ch_G11], strides=[1, 2, 2, 1])  # [?,4,4,512+512] -> [?,8,8,512]
        h11 = tf.nn.dropout(self.G_in11(h11), keep_prob=self.keep_prob)
        h11 = tf.nn.relu(h11)
        h11 = tf.concat([h11, h5_], axis=3)

        h12 = tf.nn.conv2d_transpose(h11, self.G_W12, output_shape=[batch_size, 16, 16, self.ch_G12], strides=[1, 2, 2, 1])  # [?,8,8,512+512] -> [?,16,16,512]
        h12 = self.G_in12(h12)
        h12 = tf.nn.relu(h12)
        h12 = tf.concat([h12, h4_], axis=3)

        h13 = tf.nn.conv2d_transpose(h12, self.G_W13, output_shape=[batch_size, 32, 32, self.ch_G13], strides=[1, 2, 2, 1])  # [?,16,16,512+512] -> [?,32,32,256]
        h13 = self.G_in13(h13)
        h13 = tf.nn.relu(h13)
        h13 = tf.concat([h13, h3_], axis=3)

        h14 = tf.nn.conv2d_transpose(h13, self.G_W14, output_shape=[batch_size, 64, 64, self.ch_G14], strides=[1, 2, 2, 1])  # [?,32,32,256+256] -> [?,64,64,128]
        h14 = self.G_in14(h14)
        h14 = tf.nn.relu(h14)
        h14 = tf.concat([h14, h2_], axis=3)

        h15 = tf.nn.conv2d_transpose(h14, self.G_W15, output_shape=[batch_size, 128, 128, self.ch_G15], strides=[1, 2, 2, 1])  # [?,64,64,128+128] -> [?,128,128,64]
        h15 = self.G_in15(h15)
        h15 = tf.nn.relu(h15)
        h15 = tf.concat([h15, h1_], axis=3)

        h16 = tf.nn.conv2d_transpose(h15, self.G_W16, output_shape=[batch_size, 256, 256, self.ch_G16], strides=[1, 2, 2, 1])  # [?,128,128,64+64] -> [?,256,256,3]
        h16 = tf.nn.tanh(h16)

        generated_samples = self.sess.run(h16, feed_dict={input_img: input_image, input_segimg: input_seg})
        return generated_samples

    # Train Generator and return the loss
    def train_gen(self, input_img, target_img, seg_img):
        # _, loss_val_G = self.sess.run([self.train_op_gen, self.G_loss], feed_dict={self.input_img: input_img, self.target_img: target_img})
        _, loss_val_GAN = self.sess.run([self.train_op_gen, self.G_loss_GAN], feed_dict={self.input_img: input_img, self.target_img: target_img, self.seg_img: seg_img})
        # return loss_val_G
        return loss_val_GAN

    # Train Discriminator and return the loss
    def train_discrim(self, input_img, target_img, seg_img):
        _, loss_val_D = self.sess.run([self.train_op_discrim, self.D_loss], feed_dict={self.input_img: input_img, self.target_img: target_img, self.seg_img: seg_img})
        return loss_val_D
