#以下代码修改自naturomics的GitHub实现，包含三层CapsNet和后面的重构网络
#改网络参数比较多，我们后面会只训练测试三层CapsNet。

import tensorflow as tf
import numpy as np
import os
import mnist_data
from tqdm import tqdm

epsilon = 1e-9
batch_size = 8
epoch = 1

#margin loss 中调节上margin和下margind的权重
lambda_val = 0.5
#上margin与下margin的参数值
m_plus = 0.9
m_minus = 0.1

# 路由更新c_ij所经过的迭代次数
iter_routing = 3

# Tensorboard 保存位置
logdir ='logdir'
# 数据集路径
dataset_path = 'data/mnist'
is_training= True

# 定义加载mnist的函数
def load_mnist(path, is_training):
    train_imgs, train_lbls, test_imgs, test_lbls = mnist_data.download_data(dataset_path)
    #trX将加载储存所有60000张灰度图
    fd = open(train_imgs)
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

    fd = open(train_lbls)
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.float)

    #teX将储存所有一万张测试用的图片
    fd = open(test_imgs)
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

    fd = open(test_lbls)
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.float)

    # 将所有训练图片表示为一个4维张量 [60000, 28, 28, 1]，其中每个像素值缩放到0和1之间
    trX = tf.convert_to_tensor(trX / 255., tf.float32)

    # one hot编码为 [num_samples, 10]
    trY = tf.one_hot(trY, depth=10, axis=1, dtype=tf.float32)
    teY = tf.one_hot(teY, depth=10, axis=1, dtype=tf.float32)

    # 训练和测试时返回不同的数据
    if is_training:
        return trX, trY
    else:
        return teX / 255., teY

def get_batch_data():
    trX, trY = load_mnist(dataset_path, True)

    # 每次产生一个切片
    data_queues = tf.train.slice_input_producer([trX, trY])

    # 对队列中的样本进行乱序处理
    X, Y = tf.train.shuffle_batch(data_queues,
                                  batch_size=batch_size,
                                  capacity=batch_size * 64,
                                  min_after_dequeue=batch_size * 32,
                                  allow_smaller_final_batch=False)
    return (X, Y)

# 通过定义类和对象的方式定义Capssule层级
class CapsLayer(object):
    ''' Capsule layer 类别参数有：
    Args:
        input: 一个4维张量
        num_outputs: 当前层的Capsule单元数量
        vec_len: 一个Capsule输出向量的长度
        layer_type: 选择'FC' 或 "CONV", 以确定是用全连接层还是卷积层
        with_routing: 当前Capsule是否从较低层级中Routing而得出输出向量

    Returns:
        一个四维张量
    '''
    def __init__(self, num_outputs, vec_len, with_routing=True, layer_type='FC'):
        self.num_outputs = num_outputs
        self.vec_len = vec_len
        self.with_routing = with_routing
        self.layer_type = layer_type

    def __call__(self, input, kernel_size=None, stride=None):
        '''
        当“Layer_type”选择的是“CONV”，我们将使用 'kernel_size' 和 'stride'
        '''

        # 开始构建卷积层
        if self.layer_type == 'CONV':
            self.kernel_size = kernel_size
            self.stride = stride

            # PrimaryCaps层没有Routing过程
            if not self.with_routing:
                # 卷积层为 PrimaryCaps 层（CapsNet第二层）, 并将第一层卷积的输出张量作为输入。
                # 输入张量的维度为： [batch_size, 20, 20, 256]
                assert input.get_shape() == [batch_size, 20, 20, 256]

                # # 从CapsNet输出向量的每一个分量开始执行卷积，每个分量上执行带32个卷积核的9×9标准卷积
                # capsules = []
                # for i in range(self.vec_len):
                #     # 所有Capsule的一个分量，其维度为: [batch_size, 6, 6, 32]，即6×6×1×32
                #     with tf.variable_scope('ConvUnit_' + str(i)):
                #         caps_i = tf.contrib.layers.conv2d(input, self.num_outputs,
                #                                           self.kernel_size, self.stride,
                #                                           padding="VALID")
                #
                #         # 将一般卷积的结果张量拉平，并为添加到列表中
                #         caps_i = tf.reshape(caps_i, shape=(batch_size, -1, 1, 1))
                #         capsules.append(caps_i)
                #
                # # 为将卷积后张量各个分量合并为向量做准备
                # assert capsules[0].get_shape() == [batch_size, 1152, 1, 1]
                #
                # # 合并为PrimaryCaps的输出张量，即6×6×32个长度为8的向量，合并后的维度为 [batch_size, 1152, 8, 1]
                # capsules = tf.concat(capsules, axis=2)
                # # 将每个Capsule 向量投入非线性函数squash进行缩放与激活,第二层输出的向量要经过缩放
                # capsules = squash(capsules)
                # assert capsules.get_shape() == [batch_size, 1152, 8, 1]
                # return(capsules)

                # 以下更新后的计算方法
                capsules = tf.contrib.layers.conv2d(input, self.num_outputs * self.vec_len,
                                                    self.kernel_size, self.stride, padding="VALID")
                capsules = tf.reshape(capsules, (batch_size, -1, self.vec_len, 1))

                # [batch_size, 1152, 8, 1]
                capsules = squash(capsules)
                assert capsules.get_shape() == [batch_size, 1152, 8, 1]
                return (capsules)

        if self.layer_type == 'FC':

            # DigitCaps 带有Routing过程
            if self.with_routing:
                # CapsNet 的第三层 DigitCaps 层是一个全连接网络
                # 将输入张量重建为 [batch_size, 1152, 1, 8, 1]
                self.input = tf.reshape(input, shape=(batch_size, -1, 1, input.shape[-2].value, 1))

                with tf.variable_scope('routing'):
                    # 初始化b_IJ的值为零，且维度满足: [1, 1, num_caps_l, num_caps_l_plus_1, 1]
                    b_IJ = tf.constant(np.zeros([1, input.shape[1].value, self.num_outputs, 1, 1], dtype=np.float32))
                    # 使用定义的Routing过程计算权值更新与s_j
                    capsules = routing(self.input, b_IJ)
                    # 将s_j投入 squeeze 函数以得出 DigitCaps 层的输出向量
                    capsules = tf.squeeze(capsules, axis=1)

            return(capsules)

# 定义路由算法的过程
def routing(input, b_IJ):
    ''' 路由算法

    Args:
        input: 输入张量的维度为 [batch_size, num_caps_l=1152, 1, length(u_i)=8, 1]
               其中num_caps_l为上一层（PrimaryCaps）的Capsule单元数量
    Returns:
        返回的张量维度为 [batch_size, num_caps_l_plus_1, length(v_j)=16, 1]
        表征了i+1层的输出向量 `v_j`，num_caps_l_plus_1 为DigitCaps层的输出数
    Notes:
        u_i 表示l层中 capsule i 的输出向量
        v_j 表示l+1层中 capsule j 的输出向量
     '''

    # 定义W的张量维度为 [num_caps_j, num_caps_i, len_u_i, len_v_j]
    # W_ij共有1152×10个，每一个的维度为8×16
    W = tf.get_variable('Weight', shape=(1, 1152, 10, 8, 16), dtype=tf.float32,
                        initializer=tf.random_normal_initializer(stddev=0.01))

    # 论文中的 Eq.2, 计算 u_hat
    # 在使用 W 和u_i计算u_hat前，先调整张量维度
    # input => [batch_size, 1152, 10, 8, 1]
    # W => [batch_size, 1152, 10, 8, 16]
    input = tf.tile(input, [1, 1, 10, 1, 1])
    W = tf.tile(W, [batch_size, 1, 1, 1, 1])
    assert input.get_shape() == [batch_size, 1152, 10, 8, 1]

    # 因为[8, 16].T x [8, 1] => [16, 1]，所以矩阵乘法在最后得出的维度为 [batch_size, 1152, 10, 16, 1]
    u_hat = tf.matmul(W, input, transpose_a=True)
    assert u_hat.get_shape() == [batch_size, 1152, 10, 16, 1]

    # 前面是扩展的线性组合，后面是路由的部分，以下开始迭代路由过程更新耦合系数
    # 对应论文中伪代码的第三行
    for r_iter in range(iter_routing):
        with tf.variable_scope('iter_' + str(r_iter)):
            # 原论文伪代码第四行，计算softmax(b_ij)
            # => [1, 1152, 10, 1,1]
            c_IJ = tf.nn.softmax(b_IJ, dim=3)
            c_IJ = tf.tile(c_IJ, [batch_size, 1, 1, 1, 1])
            assert c_IJ.get_shape() == [batch_size, 1152, 10, 1, 1]

            # 原论文伪代码第五行，根据更新的c_ij计算s_j
            # 先利用 c_IJ 给 u_hat 加权，即在后两个维度采用对应元素的乘积
            # => [batch_size, 1152, 10, 16, 1]
            s_J = tf.multiply(c_IJ, u_hat)
            # 在第二个维度上求和, 产生的张量维度为 [batch_size, 1, 10, 16, 1]
            s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
            assert s_J.get_shape() == [batch_size, 1, 10, 16, 1]

            # 原论文伪代码的第六行
            # 使用 Eq.1 计算squashing非线性函数
            v_J = squash(s_J)
            assert v_J.get_shape() == [batch_size, 1, 10, 16, 1]

            # 原论文伪代码的第七行
            # reshape & tile v_j from [batch_size ,1, 10, 16, 1] to [batch_size, 10, 1152, 16, 1]
            # then matmul in the last tow dim: [16, 1].T x [16, 1] => [1, 1], reduce mean in the
            # batch_size dim, resulting in [1, 1152, 10, 1, 1]
            v_J_tiled = tf.tile(v_J, [1, 1152, 1, 1, 1])
            u_produce_v = tf.matmul(u_hat, v_J_tiled, transpose_a=True)
            assert u_produce_v.get_shape() == [batch_size, 1152, 10, 1, 1]
            b_IJ += tf.reduce_sum(u_produce_v, axis=0, keep_dims=True)

    return(v_J)

def squash(vector):
    ''' 根据原论文中 Eq. 1 定义squashing函数
    Args:
        vector: 一个 5-D 张量，其维度是 [batch_size, 1, num_caps, vec_len, 1],
    Returns:
        返回一个 5-D 张量，其第四和第五个维度经过了该非线性函数据算
    '''
    vec_squared_norm = tf.reduce_sum(tf.square(vector), -2, keep_dims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
    vec_squashed = scalar_factor * vector  # element-wise
    return(vec_squashed)

# 以下定义整个 CapsNet 的架构与正向传播过程
class CapsNet():
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            if is_training:
                # 获取一个批量的训练数据
                self.X, self.Y = get_batch_data()

                self.build_arch()
                self.loss()

                # t_vars = tf.trainable_variables()
                self.optimizer = tf.train.AdamOptimizer()
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.train_op = self.optimizer.minimize(self.total_loss, global_step=self.global_step)  # var_list=t_vars)
            else:
                self.X = tf.placeholder(tf.float32,
                                        shape=(batch_size, 28, 28, 1))
                self.build_arch()

        tf.logging.info('Seting up the main structure')

    # CapsNet 类中的build_arch方法能构建整个网络的架构
    def build_arch(self):
        # 以下构建第一个常规卷积层
        with tf.variable_scope('Conv1_layer'):
            # 第一个卷积层的输出张量为： [batch_size, 20, 20, 256]
            # 以下卷积输入图像X,采用256个9×9的卷积核，步幅为1，且不使用
            conv1 = tf.contrib.layers.conv2d(self.X, num_outputs=256,
                                             kernel_size=9, stride=1,
                                             padding='VALID')
            # 是用 assert 可以在出现错误条件时就返回错误，有助于调整
            assert conv1.get_shape() == [batch_size, 20, 20, 256]

        # 以下是原论文中PrimaryCaps层的构建过程，该层的输出维度为 [batch_size, 1152, 8, 1]
        with tf.variable_scope('PrimaryCaps_layer'):
            # 调用前面定义的CapLayer函数构建第二个卷积层，该过程相当于执行八次常规卷积，
            # 然后将各对应位置的元素组合成一个长度为8的向量，这八次常规卷积都是采用32个9×9的卷积核、步幅为2
            primaryCaps = CapsLayer(num_outputs=32, vec_len=8, with_routing=False, layer_type='CONV')
            caps1 = primaryCaps(conv1, kernel_size=9, stride=2)
            assert caps1.get_shape() == [batch_size, 1152, 8, 1]

        # 以下构建 DigitCaps 层, 该层返回的张量维度为 [batch_size, 10, 16, 1]
        with tf.variable_scope('DigitCaps_layer'):
            # DigitCaps是最后一层，它返回对应10个类别的向量（每个有16个元素），该层的构建带有Routing过程
            digitCaps = CapsLayer(num_outputs=10, vec_len=16, with_routing=True, layer_type='FC')
            self.caps2 = digitCaps(caps1)

        # 以下构建论文图2中的解码结构，即由16维向量重构出对应类别的整个图像
        # 1. Do masking, how:
        with tf.variable_scope('Masking'):
            # Method 1. masking with true label, default mode

            # mask_with_y是否用真实标签蒙住目标Capsule
            mask_with_y=True
            if mask_with_y:
                self.masked_v = tf.matmul(tf.squeeze(self.caps2), tf.reshape(self.Y, (-1, 10, 1)), transpose_a=True)
                self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.caps2), axis=2, keep_dims=True) + epsilon)

        # 通过3个全连接层重构MNIST图像，这三个全连接层的神经元数分别为512、1024、784
        # [batch_size, 1, 16, 1] => [batch_size, 16] => [batch_size, 512]
        with tf.variable_scope('Decoder'):
            vector_j = tf.reshape(self.masked_v, shape=(batch_size, -1))
            fc1 = tf.contrib.layers.fully_connected(vector_j, num_outputs=512)
            assert fc1.get_shape() == [batch_size, 512]
            fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=1024)
            assert fc2.get_shape() == [batch_size, 1024]
            self.decoded = tf.contrib.layers.fully_connected(fc2, num_outputs=784, activation_fn=tf.sigmoid)

    # 定义 CapsNet 的损失函数，损失函数一共分为衡量 CapsNet准确度的Margin loss
    # 和衡量重构图像准确度的 Reconstruction loss
    def loss(self):
        # 以下先定义重构损失，因为DigitCaps的输出向量长度就为某类别的概率，因此可以借助计算向量长度计算损失
        # [batch_size, 10, 1, 1]
        # max_l = max(0, m_plus-||v_c||)^2
        max_l = tf.square(tf.maximum(0., m_plus - self.v_length))
        # max_r = max(0, ||v_c||-m_minus)^2
        max_r = tf.square(tf.maximum(0., self.v_length - m_minus))
        assert max_l.get_shape() == [batch_size, 10, 1, 1]

        # 将当前的维度[batch_size, 10, 1, 1] 转换为10个数字类别的one-hot编码 [batch_size, 10]
        max_l = tf.reshape(max_l, shape=(batch_size, -1))
        max_r = tf.reshape(max_r, shape=(batch_size, -1))

        # 计算 T_c: [batch_size, 10]，其为分类的指示函数
        # 若令T_c = Y,那么对应元素相乘就是有类别相同才会有非零输出值，T_c 和 Y 都为One-hot编码
        T_c = self.Y
        # [batch_size, 10], 对应元素相乘并构建最后的Margin loss 函数
        L_c = T_c * max_l + lambda_val * (1 - T_c) * max_r

        self.margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))

        # 以下构建reconstruction loss函数
        # 这一过程的损失函数通过计算FC Sigmoid层的输出像素点与原始图像像素点间的欧几里德距离而构建
        orgin = tf.reshape(self.X, shape=(batch_size, -1))
        squared = tf.square(self.decoded - orgin)
        self.reconstruction_err = tf.reduce_mean(squared)

        # 构建总损失函数，Hinton论文将reconstruction loss乘上0.0005
        # 以使它不会主导训练过程中的Margin loss
        self.total_loss = self.margin_loss + 0.0005 * self.reconstruction_err

        # 以下输出TensorBoard
        tf.summary.scalar('margin_loss', self.margin_loss)
        tf.summary.scalar('reconstruction_loss', self.reconstruction_err)
        tf.summary.scalar('total_loss', self.total_loss)
        recon_img = tf.reshape(self.decoded, shape=(batch_size, 28, 28, 1))
        tf.summary.image('reconstruction_img', recon_img)
        self.merged_sum = tf.summary.merge_all()

if __name__ == "__main__":
    # 训练和推断
    capsNet = CapsNet(is_training=is_training)
    tf.logging.info('Graph loaded')
    sv = tf.train.Supervisor(graph=capsNet.graph,
                             logdir=logdir,
                             save_model_secs=0)

    with sv.managed_session() as sess:
        num_batch = int(60000 / batch_size)
        for epoch in range(epoch):
            if sv.should_stop():
                break
            for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
                sess.run(capsNet.train_op)

            global_step = sess.run(capsNet.global_step)
            sv.saver.save(sess, logdir + '/model_epoch_%04d_step_%02d' % (epoch, global_step))

    tf.logging.info('Training done')
