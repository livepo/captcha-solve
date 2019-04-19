# coding: utf-8

import os
import tensorflow as tf
import numpy as np
from PIL import ImageFont, ImageDraw, Image, ImageFilter
from random import randrange, sample, choice


IMAGE_WIDTH = 100
IMAGE_HEIGHT = 34
IMAGE_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH
CAPTCHA_LEN = 4
CODES = "123456789abcdefghjkmnpqrstuvwxyzABCDEFGHJKMNPQRSTUVWXYZ"
CHAR_SET_LEN = len(CODES)
NUM_LABELS = CAPTCHA_LEN * CHAR_SET_LEN


trainer_dir = os.path.dirname(os.path.abspath(__file__))
home_dir = os.path.dirname(trainer_dir)
graph_log_dir = os.path.join(trainer_dir, 'logs')


def generate_captcha(image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT, font_size=16):
    dark_colors = ["black", "darkred", "darkgreen", "brown",
                "darkblue", "purple", "teal"]
    font_color = dark_colors
    codes = CODES
    background = (randrange(150, 255), randrange(150, 255), randrange(150, 255))
    line_color = (randrange(0, 255), randrange(0, 255), randrange(0, 255))
    sample_file = os.path.join(os.path.dirname(__file__), 'LucidaSansDemiOblique.ttf')
    font = ImageFont.truetype(sample_file, font_size)
    image = Image.new('RGB', (image_width, image_height), background)
    code = ''.join(sample(codes, 4))
    draw = ImageDraw.Draw(image)
    for i in range(randrange(5, 10)):
        xy = (randrange(0, image_width), randrange(0, image_height),
              randrange(0, image_width), randrange(0, image_height))
        draw.line(xy, fill=line_color, width=1)
    x = 2
    for i in code:
        y = randrange(0, 10)
        draw.text((x, y), i, font=font, fill=choice(font_color))
        x += font_size - 2
    image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
    # image.show()
    return image, code


def weight_variable(shape):
    initial = tf.random_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def init_model(alpha=1e-3):
    with tf.Graph().as_default() as graph:
        x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE])
        y = tf.placeholder(tf.float32, shape=[None, NUM_LABELS])
        keep_prob = tf.placeholder(tf.float32)

        x_image = tf.reshape(x, shape=[-1, IMAGE_WIDTH, IMAGE_HEIGHT, 1])

        conv_layer1_weight = weight_variable([5, 5, 1, 32])
        conv_layer1_bias = bias_variable([32])
        pool_layer1 = max_pool(
            tf.nn.relu(
                conv2d(x_image, conv_layer1_weight) + conv_layer1_bias
            )
        )

        conv_layer2_weight = weight_variable([5, 5, 32, 64])
        conv_layer2_bias = bias_variable([64])
        pool_layer2 = max_pool(
            tf.nn.relu(
                conv2d(pool_layer1, conv_layer2_weight) + conv_layer2_bias
            )
        )

        conv_layer3_weight = weight_variable([5, 5, 64, 64])
        conv_layer3_bias = bias_variable([64])
        pool_layer3 = max_pool(
            tf.nn.relu(
                conv2d(pool_layer2, conv_layer3_weight) + conv_layer3_bias
            )
        )

        # 100 * 40, 58 * 30
        # [130,220] vs. [64,220]
        fc_layer_weight = weight_variable([13 * 5 * 64, 1024])
        fc_layer_bias = bias_variable([1024])

        pool_layer3_flat = tf.reshape(pool_layer3, [-1, 13 * 5 * 64])
        fc_layer = tf.nn.relu(tf.add(tf.matmul(pool_layer3_flat, fc_layer_weight), fc_layer_bias))

        fc_layer_drop = tf.nn.dropout(fc_layer, keep_prob)

        output_layer_weight = weight_variable([1024, NUM_LABELS])
        output_layer_bias = bias_variable([NUM_LABELS])

        y_conv = tf.add(tf.matmul(fc_layer_drop, output_layer_weight), output_layer_bias)

        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_conv)
        )

        optimizer = tf.train.AdamOptimizer(alpha).minimize(loss)

        prediction = tf.argmax(tf.reshape(y_conv, [-1, CAPTCHA_LEN, CHAR_SET_LEN]), 2)
        correct = tf.argmax(tf.reshape(y, [-1, CAPTCHA_LEN, CHAR_SET_LEN]), 2)
        correct_prediction = tf.equal(prediction, correct)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        saver = tf.train.Saver(max_to_keep=2)

        model = {'x': x,
                 'y': y,
                 'optimizer': optimizer,
                 'loss': loss,
                 'keep_prob': keep_prob,
                 'accuracy': accuracy,
                 'prediction': prediction,
                 'saver': saver,
                 'graph': graph
                 }

        return model


def str2vec(_str):
    """ vectorize the captcha str """
    vec = np.zeros(4 * CHAR_SET_LEN)
    for i, ch in enumerate(_str):
        offset = CODES.find(ch)
        vec[(i*CHAR_SET_LEN) + offset] = 1
    return vec


def gen_dataset(num):
    dataset = []
    labels = []

    for _ in range(num):
        captcha, captcha_str = generate_captcha()
        dataset.append(np.asarray(captcha.convert("L")).reshape([IMAGE_HEIGHT * IMAGE_WIDTH]) / 255)
        labels.append(str2vec(captcha_str))

    return np.array(dataset), np.array(labels)


def train():
    model = init_model()
    x = model['x']
    y = model['y']
    loss = model['loss']
    optimizer = model['optimizer']
    accuracy = model['accuracy']
    keep_prob = model['keep_prob']
    saver = model['saver']
    graph = model['graph']

    save_dir = 'checkpoint'
    print("Model saved path: ", save_dir)

    def save_model(_step):
        saver.save(
            session,
            os.path.join(save_dir, 'gatling.ckpt'),
            global_step=_step
        )

    with tf.Session(graph=graph) as session:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(graph_log_dir, session.graph)
        tf.global_variables_initializer().run()
        step = 0

        while True:
            batch_data, batch_labels = gen_dataset(64)

            _, l = session.run(
                [optimizer, loss],
                feed_dict={
                    x: batch_data,
                    y: batch_labels,
                    keep_prob: 0.75
                }
            )
            step += 1
            print(("Step: %d, Loss: %4f" % (step, l)))

            if step % 50 == 0:
                test_dataset, test_labels = gen_dataset(100)
                test_accuracy = session.run(
                    accuracy,
                    feed_dict={
                        x: test_dataset,
                        y: test_labels,
                        keep_prob: 1.0
                    }
                )

                print(("Step: %d, Test Accuracy: %s" % (step, test_accuracy)))

                save_model(step)

                if test_accuracy >= 0.92 or step >= 10000:  # stop when accuracy above 92%
                    save_model(step)
                    break

        print("Test accuracy: %g" %
              session.run(
                  accuracy,
                  feed_dict={
                      x: test_dataset,
                      y: test_labels,
                      keep_prob: 1.0
                  })
              )


def find_model_ckpt(model_ckpt_dir='checkpoint'):
    """ Find Max Step model.ckpt """
    if not os.path.isdir(model_ckpt_dir):
        os.mkdir(model_ckpt_dir)

    from distutils.version import LooseVersion
    model_ckpt_tuple_list = []
    for fn in os.listdir(model_ckpt_dir):
        bare_fn, ext = os.path.splitext(fn)
        if bare_fn.startswith('gatling.ckpt') and ext == '.index':
            version = bare_fn.split('gatling.ckpt-')[1]
            model_ckpt_tuple_list.append((version, bare_fn))

    if len(model_ckpt_tuple_list) == 0:
        raise IOError('file like gatling.ckpt')
    model_ckpt_list = list(sorted(model_ckpt_tuple_list,
                                  key=lambda item: LooseVersion(item[0])))
    fn = model_ckpt_list[-1][1]
    global_step = int(model_ckpt_list[-1][0])
    path = os.path.join(model_ckpt_dir, fn)

    return path, global_step


def show_im(dataset):
    data = np.uint8(dataset[0]).reshape((IMAGE_HEIGHT, IMAGE_WIDTH)) * 255
    im = Image.fromarray(data)
    im.show()


def test_model():
    model = init_model()

    x = model['x']
    keep_prob = model['keep_prob']
    saver = model['saver']
    prediction = model['prediction']
    graph = model['graph']
    model_ckpt_path, _ = find_model_ckpt()

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        saver.restore(session, model_ckpt_path)

        dataset, labels = gen_dataset(1)

        show_im(dataset)
        for i in range(4):
            print(labels[0][i*CHAR_SET_LEN:(i+1)*CHAR_SET_LEN])

        label = prediction.eval(feed_dict={x: dataset, keep_prob: 1.0}, session=session)[0]
        print("predict label:", [CODES[i] for i in label])


if __name__ == "__main__":
    train()
    # test_model()
    # print(gen_dataset(1))
    # im, code = generate_captcha(image_width=100, image_height=40)
    # im.save('/tmp/captcha/%s.png' % code)
    # im.show()
    # print(code)
