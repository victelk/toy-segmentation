import numpy as np
import cv2
import os.path
import glob
import tensorflow as tf

LEARNING_RATE = 0.001


def my_input_fn(imagenames):
    def input_parser(img_path):
        # read the img from file
        img_file = tf.read_file(img_path[0])
        img_decoded = tf.image.decode_image(img_file, channels=1)
        img_decoded.set_shape([6, 6, 1])
        img_flatten = tf.to_float(tf.reshape(img_decoded, [-1]))
        label_img_file = tf.read_file(img_path[1])
        label_img_decoded = tf.image.decode_image(label_img_file, channels=1)
        label_img_decoded.set_shape([6, 6, 1])
        label_img_flatten = tf.to_float(tf.reshape(label_img_decoded, [-1]))
        return {'pixels': img_flatten}, label_img_flatten  # one_hot

    dataset = (tf.data.Dataset.from_tensor_slices(imagenames)
               .map(input_parser))  # Transform each elem by applying input_parser fn
    dataset = dataset.batch(2)  # Batch size to use
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels


def toy_segmentation_model_fn(features, labels, mode, params):
    """Model function for Estimator."""

    # Connect the first hidden layer to input layer
    # (features["pixels"]) with relu activation
    first_hidden_layer = tf.layers.dense(features["pixels"], 36, activation=tf.nn.relu)

    # Connect the second hidden layer to first hidden layer with relu
    second_hidden_layer = tf.layers.dense(first_hidden_layer, 36, activation=tf.nn.relu)

    # Connect the output layer to second hidden layer (no activation fn)
    output_layer = tf.layers.dense(second_hidden_layer, 36)

    # Reshape output layer to 1-dim Tensor to return predictions
    predicted = tf.reshape(output_layer, [-1,36])
    print("predicted shape = %s" % predicted.shape)
    print("labels shape = %s" % labels.shape)

    # Provide an estimator spec for `ModeKeys.PREDICT`.
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"pixels": predicted})

    # Calculate loss using mean squared error
    loss = tf.losses.mean_squared_error(labels, predicted)

    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=params["learning_rate"])
    train_op = optimizer.minimize(
        loss=loss, global_step=tf.train.get_global_step())

    # Calculate root mean squared error as additional eval metric
    eval_metric_ops = {
        "rmse": tf.metrics.root_mean_squared_error(
            tf.cast(labels, tf.float32), predicted)
    }

    # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)

def main(unused_argv):

    file_path = "./data"
    filenames = sorted([f for f in glob.glob(file_path + os.sep + "image*.jpg")])
    label_filenames = sorted([f for f in glob.glob(file_path + os.sep + "label_image*.jpg")])
    imagenames = tf.convert_to_tensor([filenames,label_filenames])

    # batch_features, batch_labels = my_input_fn(imagenames)
    # with tf.Session() as sess:
        # print(sess.run(batch_labels).shape)
        # print(sess.run(batch_features["pixels"]).shape)
        # print(sess.run(batch_labels))
        # print((batch_features)["pixels"].shape)
        # print((batch_labels).shape)
        # print(batch_labels)

    # Set model params
    model_params = {"learning_rate": LEARNING_RATE}

    # Instantiate Estimator
    nn = tf.estimator.Estimator(model_fn=toy_segmentation_model_fn, params=model_params)

    # Train
    nn.train(input_fn=lambda: my_input_fn(imagenames))


if __name__ == "__main__":
    tf.app.run(main=main)
