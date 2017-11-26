import numpy as np
import cv2
import os.path
import glob
import tensorflow as tf

LEARNING_RATE = 0.00001
tf.logging.set_verbosity(tf.logging.WARN)


def my_input_fn(imagenames, label_imagenames, repeat_count=1):
    def input_parser(img_path, label_img_path):
        # read the img from file
        img_file = tf.read_file(img_path)
        img_decoded = tf.image.decode_image(img_file, channels=1)
        img_decoded.set_shape([6, 6, 1])
        img_flatten = tf.to_float(tf.reshape(img_decoded, [-1]))
        label_img_file = tf.read_file(label_img_path)
        label_img_decoded = tf.image.decode_image(label_img_file, channels=1)
        label_img_decoded.set_shape([6, 6, 1])
        label_img_flatten = tf.to_float(tf.reshape(label_img_decoded, [-1]))
        return {'pixels': img_flatten}, label_img_flatten  # one_hot

    dataset = (tf.data.Dataset.from_tensor_slices((imagenames, label_imagenames))
               .map(input_parser))  # Transform each elem by applying input_parser fn
    dataset = dataset.repeat(repeat_count) #Epoch number
    dataset = dataset.batch(10)  # Batch size to use
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
    train_filenames = tf.convert_to_tensor(sorted([f for f in glob.glob(file_path + os.sep + "train_image*.png")]))
    train_label_filenames = tf.convert_to_tensor(sorted([f for f in glob.glob(file_path + os.sep + "train_label_image*.png")]))

    test_filenames = tf.convert_to_tensor(sorted([f for f in glob.glob(file_path + os.sep + "test_image*.png")]))
    test_label_filenames = tf.convert_to_tensor(sorted([f for f in glob.glob(file_path + os.sep + "test_label_image*.png")]))

    predict_filenames = tf.convert_to_tensor(sorted([f for f in glob.glob(file_path + os.sep + "predict_image*.png")]))
    predict_label_filenames = tf.convert_to_tensor(sorted([f for f in glob.glob(file_path + os.sep + "predict_label_image*.png")]))

    # Set model params
    model_params = {"learning_rate": LEARNING_RATE}

    # Instantiate Estimator
    nn = tf.estimator.Estimator(model_fn=toy_segmentation_model_fn, params=model_params, model_dir = "/tmp/tf_toy_segm")

    # Train
    nn.train(input_fn=lambda: my_input_fn(train_filenames, train_label_filenames, repeat_count=100))

    # Score accuracy
    ev = nn.evaluate(input_fn=lambda: my_input_fn(test_filenames, test_label_filenames))
    print("Loss: %s" % ev["loss"])
    print("Root Mean Squared Error: %s" % ev["rmse"])

    # Print out predictions
    predictions = nn.predict(input_fn=lambda: my_input_fn(predict_filenames, predict_label_filenames))
    for i, p in enumerate(predictions):
         print(p["pixels"].reshape(6,6))
'''
    dir_name = "data"
    for i, p in enumerate(predictions):
        print(p["pixels"].reshape(6,6))
        image_name = "segmentated_image" + str(i) + ".png"
        cv2.imwrite(os.path.join(dir_name, image_name), p["pixels"].reshape(6,6))
'''
if __name__ == "__main__":
    tf.app.run(main=main)
