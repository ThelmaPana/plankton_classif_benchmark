# Code for original cnn training
from absl import app
from absl import flags
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import numpy as np
from scipy import ndimage


INPUT_IMAGE_WIDTH = 224
INPUT_IMAGE_HEIGHT = 224
FLAGS = flags.FLAGS

flags.DEFINE_integer('epochs', 30, 'The number of steps to run training for.')
flags.DEFINE_integer('batch_size', 64, 'Hyperparameter: batch size.')
flags.DEFINE_integer(
    'fc1_hidden_units', 680,
    'Hyperparameter: number of hidden units for first fully connected layer.')
flags.DEFINE_integer(
    'sffc_hidden_units', 128,
    'Hyperparameter: number of hidden units for the two fully connected layers after the structured features input.'
)
flags.DEFINE_float('dropout_rate', 0.5, 'Hyperparameter: dropout rate.')
flags.DEFINE_float('learning_rate', 1e-3, 'Hyperparameter: learning rate.')
flags.DEFINE_integer('ensemble_size', 1, 'Number of models to train.')
flags.DEFINE_boolean('train_mobilenet_layer', True,
                     'Hyperparameter: Whether to train the mobile net layer.')
flags.DEFINE_boolean(
    'filter_trash_classes', False,
    'Hyperparameter: Whether to remove the trash classes from the set.')
flags.DEFINE_boolean(
    'use_structured_features', False,
    'Hyperparameter: using handcrafted features derivedd from the image.')
flags.DEFINE_boolean('image_augmentation', False,
                     'Hyperparameter: image augmentation.')
flags.DEFINE_integer('rotation_range', 180,
                     'Hyperparameter: image augmentation rotation range.')
flags.DEFINE_float('zoom_range', 0.05,
                   'Hyperparameter: image augmentation zoom range.')
flags.DEFINE_float('shift_range', 0.03,
                   'Hyperparameter: image augmentation shift range.')
flags.DEFINE_string(
    'tf_hub_layer',
    'https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4',
    'Layer loaded from tf hub.')
flags.DEFINE_string(
    'modeldir',
    'output/model.h5',
    'Path to store the trained model.')


def transform_matrix_offset_center(matrix, x, y):
  o_x = float(x) / 2 + 0.5
  o_y = float(y) / 2 + 0.5
  offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
  reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
  transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
  return transform_matrix


def apply_affine_transform(x,
                           theta=0,
                           tx=0,
                           ty=0,
                           shear=0,
                           zx=1,
                           zy=1,
                           row_axis=0,
                           col_axis=1,
                           channel_axis=2,
                           fill_mode='nearest',
                           cval=0.,
                           order=1):
  """Applies an affine transformation specified by the parameters given.

  # Arguments
      x: 2D numpy array, single image.
      theta: Rotation angle in degrees.
      tx: Width shift.
      ty: Heigh shift.
      shear: Shear angle in degrees.
      zx: Zoom in x direction.
      zy: Zoom in y direction
      row_axis: Index of axis for rows in the input image.
      col_axis: Index of axis for columns in the input image.
      channel_axis: Index of axis for channels in the input image.
      fill_mode: Points outside the boundaries of the input
          are filled according to the given mode
          (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
      cval: Value used for points outside the boundaries
          of the input if `mode='constant'`.
      order: int, order of interpolation
  # Returns
      The transformed version of the input.
  """
  transform_matrix = None
  if theta != 0:
    theta = np.deg2rad(theta)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    transform_matrix = rotation_matrix

  if tx != 0 or ty != 0:
    shift_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
    if transform_matrix is None:
      transform_matrix = shift_matrix
    else:
      transform_matrix = np.dot(transform_matrix, shift_matrix)

  if shear != 0:
    shear = np.deg2rad(shear)
    shear_matrix = np.array([[1, -np.sin(shear), 0], [0, np.cos(shear), 0],
                             [0, 0, 1]])
    if transform_matrix is None:
      transform_matrix = shear_matrix
    else:
      transform_matrix = np.dot(transform_matrix, shear_matrix)

  if zx != 1 or zy != 1:
    zoom_matrix = np.array([[zx, 0, 0], [0, zy, 0], [0, 0, 1]])
    if transform_matrix is None:
      transform_matrix = zoom_matrix
    else:
      transform_matrix = np.dot(transform_matrix, zoom_matrix)

  if transform_matrix is not None:
    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
    # x = np.rollaxis(x, channel_axis, 0)
    x = tf.transpose(x, perm=[channel_axis, row_axis, col_axis])
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]

    channel_images = [
        ndimage.interpolation.affine_transform(
            x_channel,
            final_affine_matrix,
            final_offset,
            order=order,
            mode=fill_mode,
            cval=cval) for x_channel in x
    ]
    x = np.stack(channel_images, axis=0)
    # x = np.rollaxis(x, 0, channel_axis + 1)
    x = tf.transpose(x, perm=[1, 2, 0])
  return x


def custom_affine_transform(image):
  rotation = np.random.uniform(-FLAGS.rotation_range, FLAGS.rotation_range)
  zoom = np.random.uniform(-FLAGS.zoom_range, FLAGS.zoom_range) + 1
  shift_x = np.random.uniform(-FLAGS.shift_range,
                              FLAGS.shift_range) * INPUT_IMAGE_WIDTH
  shift_y = np.random.uniform(-FLAGS.shift_range,
                              FLAGS.shift_range) * INPUT_IMAGE_HEIGHT
  return apply_affine_transform(
      image, theta=rotation, zx=zoom, zy=zoom, tx=shift_x, ty=shift_y)


def remove_legend(image):
  y, x, channels = image.shape
  legend_bottom = image.shape[0] - tf.argmax(
      tf.reverse(tf.math.reduce_mean(image, axis=(1, 2)), [0]) < 254)
  legend_top = legend_bottom - tf.argmax(
      tf.reverse(tf.math.reduce_mean(image[:legend_bottom], axis=(
          1, 2)), [0]) == 255)
  legend_slice = image[legend_top:legend_bottom + 1]
  mask = tf.math.reduce_any(legend_slice < 255, axis=(0, 2))
  legend_width = tf.boolean_mask(legend_slice, mask, axis=1).shape[1]
  white_img = tf.ones(image.shape, dtype=tf.uint8) * 255
  legend_height = tf.cast(tf.cast(legend_width, tf.float32) * 0.21, tf.int64)
  mask = tf.cast(
      tf.range(y, dtype=tf.int64) >= (legend_bottom - legend_height),
      tf.uint8)[:, tf.newaxis, tf.newaxis]
  return mask * white_img + (1 - mask) * image


def compute_scale_and_remove_legend(image):
  y, x, channels = image.shape
  legend_bottom = image.shape[0] - tf.argmax(
      tf.reverse(tf.math.reduce_mean(image, axis=(1, 2)), [0]) < 254)
  legend_top = legend_bottom - tf.argmax(
      tf.reverse(tf.math.reduce_mean(image[:legend_bottom], axis=(
          1, 2)), [0]) == 255)
  legend_slice = image[legend_top:legend_bottom + 1]
  mask = tf.math.reduce_any(legend_slice < 255, axis=(0, 2))
  legend_width = tf.boolean_mask(legend_slice, mask, axis=1).shape[1]
  white_img = tf.ones(image.shape, dtype=tf.uint8) * 255
  legend_height = tf.cast(tf.cast(legend_width, tf.float32) * 0.21, tf.int64)
  mask = tf.cast(
      tf.range(y, dtype=tf.int64) >= (legend_bottom - legend_height),
      tf.uint8)[:, tf.newaxis, tf.newaxis]
  return mask * white_img + (1 - mask) * image, tf.cast(legend_width / x,
                                                        tf.float32)


def tf_remove_legend(features, label):
  image, structured_features = features
  im_shape = image.shape
  [image, scale] = tf.py_function(compute_scale_and_remove_legend, [image],
                                  [tf.uint8, tf.float32])
  image.set_shape(im_shape)
  return (image, structured_features), label


def tf_img_augmentation(features, label):
  image, structured_features = features
  im_shape = image.shape
  [
      image,
  ] = tf.py_function(custom_affine_transform, [image], [tf.uint8])
  image.set_shape(im_shape)
  return (image, structured_features), label


def shape_input(features, label):
  if FLAGS.use_structured_features:
    return features, label
  else:
    return features[0], label  # image and label


def format_sample(sample):
  return ((sample['features']['image'],
           sample['features']['structured_features']), sample['label'])


def get_datasets():
  dataset_name = 'tara2'
  ds, info = tfds.load(
      dataset_name,
      data_dir='./tara2/',
      with_info=True)
  train_ds = ds['train'].map(format_sample)
  test_ds = ds['test'].map(format_sample)
  if FLAGS.image_augmentation:
    train_ds = train_ds.map(tf_remove_legend)
    test_ds = test_ds.map(tf_remove_legend)
    train_ds = train_ds.map(tf_img_augmentation)
  if FLAGS.use_structured_features:
    train_ds = train_ds.map(shape_input)
    test_ds = test_ds.map(shape_input)
  if FLAGS.filter_trash_classes:
    trash_classes = tf.Variable(
        [
            41,  # detritus
            48,  # fiber<detritus
            78,  # artefact
            108,  # badfocus<artefact
            125  # bubble
        ],
        dtype=tf.int64)

    def filter_fun(features, label):
      return tf.reduce_any(tf.math.equal(label, trash_classes))

    train_ds = train_ds.filter(filter_fun)
    test_ds = test_ds.filter(filter_fun)
  return train_ds, test_ds, info


def build_model(num_classes=136):
  image_input = tf.keras.layers.Input(
      shape=(INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT, 3))
  x = hub.KerasLayer(
      FLAGS.tf_hub_layer, trainable=FLAGS.train_mobilenet_layer)(
          image_input)
  x = tf.keras.layers.Dropout(FLAGS.dropout_rate)(x)
  if FLAGS.use_structured_features:
    structured_features_input = tf.keras.layers.Input(shape=(69,))
    y = tf.keras.layers.Dense(
        FLAGS.sffc_hidden_units, activation='relu')(
            structured_features_input)
    y = tf.keras.layers.Dense(FLAGS.sffc_hidden_units, activation='relu')(y)
    x = tf.keras.layers.concatenate([y, x], axis=-1)
    inputs = [image_input, structured_features_input]
  else:
    inputs = image_input
  x = tf.keras.layers.Dense(FLAGS.fc1_hidden_units, activation='relu')(x)
  outputs = tf.keras.layers.Dense(num_classes)(x)

  model = tf.keras.Model(inputs=inputs, outputs=outputs)

  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'])

  return model


def main(argv):
  del argv  # Unused

  model_path = FLAGS.modeldir

  train_ds, test_ds, info = get_datasets()

  model = build_model(num_classes=info.features['label'].num_classes)

  class Measurement(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
      print(f'Epoch: {epoch}: logs.')

  class ModelCheckpoint(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
      tf.keras.save_model(
          self.model, model_path, include_optimizer=False)

  reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
      monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

  model.fit(
      train_ds.shuffle(buffer_size=10000).batch(FLAGS.batch_size).prefetch(2),
      validation_data=test_ds.batch(FLAGS.batch_size).prefetch(2),
      epochs=FLAGS.epochs,
      callbacks=[Measurement(), ModelCheckpoint(), reduce_lr])


if __name__ == '__main__':
  app.run(main)
