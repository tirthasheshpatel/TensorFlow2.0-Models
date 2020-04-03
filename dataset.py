import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def get_batched_imdb_dataset_from_keras(buffer_size, batch_size, maxlen=500):
    r"""Loads the imdb reviews dataset from ``tf.keras.datasets`` module

    Parameters
    ----------
    buffer_size: int
        The buffer size during shuffling

    batch_size: int
        The batch size of the returned datasets

    Returns
    -------
    batched_train_data      : A batched dataset ready to be used to train a model
    batched_validation_data : A batched validation dataset
    vocabulary              : A vocabulary of words present in the dataset

    Notes
    -----
    The ``batched_train_data`` and ``batched_validation_data`` contains the padded
    sequences of shape ``[batch_size, maxlen]`` of dtype of ``tf.int32``. The coresponding
    labels have shape ``[batch_size, ]`` and dtype ``tf.int32``. The ``vocabulary`` contains
    all the words  using which the dataset can be converted to raw reviews of dtype
    ``tf.string`` and then input to a model that accepts ``tf.string`` type tensors.
    """
    # Manual Batching for V2 Support

    imdb = tf.keras.datasets.imdb

    (x_train, y_train), (x_val, y_val) = imdb.load_data()

    x_train_padded = pad_sequences(x_train, maxlen=maxlen)
    x_val_padded = pad_sequences(x_val, maxlen=maxlen)
    vocabulary = imdb.get_word_index()

    batched_train_data = (
        tf.data.Dataset.from_tensor_slices((x_train_padded, y_train))
        .batch(batch_size)
        .shuffle(buffer_size)
    )
    batched_validation_data = (
        tf.data.Dataset.from_tensor_slices((x_val_padded, y_val))
        .batch(batch_size)
        .shuffle(buffer_size)
    )

    return batched_train_data, batched_validation_data, vocabulary


def get_batched_imdb_dataset_from_hub(buffer_size, batch_size):
    r"""Loads the imdb reviews dataset from tensorflow hub

    Parameters
    ----------
    buffer_size: int
        The buffer size during shuffling

    batch_size: int
        The batch size of the returned datasets

    Returns
    -------
    batched_train_data      : A batched dataset ready to be used to train a model
    batched_validation_data : A batched validation dataset
    test_data               : A non-batched dataset to test the model

    Notes
    -----
    The ``batched_train_data`` and ``batched_validation_data`` are tensors
    of shape ``[batch_size, string_length]`` containing raw reviews of dtype
    ``tf.string``.  The coressponding labels have the shape ``[batch_size, 1]``.
    You can use ``tf.keras.preprocessing.Tokenizer`` to get the vocabulary
    and coressponding sequences and then use ``tf.keras.preprocessing.pad_sequences``
    to get padded sequences that can then be input to the model that requires the
    inputs in the form of a constant padding of sequences rather than ``tf.string``.
    """
    # Using hub for V2 support directly.

    train_data, validation_data, test_data = tfds.load(
        name="imdb_reviews",
        split=("train[:60%]", "train[60%:]", "test"),
        as_supervised=True,
    )

    batched_train_data = train_data.shuffle(buffer_size).batch(batch_size)
    batched_validation_data = validation_data.shuffle(buffer_size).batch(batch_size)

    return batched_train_data, batched_validation_data, test_data


def get_batched_imdb_dataset_from_tfds(buffer_size, batch_size):
    r"""Loads the imdb reviews sataset from tensorflow datasets.

    Parameters
    ----------
    buffer_size: int
        The buffer size during shuffling

    batch_size: int
        The batch size of the returned datasets

    Returns
    -------
    batched_train_data      : A batched dataset ready to be used to train a model
    batched_test_data       : A batched test dataset to evalute the model
    encoder                 : An object with ``encode`` and ``decode`` methods

    Notes
    -----
    The ``batched_train_data`` and ``batched_test_data`` contains the padded
    sequences of shape ``[batch_size, maxlen]`` of dtype of ``tf.int32``. The coresponding
    labels have shape ``[batch_size, ]`` and dtype ``tf.int32``. The ``encoder`` object has
    a ``encode`` method that can be used to encode a string to sequences and ``decode`` method
    that can be used to decode padded sequences to string reviews.

    .. code_block::
    
    encoded_string = encoder.encode(sample_string)
    print('Encoded string is {}'.format(encoded_string))

    original_string = encoder.decode(encoded_string)
    print('The original string: "{}"'.format(original_string))

    """
    # Using tensorflow dataset from version V2
    dataset, info = tfds.load(
        "imdb_reviews/subwords8k", with_info=True, as_supervised=True
    )

    train_data, test_data = dataset["train"], dataset["test"]

    encoder = info.features["text"].encoder

    batched_train_data = train_data.shuffle(buffer_size).padded_batch(
        batch_size, padded_shapes=([None], [])
    )

    batched_test_data = test_data.padded_batch(batch_size, padded_shapes=([None], []))

    return batched_train_data, batched_test_data, encoder
