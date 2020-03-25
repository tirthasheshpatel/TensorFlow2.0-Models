import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def get_batched_imdb_dataset_from_keras(buffer_size, batch_size, maxlen=500):
    """Loads the imdb reviews sataset from ``tf.keras.datasets`` module

    Parameters
    ----------
    buffer_size: int
        The buffer size of the
    """
    # Manual Batching for V2 Support

    imdb = tf.keras.datasets.imdb

    (x_train, y_train), (x_val, y_val) = imdb.load_data()

    x_train_padded = pad_sequences(x_train, maxlen=maxlen)
    x_val_padded   = pad_sequences(x_val  , maxlen=maxlen)
    vocabulary     = imdb.get_word_index()

    batched_train_data      = tf.data.Dataset.from_tensor_slices((x_train_padded, y_train)).batch(batch_size).shuffle(buffer_size)
    batched_validation_data = tf.data.Dataset.from_tensor_slices((x_val_padded, y_val)).batch(batch_size).shuffle(buffer_size)

    return batched_train_data, batched_validation_data, vocabulary


def get_batched_imdb_dataset_from_hub(buffer_size, batch_size):
    # Using hub for V2 support directly.

    train_data, validation_data, test_data = tfds.load(
        name="imdb_reviews",
        split=('train[:60%]', 'train[60%:]', 'test'),
        as_supervised=True
    )

    batched_train_data      = train_data.shuffle(buffer_size).batch(batch_size)
    batched_validation_data = validation_data.shuffle(buffer_size).batch(batch_size)

    return batched_train_data, test_data, batched_batched_validation_data


def get_batched_imdb_dataset_from_tfds(buffer_size, batch_size):
    # Using tensorflow dataset from version V2
    dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)

    train_data, test_data = dataset['train'], dataset['test']

    encoder = info.features['text'].encoder

    batched_train_data = (train_data
                    .shuffle(buffer_size)
                    .padded_batch(batch_size, padded_shapes=([None],[])))

    batched_test_data = (test_data
                    .padded_batch(batch_size,  padded_shapes=([None],[])))
                
    return batched_train_data, batches_test_data, encoder
