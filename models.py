import sys
import tensorflow as tf
import tensorflow_hub as hub
from tf.keras import Model
from tensorflow.keras.layers import Dense, Embedding, Bidirectional, LSTM


class ANNForSentimentAnalysis(Model):
    """ANN for Sentiment Analysis.
    Uses the pretrained embeddings from tensorflow hub. It can be any dimentional embeddings.

    Training Dataset must have a shape of : [BATCH, SAMPLES, 1]
    Labels must have the shape of         : [BATCH, LABELS]

    where SAMPLES are ``tf.string``s and LABELS are ``tf.int32``s

    Parameters
    ----------
    embedding: ``link``, optional
        Link to the pretrained embeddings.
    
    name: ``string``
        Name of your model.
    
    **kwargs: keyword arguments
        Keyword arguments for ``tf.keras.Model`` class.
    """

    def __init__(self, embedding = "https://tfhub.dev/google/nnlm-en-dim128/1", name="ANNForSentimentAnalysis", **kwargs):
        super(ANNForSentimentAnalysis, self).__init__(name=name, **kwargs)
        ####################################################################################
        self._embedding_layer = hub.KerasLayer(embedding, trainable=True, dtype=tf.string)
        self._dense_layer_1   = Dense(64, activation='relu')
        self._output_layer    = Dense(1, activation='sigmoid')
        ####################################################################################

    @tf.function
    def call(self, inputs):
        ###################################################
        embeddings   = self._embedding_layer(inputs)
        intermediate = self._dense_layer_1(embeddings)
        outputs      = self._output_layer(intermediate)
        ###################################################
        return outputs


class RNNForSentimentAnalysis(Model):
    """RNN for Sentiment Analysis.
    Uses the pretrained embeddings from tensorflow hub. It can be any dimentional embeddings.

    Training Dataset must have a shape of : [BATCH, SAMPLES, 1]
    Labels must have the shape of         : [BATCH, LABELS]

    where 

    Parameters
    ----------
    embedding: ``link``
        The link to download the embeddings from tensorflow hub.

    embedding_dims: ``int``
        The dimensions of the embedding layer

    name: ``string``
        Name of your model
    
    **kwargs: keyword arguments
        All the arguments the `tf.keras.Model` class takes
    """

    def __init__(self, embedding="https://tfhub.dev/google/nnlm-en-dim128/1", embedding_dims=128, name="RNNForSentimentAnalysis", **kwargs):
        super(RNNForSentimentAnalysis, self).__init__(name=name, **kwargs)
        ####################################################################################
        self._embedding_layer = hub.KerasLayer(embedding, trainable=False, dtype=tf.string)
        self._recurrent_layer = Bidirectional(LSTM(64), input_shape=[None, embedding_dims])
        self._dense_layer_1   = Dense(64, activation='relu')
        self._output_layer    = Dense(1, activation='sigmoid')
        ####################################################################################

    @tf.function
    def call(self, inputs):
        ###################################################
        embeddings   = self._embedding_layer(inputs)
        recurrent    = self._recurrent_layer(embeddings)
        intermediate = self._dense_layer_1(recurrent)
        outputs      = self._output_layer(intermediate)
        ###################################################
        return outputs
