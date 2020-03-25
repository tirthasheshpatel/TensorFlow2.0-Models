echo "starting..."

python -m numpydoc models > models.rst
python -m numpydoc models.ANNForSentimentAnalysis >> ./docs/models.rst
python -m numpydoc models.RNNForSentimentAnalysis >> ./docs/models.rst
python -m numpydoc dataset > dataset.rst
python -m numpydoc dataset.get_batched_imdb_dataset_from_keras >> ./docs/dataset.rst
python -m numpydoc dataset.get_batched_imdb_dataset_from_hub >> ./docs/dataset.rst
python -m numpydoc dataset.get_batched_imdb_dataset_from_tfds >> ./docs/dataset.rst
