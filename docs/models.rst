
#######################
ANNForSentimentAnalysis
#######################

ANN for Sentiment Analysis.
Uses the pretrained embeddings from tensorflow hub. It can be any dimentional embeddings.

Training Dataset must have a shape of : [BATCH, SAMPLES, 1]
Labels must have the shape of         : [BATCH, LABELS]

where SAMPLES are ``tf.string``s and LABELS are ``tf.int32``s

:Parameters:

    **embedding: ``link``, optional**
        Link to the pretrained embeddings.

    **name: ``string``**
        Name of your model.

    **\*\*kwargs: keyword arguments**
        Keyword arguments for ``tf.keras.Model`` class.












:Attributes:

    :obj:`activity_regularizer <activity_regularizer>`
        Optional regularizer function for the output of this layer.

    :obj:`distribute_strategy <distribute_strategy>`
        The `tf.distribute.Strategy` this model was created under.

    :obj:`dtype <dtype>`
        Dtype used by the weights of the layer, set in the constructor.

    :obj:`dynamic <dynamic>`
        Whether the layer is dynamic (eager-only); set in the constructor.

    :obj:`inbound_nodes <inbound_nodes>`
        Deprecated, do NOT use! Only for compatibility with external Keras.

    :obj:`input <input>`
        Retrieves the input tensor(s) of a layer.

    :obj:`input_mask <input_mask>`
        Retrieves the input mask tensor(s) of a layer.

    :obj:`input_shape <input_shape>`
        Retrieves the input shape(s) of a layer.

    :obj:`input_spec <input_spec>`
        Gets the network's input specs.

    **layers**
        ..

    :obj:`losses <losses>`
        Losses which are associated with this `Layer`.

    :obj:`metrics <metrics>`
        Returns the model's metrics added using `compile`, `add_metric` APIs.

    :obj:`metrics_names <metrics_names>`
        Returns the model's display labels for all outputs.

    :obj:`name <name>`
        Name of the layer (string), set in the constructor.

    :obj:`name_scope <name_scope>`
        Returns a `tf.name_scope` instance for this class.

    **non_trainable_variables**
        ..

    :obj:`non_trainable_weights <non_trainable_weights>`
        List of all non-trainable weights tracked by this layer.

    :obj:`outbound_nodes <outbound_nodes>`
        Deprecated, do NOT use! Only for compatibility with external Keras.

    :obj:`output <output>`
        Retrieves the output tensor(s) of a layer.

    :obj:`output_mask <output_mask>`
        Retrieves the output mask tensor(s) of a layer.

    :obj:`output_shape <output_shape>`
        Retrieves the output shape(s) of a layer.

    :obj:`run_eagerly <run_eagerly>`
        Settable attribute indicating whether the model should run eagerly.

    :obj:`state_updates <state_updates>`
        Returns the `updates` from all layers that are stateful.

    **stateful**
        ..

    :obj:`submodules <submodules>`
        Sequence of all sub-modules.

    **trainable**
        ..

    :obj:`trainable_variables <trainable_variables>`
        Sequence of trainable variables owned by this module and its submodules.

    :obj:`trainable_weights <trainable_weights>`
        List of all trainable weights tracked by this layer.

    **updates**
        ..

    :obj:`variables <variables>`
        Returns the list of all layer variables/weights.

    :obj:`weights <weights>`
        Returns the list of all layer variables/weights.

.. rubric:: Methods

.. autosummary::
   :toctree:

   __call__
   add_loss
   add_metric
   add_update
   add_variable
   add_weight
   apply
   build
   compile
   compute_mask
   compute_output_shape
   compute_output_signature
   count_params
   evaluate
   evaluate_generator
   fit
   fit_generator
   from_config
   get_config
   get_input_at
   get_input_mask_at
   get_input_shape_at
   get_layer
   get_losses_for
   get_output_at
   get_output_mask_at
   get_output_shape_at
   get_updates_for
   get_weights
   load_weights
   make_predict_function
   make_test_function
   make_train_function
   predict
   predict_generator
   predict_on_batch
   predict_step
   reset_metrics
   save
   save_weights
   set_weights
   summary
   test_on_batch
   test_step
   to_json
   to_yaml
   train_on_batch
   train_step
   with_name_scope


================  ==========
        **call**    
**reset_states**    
================  ==========


#######################
RNNForSentimentAnalysis
#######################

RNN for Sentiment Analysis.
Uses the pretrained embeddings from tensorflow hub. It can be any dimentional embeddings.

Training Dataset must have a shape of : [BATCH, SAMPLES, 1]
Labels must have the shape of         : [BATCH, LABELS]

where 

:Parameters:

    **embedding: ``link``**
        The link to download the embeddings from tensorflow hub.

    **embedding_dims: ``int``**
        The dimensions of the embedding layer

    **name: ``string``**
        Name of your model

    **\*\*kwargs: keyword arguments**
        All the arguments the `tf.keras.Model` class takes












:Attributes:

    :obj:`activity_regularizer <activity_regularizer>`
        Optional regularizer function for the output of this layer.

    :obj:`distribute_strategy <distribute_strategy>`
        The `tf.distribute.Strategy` this model was created under.

    :obj:`dtype <dtype>`
        Dtype used by the weights of the layer, set in the constructor.

    :obj:`dynamic <dynamic>`
        Whether the layer is dynamic (eager-only); set in the constructor.

    :obj:`inbound_nodes <inbound_nodes>`
        Deprecated, do NOT use! Only for compatibility with external Keras.

    :obj:`input <input>`
        Retrieves the input tensor(s) of a layer.

    :obj:`input_mask <input_mask>`
        Retrieves the input mask tensor(s) of a layer.

    :obj:`input_shape <input_shape>`
        Retrieves the input shape(s) of a layer.

    :obj:`input_spec <input_spec>`
        Gets the network's input specs.

    **layers**
        ..

    :obj:`losses <losses>`
        Losses which are associated with this `Layer`.

    :obj:`metrics <metrics>`
        Returns the model's metrics added using `compile`, `add_metric` APIs.

    :obj:`metrics_names <metrics_names>`
        Returns the model's display labels for all outputs.

    :obj:`name <name>`
        Name of the layer (string), set in the constructor.

    :obj:`name_scope <name_scope>`
        Returns a `tf.name_scope` instance for this class.

    **non_trainable_variables**
        ..

    :obj:`non_trainable_weights <non_trainable_weights>`
        List of all non-trainable weights tracked by this layer.

    :obj:`outbound_nodes <outbound_nodes>`
        Deprecated, do NOT use! Only for compatibility with external Keras.

    :obj:`output <output>`
        Retrieves the output tensor(s) of a layer.

    :obj:`output_mask <output_mask>`
        Retrieves the output mask tensor(s) of a layer.

    :obj:`output_shape <output_shape>`
        Retrieves the output shape(s) of a layer.

    :obj:`run_eagerly <run_eagerly>`
        Settable attribute indicating whether the model should run eagerly.

    :obj:`state_updates <state_updates>`
        Returns the `updates` from all layers that are stateful.

    **stateful**
        ..

    :obj:`submodules <submodules>`
        Sequence of all sub-modules.

    **trainable**
        ..

    :obj:`trainable_variables <trainable_variables>`
        Sequence of trainable variables owned by this module and its submodules.

    :obj:`trainable_weights <trainable_weights>`
        List of all trainable weights tracked by this layer.

    **updates**
        ..

    :obj:`variables <variables>`
        Returns the list of all layer variables/weights.

    :obj:`weights <weights>`
        Returns the list of all layer variables/weights.

.. rubric:: Methods

.. autosummary::
   :toctree:

   __call__
   add_loss
   add_metric
   add_update
   add_variable
   add_weight
   apply
   build
   compile
   compute_mask
   compute_output_shape
   compute_output_signature
   count_params
   evaluate
   evaluate_generator
   fit
   fit_generator
   from_config
   get_config
   get_input_at
   get_input_mask_at
   get_input_shape_at
   get_layer
   get_losses_for
   get_output_at
   get_output_mask_at
   get_output_shape_at
   get_updates_for
   get_weights
   load_weights
   make_predict_function
   make_test_function
   make_train_function
   predict
   predict_generator
   predict_on_batch
   predict_step
   reset_metrics
   save
   save_weights
   set_weights
   summary
   test_on_batch
   test_step
   to_json
   to_yaml
   train_on_batch
   train_step
   with_name_scope


================  ==========
        **call**    
**reset_states**    
================  ==========

