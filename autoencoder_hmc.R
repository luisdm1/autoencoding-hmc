# autoencoder for hmc samples
library(keras)
use_session_with_seed(42)
autoencoder_hmc = function(x_train, x_test, model_savepath, ncol_hidden_layer, ncol_latent_layer, num_epoch, num_batchsize, early.stop, patience){

  # Model definition
  input <- layer_input(shape = c(ncol(x_train)))
  hidden_encoded <- layer_dense(input, units = ncol_hidden_layer, activation = "tanh")
  latent <- layer_dense(hidden_encoded, units = ncol_latent_layer, activation = "tanh")
  decoder_hidden_layer <- layer_dense(units = ncol_hidden_layer, activation = "tanh")

  decoder_input_layer <- layer_dense(units = ncol(x_train))

  hidden_decoded <- decoder_hidden_layer(latent)
  reconstruction <- decoder_input_layer(hidden_decoded)

  # end-to-end autoencoder
  model <- keras_model(inputs = input, outputs = reconstruction)

  # encoder, from inputs to latent space
  encoder <- keras_model(inputs = input, outputs = latent)

  # generator, from latent space to reconstructed inputs
  decoder_input <- layer_input(shape = ncol_latent_layer)
  hidden_decoded_2 <- decoder_hidden_layer(decoder_input)
  reconstruction_2 <- decoder_input_layer(hidden_decoded_2)
  decoder <- keras_model(inputs = decoder_input, outputs = reconstruction_2)

  # create and compile model
  model %>% compile(
    loss = "mean_squared_error",
    # optimizer = optimizer_adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    optimizer = optimizer_nadam(lr = 0.002, beta_1 = 0.9, beta_2 = 0.999, epsilon = NULL, schedule_decay = 0.004)

  )

  # Training
  checkpoint <- callback_model_checkpoint(
    filepath = model_savepath,
    save_best_only = TRUE,
    period = 1,
    verbose = 1
  )

  early_stopping <- callback_early_stopping(patience = patience)

  if (early.stop){
    hist <- model %>% fit(
      x = x_train,
      y = x_train,
      epochs = num_epoch,
      batch_size = num_batchsize,
      validation_data = list(x_test, x_test),
      callbacks = list(checkpoint, early_stopping)
    )
  } else{
    hist <- model %>% fit(
      x = x_train,
      y = x_train,
      epochs = num_epoch,
      batch_size = num_batchsize,
      validation_data = list(x_test, x_test),
      callbacks = list(checkpoint)
    )
  }

  hist_df <- as.data.frame(hist)
  str(hist_df)

  # browser()
  # loss for test data
  loss <- evaluate(model, x = x_test, y = x_test)
  cat("Test data loss: ", loss)

  return(list(Hist = hist, Model = model, Encoder = encoder, Decoder = decoder))
}

