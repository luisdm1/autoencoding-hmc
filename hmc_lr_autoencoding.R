# HMC with autoencoder for regression
hmc_lr_autoencoding = function(U, gU_latent, gK_latent, encoder, decoder,norm_param_autoencoder_input, autoencoder_weights, epsilon, L, current_q, data, new_data, param){

  q = current_q
  p = rnorm(length(q),0,1)
  current_p = p

  # obtain normalized q
  normalized_q = (q-norm_param_autoencoder_input$mean)/(norm_param_autoencoder_input$sd)
  # latent representation of q
  latent_q = encoder %>%
    predict(matrix(normalized_q,1,length(normalized_q))) %>%
    as.vector()

  # obtain normalized p
  latent_p = encoder %>%
    predict(matrix(p,1,length(p))) %>%
    as.vector()


  current_normalized_q = normalized_q
  current_latent_q = latent_q
  current_latent_p = latent_p

  for (i in 1:L){
    half.latent_p = latent_p - epsilon/2 * gU_latent(latent_q, new_data, param)
    latent_q = latent_q + epsilon * gK_latent(half.latent_p, new_data)
    latent_p = half.latent_p - epsilon/2 * gU_latent(latent_q, new_data, param)
  }

  # reconstruct q from latent q
  # tic('calculate reconstructed')
  normalized_q = decoder %>%
    predict(matrix(latent_q,1,length(latent_q))) %>%
    as.vector()
  # unormalize q
  q = normalized_q*norm_param_autoencoder_input$sd + norm_param_autoencoder_input$mean

  p = decoder %>%
    predict(matrix(latent_p,1,length(latent_p))) %>%
    as.vector()

  current_U = U(current_q, data, param)
  current_K = sum(current_p^2) / 2
  proposed_U = U(q, data, param)
  proposed_K = sum(p^2) / 2

  # # Calculate volume correction term
  # W1 = autoencoder_weights$W1
  # b1 = autoencoder_weights$b1
  # W2 = autoencoder_weights$W2
  #
  # U1 = autoencoder_weights$U1
  # d1 = autoencoder_weights$d1
  # U2 = autoencoder_weights$U2
  #
  # v1 = 1-tanh(as.vector(latent_q %*% W1+ t(b1)))^2
  # v2 = 1-current_latent_q^2
  # v3 = 1-tanh(as.vector(current_normalized_q %*% U1+ t(d1)))^2
  # E1 = matrix(v1,nrow=dim(W2)[1],ncol=dim(W2)[2])
  # E2 = matrix(v2,nrow=dim(U2)[2],ncol=dim(U2)[1])
  # E3 = matrix(v3,nrow=dim(U2)[1],ncol=dim(U2)[2])
  # mat1 = (t(W2) * t(E1)) %*% t(W1)
  # mat2 = (E2 * t(U2) * t(E3)) %*% t(U1)
  # correction_term_q = sqrt(det(t(mat1) %*% mat1)) * sqrt(det(mat2 %*% t(mat2)))
  #
  #
  # v1 = 1-tanh(as.vector(latent_p %*% W1+ t(b1)))^2
  # v2 = 1-current_latent_p^2
  # v3 = 1-tanh(as.vector(current_p %*% U1+ t(d1)))^2
  # E1 = matrix(v1,nrow=dim(W2)[1],ncol=dim(W2)[2])
  # E2 = matrix(v2,nrow=dim(U2)[2],ncol=dim(U2)[1])
  # E3 = matrix(v3,nrow=dim(U2)[1],ncol=dim(U2)[2])
  # mat1 = (t(W2) * t(E1)) %*% t(W1)
  # mat2 = (E2 * t(U2) * t(E3)) %*% t(U1)
  # correction_term_p = sqrt(det(t(mat1) %*% mat1)) * sqrt(det(mat2 %*% t(mat2)))
  #
  # correction_term = correction_term_q * correction_term_p

  correction_term = 1

  # cat('current_U-proposed_U:', current_U-proposed_U,'\n')
  # cat('current_K-proposed_K:', current_K-proposed_K,'\n')
  # cat('correction:', correction_term,'\n')
  # cat('',exp(current_U-proposed_U+current_K-proposed_K)*correction_term, '\n')

  if (runif(1) < exp(current_U-proposed_U+current_K-proposed_K)*correction_term){
    # print('accepted')
    return(list("sample" = q, "acc" = 1))
  } else {
    # print('rejected')
    return(list("sample" = current_q, "acc" = 0))
  }

  # return(list("sample" = q, "acc" = 1))
}
