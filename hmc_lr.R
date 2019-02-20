# HMC for regression

hmc_lr = function(U, grad_U, epsilon, L, current_q, data, param){

  # tic()
  q = current_q

  # tic('propose q')
  p = rnorm(length(q),0,1)
  current_p = p

  for (i in 1:L){
    # tic('calculate gU')
    gU = grad_U(q, data, param)
    # toc()

    # tic('one proposal')
    half.p = p - epsilon/2 * grad_U(q, data, param)
    # toc()

    q = q + epsilon * half.p
    p = half.p - epsilon/2 * grad_U(q, data, param)
  }

  p = -p

  # toc()

  # plot(current_q, ylim = c(min(q,current_q), max(q, current_q)))
  # points(q, col = 'red')

  current_U = U(current_q, data, param)
  current_K = sum(current_p^2) / 2
  proposed_U = U(q, data, param)
  proposed_K = sum(p^2) / 2

  # cat('current_U-proposed_U:', current_U-proposed_U,'\n')
  # cat('current_K-proposed_K:', current_K-proposed_K,'\n')
  # cat('exp:', exp(current_U-proposed_U+current_K-proposed_K),'\n')
  #

  if (runif(1) < exp(current_U-proposed_U+current_K-proposed_K)){
    return(list("sample" = q, "acc" = 1))
  } else {
    return(list("sample" = current_q, "acc" = 0))
  }
}
