# HMC for logistic regression

hmc_lr_presample = function(U, grad_U, epsilon, L, current_q, data, param){

  q = current_q
  p = rnorm(length(q),0,1)
  current_p = p

  for (i in 1:L){
    half.p = p - epsilon/2 * grad_U(q, data, param)
    q = q + epsilon * half.p
    p = half.p - epsilon/2 * grad_U(q, data, param)
  }

  p = -p

  current_U = U(current_q, data, param)
  current_K = sum(current_p^2) / 2
  proposed_U = U(q, data, param)
  proposed_K = sum(p^2) / 2

  # browser()

  if (runif(1) < exp(current_U-proposed_U+current_K-proposed_K)){
    return(list("sample" = q, "acc" = 1))
  } else {
    return(list("sample" = current_q, "acc" = 0))
  }
}
