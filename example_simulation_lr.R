rm(list=ls())
library("bayesplot")
library("ggplot2")
library("GGally")
library("rstanarm")
library("MASS")
library("purrr")
library("tictoc")
library("mcmcse")
library("reshape2")
library("coda")

#####################################################################
#                       Simulate Data                               #
#####################################################################
set.seed(12345)
# simulate data
n = 1000
# number of parameters
m = 500
latent_dim = 50
# hidden_dim = round((m+latent_dim)/2)
hidden_dim = 100
nn_hidden_dim = 100

var_X = 0.01
scale_intercept = 0.1
var_beta = 100
beta_min = -1
beta_max = 1

x1 = mvrnorm(n = n, mu = rep(0,m-1), Sigma = var_X*diag(m-1))
X = cbind(rep(scale_intercept,n), x1)
beta.T = runif(m, min = beta_min, max = beta_max)

cat('True parameters beta: ', beta.T)
prob.T = exp(X %*% beta.T)/(1+exp(X %*% beta.T))

y = numeric()
for(i in 1:n){y[i] = rbinom(1, size = 1, prob = prob.T[i])}

data = cbind(y,X)

#prior for beta
mu = rep(0, m)
sigma2 = rep(1, m)*var_beta
param = c(mu, sigma2)

U = function(beta, data, param){
  y.U = data[,1]
  X.U = data[,2:(m+1)]
  mu.U = param[1:m]
  sigma2.U = param[(m+1):(2*m)]
  return(sum((beta-mu.U)^2/(2*sigma2.U)) - t(y.U)%*%(X.U%*%beta) + sum(log(1+exp(X.U%*%beta))))
}

grad_U = function(beta, data, param){
  y.gU = data[,1]
  X.gU = data[,2:(m+1)]
  mu.gU = param[1:m]
  sigma2.gU = param[(m+1):(2*m)]
  return((beta -mu.gU)/sigma2.gU -t(X.gU)%*%y.gU + t(X.gU) %*% (exp(X.gU%*%beta)/(1+exp(X.gU%*%beta))))
}

#####################################################################
#                      Pretrain HMC samples                         #
#####################################################################
logistic_hmc_presample = function(start, data, param, L, epsilon, max.iter){

  beta.vec = matrix(0, max.iter, m)
  acc.vec = rep(0, max.iter)

  acc.vec[1] = 0
  beta.vec[1,] = start
  current_beta = beta.vec[1,]

  for (i in 2:max.iter){
    result = hmc_lr_presample(U, grad_U, epsilon, L, current_beta, data, param)
    beta.vec[i,] = result$sample
    current_beta = beta.vec[i,]
    acc.vec[i] = result$acc
  }

  return(list(samples = beta.vec, acc = acc.vec))
}

source("hmc_lr_presample.R")
presample_size = 1000

tic("HMC pretrain")
presample_results = logistic_hmc_presample(rep(1,m), data, param, 10, 0.1,presample_size)
saveRDS(presample_results, "data/presample_results.rds")
toc()

presample_results <- readRDS("data/presample_results.rds")
presample_accepted = presample_results$samples[presample_results$acc == 1,]
presample_accepted_size = dim(presample_accepted)[1]
cat("Acceptance ratio: ", sum(presample_results$acc)/(presample_size-1), '\n')
next_hmc_init = presample_results$samples[presample_size,]
pretrain_samples <- presample_results$samples


#####################################################################
#                       Auto-encoding HMC                          #
#####################################################################
# # mean-sd normalization of training samples
presample_train = presample_accepted[1:round(presample_accepted_size*0.9),]
presample_test = presample_accepted[(round(presample_accepted_size*0.9)+1):presample_accepted_size,]

mean_train = apply(presample_train,2,mean)
sd_train = apply(presample_train,2,sd)
norm_param = list(mean = mean_train, sd = sd_train)
#
x_train = t(apply(presample_train, 1, function(x)(x-norm_param$mean)/norm_param$sd))
x_test = t(apply(presample_test, 1, function(x)(x-norm_param$mean)/norm_param$sd))

# # no normalization
# x_train = presample_accepted[1:round(presample_accepted_size*0.9),]
# x_test = presample_accepted[(round(presample_accepted_size*0.9)+1):presample_accepted_size,]

# ##############################################
# Train the autoencoder with the pretrained samples
source("autoencoder_trainer.R")
tic("Training autoencoder")
autoencoder_results = autoencoder_trainer(x_train, x_test, model_savepath = "model_pretrain_autoencoder.hdf5", ncol_hidden_layer = hidden_dim, ncol_latent_layer = latent_dim, num_epoch = 1000, num_batchsize = 64, early.stop = TRUE, patience = 10)
toc()

loss_history = autoencoder_results$Hist
plot(loss_history)

model_pretrain_autoencoder = autoencoder_results$Model
encoder = autoencoder_results$Encoder
decoder = autoencoder_results$Decoder

# model_pretrain_autoencoder <- load_model_hdf5("model_pretrain_autoencoder.hdf5", compile = FALSE)

##### Check reconstruction of the test data####
# ggpairs(data=data.frame(x_test), columns = 1:5, lower = list(continuous = wrap("points", size=0.1)))
#
# reconstruction <- predict(model_pretrain_autoencoder, x_test)
# ggpairs(data=data.frame(reconstruction), columns = 1:5, lower = list(continuous = wrap("points", size=0.1)))

# so far: take presample_train samples (those which lead acc rate=1), 
# stardardadize these samples, train autoencoder, predict for same samples,
# un-standardize by same mean and sd and compare densities

x_trained = presample_test
x_trained_predicted = predict(model_pretrain_autoencoder, x_trained)
# mean_pred = apply(x_trained_predicted,2,mean)
# sd_pred = apply(x_trained_predicted,2,sd)
# norm_param_pred = list(mean = mean_pred, sd = sd_pred)
x_trained_predicted = t(apply(x_trained_predicted, 1, function(x)(x*norm_param$sd+norm_param$mean)))

colnames(x_trained) = paste0("Beta", 1:m)
colnames(x_trained_predicted) = paste0("Beta", 1:m)
names(beta.T)= paste0("Beta", 1:m)

set.seed(12345)
ix_params = sort(sample(1:m, 20))
x_trained_sub = melt(x_trained[,ix_params])
x_trained_sub$type = "input"
x_trained_predicted_sub = melt(x_trained_predicted[,ix_params])
x_trained_predicted_sub$type = "decoded"

true_params = melt(beta.T[ix_params])
true_params$iter = 0
true_params$param = names(beta.T[ix_params])
true_params$type = "true"
true_params = true_params[,c("iter", "param", "value", "type")]

df_pretrain_reconstructed = data.frame(rbind(x_trained_sub, x_trained_predicted_sub))

names(df_pretrain_reconstructed)=c("iter", "param", "value", "type")
df_pretrain_reconstructed = rbind(df_pretrain_reconstructed, true_params)

ggplot(data=df_pretrain_reconstructed, 
       aes(x=value))+
  geom_density(data=df_pretrain_reconstructed[df_pretrain_reconstructed$type!="true",],
               aes(fill=type),alpha=0.4)+
  geom_segment(data=df_pretrain_reconstructed[df_pretrain_reconstructed$type=="true",],
             aes(x=value,y=0, xend=value,yend=1.5),size=1,col="blue",linetype="dashed")+
  facet_wrap(~param)+
  #coord_cartesian(ylim = c(0, 2),xlim=c(-12.5,12.5))+
  ggtitle(label = "Comparing input vs decoded output of trained autoencoder (98 HMC input samples)")->p

ggsave(p,filename = "input_vs_decoded_hmc_train_samples.pdf",width = 15,height = 10,dpi = 500)


# Get encoder and decoder weights
library(keras)
library(tensorflow)

# Build tensors from the model
encoder_weights =get_weights(encoder)
U1 = encoder_weights[[1]]
d1 = encoder_weights[[2]]
U2 = encoder_weights[[3]]

decoder_weights =get_weights(decoder)
W1 = decoder_weights[[1]]
b1 = decoder_weights[[2]]
W2 = decoder_weights[[3]]

autoencoder_weights = list(U1 = U1, d1 = d1, U2 = U2, W1 = W1, b1 = b1, W2 = W2)

# ##############################################
# Calculate new gradient
# pre-calculate data and big matrix
y.gU = data[,1]
X.gU = data[,2:(m+1)]
XW2 = X.gU %*% t(W2)
sq_W2 = W2 %*% t(W2)
new_data = list(y.gU = y.gU, Z.gU = XW2, sq_W2 = sq_W2, W1 = W1, b1 = b1)

# gradient of latent U
gU_latent = function(beta_h, data, param){
  y.gU = data$y.gU
  Z.gU = data$Z.gU
  sq_W2 = data$sq_W2
  W1 = data$W1
  b1 = data$b1
  mu.gU = param[1:m]
  sigma2.gU = param[(m+1):(2*m)]
  tanh_term = tanh(as.vector(beta_h %*% W1+ t(b1)))
  grad_1 = t(tanh_term%*%sq_W2) - t(Z.gU)%*%y.gU+ t(Z.gU) %*% (1/(1+exp(-Z.gU%*%tanh_term)))
  v1 = 1-tanh_term^2
  E1 = matrix(v1,nrow=dim(grad_1)[1],ncol=dim(grad_1)[2])
  return((t(grad_1) * t(E1)) %*% t(W1))
}

# gradient of latent K
gK_latent = function(p_h, data){
  sq_W2 = data$sq_W2
  W1 = data$W1
  b1 = data$b1
  tanh_term = tanh(as.vector(p_h %*% W1+ t(b1)))
  grad_1 = t(tanh_term%*%sq_W2)
  v1 = 1-tanh_term^2
  E1 = matrix(v1,nrow=dim(grad_1)[1],ncol=dim(grad_1)[2])
  return((t(grad_1) * t(E1)) %*% t(W1))
}

## Implement HMC with autoencoder with correction
source("hmc_lr_autoencoding.R")
logistic_hmc_autoencoder = function(start, data, new_data, param, encoder,decoder, norm_param_autoencoder_input, autoencoder_weights, L, epsilon,max.iter){
  acc_sum = 0
  beta.vec = matrix(0, max.iter, m)

  beta.vec[1,] = start
  current_beta = beta.vec[1,]

  for (i in 2:max.iter){
    result = hmc_lr_autoencoding(U, gU_latent, gK_latent, encoder, decoder,norm_param_autoencoder_input, autoencoder_weights, epsilon, L, current_beta, data, new_data, param)
    beta.vec[i,] = result$sample
    current_beta = beta.vec[i,]
    acc_sum = acc_sum + result$acc
  }

  cat("Acceptance ratio: ", acc_sum/(max.iter-1))
  return(beta.vec)
}

tic("HMC with autoencoder sampling")
source("hmc_lr_autoencoding.R")
hmc_samples = logistic_hmc_autoencoder(start = next_hmc_init, data, new_data, param, encoder, decoder,norm_param_autoencoder_input = norm_param, autoencoder_weights, 10, 0.032, 2000)
toc()

saveRDS(hmc_samples, "data/samples_auto_encoding_hmc_lr_correction.rds")
ae_hmc_samples <- readRDS("data/samples_auto_encoding_hmc_lr_correction.rds")

# Plot and check sample autocorrelaiton
plot(ae_hmc_samples[,1], type = 'l')

#####################################################################
#                       Standard HMC                                #
#####################################################################
# Standard HMC distribution
source("hmc_lr.r")
logistic_hmc = function(start, data, param, L, epsilon,max.iter){
  acc_sum = 0
  beta.vec = matrix(0, max.iter, m)
  beta.vec[1,] = start
  current_beta = beta.vec[1,]
  comp_times = numeric(0)

  for (i in 2:max.iter){
    tic()
    result = hmc_lr(U, grad_U, epsilon, L, current_beta, data, param)
    cpu_time= toc()
    comp_times[i-1] = cpu_time$toc-cpu_time$tic
    beta.vec[i,] = result$sample
    current_beta = beta.vec[i,]
    acc_sum = acc_sum + result$acc
  }
  cat("Acceptance ratio: ", acc_sum/(max.iter-1))
  print(mean(comp_times))
  return(beta.vec)
}

tic("HMC sampling")
hmc_samples = logistic_hmc(start = next_hmc_init, data, param, 10, 0.5, 2000)
toc()

saveRDS(hmc_samples, "data/samples_hmc_lr.rds")
hmc_samples <- readRDS("data/samples_hmc_lr.rds")

# Plot and check sample autocorrelaiton
plot(hmc_samples[,1], type = 'l')
