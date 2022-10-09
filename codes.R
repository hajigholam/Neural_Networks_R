
# Neural Networks and Deep Learners

# Amirhossein H. Saryazdi
# Mahdi Javadi
# Hamed Rismanian

# 25/04/2021



# 5.1 FNN Analysis
# inserting data for regression
set.seed(12345)
library(readr)
library(dplyr)

#inserting data
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data'

yacht = read_table(file = url,
                   col_names = c('LongPos_COB', 'Prismatic_Coeff',
                                 'Len_Disp_Ratio', 'Beam_Draut_Ratio', 
                                 'Length_Beam_Ratio','Froude_Num', 
                                 'Residuary_Resist')) %>%
  na.omit()

# Scale the Data
scale01 = function(x){
  (x - min(x)) / (max(x) - min(x))
}

yacht = yacht %>%
  mutate_all(scale01)

# Split into test and train sets, 80% of the observations for training
set.seed(12345)
yacht_train = sample_frac(tbl = yacht, replace = FALSE, size = 0.80)
yacht_test = anti_join(yacht, yacht_train)

yacht_train_x=yacht_train
yacht_train_x$Residuary_Resist=NULL
yacht_train_x=as.matrix(yacht_train_x)

yacht_test_x=yacht_test
yacht_test_x$Residuary_Resist=NULL
yacht_test_x=as.matrix(yacht_test_x)


# 5.1.1 neuralnet package, Regression
# Uncomment for installing the following package on your machine
#install.packages('neuralnet')
library(neuralnet)

# Training a FNN-1
set.seed(12345)
neuralnet_NN1 = neuralnet(Residuary_Resist ~ LongPos_COB + 
                            Prismatic_Coeff + Len_Disp_Ratio + 
                            Beam_Draut_Ratio + Length_Beam_Ratio +
                            Froude_Num, data = yacht_train,
                          hidden = 1, act.fct = "logistic",
                          err.fct = "sse", learningrate=0.001,
                          algorithm='backprop')
plot(neuralnet_NN1, rep = 'best')


# predidct FNN-1
set.seed(12345)
neuralnet_NN1_pred = compute(neuralnet_NN1, yacht_test[, 1:6])$net.result
neuralnet_NN1_test_SSE = sum((neuralnet_NN1_pred - yacht_test[, 7])^2)/2
neuralnet_NN1_test_SSE


# 2-Hidden Layers, Layer-1 3-neurons, Layer-2, 2-neuron
set.seed(12345)
neuralnet_NN2 = neuralnet(Residuary_Resist ~ LongPos_COB + 
                            Prismatic_Coeff + Len_Disp_Ratio + 
                            Beam_Draut_Ratio + Length_Beam_Ratio +
                            Froude_Num, data = yacht_train,
                          hidden = c(3,2), act.fct = "logistic",
                          err.fct = "sse", learningrate=0.001,
                          algorithm='backprop')

# Train SSE
neuralnet_NN2_train_SSE <- sum((neuralnet_NN2$net.result - yacht_train[, 7])^2)/2
paste("Train SSE: ", round(neuralnet_NN2_train_SSE, 4))

# predidct FNN-2
set.seed(12345)
neuralnet_NN2_pred = compute(neuralnet_NN2, yacht_test[, 1:6])$net.result
neuralnet_NN2_test_SSE = sum((neuralnet_NN2_pred - yacht_test[, 7])^2)/2
paste("Test SSE: ", round(neuralnet_NN2_test_SSE, 4))



# 5.1.2 h2o package, Classification
# Importing  dataset for classification
dataset = read.csv("C:/Users/canbec/Documents/Cours BI/Advanced statistical learning/Final project/Churn.csv")
dataset = dataset[4:14]

# Encoding the categorical variables as factors
dataset$Geography = as.numeric(factor(dataset$Geography,
                                      levels = c('France', 'Spain', 'Germany'),
                                      labels = c(1, 2, 3)))
dataset$Gender = as.numeric(factor(dataset$Gender,
                                   levels = c('Female', 'Male'),
                                   labels = c(1, 2)))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(12345)
split = sample.split(dataset$Exited, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[-11] = scale(training_set[-11])
test_set[-11] = scale(test_set[-11])

# Fitting ANN to the Training set
# install.packages('h2o')
library(h2o)

set.seed(12345)
h2o.init()
h2o_NN1 = h2o.deeplearning(y = 'Exited',
                           training_frame = as.h2o(training_set),
                           activation = 'Rectifier',
                           hidden = c(4,2),
                           epochs = 10)

# Predicting the Test set results
y_pred = h2o.predict(h2o_NN1, newdata = as.h2o(test_set[-11]))
# using 0.5 threshold
y_pred = (y_pred > 0.5)
y_pred = as.vector(y_pred)

# Making the Confusion Matrix
cm1 = table(test_set[, 11], y_pred)

cm1
h2o.shutdown()



h2o_NN2 = h2o.deeplearning(y = 'Exited',
                           training_frame = as.h2o(training_set),
                           activation = 'Rectifier',
                           hidden = c(4,2),
                           epochs = 100)

# Predicting the Test set results
y_pred = h2o.predict(h2o_NN2, newdata = as.h2o(test_set[-11]))
# using 0.5 threshold
y_pred = (y_pred > 0.5)
y_pred = as.vector(y_pred)

# Making the Confusion Matrix
cm2 = table(test_set[, 11], y_pred)

cm2

h2o.shutdown()


# 5.1.3 deepnet package, Classification
# preparing data
churn_train_x=training_set
churn_train_x$Exited=NULL
churn_train_x=as.matrix(churn_train_x)

churn_test_x=test_set
churn_test_x$Exited=NULL
churn_test_x=as.matrix(churn_test_x)

#install.packages("deepnet")
library(deepnet)

set.seed(12345)
deepnet_NN1 = nn.train(churn_train_x, training_set$Exited,hidden = c(4,2), learningrate = 0.001, momentum = 0.5, activationfun = "sigm", numepochs = 100, batchsize = 100, hidden_dropout=0)

# Test new samples by Trainded NN,return error rate for classification
deepnet_NN1_error=nn.test(deepnet_NN1,churn_test_x, test_set$Exited, t=0.5)

#classification error
deepnet_NN1_error


# 5.1.4 keras package, Regression

#install.packages("keras")
set.seed(12345)
library(keras)

#install_keras()
# Building network
set.seed(12345)
keras_NN1 = keras_model_sequential() %>%
  layer_dense(units = 3, activation = "relu", input_shape =ncol(yacht_train_x))%>%
  layer_dense(units = 2,  activation = "relu",) %>%
  layer_dense(units = 1)

# compiling required hyperparameters
set.seed(12345)
keras_NN1 %>% compile(
  optimizer = "rmsprop",
  loss = "mse",
  metrics = c("mae")
)

# train model
set.seed(12345)
history <- keras_NN1 %>% fit(
  x = yacht_train_x,
  y = yacht_train$Residuary_Resist,
  epochs = 10,
  batch_size = 5,
  validation_split = .2,
  verbose = FALSE
)

history


keras_NN1 %>% predict(yacht_test_x)


results <- keras_NN1 %>% evaluate(yacht_test_x, yacht_test$Residuary_Resist)

results

# 5.2

# 5.2.1) Prediction on a generated dataset usuing Recurrent Neural Network with
# package "rnn"
# install.packages("rnn")
rm(list=ls())

require(rnn)
(.packages())
set.seed(1234)


T <- seq(0,20,length=400)
X <- 5 + 5*cos(4*T+2) +.2*T^2 + rnorm(400)
Y <- 5*sin(4*T+2) +.2*T^2 
plot(T,X,type="l")
plot(T,Y,type="l")



X <- matrix(X, nrow = 40)
Y <- matrix(Y, nrow = 40)


plot(as.vector(X), col='deepskyblue1', type='l', ylab = "X,Y", main = "Noisy data")
lines(as.vector(Y), col = "deeppink1")
legend("topright", c("X", "Y"), col = c("deepskyblue1","deeppink1"), lty = c(1,1), lwd = c(1,1))


X <- (X - min(X)) / (max(X) - min(X))
Y <- (Y - min(Y)) / (max(Y) - min(Y))


# Transposing the matrix
X <- t(X)
Y <- t(Y)


# Splitting the data to Train and Test sets
train <- 1:8
test <- 9:10


model <- trainr(Y = Y[train,],
                X = X[train,],
                learningrate = 0.02,
                hidden_dim = 16,
                numepochs = 1500,network_type='rnn',use_bias = T, update_rule = "sgd")


pred <- predictr(model, X)


# Plotting the predicted vs actual values. Both Train and Test set
plot(as.vector(t(Y)), col = 'deeppink1', type = 'l', main = "Actual vs Predicted Values", ylab = "Y,pred")
lines(as.vector(t(pred)), type = 'l', col = 'deepskyblue1')
legend("topright", c("Predicted", "Real"), col = c("deepskyblue1","deeppink1"), lty = c(1,1), lwd = c(1,1))


# Plotting the predicted vs actual values. This time the Test set only.
plot(as.vector(t(Y[test,])), col = 'deeppink1', type='l', main = "Actual vs Predicted: Test set", ylab = "Y,pred")
lines(as.vector(t(pred[test,])), type = 'l', col = 'deepskyblue1')
legend("topleft", c("Predicted", "Actual"), col = c("deepskyblue1","deeppink1"), lty = c(1,1), lwd = c(1,1))


(.packages())

#install.packages("NCmisc")
library(NCmisc)
list.functions.in.file("FilePath/Filename.R")

library("rnn")
run.rnn_demo(port = NULL)


# 5.2.2) News classification (Reutrers Data set) usuing LSTM with package "keras"
# install.packages("keras")
require(keras)

max_features <- 20000
maxlen <- 100
batch_size <- 32


reuters <- dataset_reuters(num_words = max_features)


x_train <- reuters$train$x
y_train <- reuters$train$y
y_train <- to_categorical(y_train)
x_test <- reuters$test$x
y_test <- reuters$test$y
y_test <- to_categorical(y_test)

cat(length(x_train), 'Sequences in the Train set\n')
cat(length(x_test), 'Sequences in the Test set\n')


# Pad training and test inputs
x_train <- pad_sequences(x_train, maxlen = maxlen)
x_test <- pad_sequences(x_test, maxlen = maxlen)


# Initialize model
model <- keras_model_sequential()
model %>%
  layer_embedding(input_dim = max_features, 
                  output_dim = 256, 
                  input_length = maxlen) %>% 
  bidirectional(layer_lstm(units = 128)) %>%
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 46, activation = 'softmax')

model


model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)


cat('Train...\n')
model %>% fit(
  x_train, y_train,
  batch_size = batch_size,
  epochs = 8,
  verbose = 1,
  validation_split = 0.1
  
)



pred <- model %>% predict(x_test, batch_size = batch_size) 

score <- model %>% evaluate(
  pred, y_test,
  batch_size = batch_size,
  verbose = 1
)


(.packages())


# 5.3 Image classification with convolutional neural networks
library(keras)

# Download data set:
data = dataset_cifar10()


# 5.3.2 Model's architecture
# Define model's architecture:
CNN = keras_model_sequential()

CNN %>%
  layer_conv_2d(filters = 50, kernel_size = c(3,3), activation = "relu",
                input_shape = c(32,32,3)) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 60, kernel_size = c(2,2), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units = 65, activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")

summary(CNN)


# Loss function, optimization method, and evaluation metric:
CNN %>% compile(
  optimizer = 'adam',
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy'))


# 5.3.3 Train the model
# Train the model:
set.seed(112497)
CNN_fit = CNN %>%
  fit(
    x = data$train$x, y = data$train$y,
    epochs = 15,
    validation_data = unname(data$test),
    verbose = 2)


# 5.3.4 Evaluate the model
# Plot the history of the model train:
plot(CNN_fit)

