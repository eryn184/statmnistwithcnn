# MNIST Digits Recognition with CNN
# Nicole Tucker

library(keras)
library(tensorflow)
library(tidyverse)


mnist <- dataset_mnist()

train_images <- mnist$train$x
train_labels <- mnist$train$y
test_images <- mnist$test$x
test_labels <- mnist$test$y

dim(train_images) #60,000 images; 28x28 pixels
dim(test_images) #10,000 images; 28x28 pixels

#View First Image
image_1 <- as.data.frame(train_images[1, , ])
colnames(image_1) <- seq_len(ncol(image_1))
image_1$y <- seq_len(nrow(image_1))
image_1 <- gather(image_1, "x", "value", -y)
image_1$x <- as.integer(image_1$x)

ggplot(image_1, aes(x = x, y = y, fill = value)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "black", na.value = NA) +
  scale_y_reverse() +
  theme_minimal() +
  theme(panel.grid = element_blank())   +
  theme(aspect.ratio = 1) +
  xlab("") +
  ylab("")

#Pre-Processing
train_images <- array_reshape(train_images, c(60000, 28 * 28))
train_images <- train_images / 255
test_images <- array_reshape(test_images, c(10000, 28 * 28))
test_images <- test_images / 255
train_labels <- to_categorical(train_labels)
test_labels <- to_categorical(test_labels)


#Set Up Neural Network Layers
network <- keras_model_sequential() %>% 
  layer_dense(units = 512, activation = "relu", input_shape = c(28 * 28)) %>% 
  layer_dense(units = 10, activation = "softmax")


#Compile Network
network %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)


#Train Network
network %>% fit(train_images, train_labels, epochs = 5, batch_size = 128)


metrics <- network %>% evaluate(test_images, test_labels, verbose = 0)
metrics

Error <- 1 - metrics[2]
Error

#Loss: 0.06359591
#Accuracy: 0.98049998
#Error: 0.01950002; 1.95%
