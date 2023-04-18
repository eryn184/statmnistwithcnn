# Fashion Mnist Basic CNN
# Nicole Tucker

library(keras)
library(tensorflow)
library(tidyverse)

fashion_mnist <- dataset_fashion_mnist()

train_images <- mnist$train$x
train_labels <- mnist$train$y
test_images <- mnist$test$x
test_labels <- mnist$test$y


class_names = c('T-shirt/top',
                'Trouser',
                'Pullover',
                'Dress',
                'Coat',
                'Sandal',
                'Shirt',
                'Sneaker',
                'Bag',
                'Ankle boot')

#Explore Data
dim(train_images) #60,000 images; 28x28 pixels
dim(test_images) #10,000 images; 28x28 pixels


#Inspect Image 1; Check for Neccessary Pre-Processing

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


#We observe the values range from 0 to 250
#Want: Binary- 0 to 1

train_images <- train_images / 255
test_images <- test_images / 255

#View first 25 images
par(mfcol=c(5,5))
par(mar=c(0, 0, 1.5, 0), xaxs='i', yaxs='i')
for (i in 1:25) {
  img <- train_images[i, , ]
  img <- t(apply(img, 2, rev))
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
        main = paste(class_names[train_labels[i] + 1]))
}


#Set Up Layers of the Model [First Attempt] [This is the part that we can manipulate throughout project]

model <- keras_model_sequential()
model %>%
  layer_flatten(input_shape = c(28, 28)) %>% #transforms data format only
  #transforms the format of the images from a 2d-array (of 28 by 28 pixels), to a 1d-array of 28 * 28 = 784 pixels
  layer_dense(units = 128, activation = 'relu') %>% #128 neurons
  layer_dense(units = 10, activation = 'softmax') #returns an array of 10 probability scores that sum to 1


#Compile Model [Also could be changed- optimizer, loss, metrics]
model %>% compile(
  optimizer = 'adam',
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
)


#Train the Model
#We can change the epochs and verbose
model %>% fit(train_images, train_labels, epochs = 5, verbose = 2)


#Evaluate on Test Data
score <- model %>% evaluate(test_images, test_labels, verbose = 0)

cat('Test loss:', score["loss"], "\n") #0.3368211
cat('Test accuracy:', score["accuracy"], "\n") #0.8804

metrics <- network %>% evaluate(test_images, test_labels, verbose = 0)
metrics

Error <- 1 - metrics[2]
Error


#Look at predictions
predictions <- model %>% predict(test_images)


#check out a prediction
predictions[1, ]
#A prediction is comprised of 10 values because the last layer has 10 neurons
#a confidence value for each of ten fashion items
which.max(predictions[1, ])
test_labels[1] #test labels are zero based so add one to this result

predictions[5, ]
#A prediction is comprised of 10 values because the last layer has 10 neurons
#a confidence value for each of ten fashion items 
which.max(predictions[5, ])
test_labels[5] #test labels are zero based so add one to this result
