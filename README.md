# Forest-fire-classification

## 1. Introduce

This project presents a program designed to detect and alert on the presence of wildfires by classifying images captured directly from cameras using deep learning models. To address this challenge, I compiled wildfire images from publicly available datasets and applied data augmentation techniques to create a comprehensive dataset consisting of 6,967 images. This dataset was then tested on deep learning models such as VGG16 and ResNet50 to evaluate their performance and classification speed in identifying whether the images contain signs of a wildfire.

## 2. Dataset

### 2.1. Dataset Compilation

The dataset used in this project is compiled from various wildfire data sources on Kaggle. It includes images captured from different angles, enabling the model to detect wildfires through two methods: (1) identifying flames or (2) detecting smoke from the fire. Specifically, the dataset is categorized into:

- **fire_images:** Images of wildfires and mountains where flames and/or smoke are present.

![](https://github.com/tnhi1821/Forest-fire-classification/blob/main/image%20source/fire.jpg)
  
- **non-fire_images:** Images of forests, mountains from various perspectives, as well as images of houses, interiors, and other objects without any signs of fire or smoke.

![](https://github.com/tnhi1821/Forest-fire-classification/blob/main/image%20source/non_fire.jpg)

### 2.2. Data Augmentation

To enhance the diversity and effectiveness of the training dataset, we applied data augmentation techniques using the ImageDataGenerator from the Keras library. These techniques include randomly rotating images from -30 to +30 degrees, randomly shifting horizontally and vertically within a range of -20% to +20%, resizing randomly from 80% to 120% of the original size, and randomly flipping horizontally.

![](https://github.com/tnhi1821/Forest-fire-classification/blob/main/image%20source/fire_augmentation.jpg)

![](https://github.com/tnhi1821/Forest-fire-classification/blob/main/image%20source/non_fire_augmentation.jpg)

### 2.3. Train/Test split

For model training, 80% of the dataset (equivalent to 5,574 images) was used for training, while the remaining 20% (equivalent to 1,393 images) was set aside for validation.

## 3. Method

### 3.1. VGG16

The VGG16 architecture, originally trained on the ImageNet dataset, produces outputs for 1,000 different labels. However, the project's objective requires predicting only two labels. Therefore, the team employed the pre-trained VGG16 model for feature extraction and then fine-tuned the model to meet the project’s specific needs.

![](https://github.com/tnhi1821/Forest-fire-classification/blob/main/image%20source/VGG16.jpg)

The model structure used includes:

1. **VGG16:** The VGG16 model from the Keras Applications library was applied. The output has a shape of (None, 7, 7, 512) with a total of 14,714,688 untrained parameters.
2. **Flatten:** This layer transforms the output from the VGG16 model into a 1D vector with a size of (None, 25,088).
3. **Dropout:** A Dropout layer with a rate of 0.2 is used to minimize the risk of overfitting.
4. **Dense:** A fully connected layer with 1 neuron and a sigmoid activation function. This is the output layer for binary classification of *fire_images* and *non_fire_images*.

The total number of parameters in the model is 14,739,777, of which only 25,089 parameters are trainable (belonging to the Dense layer). The remaining parameters were pre-trained on ImageNet and remain unchanged during the current training process.

### 3.2. ResNet50

ResNet has a more compact and less complex architecture compared to VGGNet. The ResNet architecture adheres to two fundamental principles: (1) the number of filters in each layer remains constant depending on the output feature map size, and (2) if the feature map size is reduced by half, the number of filters is doubled to maintain the complexity of each layer.

The key feature of ResNet is its ability to add more convolutional layers to the CNN without encountering the vanishing gradient problem, thanks to the concept of *skip connection*. The *skip connection* allows certain layers to be bypassed, transforming the standard network into a Residual Network.

![](https://github.com/tnhi1821/Forest-fire-classification/blob/main/image%20source/Resnet50.jpg)

Similar to VGG16, ResNet50, when trained on ImageNet, also produces outputs for 1,000 labels. Consequently, the team utilized the pre-trained ResNet50 model for feature extraction and then fine-tuned it to align with the project's requirements. This fine-tuning process is similar to that used with the VGG16 model.

## 4. Result
Both models yielded very promising results, with an accuracy on the validation dataset of approximately 0.99.
| Model      | Class               | Precision        | Recall          |
|------------|---------------------|------------------|-----------------|
| VGG16      | fire_images         | 0.99             | 0.98            |
|            | non_fire_images     | 0.98             | 0.99            |
|Resnet50    | fire_images         | 0.99             | 0.98            |
|            | non_fire_images     | 0.99             | 0.99            |

![](https://github.com/tnhi1821/Forest-fire-classification/blob/main/image%20source/fire_result.jpg)
![](https://github.com/tnhi1821/Forest-fire-classification/blob/main/image%20source/non_fire_result.jpg)

However, both models encountered difficulties when handling certain images that could easily be mistaken for wildfires, such as images of sunsets, sunrises, or forests in autumn.

![](https://github.com/tnhi1821/Forest-fire-classification/blob/main/image%20source/wrong_result.jpg)

To improve the model's performance, the following development directions can be considered:

- **Specialized Data Augmentation:** Create additional variations of confusing images by applying data augmentation techniques.

- **Incorporating Additional Features (Feature Engineering):** Integrate additional features such as color analysis, brightness, or spectral factors to help the model distinguish between natural light (sunset, sunrise) and light from fire.




