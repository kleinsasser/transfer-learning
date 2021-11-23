# An Exercise in Transfer Learning

Hypothesis: Intermediate representations of images from vision models pre-trained on broad tasks (ImageNet in particular) can be used as input to simpler ML models with reasonable success on other broad object recognition tasks (CIFAR100 in particular).

This is a hypothesis whose truth has repercussions on a larger research project I'm working on (github.com/kleinsasser/cortex), so it was tested only to the point where I was informally satisfied with the light shed on the idea. Surely much better research exists on this topic, but I wanted to try out some transfer learning for myself.

To test this hypothesis, I created a dataset of $(x_i, y_i)$ pairs where $x_i$ is the activations of the penultimate layers of either a ResNet50 model and an EfficientNetB5 model (both pre-trained on ImageNet) after a forward pass on a resized sample from CIFAR100, and $y_i$ is the CIFAR100 label corresponding to said sample. If the hypothesis is true, simpler ML models should be able to achieve some not-terrible accuracy level on this dataset.

### Some Baseline Statistics

Here are how each pre-trained model performs on ImageNet (the dataset they were trained on).

|Model|Params|Top1|Top5|
|---|---|---|---|
|ResNet50|26M|76.0|93.0|
|EfficientNetB5|30M|83.6|96.7|

These models in particular were chosen mainly because they weren't too large to work with locally, and each is a fairly common choice for transfer learning applications.

### Results

I tested the performance of several popular classical ML classification models, as well as a single-layer neural network to include the baseline transfer learning procedure. The other models tested are:

Naive Bayes Classifier
K Nearest Neighbor Classifier
Support Vector Machine

#### Models Trained on ResNet50 Representations

|Model|Top1|Top5|Top10|
|---|---|---|---|
|Naive Bayes|0.313|0.584|0.701|
|KNN|0.243|0.474|0.589|
|SVM|0.142|NA|NA|
|Neural Net|0.635|0.889|0.945|

#### Models Trained on EfficientNetB5 Representations

|Model|Top1|Top5|Top10|
|---|---|---|---|
|Naive Bayes|0.097|0.261|0.375|
|KNN|0.036|0.130|0.217|
|SVM|0.085|NA|NA|
|Neural Net|0.071|0.218|0.331|

## Sources
torchvision models reference [https://pytorch.org/vision/stable/models.html]
ResNet Paper [https://arxiv.org/abs/1512.03385]
EfficientNet Paper [https://arxiv.org/abs/1905.11946]