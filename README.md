# An Exercise in Transfer Learning

Hypothesis: Intermediate representations of images from vision models pre-trained on broad tasks (ImageNet in particular) can be used as input to simpler ML models with reasonable success on other broad object recognition tasks (CIFAR100 in particular).

This is a hypothesis whose truth has repercussions on a larger research project I'm working on, so it was tested only to the point where I was informally satisfied with the light shed on the idea. Surely much better research exists on this topic, but I wanted to try out some transfer learning for myself.

To test this hypothesis, I created a dataset of $(x_i, y_i)$ pairs where $x_i$ is the activation of the penultimate layer of a ResNet50 model (pre-trained on ImageNet) after a forward pass on a sample from CIFAR100, and $y_i$ is the CIFAR100 label corresponding to said sample. If the hypothesis is true, simpler ML models should be able to achieve some not-terrible accuracy level on this dataset.

### Results
