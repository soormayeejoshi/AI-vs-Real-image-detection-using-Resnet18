# AI-vs-Real-image-detection-using-Resnet18
Binary classifier distinguishing real vs AI-generated images using the CIFAKE dataset (60,000 CIFAR-10 real + 60,000 AI-generated equivalents). Uses pretrained ResNet18 with all base layers frozen; only the custom fc classification head is retrained. Includes Grad-CAM for interpretability and Gaussian noise robustness testing.
