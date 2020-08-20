# Udacity Computer Vision ND Image Captioning Project

Image Captioning is the process of generating textual description of an image. In this project, I have implemented
a Deep Learning Model inspired by [this paper](https://arxiv.org/pdf/1502.03044.pdf) and [this paper](https://arxiv.org/pdf/1411.4555.pdf) using [COCO data](http://cocodataset.org/) set by Microsoft and trained the network for nearly 10 hrs using GPU.
<p align="center">
  <img src="./images/encoder-decoder.png" width=100% height=100% />
</p>

The architecture consists of:
1. CNN based on the ResNet architecture encoder, which encodes the images into the embedded feature vectors
<p align="center">
  <img src="./images/encoder.png" width=100% height=100% />
</p>
2. RNN decoder consisting of LSTM units, which translates the feature vector into a sequence of tokens 
<p align="center">
  <img src="./images/decoder.png" width=100% height=100% />
</p>

## Output results
<p align="center">
  <img src="./images/sample image4.bmp" width=50% height=50% />
  <img src="./images/sample image8.bmp" width=50% height=50% />
  <img src="./images/sample image6.bmp" width=50% height=50% />
  <img src="./images/sample image3.bmp" width=50% height=50% />
  <img src="./images/sample image7.bmp" width=50% height=50% />
  <img src="./images/sample image2.bmp" width=50% height=50% />
  <img src="./images/sample image.bmp" width=50% height=50% />
  <img src="./images/sample image5.bmp" width=50% height=50% />
</p>
