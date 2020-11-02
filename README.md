# CVPR 2018: What do deep Networks like to See?

Implementation from the CVPR 2018 paper _"What do Deep Networks Like to See?"_.

This is a simple proof of concept that uses a fine-tuned autoencoder on ResNet50 to reconstruct input images.

1. Download the Torch model [from here](https://cloud.dfki.de/owncloud/index.php/s/3fdqbLXcEx7kEaK) and store it with the root directory of the repo.
2. Call 
```
python plot_ae_reconstruction.py -i PATH
```
where ```PATH``` is the path to the input image.
3. An output image should be saved in the root directory of the repo with the reconstruction.

### Example:
![Original input and its reconstruction using an autoencoder fine-tuned on ResNet 50](./lenas.png)

#### For more info, please check out [the paper's website](https://spalaciob.github.io/s2snets.html).

### UPDATES:
#### 02.11.2020
 - Weights for the original SegNet (pre-trained on YFCC100m) are now [available here](https://cloud.dfki.de/owncloud/index.php/s/ccSAQnxjZS384p6) and can be used by `plot_ae_reconstruction.py`. Make sure the path is correctly loaded by modifying the global variable [`RESNET_PATH`](https://github.com/spalaciob/s2snets-reconstruction/blob/master/plot_ae_reconstruction.py#L24).
