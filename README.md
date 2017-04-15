# GenerativeFaceCompletion
Matcaffe implementation for our CVPR17 [paper](https://drive.google.com/file/d/0B8_MZ8a8aoSeMjRFM2VYdVR4Q1U/view) on face completion

## Setup

## Training
- Follow the [DCGAN](https://github.com/soumith/dcgan.torch) framework to prepare the CelebA dataset for training. The only differece is that the face we cropped is of size 128. Please modify their [crop_celebA.lua](https://github.com/soumith/dcgan.torch/blob/master/data/crop_celebA.lua) file.


- Download our face parsing model [Model_parsing](https://drive.google.com/open?id=0B8_MZ8a8aoSeaXlUR296TzM2NW8) and put it under ./matlab/FaceCompletion_training/model/ 

## Testing
- Download our face completion model [Model_G](https://drive.google.com/open?id=0B8_MZ8a8aoSeQlNwY2pkRkVIVmM) and put it under ./matlab/FaceCompletion_testing/model/ 
- Run ./matlab/FaceCompletion_testing/demo_face128.m

## Citation
If you use this code for your research, please cite our paper <a href="https://drive.google.com/file/d/0B8_MZ8a8aoSeMjRFM2VYdVR4Q1U/view">Generative Face Completion</a>:

```
@inproceedings{GFC-CVPR-2017,
    author = {Li, Yijun and Liu, Sifei and Yang, Jimei and Yang, Ming-Hsuan},
    title = {Generative Face Completion},
    booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
    year = {2017}
}
```

## Acknowledgement
Gratitude goes to [Sifei Liu](https://github.com/Liusifei) for the great help on code.
