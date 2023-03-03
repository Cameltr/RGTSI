# Reference-Guided Texture and Structure Inference for Image Inpainting
![visitors](https://visitor-badge.glitch.me/badge?page_id=Cameltr/RGTSI)


 This is the repository of the paper **Reference-Guided Texture and Structure Inference for Image Inpainting**, accepted by [ICIP 2022](https://2022.ieeeicip.org/).
 
 > **Abstract:** *Existing learning-based image inpainting methods are still in challenge when facing complex semantic environments and diverse hole patterns. The prior information learned from the large scale training data is still insufficient for these situations. Reference images captured covering the same scenes share similar texture and structure priors with the corrupted images, which offers new prospects for the image inpainting tasks. Inspired by this, we first build a benchmark dataset containing 10K pairs of input and reference images for reference-guided inpainting. Then we adopt an encoder-decoder structure to separately infer the texture and structure features of the input image considering their pattern discrepancy of texture and structure during inpainting. A feature alignment module is further designed to refine these features of the input image with the guidance of a reference image. Both quantitative and qualitative evaluations demonstrate the superiority of our method over the state-of-the-art methods in terms of completing complex holes.* 

![](./imgs/pipeline.png)


## Usage Instructions

### Environment
Please install Anaconda, Pytorch. For other libs, please refer to the file requirements.txt.
```
git clone https://github.com/Cameltr/RGTSI.git
conda create -n RGTSI python=3.8
conda activate RGTSI
pip install -r requirements.txt
```

## Dataset Preparation

Please download **DPED10K** dataset from [Google Netdisk](https://pan.baidu.com/s/17HmDXmStYRhAErpYjLFkJA),

Our model is trained on the irregular mask dataset provided by [Liu et al](https://arxiv.org/abs/1804.07723). You can download publically available Irregular Mask Dataset from their [website](http://masc.cs.gmu.edu/wiki/partialconv).

For Structure image of datasets, we follow the [Structure flow](https://github.com/RenYurui/StructureFlow) and utlize the [RTV smooth method](http://www.cse.cuhk.edu.hk/~leojia/projects/texturesep/).Run generation function [data/Matlab/generate_structre_images.m](./data/Matlab/generate_structure_images.m) in your matlab. For example, if you want to generate smooth images for Places2, you can run the following code:

```
generate_structure_images("path to dataset root", "path to output folder");
```


## Training New Models
```bash
# To train on the you dataset, for example.
python train.py --st_root=[the path of structure images] --de_root=[the path of ground truth images] --mask_root=[the path of mask images]
```
There are many options you can specify. Please use `python train.py --help` or see the options

For the current version, the batchsize needs to be set to 1.

To log training, use `--./logs` for Tensorboard. The logs are stored at `logs/[name]`.

## Pre-trained weights and test model
 You can download the pre-trained model [here](https://drive.google.com/drive/folders/1uLC9YN_34mLod5kIE1nMb9P5L40Iqbkp?usp=sharing)，and unzip it to `checkpoints`

## Thanks

## Citation
If you use this code for your research, please cite our papers.
```
@inproceedings{RGTSI,
  title={Reference-guided texture and structure inference for image inpainting},
  author={Liu, Taorong and Liao, Liang and Wang, Zheng and Satoh, Shin’Ichi},
  booktitle={Proc. IEEE Int. Conf. Image Process.},
  pages={1996--2000},
  year={2022},
}
```