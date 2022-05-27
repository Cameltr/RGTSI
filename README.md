# RGTSI
 This is the repository of the paper **Reference-Guided Texture and Structure Inference for Image Inpainting**, submitted at [ICIP 2022](https://2022.ieeeicip.org/).
 

## Pipeline

![](./imgs/pipeline.png)
The overview of the proposed pipeline. We adopt a referenced-based encoder-decoder to jointly fill image holes. The features of the input and reference images are aligned and fused by the Feature Alignment Module (FAM).We recover the holes in multi-scale within the aligned features and equalize the output features. The equalized features contain consistent structure
and texture features and are supplemented to the decoder by skip connections to generate the output image.

