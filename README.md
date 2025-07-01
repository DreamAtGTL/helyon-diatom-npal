# HELIYON Research Article: 
The code in this repository suppport the following article published in the HELIYON journal

*Title*: Long-term effects of the herbicide glyphosate and its main metabolite (aminomethylphosphonic acid) on the growth, chlorophyll a and morphology of freshwater benthic diatoms 

*Authors*: Sarah Chéron, Vincent Felten, Aishwarya Venkataramanan, Carlos Eduardo Wetzel, David Heudre, Cédric Pradalier, Philippe Usseglio-Polatera, Simon Devin, Martin Laviale.

*Year*: 2025.

The main author of this code is [Aishwarya Venkataramanan](https://scholar.google.com/citations?user=IqlI43cAAAAJ&hl=en&oi=sra).


This code is based on DeepLabV3 and DeepLabV3+ with MobileNetv2 and ResNet backbones for Pytorch.

The code for segmentation is borrowed from https://github.com/VainF/DeepLabV3Plus-Pytorch. 



### Morphometric parameters extraction
The code for automatically extracting the individual diatoms is available in `automatic_diatom_extraction.py`. It does instance segmentation by combining the predictions from both semantic segmentation and rotated bounding box detection. It requires path to 3 folders: 
1. image_path - This is the path to the full diatom microscopy images
2. seg_mask_path - This is the path to the segmentation masks. It assumes a binary mask (diatom and non-diatom).
3. rbb_labels_path - This is the path to the labels for the rotated bounding box detection. The label files are obtained from the Rotated YOLOv5 code (https://github.com/XinzeLee/RotateObjectDetection).

Note that sometimes the masks or the bounding box labels can be incorrect which can result in incorrectly extracted diatom images. So a filtering is done based on the eccentricity values of the indiavidiual diatom masks. 

Finally, the images are saved in the folder `extracted_thumbnails`.

The code for comparing the automatically extracted morphological descriptors with the ground truth labelled ones is provided in `thumbnail_comparison.py`. The comparison is performed on 5 descriptors: length, width, area, perimeter and roundness.


## Training instructions for semantic segmentation

#### Available Architectures
Specify the model architecture with '--model ARCH_NAME' and set the output stride with '--output_stride OUTPUT_STRIDE'.

| DeepLabV3    |  DeepLabV3+        |
| :---: | :---:     |
|deeplabv3_resnet50|deeplabv3plus_resnet50|
|deeplabv3_resnet101|deeplabv3plus_resnet101|
|deeplabv3_mobilenet|deeplabv3plus_mobilenet |

All available pretrained models: [Dropbox](https://www.dropbox.com/sh/w3z9z8lqpi8b2w7/AAB0vkl4F5vy6HdIhmRCTKHSa?dl=0)

Load the pretrained model:
```python
model.load_state_dict( torch.load( CKPT_PATH )['model_state']  )
```

#### Atrous Separable Convolution
Atrous Separable Convolution is supported in this repo. We provide a simple tool ``network.convert_to_separable_conv`` to convert ``nn.Conv2d`` to ``AtrousSeparableConvolution``. **Please run main.py with '--separable_conv' if it is required**. See 'main.py' and 'network/_deeplab.py' for more details. 



## Quick Start

### 1. Requirements

```bash
pip install -r requirements.txt
```

### 2. Prepare Datasets

#### diatom

Download the images and the binary segmentation masks for the diatoms.

#### Visualize training (Optional)

Start visdom sever for visualization. Please remove '--enable_vis' if visualization is not needed. 

```bash
# Run visdom server on port 28333
visdom -port 28333
```

#### Train with OS=16

Run main.py with *"--year 2012_aug"* to train your model on Pascal VOC2012 Aug. You can also parallel your training on 4 GPUs with '--gpu_id 0,1,2,3'

```bash
python main.py --model deeplabv3plus_mobilenet --enable_vis --vis_port 28333 --gpu_id 0 --year 2012_aug --crop_val --lr 0.01 --crop_size 513 --batch_size 16 --output_stride 16
```

#### Continue training

Run main.py with '--continue_training' to restore the state_dict of optimizer and scheduler from YOUR_CKPT.

```bash
python main.py ... --ckpt YOUR_CKPT --continue_training
```

### 4. Test

Results will be saved at ./results.

```bash
python main.py --model deeplabv3plus_mobilenet --enable_vis --vis_port 28333 --gpu_id 0 --year 2012_aug --crop_val --lr 0.01 --crop_size 513 --batch_size 16 --output_stride 16 --ckpt checkpoints/best_deeplabv3plus_mobilenet_voc_os16.pth --test_only --save_val_results
```

## Reference

[1] [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)

[2] [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)
