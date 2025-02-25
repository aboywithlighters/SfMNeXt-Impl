# SfMNeXt: The NeXt Series of Learning Structure Prior from Motion

## 👩‍⚖️ Demo

Online demo is available at [HERE](http://cn-nd-plc-1.openfrp.top:56789/)

## 👀 Training

To train on KITTI, run:

```bash
python train.py ./args_files/args_res50_kitti_192x640_train.txt
```
For instructions on downloading the KITTI dataset, see [Monodepth2](https://github.com/nianticlabs/monodepth2)

To train on CityScapes, run:

```bash
python train.py ./args_files/args_cityscapes_train.txt
```
To finetune on CityScapes, run:

```bash
python train.py ./args_files/args_cityscapes_finetune.txt
```

For preparing cityscapes dataset, please refer to SfMLearner's [prepare_train_data.py](https://github.com/tinghuiz/SfMLearner/blob/master/data/prepare_train_data.py) script.
We used the following command:

```bash
python prepare_train_data.py \
    --img_height 512 \
    --img_width 1024 \
    --dataset_dir <path_to_downloaded_cityscapes_data> \
    --dataset_name cityscapes \
    --dump_root <your_preprocessed_cityscapes_path> \
    --seq_length 3 \
    --num_threads 8
```

## 💾 Pretrained weights and evaluation

You can download weights for some pretrained models here:

| Methods |WxH|abs rel| RMSE |
| :----------- | :---: | :-----: | :---: |
[KITTI (ResNet-50)](https://drive.google.com/file/d/1_BHfGXqVsE4RrCM58_5nEzQHJmyxDsQO/view?usp=drive_link)|640x192|0.088|4.175|
[KITTI (ResNet-50)](https://drive.google.com/file/d/1NaN8F3gPU2vY38KtYFDSAw4Hbye5Z5AG/view?usp=drive_link)|1024x320|0.082|3.914|
[CityScapes (ResNet-50)](https://drive.google.com/file/d/1nLwTQnXV_9IURUqfCfoGZyVHb4U5XwYI/view?usp=sharing)|512x192|0.106|6.237|
[KITTI (ConvNeXt-L)](https://drive.google.com/file/d/14Hb8UsjraLWk1TvtMPd5moEDMlDxiHkC/view?usp=drive_link)|1024x320|0.043|1.698|

To evaluate a model on KITTI, run:

```bash
python evaluate_depth_config.py args_files/hisfog/kitti/resnet_320x1024.txt
```

Make sure you have first run `export_gt_depth.py` to extract ground truth files.

And to evaluate a model on Cityscapes, run:

```bash
python ./tools/evaluate_depth_cityscapes_config.py args_files/args_res50_cityscapes_finetune_192x640_eval.txt
```

The ground truth depth files can be found at [HERE](https://storage.googleapis.com/niantic-lon-static/research/manydepth/gt_depths_cityscapes.zip),
Download this and unzip into `splits/cityscapes`.

## 🖼 Inference with your own iamges

```bash
python test_simple_SQL_config.py ./args_files/args_test_simple_kitti_320x1024.txt
```
## Future Works

- [x] release code for training in outdoor scenes (KITTI, Cityscapes)
- [x] model release (KITTI, Cityscapes)
- [x] code for training in indoor scenes (NYU-Depth-v2, MannequinChallenge)
- [x] code for finetuning self-supervised model using metric depth
- [ ] model release for indoor scenes and metric fine-tuned model

## License

All rights reserved.
Please see the [license file](LICENSE) for terms.
