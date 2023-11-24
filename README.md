# wsi_seg

### Environment:
- Cone this repo
- Create Python 3.10.13 environment
  ```
  conda create --name wsi_seg python=3.10
  conda activate wsi_seg
  ```
- Install Dependencies
  ```
  pip install -r requirements.txt
  ```
- Copy the Dataset directory in the same folder and then change variables in config and constants file as per need.

### Work directory Explaination:
- **config.json**: Stores params, hyper-params and other configs for the whole pipeline. Update this file accordingly before running your experiment.
- **constants.py**: Stores most of the constant values for this problem statement. Change the experiment name directory accordingly.
- **data.py**: Dataset and data loader definition.
- **main.py**: main script to run the training, evaluation and inference.
- **model.py**: Model architecture and compilation methods.
- **pre_process.py**: Scripts to extract patches from WSI images.
- **requirements.txt**: Lists packages/dependencies names for the environment.
- **utils.py**: Contains Various utility files.

### Approach:
- Since WSI are huge in dimension so I have used *patchify* to create the patches of smaller size.
- The train and validation images and masks were splitted into smaller patches. The validation unpatched images were also used for prediction in the last based on best model.
- I have used pytorch and segmentation models library to train different models for this problem statement.

### Steps:
- Create smaller patches. Update patch size in the config. I have tried with patch size of 256 and 512.
  ```
  python pre_process.py
  ```
- Run main script to perform training, evaluation and inference. This script creates a experiment directory based on the name specified in the constants.py file and then trains the models based on the params and hyper-params specified in the config.json file. This will save the latest and best model weights and also save the model predictions on two valid images provided.
  ```
  python main.py
  ```

### Results:
- WandB Project link for accuracy and loss curves: https://wandb.ai/shubh19/exp?workspace=user-shubh19
- Google Drive link for model weights, logs and Predictions: https://drive.google.com/drive/folders/1rPszBQuqHpFWpIAYKM0o6rJPHwdkPClc?usp=sharing
  
- UNet with Resnet34 with LR scheduler and patch size = 512
    ```
    Running Prediction for Test/Image/7105b8ee8d4c00c513b334fdfdcd6c49.png
    Dice coefficient for class Black: 0.9930900643150197
    Dice coefficient for class Yellow: 0.41872146620505857
    Dice coefficient for class Red: 0.3816181222900247
    Average Dice coefficient: 0.5978098842700343

    Running Prediction for Test/Image/d0cf594c5106fb84e894c0b12013f367.png
    Dice coefficient for class Black: 0.9673794184824136
    Dice coefficient for class Yellow: 0.5327590964151333
    Dice coefficient for class Red: 0.7543495720963054
    Average Dice coefficient: 0.7514960289979508
    ```

- FPN with Resnet34 and patch size = 512
    ```
    Running Prediction for Test/Image/7105b8ee8d4c00c513b334fdfdcd6c49.png
    Dice coefficient for class Black: 0.9931033342601725
    Dice coefficient for class Yellow: 0.5742875592859272
    Dice coefficient for class Red: 0.43456890660454395
    Average Dice coefficient: 0.6673199333835479

    Running Prediction for Test/Image/d0cf594c5106fb84e894c0b12013f367.png
    Dice coefficient for class Black: 0.9692450931760175
    Dice coefficient for class Yellow: 0.5684065583394063
    Dice coefficient for class Red: 0.7557470667546216
    Average Dice coefficient: 0.7644662394233483
    ```

- UNet with Resnet18 without LR scheduler and patch size = 512
    ```
    Running Prediction for Test/Image/7105b8ee8d4c00c513b334fdfdcd6c49.png
    Dice coefficient for class Black: 0.987933335585651
    Dice coefficient for class Yellow: 0.49366078563947086
    Dice coefficient for class Red: 0.4210110024759401
    Average Dice coefficient: 0.634201707900354

    Running Prediction for Test/Image/d0cf594c5106fb84e894c0b12013f367.png
    Dice coefficient for class Black: 0.9631788424254311
    Dice coefficient for class Yellow: 0.5651457952011617
    Dice coefficient for class Red: 0.7642635364989824
    Average Dice coefficient: 0.7641960580418584
    ```

- UNetPlus with Resnet34 and patch size = 256
    ```
    Running Prediction for Test/Image/7105b8ee8d4c00c513b334fdfdcd6c49.png

    Dice coefficient for class Black: 0.9906353220541534
    Dice coefficient for class Yellow: 0.49749753596106006
    Dice coefficient for class Red: 0.411891092885325
    Average Dice coefficient: 0.6333413169668461

    Running Prediction for Test/Image/d0cf594c5106fb84e894c0b12013f367.png
    Dice coefficient for class Black: 0.961001661053341
    Dice coefficient for class Yellow: 0.5259151811798194
    Dice coefficient for class Red: 0.7546168068268623
    Average Dice coefficient: 0.7471778830200075
    ```

### Disclaimer & References:
- I have used few open source codes and packages for the end-to-end implementation of this problem statement.
- [Link](https://github.com/bnsreenu/python_for_microscopists/tree/master/228_semantic_segmentation_of_aerial_imagery_using_unet)
- [Link](https://github.com/qubvel/segmentation_models.pytorch/tree/master)

### Scope of Improvement:
- Following paper can be explored and other custom or open source models can be used for improved results.
- [Paperswithcode](https://paperswithcode.com/search?q_meta=&q_type=&q=Segment+Breast+Biopsy+Whole+Slide+Images)
- [Paper](https://arxiv.org/pdf/1709.02554v2.pdf)