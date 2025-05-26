# Segmentation_models

Accurate water segmentation plays a vital role in ensuring the safe and autonomous navigation of unmanned surface vehicles (USVs), allowing them to effectively differentiate between water surfaces and potential obstacles. This functionality is essential for key operational tasks such as route planning and collision prevention in dynamic maritime environments. While RADAR and LIDAR technologies are widely used, vision-based systems present a more cost-efficient yet dependable alternative.

This project explores computer vision-based water segmentation techniques tailored to the marine conditions of the Gulf of Finland, focusing on their application in USVs.

This dataset combines images from the Tampere-WaterSeg and USVInland collections, supplemented by 226 frames captured along the Estonian coastline. 

Dataset:
- [] train: 781 images
- [] validation: 390 images
- [] test: 129 images

The trained models are in the file:

```
model_name/pretrained_models
```

all models are stored in these folders with these names - **backbone_model.pth**
The prefix **_new_validation** - means that the model was trained with new data from the coast of the Gulf of Finland, which were added only to the validation folder:
- [] train: 910 images
- [] validation: 515 images

The prefix **_rebalanced_dataset** - means that the model was trained with a new dataset configuration:
- [] train: 1205 images
- [] validation: 220 images