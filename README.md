# UnetPointAnnotationSegmentation

## UserInteraction
![Interaction](./img/interaction.gif)

## Model
![Model](./img/model.png)
## MaskingTask
![MaskingTask](./img/maskingTask.JPG)
## refine mask (Threshold: RGB 200)
![mask](./img/mask.png)
## removeTarget(Background)
![removeTarget](./img/remove.png)
## target(foreground)
![target](./img/target.png)
## Background blur
![outFocusing](./img/outFocusing.png)

# Used Dataset
COCO: Common Objects in Context(COCO) 2017 Train/Val Dataset https://cocodataset.org/#home

============================================================================================

Two Point Interaction Model
Foreground Point(target) ,Background Point 
Point : Set of applying a Gaussian Filter(shape(65*65), segma:7)*k(255*/center_value) about random pixel in mask
