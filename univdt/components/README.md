# Datasets
- [Datasets](#datasets)
    - [1. CamVid](#1-camvid)
    - [2. PASCAL VOC](#2-pascal-voc)
    - [3. Public Tuberculosis](#3-public-tuberculosis)
      - [3.1 TB-Predict](#31-tb-predict)
      - [3.2 Montgomery](#32-montgomery)
      - [3.3 Shenzhen](#33-shenzhen)
      - [3.4 TBX11k](#34-tbx11k)
    - [4. NIH CXR (Chest X-Ray14)](#4-nih-cxr-chest-x-ray14)
  
---


### [1. CamVid][CAMVID]
CamVid is a road scene dataset consisting of a total of 701 images, with 367 in the training set, 101 in the validation set, and 233 in the test set. The annotations for the test set are publicly available. There are a total of 32 classes, but only 11 classes were mainly used in the experiment. The 11 classes primarily used in the CamVid training dataset are as follows: **sky, building, columnpole, road, sidewalk, tree, sign/symbol, fence, car, pedestrian, and bicyclist** <br>
You can download the dataset from the [official source][CAMVID] or execute [custom source](../../scripts/camvid.sh).


[CAMVID]:http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/
[CAMVID_LINK]:https://drive.google.com/file/d/15e7J7bLBosM8Aqb6LtkbD7gQFzbZ9TbY/view?usp=drive_link

---

### [2. PASCAL VOC][VOC]
PASCAL VOC (Visual Object Classes) dataset is one of the widely used datasets for tasks like object recognition and object detection in the field of computer vision. The dataset consists of a total of 20 object classes. For semantic segmentation, the dataset is divided into train, val, and test sets, consisting of 1464, 1449, and 1456 images respectively. Labels for the test set are not publicly available, but evaluation can be performed using the [test server][VOC_EVALUATION_SERVER]. Further details can be found in the [official documentation][VOC_DOC].  <br>
You can download the dataset from the [official source][VOC] or execute [custom source](../../scripts/pascal_voc.sh).

**Currently, the dataset is only supported for the semantic segmentation task in this repository.**

[VOC]:http://host.robots.ox.ac.uk/pascal/VOC/
[VOC_LINK]:https://drive.google.com/file/d/15e7J7bLBosM8Aqb6LtkbD7gQFzbZ9TbY/view?usp=sharing
[VOC_EVALUATION_SERVER]:http://host.robots.ox.ac.uk/pascal/VOC/
[VOC_DOC]:http://host.robots.ox.ac.uk/pascal/VOC/voc2012/devkit_doc.pdf

--- 

### 3. Public Tuberculosis
#### [3.1 TB-Predict][DADB]
The **TB-Predict** includes two tuberculosis datasets, namely **DADB**. These datasets were acquired from two distinct X-ray machines located at the National Institute of Tuberculosis and Respiratory Diseases in New Delhi. DA comprises 104 images in the training set and 52 images in the test set, while DB consists of 100 training images and 50 test images. **However, currently, 28 out of the 50 training images in DB-training are unavailable. Therefore, in this paper, the remaining 22 images from DB-train are utilized.** This dataset does not have a separate validation set, so it is manually constructed. <br>
You can download the dataset from the [official source][DADB] and the [custom source][DADB_LINK]. The images in the custom source have been resized to 1024x1024 and normalized to the range of 0 to 255 from the original. This custom source includes data split information.

[DADB]:https://sourceforge.net/projects/tbxpredict/
[DADB_LINK]:https://drive.google.com/file/d/1lhzxuvtmY4hYGC0ZdoGgERf-KTFwCtiW/view?usp=sharing

#### [3.2 Montgomery][MONTGOMERY]
The **Montgomery** dataset were acquired in collaboration with The Department of Health and Human Services of Montgomery County, MD, USA.  Montgomery consists of a **total of 138 images, with 58 images related to tuberculosis (tb) and 80 images** related to non-tuberculosis (non-tb). The dataset includes additional information such as image-level annotation, gender, age, and report. The data split is not included, so it is constructed manually.<br>
You can download the dataset from the [official source][MONTGOMERY_LINK1] and the [custom source][MONTGOMERY_LINK2]. The images in the custom source have been resized to 1024x1024 and normalized to the range of 0 to 255 from the original. This custom source includes data split information.

<details>
  <summary>Data Split Details</summary>
    tbu
</details>

[MONTGOMERY]:https://openi.nlm.nih.gov/faq#collection
[MONTGOMERY_LINK1]:https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip
[MONTGOMERY_LINK2]:https://drive.google.com/file/d/1VsDzFxXASNl_J0DKksQgpusxOfbQ4iR1/view?usp=drive_link

#### [3.3 Shenzhen][SHENZHEN]
The **Shenzhen** dataset was collected at Shenzhen No.3 People’s Hospital, Guangdong providence, China. The Shenzhen dataset comprises a total of 662 images, with 336 images related to TB and 326 images related to Non-TB. The dataset includes additional information such as image-level annotation, gender, age, and report. The data split is not included, so it is constructed manually.<br>
You can download the dataset from the [official source][SHENZHEN_LINK1] and the [custom source][SHENZHEN_LINK2]. The images in the custom source have been resized to 1024x1024 and normalized to the range of 0 to 255 from the original. This custom source includes data split information.

<details>
  <summary>Data Split Details</summary>
    tbu
</details>

[SHENZHEN]:https://openi.nlm.nih.gov/faq#collection
[SHENZHEN_LINK1]:https://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip
[SHENZHEN_LINK2]:https://drive.google.com/file/d/16uK-BQJtecqO8_UY9hQL1UwpeaBMB_BT/view?usp=sharing

#### [3.4 TBX11k][TBX11K]

The TBX11k is a large-scale dataset specifically constructed by the authors for Computer-Aided Tuberculosis Diagnosis (CTD). It consists of 11,200 images, including 1,200 images related to TB, 5,000 sick but non-TB images, and 5,000 healthy images, along with bounding box-level annotations for each. The dataset also provides data splits, with 6,600 images for training, 1,800 for validation, and 2,800 for testing. Furthermore, to ensure fair comparisons, the annotations for the test set are not publicly disclosed. Instead, measurements can be obtained through a [server][TBX11K_SERVER] for evaluation purposes.<br>
You can download the dataset from the [official source][TBX11K_LINK] and the [custom source][TBX11K_LINK2]. The images in the custom source have been resized to 1024x1024 and normalized to the range of 0 to 255 from the original.

[TBX11K]:https://mmcheng.net/tb
[TBX11K_SERVER]:https://codalab.lisn.upsaclay.fr/competitions/7916
[TBX11K_LINK]:https://drive.google.com/file/d/1r-oNYTPiPCOUzSjChjCIYTdkjBTugqxR/view?usp=sharing
[TBX11K_LINK2]:https://drive.google.com/file/d/1R4-4uOtDQBQO6Iqsnf55hvCxpqLDjAu3/view?usp=sharing

---

### [4. NIH CXR (Chest X-Ray14)][NIH]

The NIH CXR (Chest X-Ray14) dataset is a widely used collection for chest X-ray images, introduced by the National Institutes of Health (NIH). It comprises a total of 112,120 frontal-view X-ray images from 30,805 unique patients, including image-level labels for 14 different pathologies. The dataset is specifically designed for various research purposes, including the development and evaluation of machine learning algorithms for chest X-ray interpretation. Among the total of 112,120, 25,596 are allocated for testset, with the remaining 86,524 intended for training and validation. However, the training and validation sets are not separated. <br>
You can download the dataset from the [official source][NIH_LINK] and the [custom source][NIH_LINK2]. The images in the custom source have been resized to 1024x1024 and normalized to the range of 0 to 255 from the original.

<details>
  <summary>Click to expand</summary>
    tbu
</details>

[NIH]:https://www.cc.nih.gov/drd/summers.html
[NIH_LINK]:https://nihcc.app.box.com/v/ChestXray-NIHCC
[NIH_LINK2]:https://drive.google.com/file/d/11YNGiwTkDASqEK0t6amGmJu-X0wp3evL/view?usp=drive_link

---
<!-- 
### [5. Cityscapes][CITYSCAPES]

Cityscapes is a large-scale dataset that focuses on semantic understanding of urban street scenes. It provides pixel-level annotations for 5,000 images in the training set, 500 images in the validation set, and 1,525 images in the test set. The dataset includes 30 classes, but only 19 classes were mainly used in the experiment. The 19 classes primarily used in the Cityscapes training dataset are as follows: **road, sidewalk, building, wall, fence, pole, traffic light, traffic sign, vegetation, terrain, sky, person, rider, car, truck, bus, train, motorcycle, and bicycle** <br>
You can download the dataset from the [official source][CITYSCAPES].

[CITYSCAPES]:https://www.cityscapes-dataset.com/

---

### [6. COCO][COCO]

COCO is a large-scale dataset that focuses on object detection, segmentation, and captioning. It provides pixel-level annotations for 118,287 images in the training set, 5,000 images in the validation set, and 40,670 images in the test set. The dataset includes 80 classes. <br>

[COCO]:https://cocodataset.org/#home

---

### [VinDr CXR][VINDRCXR]



kaggle : https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/

[VINDRCXR]:https://vinlab.io/dataset/vindr-chest-x-ray.html/   

--- -->