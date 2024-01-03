# datasets
1. [Camvid](#1-camvid)
2. [PASCAL VOC](#2-pascal-voc)
3. [Public Tuberculosis](#3-public-tuberculosis)   
    3.1 [TB-Predict](#31-tb-predict)   
    3.2 [Montgomery](#32-montgomery)   
    3.3 [Shenzhen](#33-shenzhen)   
    3.4 [TBX11k](#34-tbx11k)   


### [1. CamVid][CAMVID]

CamVid는 road scene dataset이다. 총 701장으로 구성되어있으며, train, val ,test 각각 367,101, 233 이다. testset의 annotation은 공개되어있다. 32개의 클래스가 존재하지만 실험에서는 주로 11개의 클래스를 사용한다. 홈페이지에서 다운로드가 불편하기때문에 [여기][CAMVID_LINK]에서 다운로드 혹은 [prepare_dataset/camvid.sh](../../prepare_dataset/camvid.sh) 실행합니다.

CamVid is a road scene dataset consisting of a total of 701 images, with 367 in the training set, 101 in the validation set, and 233 in the test set. The annotations for the test set are publicly available. There are a total of 32 classes, but only 11 classes were mainly used in the experiment. You can download the dataset from [here][CAMVID_LINK] or execute [prepare_dataset/camvid.sh](../../prepare_dataset/camvid.sh)

The 11 classes primarily used in the CamVid training dataset are as follows: **sky, building, columnpole, road, sidewalk, tree, sign/symbol, fence, car, pedestrian, and bicyclist**


[CAMVID]:http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/
[CAMVID_LINK]:https://drive.google.com/file/d/15e7J7bLBosM8Aqb6LtkbD7gQFzbZ9TbY/view?usp=drive_link

---

### [2. PASCAL VOC][VOC]
TBU
<!-- PASCAL VOC Dataset은 20개의 클래스로 구성되어있으며, 2007년부터 2012년까지의 데이터셋이 존재한다. 07년도까지는 testset anootation이 공개되었지만 이후로는 공개되지 않았다. 
데이터셋은 [여기][VOC_LINK]에서 다운로드 혹은 [prepare_dataset/voc.sh](../../prepare_dataset/voc.sh) 실행합니다.
-->
[VOC]:http://host.robots.ox.ac.uk/pascal/VOC/
[VOC_LINK]:https://drive.google.com/file/d/15e7J7bLBosM8Aqb6LtkbD7gQFzbZ9TbY/view?usp=sharing

--- 

### 3. Public Tuberculosis
#### [3.1 TB-Predict][DADB]
The **TB-Predict** includes two tuberculosis datasets, namely **DADB**. These datasets were acquired from two distinct X-ray machines located at the National Institute of Tuberculosis and Respiratory Diseases in New Delhi. DA comprises 104 images in the training set and 52 images in the test set, while DB consists of 100 training images and 50 test images.    
**However, currently, 28 out of the 50 training images in DB-training are unavailable. Therefore, in this paper, the remaining 22 images from DB-train are utilized.**    
You can download the dataset from the [official source][DADB] and the [custom source][DADB_LINK]. The images in the custom source have been resized to 1024x1024 and normalized to the range of 0 to 255 from the original. This custom source includes data split information.

[DADB]:https://sourceforge.net/projects/tbxpredict/
[DADB_LINK]:https://drive.google.com/file/d/1lhzxuvtmY4hYGC0ZdoGgERf-KTFwCtiW/view?usp=sharing

#### [3.2 Montgomery][MONTGOMERY]
The **Montgomery** dataset were acquired in collaboration with The Department of Health and Human Services of Montgomery County, MD, USA.  Montgomery consists of a **total of 138 images, with 58 images related to tuberculosis (tb) and 80 images** related to non-tuberculosis (non-tb). The dataset includes additional information such as image-level annotation, gender, age, and report. The data split is not included, so it is constructed manually. You can download the dataset from the [official source][MONTGOMERY_LINK1] and the [custom source][MONTGOMERY_LINK2]. The images in the custom source have been resized to 1024x1024 and normalized to the range of 0 to 255 from the original. This custom source includes data split information.

[MONTGOMERY]:https://openi.nlm.nih.gov/faq#collection
[MONTGOMERY_LINK1]:https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip
[MONTGOMERY_LINK2]:https://drive.google.com/file/d/1VsDzFxXASNl_J0DKksQgpusxOfbQ4iR1/view?usp=drive_link

#### [3.3 Shenzhen][SHENZHEN]
The **Shenzhen** dataset was collected at Shenzhen No.3 People’s Hospital, Guangdong providence, China. The Shenzhen dataset comprises a total of 662 images, with 336 images related to TB and 326 images related to Non-TB. The dataset includes additional information such as image-level annotation, gender, age, and report. The data split is not included, so it is constructed manually. You can download the dataset from the [official source][SHENZHEN_LINK1] and the [custom source][SHENZHEN_LINK2]. The images in the custom source have been resized to 1024x1024 and normalized to the range of 0 to 255 from the original. This custom source includes data split information.

[SHENZHEN]:https://openi.nlm.nih.gov/faq#collection
[SHENZHEN_LINK1]:https://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip
[SHENZHEN_LINK2]:https://drive.google.com/file/d/16uK-BQJtecqO8_UY9hQL1UwpeaBMB_BT/view?usp=sharing

#### [3.4 TBX11k][TBX11K]

The TBX11k is a large-scale dataset specifically constructed by the authors for Computer-Aided Tuberculosis Diagnosis (CTD). It consists of 11,200 images, including 1,200 images related to TB, 5,000 sick but non-TB images, and 5,000 healthy images, along with bounding box-level annotations for each. The dataset also provides data splits, with 6,600 images for training, 1,800 for validation, and 2,800 for testing.   
Furthermore, to ensure fair comparisons, the annotations for the test set are not publicly disclosed. Instead, measurements can be obtained through a [server][TBX11K_SERVER] for evaluation purposes.    
You can download the dataset from the [official source][TBX11K_LINK] and the [custom source][TBX11K_LINK2]. The images in the custom source have been resized to 1024x1024 and normalized to the range of 0 to 255 from the original. This custom source includes data split information.

[TBX11K]:https://mmcheng.net/tb
[TBX11K_SERVER]:https://codalab.lisn.upsaclay.fr/competitions/7916
[TBX11K_LINK]:https://drive.google.com/file/d/1r-oNYTPiPCOUzSjChjCIYTdkjBTugqxR/view?usp=sharing
[TBX11K_LINK2]:https://drive.google.com

---