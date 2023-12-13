### [1. CamVid][CAMVID]

CamVid는 road scene dataset이다. 총 701장으로 구성되어있으며, train, val ,test 각각 367,101, 233 이다. testset의 annotation은 공개되어있다. 32개의 클래스가 존재하지만 실험에서는 주로 11개의 클래스를 사용한다. 홈페이지에서 다운로드가 불편하기때문에 [여기][CAMVID_LINK]에서 다운로드 혹은 [prepare_dataset/camvid.sh](../../prepare_dataset/camvid.sh) 실행합니다.

CamVid is a road scene dataset consisting of a total of 701 images, with 367 in the training set, 101 in the validation set, and 233 in the test set. The annotations for the test set are publicly available. There are a total of 32 classes, but only 11 classes were mainly used in the experiment. You can download the dataset from [here][CAMVID_LINK] or execute [prepare_dataset/camvid.sh](../../prepare_dataset/camvid.sh)

The 11 classes primarily used in the CamVid training dataset are as follows: **sky, building, columnpole, road, sidewalk, tree, sign/symbol, fence, car, pedestrian, and bicyclist**


[CAMVID]:http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/
[CAMVID_LINK]:https://drive.google.com/file/d/15e7J7bLBosM8Aqb6LtkbD7gQFzbZ9TbY/view?usp=drive_link
