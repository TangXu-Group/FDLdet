# FDLdet
This repository provides the code for the methods and experiments presented in our paper 'FDLdet: A Change Detector Based on Forward Dictionary Learning for Remote Sensing Images'. (TGRS2024)

![微信截图_20240627012458](https://github.com/TangXu-Group/FDLdet/assets/74549002/aacbd4cc-cd0f-4276-94dd-07c5c3e3db01)
If you have any questions, you can send me an email. My mail address is yqunyang@163.com

Training
=
Prepare
---
The based dependencies of runing codes:
```
torchvision: 0.9.0
torch: 1.8.0
python: 3.8.17
```
Start
---
First, use ```SLIC.ipynb``` to generate SLIC segmentation results and save them in the ```/SLIC_DATA``` folder offline.
Then, use ```Train.ipynb``` to train and test the model.
