# EMANet
The changed evaluation code for comparison with EMANet

## Usage
The problem encountered is, the predicted mask does not match the loaded label in size some times. The solution we adopted is to unify their sizes using  F.interpolate().
Please train the model with dataset and strategies provided in <https://github.com/XiaLiPKU/EMANet>, and evaluate the trained model with the eval.py provided above.

## Preview

<img src="https://user-images.githubusercontent.com/38877851/222956565-e42ae846-bb56-44f8-a186-ce1b18a89bc2.png" width="400">


## DGOD Experiments
[train_epo11.txt](https://github.com/countytown/EMANet/files/12035537/train_epo11.txt)  
[test_on_Dust_rainy.log](https://github.com/countytown/EMANet/files/12035445/test_on_Dust_rainy.log)  
[test_on_Night_rainy.log](https://github.com/countytown/EMANet/files/12035458/test_on_Night_rainy.log)  
[test_on_Night_sunny.log](https://github.com/countytown/EMANet/files/12035540/test_on_Night_sunny.log)


