# EMANet
The changed evaluation code for comparison with EMANet

## Usage
The problem encountered is, the predicted mask does not match the loaded label in size some times. The solution we adopted is to unify their sizes using  F.interpolate().
Please train the model with dataset and strategies provided in <https://github.com/XiaLiPKU/EMANet>, and evaluate the trained model with the eval.py provided above.

## Preview

<img src="https://user-images.githubusercontent.com/38877851/222956565-e42ae846-bb56-44f8-a186-ce1b18a89bc2.png",width="200",height="200">


## Citation
Coming soon~
