# EMANet
The changed evaluation code for comparison with EMANet

## Usage
The problem encountered is, the predicted mask does not match the loaded label in size some times. The solution we adopted is to unify their sizes using  F.interpolate().
Please train the model with dataset and strategies provided in <https://github.com/XiaLiPKU/EMANet>, and evaluate the trained model with the eval.py provided above.

## Citation
Coming soon~
