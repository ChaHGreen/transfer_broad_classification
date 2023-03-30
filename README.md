This is the [original repository](https://github.com/asrafulashiq/transfer_broad)

### Brief Introduction of Models
The feature of this model is that it applys Contrastive Loss to train the model and use Cross Entropy Loss as baseline.

Their vision backbone is ResNet 50, but trained with different loss
1. Trained with **Cross Entropy,** refered as **CE**.
2. Trained with **Supervised Contrastive Entropy**, refered as **SupCon**.
3. Trained with both **Supervised Contrastive Entropy and Self-supervised Contrastive Entropy**, refered as **SupCon+SelfSupCon**.

There are 2 differnet training strategies:
1. **Freeze the backbone**, only train a classification head on the frozen backnone, refered as **Linear Evaluation**.
2. **Finetunre the backbone** while training the classification layer, referred as **Linear Transfer**.

It's repository provides some [pretrain models](https://drive.google.com/drive/folders/1MXD47VqofZnfQU7iKHE0wL08HuTqGuaK?usp=sharing), more pretrain models can be find in original repository. Pretain model and its correspoding names are as follows:
1. base_finetune --> CE
2. moco --> Self-SupCon
3. moco-mit --> SupCon
4. base_plus_moco --> CE+SelfSupCon
5. supervised_mean2 --> SupCon+SelfSupCon

---
The key to apply transfer board is to register your own datsset.  To register successfully, you need to **do the following:**

## Prepare your dataset

The most convinient way is to prepare your dataset in the format that *pytorch dataloder* can process directly, for example:
```
├── CropDiseases   ## Dataset Name
│   ├── test    ## must contain the file "test" for testing images
│   │   ├── Apple___Apple_scab   ## Class File: File names are names for your classes
│   │   ├── Apple___Black_rot
│   │   ├── Apple___Cedar_apple_rust
│   │   ├── Apple___healthy

│   └── train       ## must contain the file "train" for training images
│       ├── Apple___Apple_scab    ## File names are names for your classes
│       ├── Apple___Black_rot
│       ├── Apple___Cedar_apple_rust
│       ├── Apple___healthy

│   └── all       ## all = train + test --> this may not be used, so you don't have to prepare it
│       ├── Apple___Apple_scab    ## File names are names for your classes
│       ├── Apple___Black_rot
│       ├── Apple___Cedar_apple_rust
│       ├── Apple___healthy
```
	Each class file containing the images for that class

- And I suggest yout creat a  directory like this and put your dataset under it. It's the same as other dataset, so less things need to be changed and less likely to go wrong
```
├── transfer_board_classification   
│   ├── datasets
│   │   ├── cdfsl
│   │   │   ├── $ your dataset
```

## Register your dataset

1. Register your dataset in transfer_board_classification/dataloder/config_path.yaml
	 -  Copy the format of other dataset and write your own, like this:
```
ImageNet1K:    ##Your Dataset Name
	data_path: "~/datasets/cdfsl/imagenet1k"   ##Path to your dataset
	num_class: 1000       ##The number of classes in your own dataset
	split: 43456,10849     ## how do you split your train and test  set --> train num, test num
```

2. Register your dataset in transfer_board_classification/dataloder/aditional_data.py
	- Our dataset register the same as CropDisease, so you can copy the format of it, like this:
```
class CropDisease(datasets.ImageFolder):
	def __init__(self, data_root: str, mode='train'):
		if mode == 'train':
			path = os.path.join(data_root, 'train')
		elif mode == "test":
			path = os.path.join(data_root, 'test')
		else:
			path = os.path.join(data_root, 'all')
		super().__init__(path)
```