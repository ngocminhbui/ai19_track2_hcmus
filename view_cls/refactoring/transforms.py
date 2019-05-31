from torchvision import datasets, models, transforms
import torch

data_transforms = {
        'train': transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
        'val': transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
}
mask_tensor = mask_tensor = torch.Tensor([
            [1,0,0],
            [1,0,1],
            [0,1,1],
            [0,1,0],
            [0,0,1]
        ])


