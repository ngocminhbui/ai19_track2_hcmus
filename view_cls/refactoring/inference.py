from .dataset_dataloaders import CSVDataset
from .model_99 import model_ft
#from .model_9646 import model_ft

import torch
from .transforms import data_transforms
import numpy as np
from torch.utils.data import DataLoader, Dataset
import ipdb

def inference_dataloader(dataloader, save_path=None):
    outputs_accumulated = []
    for inputs in dataloader:
        inputs = inputs.to(device)
        with torch.set_grad_enabled(False):
            outputs = model_ft(inputs)
            outputs = torch.sigmoid(outputs)
            outputs_accumulated.append(outputs)
    outputs_accumulated = torch.cat(outputs_accumulated,dim=0)
    outputs_accumulated_np = outputs_accumulated.cpu().numpy()
    if save_path is not None:
        np.savetxt(save_path,outputs_accumulated_np,fmt="%.4f")
    return outputs_accumulated_np

dts_query = CSVDataset(csv_file='/home/bnminh/projects/ai2/SOURCE/All_notebook/view_cls/all_query.txt',
                           root_dir='/home/bnminh/projects/ai2/datasets/ai_dataset/DATA/image_query_root/image_query/',
                           transform=data_transforms['val'])
dtl_query = DataLoader(dts_query,batch_size=64, shuffle=False)

dts_test_best = CSVDataset(csv_file='/home/bnminh/projects/ai2/SOURCE/All_notebook/view_cls/track2_test_best_imgs.txt',
                           root_dir='/home/bnminh/projects/ai2/datasets/ai_dataset/DATA/image_test_root/image_test/',
                           transform=data_transforms['val'], only_filename=False)
dtl_test_best = DataLoader(dts_test_best,batch_size=64, shuffle=False)

dts_train = CSVDataset(csv_file='/home/bnminh/projects/ai2/SOURCE/All_notebook/view_cls/all_train_images.txt',
                           root_dir='/home/bnminh/projects/ai2/datasets/ai_dataset/DATA/image_train/' ,
                           transform=data_transforms['val'])
dtl_train = DataLoader(dts_train,batch_size=64, shuffle=False)

dts_test = CSVDataset(csv_file='/home/bnminh/projects/ai2/SOURCE/All_notebook/view_cls/all_test.txt',
                           root_dir='/home/bnminh/projects/ai2/datasets/ai_dataset/DATA/image_test_root/image_test/',
                           transform=data_transforms['val'])
dtl_test = DataLoader(dts_test,batch_size=64, shuffle=False)
#ipdb.set_trace()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

best = torch.load('/home/bnminh/projects/ai2/SOURCE/All_notebook/view_cls/best_model_03_05_resnet50_0.9909/model_best.pth.tar')
model_ft.load_state_dict(best['state_dict'])
model_ft = model_ft.to(device)
model_ft.eval()
print(best['best_acc'])
print(best['epoch'])

inference_dataloader(dtl_query,'query_view_scores.txt')
#inference_dataloader(dtl_test_best,'test_best_view_scores.txt')
#inference_dataloader(dtl_train,'train_view_scores.txt')
inference_dataloader(dtl_test,'test_view_scores.txt')

