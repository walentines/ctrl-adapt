import os
from glob import glob
import sys

import cv2
import numpy as np

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt


# local imports
sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append('/bigdata/userhome/ionut.serban/shared/MIRPR-proiectCONTROL-NET/CREStereo-Pytorch')
from nets import Model

device = 'cuda'
root = "/bigdata/userhome/ionut.serban/shared/MIRPR-proiectCONTROL-NET/dataset_cityscapes_full"
model_path = "/bigdata/userhome/ionut.serban/shared/crestereo_eth3d.pth"
OG_W, OG_H = 2048, 1024   # OG width adn height
W1, H1 = OG_W, OG_H # resize w,h for dataloader
W2, H2 = W1, H1     # resize w,h for model inference

batch_size = 1
n_iter=20


baseline = 0.209313 # meters
f = 2262.52 # x focal length (pixels)


from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CityScapesStereo(Dataset):
    def __init__(self, root, transform=None, split='train', return_paths=False):
        self.root = root
        self.transform = transform
        self.return_paths = return_paths

        self.left_paths = sorted(glob(os.path.join(root, 'leftImg8bit', split, '**/*.png')))
        self.right_paths = sorted(glob(os.path.join(root, 'rightImg8bit', split, '**/*.png')))
        # print(self.left_paths)
        # print(self.right_paths)

    def __getitem__(self, idx):
        left = cv2.cvtColor(cv2.imread(self.left_paths[idx]), cv2.COLOR_BGR2RGB)
        right = cv2.cvtColor(cv2.imread(self.right_paths[idx]), cv2.COLOR_BGR2RGB)

        if self.transform:
            left = self.transform(left)
            right = self.transform(right)

        if self.return_paths:
            return left, right, self.left_paths[idx], self.right_paths[idx]

        return left, right
    
    def __len__(self):
        return len(self.left_paths)
    

# helper to convert from pytorch to numpy for display
def convert_to_numpy(image):
    return image.detach().cpu().numpy().transpose(1, 2, 0)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((H1, W1)),
])

model = Model(max_disp=256, mixed_precision=False, test_mode=True)
model.load_state_dict(torch.load(model_path), strict=True)
model.to(device)
model.eval()

# zero out all gradients
for param in model.parameters():
    param.grad = None

# find optimal backend for performning convolutions
torch.backends.cudnn.benchmark = True



def inference(left, right, model, n_iter=20):

	imgL = left.transpose(2, 0, 1)
	imgR = right.transpose(2, 0, 1)
	imgL = np.ascontiguousarray(imgL[None, :, :, :])
	imgR = np.ascontiguousarray(imgR[None, :, :, :])

	imgL = torch.tensor(imgL.astype("float32"), device='cuda')
	imgR = torch.tensor(imgR.astype("float32"), device='cuda')

	imgL_dw2 = F.interpolate(
		imgL,
		size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
		mode="bilinear",
		align_corners=True,
	)
	imgR_dw2 = F.interpolate(
		imgR,
		size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
		mode="bilinear",
		align_corners=True,
	)
	# print(imgR_dw2.shape)
	with torch.inference_mode():
		pred_flow_dw2 = model(imgL_dw2, imgR_dw2, iters=n_iter, flow_init=None)
		del imgL_dw2, imgR_dw2
		pred_flow = model(imgL, imgR, iters=n_iter, flow_init=pred_flow_dw2)
	pred_disp = torch.squeeze(pred_flow[:, 0, :, :]).cpu().detach().numpy()

	return pred_disp

train_dataset = CityScapesStereo(root, None, 'train')
# print(train_dataset)
left, right = train_dataset[0]

pred_disp = inference(left, right, model, n_iter=20)


_, ax = plt.subplots(1, 3, figsize=(25, 15))
ax[0].imshow(left)
ax[0].set_title("Left")
ax[1].imshow(right, cmap='jet')
ax[1].set_title("Right")
ax[2].imshow(np.log(pred_disp), cmap='jet')
ax[2].set_title("Predicted Disparity")


def get_disparity(left, right):

	# place resize and place on device
	left_dw2 = F.interpolate(
		left,
		size=(H2, W2),
		mode="bilinear",
		align_corners=True,
	).to(device)

	right_dw2 = F.interpolate(
		right,
		size=(H2, W2),
		mode="bilinear",
		align_corners=True,
	).to(device)
	
	# perform inference
	with torch.inference_mode():
		pred_flow_dw2 = model(left_dw2, right_dw2, iters=n_iter, flow_init=None)
		del left_dw2, right_dw2
		pred_flow = model(left.to(device), right.to(device), iters=n_iter, flow_init=pred_flow_dw2)

	
	return torch.squeeze(pred_flow[:, 0, :, :]).cpu().detach().numpy()

def make_dir(dirpath):
    try:
        os.mkdir(dirpath)
    except:
        try:
            os.mkdir(os.path.split(dirpath)[0])
        except:
            os.mkdir(os.path.split(os.path.split(dirpath)[0])[0])


disp_dir = os.path.join(root, 'crestereo_disparity')
depth_dir = os.path.join(root, 'crestereo_depth')

if not os.path.exists(disp_dir): os.mkdir(disp_dir)
if not os.path.exists(depth_dir): os.mkdir(depth_dir)

for split in ['train', 'val', 'test']: # ['train', 'val']: # ('train', 'val', 'test'):
    # create split directories
    disp_split_dir = os.path.join(disp_dir, split)
    depth_split_dir = os.path.join(depth_dir, split)

    if not os.path.exists(disp_split_dir): os.mkdir(disp_split_dir)
    if not os.path.exists(depth_split_dir): os.mkdir(depth_split_dir)

    # get dataloader
    data_loader = DataLoader(CityScapesStereo(root, transform, split, True), batch_size, pin_memory=True)
    
    # get depth and disparity for all image pairs
    for i, (left, right, left_path, _) in enumerate(data_loader):

        disp_savepath = left_path[0].replace('leftImg8bit', 'crestereo_disparity')
        depth_savepath = left_path[0].replace('leftImg8bit', 'crestereo_depth')

        if i < 2:
             print(left, right, left_path)
        # special case for when the file already exist
        if os.path.exists(disp_savepath) and os.path.exists(depth_savepath):
            continue
        if i % 10 == 0:
            print(f"{i}...")

        disp_dirname, _= os.path.split(disp_savepath)
        depth_dirname, _= os.path.split(depth_savepath)
        if not os.path.exists(disp_dirname): os.mkdir(disp_dirname)
        if not os.path.exists(depth_dirname): os.mkdir(depth_dirname)
            
        pred_disparity = get_disparity(left, right)
        pred_depth = baseline * f / (pred_disparity + 0.1)

        # cv2.imwrite(disp_savepath, pred_disparity)
        cv2.imwrite(depth_savepath, pred_depth)
