import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import random

class LiveCellDataset(Dataset):
    def __init__(self, img_dir, ann_file, image_size=512):
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.ids = list(self.coco.imgs.keys())
        self.image_size = image_size

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]

        img = cv2.imread(f"{self.img_dir}/{img_info['file_name']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        ann = random.choice(anns)
        mask = self.coco.annToMask(ann)

        img = cv2.resize(img, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size),
                          interpolation=cv2.INTER_NEAREST)

        # generate point prompt inside object
        ys, xs = np.where(mask > 0)
        idx = random.randint(0, len(xs) - 1)
        point = np.array([[xs[idx], ys[idx]]])

        return (
            torch.from_numpy(img).permute(2, 0, 1).float() / 255.0,
            torch.from_numpy(mask).unsqueeze(0).float(),
            torch.from_numpy(point).float()
        )
