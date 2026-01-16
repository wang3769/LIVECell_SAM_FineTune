import torch
from torch.utils.data import DataLoader
from segment_anything import sam_model_registry
from dataset import LiveCellDataset
from losses import seg_loss
from tqdm import tqdm

device = "cuda"

sam = sam_model_registry["vit_b"](checkpoint="sam/sam_vit_b_01ec64.pth")
sam.to(device)

# freeze everything except mask decoder
for p in sam.image_encoder.parameters():
    p.requires_grad = False
for p in sam.prompt_encoder.parameters():
    p.requires_grad = False

optimizer = torch.optim.AdamW(
    sam.mask_decoder.parameters(), lr=1e-4
)

dataset = LiveCellDataset(
    img_dir="data/livecell/images/train",
    ann_file="data/livecell/annotations/train.json"
)

loader = DataLoader(dataset, batch_size=4, shuffle=True)

sam.train()

for epoch in range(10):
    pbar = tqdm(loader)
    for img, mask, point in pbar:
        img = img.to(device)
        mask = mask.to(device)
        point = point.to(device)

        with torch.no_grad():
            image_embedding = sam.image_encoder(img)

        sparse, dense = sam.prompt_encoder(
            points=(point, torch.ones(len(point), 1).to(device)),
            boxes=None,
            masks=None
        )

        low_res_masks, _ = sam.mask_decoder(
            image_embedding,
            sam.prompt_encoder.get_dense_pe(),
            sparse,
            dense,
            multimask_output=False
        )

        loss = seg_loss(low_res_masks, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(f"epoch {epoch} | loss {loss.item():.4f}")
