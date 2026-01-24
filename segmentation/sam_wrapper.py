import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor


class SAMSegmenter:
    def __init__(self, encoder_ckpt, decoder_ckpt, model_type="vit_b", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load base SAM (image encoder + prompt encoder + default decoder)
        sam = sam_model_registry[model_type](checkpoint=encoder_ckpt)

        # Load your fine-tuned mask decoder weights
        decoder_state = torch.load(decoder_ckpt, map_location="cpu")
        sam.mask_decoder.load_state_dict(decoder_state, strict=False)

        sam.to(self.device)
        sam.eval()

        self.predictor = SamPredictor(sam)

    def segment_with_box(self, image, box_xyxy):
        """
        image: HxW or HxWx3 (uint8)
        box_xyxy: [x1, y1, x2, y2] in image coordinates (1024-space)
        """
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)

        self.predictor.set_image(image)

        box = np.array(box_xyxy, dtype=np.float32)
        masks, _, _ = self.predictor.predict(
            box=box,
            multimask_output=False
        )
        return masks[0]

