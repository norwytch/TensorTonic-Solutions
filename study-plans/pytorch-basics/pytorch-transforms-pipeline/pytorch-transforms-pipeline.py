import torch

class TransformPipeline:
    """
    Returns: float32 tensor of shape (C, H, W) from __call__
    """

    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)


    def __call__(self, image):
        image = torch.tensor(image, dtype=torch.float32)
        image = image/255
        image = (image - self.mean)/self.std
        image = image.permute(2, 0, 1)
        return image