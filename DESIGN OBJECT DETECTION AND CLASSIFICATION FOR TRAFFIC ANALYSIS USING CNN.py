import torch
from torchvision import models
from torchvision.ops import nms

boxes = torch.tensor([[0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.6, 0.6]])
scores = torch.tensor([0.9, 0.75])
labels = torch.tensor([1, 2])

model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

prediction = {'boxes': boxes, 'scores': scores, 'labels': labels}
nms_indices = nms(prediction['boxes'], prediction['scores'], 0.5)

print("Detected Classes:", prediction['labels'][nms_indices])
