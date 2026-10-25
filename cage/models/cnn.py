"""Intentionally under-parameterized CNN classifier for CAGE."""

import torch
import torch.nn as nn


class WeakCNN(nn.Module):
    """2-conv classifier designed to be weak by construction.

    Intentionally small to establish a low accuracy ceiling so CAGE's
    improvement is measurable. Named conv layers (conv1, conv2) are
    required for GradCAM hooks in cage/gradcam.py.

    Spatial flow (no padding, RGB 128×128 input):
        3×128×128 → conv1 → 8×126×126 → pool → 8×63×63
                  → conv2 → 16×61×61  → pool → 16×30×30
                  → flatten → Linear(14400, num_classes)
    """

    def __init__(self, num_classes: int = 10, channels: int = 3) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 8, kernel_size=3)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.25)
        self.classifier = nn.Linear(16 * 30 * 30, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))  # 8×63×63
        x = self.pool(self.relu(self.conv2(x)))  # 16×30×30
        x = x.view(x.size(0), -1)  # 14400
        x = self.dropout(x)
        return self.classifier(x)

    def logit_contributions(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Per-position terms of each image's logit for `labels`, shape (N, 30, 30).

        The head is linear, so the terms plus the class bias equal the logit
        exactly (in eval mode, where dropout is off).
        """
        features = self.pool(self.relu(self.conv1(x)))
        features = self.pool(self.relu(self.conv2(features)))
        weights = self.classifier.weight[labels].view(-1, *features.shape[1:])
        return (weights * features).sum(1)

    def freeze(self) -> None:
        """Freeze all parameters and set eval mode.

        Sets requires_grad=False on all params and calls eval().
        Gradients still flow THROUGH frozen computations into U-Net outputs.
        """
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
