import torch
from monai.networks.nets import ResNet, resnet50


class ResNet50(ResNet):
    """A simple fully-connected neural net for computing predictions."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        if self.fc is not None:
            x = self.fc(x)

        return x


if __name__ == "__main__":
    model = ResNet50(
        block="bottleneck",
        layers=[3, 4, 6, 3],
        block_inplanes=[64, 128, 256, 512],
        spatial_dims=3,
        n_input_channels=1,
        num_classes=2,
    )
    # model = resnet50(spatial_dims=3, n_input_channels=1, num_classes=2)
    img = torch.randn(2, 1, 256, 256, 32)
    preds = model(img)
    print(preds.shape)
