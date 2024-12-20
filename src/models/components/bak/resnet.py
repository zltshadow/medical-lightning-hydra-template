import torch
from monai.networks import nets
from torchinfo import summary
import yaml

from src.utils.utils import add_torch_shape_forvs


class ResNet(nets.ResNet):
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
    add_torch_shape_forvs()
    with open("configs/data/lbl.yaml", "r", encoding="utf-8") as f:
        data_config = yaml.load(f.read(), Loader=yaml.FullLoader)
    input_size = data_config["input_size"]
    batch_size = data_config["batch_size"]
    in_channels = data_config["in_channels"]
    num_classes = data_config["num_classes"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # resnet34
    # model = ResNet(
    #     block="basic",
    #     layers=[3, 4, 6, 3],
    #     block_inplanes=[64, 128, 256, 512],
    #     shortcut_type="A",
    #     bias_downsample=True,
    #     spatial_dims=3,
    #     n_input_channels=1,
    #     num_classes=2,
    # )
    # resnet50
    model = ResNet(
        block="bottleneck",
        layers=[3, 4, 6, 3],
        block_inplanes=[64, 128, 256, 512],
        shortcut_type="B",
        bias_downsample=False,
        spatial_dims=2,
        n_input_channels=3,
        num_classes=2,
    ).to(device)
    # # resnet10
    # model = ResNet(
    #     block="basic",
    #     layers=[1, 1, 1, 1],
    #     block_inplanes=[64, 128, 256, 512],
    #     shortcut_type="B",
    #     bias_downsample=False,
    #     spatial_dims=3,
    #     n_input_channels=1,
    #     num_classes=2,
    # ).to(device)
    # # resnet18
    # model = ResNet(
    #     block="basic",
    #     layers=[2, 2, 2, 2],
    #     block_inplanes=[64, 128, 256, 512],
    #     shortcut_type="A",
    #     bias_downsample=True,
    #     spatial_dims=3,
    #     n_input_channels=in_channels,
    #     num_classes=2,
    # ).to(device)
    img = torch.randn(
        batch_size, in_channels, input_size[0], input_size[1], input_size[2]
    ).to(device)
    summary(model=model, input_size=img.shape)
    preds = model(img)
    print(preds, preds[0].shape)
