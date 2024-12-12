import timm
import torch
from torchinfo import summary
import yaml
from src.utils.utils import add_torch_shape_forvs
import torch
from torch import nn


class SENet(nn.Module):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__()

        self.model = timm.create_model("seresnet152", **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        return self.model(x)


if __name__ == "__main__":
    add_torch_shape_forvs()
    with open("configs/data/lbl.yaml", "r", encoding="utf-8") as f:
        data_config = yaml.load(f.read(), Loader=yaml.FullLoader)
    input_size = data_config["input_size"]
    batch_size = data_config["batch_size"]
    in_channels = data_config["in_channels"]
    num_classes = data_config["num_classes"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SENet(
        in_chans=in_channels,
        num_classes=num_classes,
    ).to(device)
    summary(
        model,
        input_size=(
            batch_size,
            in_channels,
            input_size[0],
            input_size[1],
        ),
    )
    img = torch.randn((batch_size, in_channels, input_size[0], input_size[1])).to(
        device
    )
    preds = model(img)
    print(preds, preds[0].shape)
    print(model)
