import timm
import torch
from torchinfo import summary
import yaml
from src.utils.utils import add_torch_shape_forvs
import torch
from torch import nn
from timm.models import SwinTransformerV2


class SwinTransformer(nn.Module):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__()

        # self.model = timm.create_model("swin_base_patch4_window7_224", **kwargs)
        # swin_base_patch4_window7_224
        self.model = SwinTransformerV2(
            patch_size=4,
            window_size=7,
            embed_dim=128,
            depths=(2, 2, 18, 2),
            num_heads=(4, 8, 16, 32),
            **kwargs,
        )

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
    model = SwinTransformer(
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
    for name, _ in model.named_modules():
        print(name)
