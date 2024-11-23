import torch
from monai.networks import nets
from torchinfo import summary
import yaml

from src.utils.utils import add_torch_shape_forvs


class ViT(nets.ViT):
    """A simple fully-connected neural net for computing predictions."""

    def forward(self, x):
        x = self.patch_embedding(x)
        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)
        if hasattr(self, "classification_head"):
            x = self.classification_head(x[:, 0])
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
    model = ViT(
        img_size=[224, 224],
        patch_size=[16, 16],
        spatial_dims=2,
        in_channels=in_channels,
        num_classes=2,
        classification=True,
        dropout_rate=0,
        post_activation=None,
    ).to(device)
    img = torch.randn(
        batch_size, in_channels, input_size[0], input_size[1], input_size[2]
    ).to(device)
    summary(model=model, input_size=img.shape)
    preds = model(img)
    print(preds, preds[0].shape)
