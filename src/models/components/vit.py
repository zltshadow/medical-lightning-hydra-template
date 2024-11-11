import torch
from monai.networks import nets


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
    model = ViT(
        img_size=[128, 128, 32],
        patch_size=[16, 16, 16],
        spatial_dims=3,
        in_channels=1,
        num_classes=2,
        classification=True,
        dropout_rate=0,
        post_activation=None,
    )
    img = torch.randn(2, 1, 128, 128, 32)
    preds = model(img)
    print(preds, preds[0].shape)
