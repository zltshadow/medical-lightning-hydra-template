import os
from typing import Any, Dict, Tuple

import cv2
from matplotlib import cm, pyplot as plt
import numpy as np
import pandas as pd
import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryConfusionMatrix,
)

# 初始化指标
import torch.nn.functional as F

# from monai.visualize import CAM, GradCAM, GradCAMpp
from pytorch_grad_cam import (
    GradCAM,
    HiResCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    FullGrad,
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


class LBLLitModule(LightningModule):
    """Example of a `LightningModule` for LBL classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        model_name: str = "",
        loss_name: str = "bce",
    ) -> None:
        """Initialize a `LBLLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net
        self.model_name = model_name
        self.loss_name = loss_name

        # loss function
        if loss_name == "ce":
            self.criterion = torch.nn.CrossEntropyLoss()
        elif loss_name == "bce":
            self.criterion = torch.nn.BCEWithLogitsLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.test_acc = BinaryAccuracy()

        self.train_auc = BinaryAUROC()
        self.val_auc = BinaryAUROC()
        self.val_precision = BinaryPrecision()
        self.val_recall = BinaryRecall()
        self.val_f1 = BinaryF1Score()

        self.test_auc = BinaryAUROC()
        self.test_precision = BinaryPrecision()
        self.test_recall = BinaryRecall()
        self.test_f1 = BinaryF1Score()
        self.test_confusion_matrix = BinaryConfusionMatrix()
        self.test_probs = torch.Tensor([])
        self.test_preds = torch.Tensor([])
        self.test_targets = torch.Tensor([])

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()
        self.val_auc_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        # if self.model_name == "SFMamba":
        #     # 计算最后一个维度的大小
        #     last_dim_size = x.shape[-1]
        #     # 计算每份的大小
        #     chunk_size = last_dim_size // 3
        #     # 分割最后一个维度
        #     t1_sample = x[:, :, :, :, :chunk_size]
        #     t2_sample = x[:, :, :, :, chunk_size : 2 * chunk_size]
        #     t1c_sample = x[:, :, :, :, 2 * chunk_size :]
        #     # print(t1_sample.shape, t2_sample.shape, t1c_sample.shape)
        #     return self.net(t1_sample, t2_sample, t1c_sample)
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()
        self.val_auc.reset()
        self.val_auc_best.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        # 3d
        # image = batch["image"]
        # seg = batch["seg"]
        # label = batch["label"]
        image = batch[0]
        label = batch[1]
        # image, seg, label = batch
        x = image
        y = label
        logits = self.forward(x)
        if self.loss_name == "ce":
            loss = self.criterion(logits, y)
        if self.loss_name == "bce":
            one_hot_y = F.one_hot(y, num_classes=2).float()
            # BCE需要使用one_hot
            loss = self.criterion(logits, one_hot_y)
        # 应用softmax函数，dim=1表示对每一行进行操作：
        probs = F.softmax(logits, dim=1)
        # 提取正类概率（正类的索引为 1）
        positive_class_probs = probs[:, 1]
        preds = torch.argmax(logits, dim=1)
        return loss, positive_class_probs, preds, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, probs, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.train_auc(probs, targets)
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/auc", self.train_auc, on_step=False, on_epoch=True, prog_bar=True
        )

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, probs, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.val_auc(probs, targets)
        self.val_precision(preds, targets)
        self.val_recall(preds, targets)
        self.val_f1(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auc", self.val_auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val/precision",
            self.val_precision,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/recall", self.val_recall, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        auc = self.val_auc.compute()  # get current val acc
        self.val_auc_best(auc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log(
            "val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True
        )
        self.log(
            "val/auc_best", self.val_auc_best.compute(), sync_dist=True, prog_bar=True
        )

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, probs, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.test_auc(probs, targets)
        self.test_precision(preds, targets)
        self.test_recall(preds, targets)
        self.test_f1(preds, targets)
        self.test_probs = torch.concat((self.test_probs.to(probs.device), probs), dim=0)
        self.test_preds = torch.concat((self.test_preds.to(preds.device), preds), dim=0)
        self.test_targets = torch.concat(
            (self.test_targets.to(targets.device), targets), dim=0
        )
        # confusion_matrix = self.test_confusion_matrix(preds, targets)
        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/auc", self.test_auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "test/precision",
            self.test_precision,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test/recall", self.test_recall, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("test/confusion_matrix", confusion_matrix, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        # # Create a DataFrame from the test results
        # test_results = pd.DataFrame(
        #     {
        #         "predictions": self.test_preds.tolist(),
        #         "targets": self.test_targets.tolist(),
        #         "probabilities": self.test_probs.tolist(),
        #     }
        # )

        # os.makedirs("outputs", exisits_ok=True)
        # # Save the results to an Excel file
        # test_results.to_excel(
        #     f"{self.hparams.model_name.lower()}_test_results.xlsx", index=False
        # )

        # Optionally, print confusion matrix and other details
        print(self.test_confusion_matrix(self.test_preds, self.test_targets))
        print(self.test_preds.tolist())
        print(self.test_targets.tolist())

    def predict_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, probs, preds, targets = self.model_step(batch)
        # image = batch[0]
        # label = batch[1]
        # x = image
        # y = label
        # logits = self.net(x)
        model = self.net
        # model_name = "resnet"
        model_name = model._get_name()
        model_name = model_name.lower()
        # for name, _ in model.named_modules():
        #     print(name)
        # 模型层配置字典
        layer_configs = {
            "resnet": {
                "index": 0,
                "target_layers": "model.layer4",
                "fc_layers": "model.fc",
            },
            "senet": {
                "index": 1,
                "target_layers": "model.layer4",
                "fc_layers": "model.fc",
            },
            "convnext": {
                "index": 2,
                "target_layers": "model.stages.3",
                "fc_layers": "model.head",
            },
            "visiontransformer": {
                "index": 3,
                "target_layers": "model.blocks.11",
                "fc_layers": "model.head",
            },
            "globalcontextvit": {
                "index": 4,
                "target_layers": "model.stages.3",
                "fc_layers": "model.head",
            },
            "swintransformer": {
                "index": 5,
                "target_layers": "model.layers.3",
                "fc_layers": "model.head",
            },
            "vssm": {
                "index": 6,
                "target_layers": "layers.3",
                "fc_layers": "head",
            },
            "nnmambaencoder": {
                "index": 7,
                "target_layers": "layer3",
                "fc_layers": "mlp",
            },
            "visionmamba": {
                "index": 8,
                "target_layers": "layers.23",
                "fc_layers": "norm_f",
            },
            "multiplesequencehybridmamba": {
                "index": 9,
                "target_layers": "feature_extractor.msf_encoder4",
                "fc_layers": "cls_head",
            },
        }
        model_names_map = {
            "resnet": "ResNet",
            "senet": "SENet",
            "convnext": "ConvNeXt",
            "vit": "ViT",
            "gcvit": "GCViT",
            "swint": "SwinT",
            "medmamba": "MedMamba",
            "nnmamba": "nnMamba",
            "vim": "ViM",
            "mshm": "MSHM",
        }
        # with torch.set_grad_enabled(True):
        with torch.enable_grad():
            with torch.inference_mode(False):
                # model.train()
                # image = image.requires_grad_(True)
                # torchinfo.summary(model, input_data=img)
                grad_image = batch[0].clone().requires_grad_(True)
                model = model

                input_tensor = grad_image
                if model_name == "resnet":
                    target_layers = [model.model.layer4]
                    cam = GradCAM(model=model, target_layers=target_layers)
                elif model_name == "senet":
                    target_layers = [model.model.layer4]
                    cam = GradCAM(model=model, target_layers=target_layers)
                elif model_name == "convnext":
                    target_layers = [model.model.stages[-1]]
                    cam = GradCAM(model=model, target_layers=target_layers)
                elif model_name == "visiontransformer":
                    target_layers = [model.model.blocks[-1].norm1]

                    def reshape_transform(tensor, height=14, width=14):
                        result = tensor[:, 1:, :].reshape(
                            tensor.size(0), height, width, tensor.size(2)
                        )

                        # Bring the channels to the first dimension,
                        # like in CNNs.
                        result = result.transpose(2, 3).transpose(1, 2)
                        return result

                    cam = GradCAM(
                        model=model,
                        target_layers=target_layers,
                        reshape_transform=reshape_transform,
                    )
                elif model_name == "globalcontextvit":

                    # def reshape_transform(tensor, height=14, width=14):
                    #     result = tensor[:, 1:, :].reshape(
                    #         tensor.size(0), height, width, tensor.size(2)
                    #     )

                    #     # Bring the channels to the first dimension,
                    #     # like in CNNs.
                    #     result = result.transpose(2, 3).transpose(1, 2)
                    #     return result

                    target_layers = [model.model.stages[-1]]
                    cam = GradCAM(
                        model=model,
                        target_layers=target_layers,
                        # reshape_transform=reshape_transform,
                    )
                elif model_name == "swintransformer":
                    target_layers = [model.model.layers[-1].blocks[-1].norm2]

                    def reshape_transform(tensor, height=7, width=7):
                        result = tensor.reshape(
                            tensor.size(0), height, width, tensor.size(2)
                        )

                        # Bring the channels to the first dimension,
                        # like in CNNs.
                        result = result.transpose(2, 3).transpose(1, 2)
                        return result

                    cam = GradCAM(
                        model=model,
                        target_layers=target_layers,
                        reshape_transform=reshape_transform,
                    )
                elif model_name == "vssm":

                    # def reshape_transform(tensor, height=14, width=14):
                    #     result = tensor[:, 1:, :].reshape(
                    #         tensor.size(0), height, width, tensor.size(2)
                    #     )

                    #     # Bring the channels to the first dimension,
                    #     # like in CNNs.
                    #     result = result.transpose(2, 3).transpose(1, 2)
                    #     return result

                    target_layers = [model.layers[-1].blocks[-1].conv33conv33conv11[0]]
                    cam = GradCAM(
                        model=model,
                        target_layers=target_layers,
                        # reshape_transform=reshape_transform,
                    )
                elif model_name == "nnmambaencoder":
                    target_layers = [model.layer3]
                    cam = GradCAM(model=model, target_layers=target_layers)
                elif model_name == "visionmamba":

                    def reshape_transform(tensor, height=14, width=14):
                        result = tensor[:, 1:, :].reshape(
                            tensor.size(0), height, width, tensor.size(2)
                        )

                        # Bring the channels to the first dimension,
                        # like in CNNs.
                        result = result.transpose(2, 3).transpose(1, 2)
                        return result

                    target_layers = [model.layers[-1].drop_path]
                    cam = GradCAM(
                        model=model,
                        target_layers=target_layers,
                        reshape_transform=reshape_transform,
                    )
                elif model_name == "multiplesequencehybridmamba":
                    target_layers = [model.feature_extractor.msf_encoder4]
                    cam = GradCAM(model=model, target_layers=target_layers)

                else:
                    raise ValueError(f"Unsupported model_name: {model_name}")

                grayscale_cam_batch = cam(input_tensor=input_tensor, targets=None)
                # In this example grayscale_cam has only one image in the batch:
                for i in range(grayscale_cam_batch.shape[0]):
                    grayscale_cam = grayscale_cam_batch[i, :]
                    for seq_idx in range(3):
                        gray_img = grad_image[i, seq_idx].cpu().detach().numpy()
                        rgb_img = np.stack([gray_img] * 3, axis=-1)
                        output_dir = f"outputs/cam/{batch_idx}/{i}"
                        os.makedirs(output_dir, exist_ok=True)
                        # cv2.imwrite(f"{output_dir}/{seq_idx}_0_gt.jpg", np.uint8(255 * rgb_img))
                        cam_image = show_cam_on_image(rgb_img, grayscale_cam)
                        cv2.imwrite(f"{output_dir}/{seq_idx}_{layer_configs[model_name]['index']}_{model_name}_cam.jpg", cam_image)

                # gradcampp = GradCAM(
                #     nn_module=model,
                #     target_layers=layer_configs[model_name]["target_layers"],
                # )
                # cam_result = gradcampp(x=grad_image, class_idx=None)
                # # model = model.to(cpu)
                # # cam_result = gradcampp(x=torch.rand((32, 3, 224, 224)), class_idx=None)

                # cam_result = 1 - cam_result
                # seq_idx = 0
                # batch_idx = 0
                # # plt.imshow(1 - np.squeeze(cam(x=img,class_idx=1)[:, :, :, :, depth_slice]).detach().cpu().numpy(), cmap='jet')
                # # plt.imshow(np.squeeze(cam(x=img,class_idx=1)[:, :, :, :, depth_slice]).detach().cpu().numpy(), cmap='jet')
                # # 提取原始图像的单个切片并转换为灰度图
                # original_slice = np.squeeze(
                #     image[: batch_idx + 1, seq_idx, :, :].detach().cpu().numpy()
                # )
                # # # 归一化到0..1范围
                # # original_slice = (original_slice - original_slice.min()) / (
                # #     original_slice.max() - original_slice.min()
                # # )
                # # 将灰度图转换为伪RGB格式
                # original_slice_gray = (original_slice * 255).astype(np.uint8)
                # original_slice_rgb = np.stack([original_slice_gray] * 3, axis=-1)
                # attention_slice = (
                #     cam_result[: batch_idx + 1, seq_idx, :, :].detach().cpu().numpy()
                # )
                # attention_slice = np.squeeze(
                #     (attention_slice - attention_slice.min())
                #     / (attention_slice.max() - attention_slice.min())
                # )
                # # 忽略jet中透明度的通道
                # attention_rgb = cm.jet(attention_slice)[:, :, :3]
                # # 调整热力图的透明度
                # alpha = 0.5
                # # 创建一个掩码，其中 attention_rgb 为 0 的地方掩码值为 0，其他地方为 1
                # mask = np.where(attention_rgb > 0, 1, 0)
                # # 将掩码应用于 attention_rgb，这样在 attention_rgb 为 0 的地方，掩码值为 0
                # # 这样在这些地方叠加时不会改变 original_slice_rgb 的值
                # masked_attention_rgb = attention_rgb * mask
                # # # 叠加图像
                # # overlay_result = (
                # #     original_slice_rgb / 255 * (1 - alpha * mask) + masked_attention_rgb * alpha
                # # )
                # # 不处理的叠加方式
                # overlay_result = (
                #     original_slice_rgb / 255 * (1 - alpha) + attention_rgb * alpha
                # )

                # # 绘制三个图像：原始图像、注意力图、叠加后的图像
                # fig, axes = plt.subplots(1, 3, figsize=(30, 15), facecolor="white")

                # # 原始图像
                # cmap_gray = "gray"
                # ax = axes[0]
                # im_show = ax.imshow(original_slice, cmap=cmap_gray)
                # ax.axis("off")
                # fig.colorbar(im_show, ax=ax)
                # ax.set_title("Original Image")

                # # 注意力图
                # cmap_jet = "jet"
                # ax = axes[1]
                # im_show = ax.imshow(attention_slice, cmap=cmap_jet)
                # ax.axis("off")
                # fig.colorbar(im_show, ax=ax)
                # ax.set_title("Attention Map")

                # # 叠加后的图像
                # ax = axes[2]
                # ax.imshow(overlay_result)
                # ax.axis("off")
                # ax.set_title("Overlay Image")

                # # 保存叠加后的图像
                # # overlay_img_path = (
                # #     f"{hm_output_dir}/hm_{mode}_{sequence_name}_{model_name}.jpg"
                # # )
                # overlay_img_path = f"{model_name}_test_hm.jpg"
                # plt.imsave(overlay_img_path, overlay_result)
                # # # 显示并关闭图像
                # # plt.show()
                # # plt.close(fig)
                # print(f"Overlay image saved to {overlay_img_path}")
                # cam = CAM(
                #     nn_module=model,
                #     target_layers=layer_configs[model_name]["target_layers"],
                #     fc_layers=layer_configs[model_name]["fc_layers"],
                # )
                # cam_result = cam(x=img, class_idx=[1])
                # gradcam = GradCAM(
                #     nn_module=model,
                #     target_layers=layer_configs[model_name]["target_layers"],
                # )
                # cam_result = gradcam(x=img, class_idx=[1])

        return probs, preds, targets

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = LBLLitModule(None, None, None, None)
