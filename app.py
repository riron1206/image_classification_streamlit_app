"""
画像を Streamlit 上で表示して、Pytorchのimagenetの学習済みモデルで分類
grad-cam などの可視化選べる
https://towardsdatascience.com/create-an-image-classification-web-app-using-pytorch-and-streamlit-f043ddf00c24

Usage:
    $ conda activate lightning
    $ streamlit run ./app.py
"""

# ==============================================
# https://github.com/jacobgil/pytorch-grad-cam/tree/master/pytorch_grad_cam/utils/
# image.py
# ==============================================
import cv2
import numpy as np
import torch
from torchvision.transforms import Compose, Normalize, ToTensor


def preprocess_image(img: np.ndarray, mean=None, std=None) -> torch.Tensor:
    if std is None:
        std = [0.5, 0.5, 0.5]
    if mean is None:
        mean = [0.5, 0.5, 0.5]

    preprocessing = Compose([ToTensor(), Normalize(mean=mean, std=std)])

    return preprocessing(img.copy()).unsqueeze(0)


def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)


def show_cam_on_image(
    img: np.ndarray,
    mask: np.ndarray,
    use_rgb: bool = False,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception("The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


# ==============================================
# https://github.com/jacobgil/pytorch-grad-cam/tree/master/pytorch_grad_cam
# guided_backprop.py
# ==============================================
import numpy as np
import torch
from torch.autograd import Function


class GuidedBackpropReLU(Function):
    @staticmethod
    def forward(self, input_img):
        positive_mask = (input_img > 0).type_as(input_img)
        output = torch.addcmul(
            torch.zeros(input_img.size()).type_as(input_img), input_img, positive_mask
        )
        self.save_for_backward(input_img, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input_img, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input_img > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(
            torch.zeros(input_img.size()).type_as(input_img),
            torch.addcmul(
                torch.zeros(input_img.size()).type_as(input_img),
                grad_output,
                positive_mask_1,
            ),
            positive_mask_2,
        )
        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == "ReLU":
                    module_top._modules[idx] = GuidedBackpropReLU.apply

        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, input_img, target_category=None):
        if self.cuda:
            input_img = input_img.cuda()

        input_img = input_img.requires_grad_(True)

        output = self.forward(input_img)

        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = one_hot.cuda()

        one_hot = torch.sum(one_hot * output)
        one_hot.backward(retain_graph=True)

        output = input_img.grad.cpu().data.numpy()
        output = output[0, :, :, :]
        output = output.transpose((1, 2, 0))
        return output


# ==============================================
# https://github.com/jacobgil/pytorch-grad-cam/tree/master/pytorch_grad_cam
# activations_and_gradients.py
# ==============================================
class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layer, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu())

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients = [grad.cpu()] + self.gradients

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)


# ==============================================
# https://github.com/jacobgil/pytorch-grad-cam/tree/master/pytorch_grad_cam
# base_cam.py
# ==============================================
import cv2
import numpy as np
import torch

# from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients


class BaseCAM:
    def __init__(self, model, target_layer, use_cuda=False, reshape_transform=None):
        self.model = model.eval()
        self.target_layer = target_layer
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.reshape_transform = reshape_transform
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layer, reshape_transform
        )

    def forward(self, input_img):
        return self.model(input_img)

    def get_cam_weights(self, input_tensor, target_category, activations, grads):
        raise Exception("Not Implemented")

    def get_loss(self, output, target_category):
        return output[:, target_category]

    def get_cam_image(self, input_tensor, target_category, activations, grads):
        weights = self.get_cam_weights(
            input_tensor, target_category, activations, grads
        )
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        return cam

    def __call__(self, input_tensor, target_category=None):
        if self.cuda:
            input_tensor = input_tensor.cuda()

        output = self.activations_and_grads(input_tensor)

        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy())

        self.model.zero_grad()
        loss = self.get_loss(output, target_category)
        loss.backward(retain_graph=True)

        activations = (
            self.activations_and_grads.activations[-1].cpu().data.numpy()[0, :]
        )
        grads = self.activations_and_grads.gradients[-1].cpu().data.numpy()[0, :]
        cam = self.get_cam_image(input_tensor, target_category, activations, grads)

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input_tensor.shape[2:][::-1])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


# ==============================================
# https://github.com/jacobgil/pytorch-grad-cam/tree/master/pytorch_grad_cam
# grad_cam.py
# ==============================================
import cv2
import numpy as np
import torch

# from pytorch_grad_cam.base_cam import BaseCAM


class GradCAM(BaseCAM):
    """GradCAM:	Weight the 2D activations by the average gradient.
    平均勾配で2Dアクティベーションに重みを付け
    """

    def __init__(self, model, target_layer, use_cuda=False, reshape_transform=None):
        super(GradCAM, self).__init__(model, target_layer, use_cuda, reshape_transform)

    def get_cam_weights(self, input_tensor, target_category, activations, grads):
        return np.mean(grads, axis=(1, 2))


# ==============================================
# https://github.com/jacobgil/pytorch-grad-cam/tree/master/pytorch_grad_cam
# grad_cam_plusplus.py
# ==============================================
import cv2
import numpy as np
import torch

# from pytorch_grad_cam.base_cam import BaseCAM


class GradCAMPlusPlus(BaseCAM):
    """GradCAM++: Like GradCAM but uses second order gradients.
    GradCAMと同様ですが、2次グラデーションを使用
    """

    def __init__(self, model, target_layer, use_cuda=False, reshape_transform=None):
        super(GradCAMPlusPlus, self).__init__(
            model, target_layer, use_cuda, reshape_transform
        )

    def get_cam_weights(self, input_tensor, target_category, activations, grads):
        grads_power_2 = grads ** 2
        grads_power_3 = grads_power_2 * grads
        # Equation 19 in https://arxiv.org/abs/1710.11063
        sum_activations = np.sum(activations, axis=(1, 2))
        eps = 0.000001
        aij = grads_power_2 / (
            2 * grads_power_2 + sum_activations[:, None, None] * grads_power_3 + eps
        )

        # Now bring back the ReLU from eq.7 in the paper,
        # And zero out aijs where the activations are 0
        aij = np.where(grads != 0, aij, 0)

        weights = np.maximum(grads, 0) * aij
        weights = np.sum(weights, axis=(1, 2))
        return weights


# ==============================================
# https://github.com/jacobgil/pytorch-grad-cam/tree/master/pytorch_grad_cam
# xgrad_cam.py
# ==============================================
import cv2
import numpy as np
import torch

# from pytorch_grad_cam.base_cam import BaseCAM


class XGradCAM(BaseCAM):
    """XGradCAM: Like GradCAM but scale the gradients by the normalized activations.
    GradCAMと同様ですが、正規化されたアクティベーションによってグラデーションをスケーリング
    """

    def __init__(self, model, target_layer, use_cuda=False, reshape_transform=None):
        super(XGradCAM, self).__init__(model, target_layer, use_cuda, reshape_transform)

    def get_cam_weights(self, input_tensor, target_category, activations, grads):
        sum_activations = np.sum(activations, axis=(1, 2))
        eps = 1e-7
        weights = grads * activations / (sum_activations[:, None, None] + eps)
        weights = weights.sum(axis=(1, 2))
        return weights


# ==============================================
# https://github.com/jacobgil/pytorch-grad-cam/tree/master/pytorch_grad_cam
# ablation_cam.py
# ==============================================
import cv2
import numpy as np
import torch

# from pytorch_grad_cam.base_cam import BaseCAM


class AblationLayer(torch.nn.Module):
    def __init__(self, layer, reshape_transform, indices):
        super(AblationLayer, self).__init__()

        self.layer = layer
        self.reshape_transform = reshape_transform
        # The channels to zero out:
        self.indices = indices

    def forward(self, x):
        self.__call__(x)

    def __call__(self, x):
        output = self.layer(x)

        # Hack to work with ViT,
        # Since the activation channels are last and not first like in CNNs
        # Probably should remove it?
        if self.reshape_transform is not None:
            output = output.transpose(1, 2)

        for i in range(output.size(0)):

            # Commonly the minimum activation will be 0,
            # And then it makes sense to zero it out.
            # However depending on the architecture,
            # If the values can be negative, we use very negative values
            # to perform the ablation, deviating from the paper.
            if torch.min(output) == 0:
                output[i, self.indices[i], :] = 0
            else:
                ABLATION_VALUE = 1e5
                output[i, self.indices[i], :] = torch.min(output) - ABLATION_VALUE

        if self.reshape_transform is not None:
            output = output.transpose(2, 1)

        return output


def replace_layer_recursive(model, old_layer, new_layer):
    for name, layer in model._modules.items():
        if layer == old_layer:
            model._modules[name] = new_layer
            return True
        elif replace_layer_recursive(layer, old_layer, new_layer):
            return True
    return False


class AblationCAM(BaseCAM):
    """AblationCAM: Zero out activations and measure how the output drops (this repository includes a fast batched implementation)
    アクティベーションをゼロにし、出力がどのように低下​​するかを測定します（このリポジトリには高速バッチ実装が含まれています）
    """

    def __init__(self, model, target_layer, use_cuda=False, reshape_transform=None):
        super(AblationCAM, self).__init__(
            model, target_layer, use_cuda, reshape_transform
        )

    def get_cam_weights(self, input_tensor, target_category, activations, grads):
        with torch.no_grad():
            original_score = self.model(input_tensor)[0, target_category].cpu().numpy()

        ablation_layer = AblationLayer(
            self.target_layer, self.reshape_transform, indices=[]
        )
        replace_layer_recursive(self.model, self.target_layer, ablation_layer)

        weights = []

        if hasattr(self, "batch_size"):
            BATCH_SIZE = self.batch_size
        else:
            BATCH_SIZE = 32

        with torch.no_grad():
            batch_tensor = input_tensor.repeat(BATCH_SIZE, 1, 1, 1)
            for i in range(0, activations.shape[0], BATCH_SIZE):
                ablation_layer.indices = list(range(i, i + BATCH_SIZE))

                if i + BATCH_SIZE > activations.shape[0]:
                    keep = i + BATCH_SIZE - activations.shape[0] - 1
                    batch_tensor = batch_tensor[:keep]
                    ablation_layer.indices = ablation_layer.indices[:keep]
                weights.extend(
                    self.model(batch_tensor)[:, target_category].cpu().numpy()
                )

        weights = np.float32(weights)
        weights = (original_score - weights) / original_score

        # replace the model back to the original state
        replace_layer_recursive(self.model, ablation_layer, self.target_layer)
        return weights


# ==============================================
# https://github.com/jacobgil/pytorch-grad-cam/tree/master/pytorch_grad_cam
# score_cam.py
# ==============================================
import cv2
import numpy as np
import torch

# from pytorch_grad_cam.base_cam import BaseCAM


class ScoreCAM(BaseCAM):
    """ScoreCAM: Perbutate the image by the scaled activations and measure how the output drops.
    スケーリングされたアクティベーションによって画像にパービュートし、出力がどのように低下​​するかを測定
    """

    def __init__(self, model, target_layer, use_cuda=False, reshape_transform=None):
        super(ScoreCAM, self).__init__(
            model, target_layer, use_cuda, reshape_transform=reshape_transform
        )

    def get_cam_weights(self, input_tensor, target_category, activations, grads):
        with torch.no_grad():
            upsample = torch.nn.UpsamplingBilinear2d(size=input_tensor.shape[2:])
            activation_tensor = torch.from_numpy(activations).unsqueeze(0)
            if self.cuda:
                activation_tensor = activation_tensor.cuda()

            upsampled = upsample(activation_tensor)
            upsampled = upsampled[
                0,
            ]

            maxs = upsampled.view(upsampled.size(0), -1).max(dim=-1)[0]
            mins = upsampled.view(upsampled.size(0), -1).min(dim=-1)[0]
            maxs, mins = maxs[:, None, None], mins[:, None, None]
            upsampled = (upsampled - mins) / (maxs - mins)

            input_tensors = input_tensor * upsampled[:, None, :, :]

            if hasattr(self, "batch_size"):
                BATCH_SIZE = self.batch_size
            else:
                BATCH_SIZE = 16

            scores = []
            for i in range(0, input_tensors.size(0), BATCH_SIZE):
                batch = input_tensors[i : i + BATCH_SIZE, :]
                outputs = self.model(batch).cpu().numpy()[:, target_category]
                scores.append(outputs)
            scores = torch.from_numpy(np.concatenate(scores))
            weights = torch.nn.Softmax(dim=-1)(scores).numpy()
            return weights


# ==============================================
# https://github.com/jacobgil/pytorch-grad-cam/tree/master/pytorch_grad_cam
# eigen_cam.py
# ==============================================
import cv2
import cv2
import numpy as np
import torch

# from pytorch_grad_cam.base_cam import BaseCAM

# https://arxiv.org/abs/2008.00299
class EigenCAM(BaseCAM):
    """EigenCAM:Takes the first principle component of the 2D Activations (no class discrimination, but seems to give great results).
    2Dアクティベーションの最初の主成分を取ります（クラスの識別はありませんが、素晴らしい結果が得られるようです）
    """

    def __init__(self, model, target_layer, use_cuda=False, reshape_transform=None):
        super(EigenCAM, self).__init__(model, target_layer, use_cuda, reshape_transform)

    def get_cam_image(self, input_tensor, target_category, activations, grads):
        reshaped_activations = (
            (activations).reshape(activations.shape[0], -1).transpose()
        )
        # Centering before the SVD seems to be important here,
        # Otherwise the image returned is negative
        reshaped_activations = reshaped_activations - reshaped_activations.mean(axis=0)
        U, S, VT = np.linalg.svd(reshaped_activations, full_matrices=True)
        projection = reshaped_activations @ VT[0, :]
        projection = projection.reshape(activations.shape[1:])
        return projection


# ==============================================
# https://github.com/jacobgil/pytorch-grad-cam/tree/master
# cam.py
# ==============================================
import argparse
import cv2
import numpy as np
import torch
from torchvision import models

# from pytorch_grad_cam import (GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,)
# from pytorch_grad_cam import GuidedBackpropReLUModel
# from pytorch_grad_cam.utils.image import (show_cam_on_image, deprocess_image, preprocess_image,)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-cuda",
        action="store_true",
        default=False,
        help="Use NVIDIA GPU acceleration",
    )
    # parser.add_argument("--image-path", type=str, default="./image/dog.jpg", help="Input image path")
    parser.add_argument(
        "--method",
        type=str,
        default="gradcam",
        help="Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam",
    )

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    # if args.use_cuda:
    #    print("Using GPU for acceleration")
    # else:
    #    print("Using CPU for computation")

    return args


def cam_main(args, model, target_layer, cv2_img, target_category=None):
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """
    methods = {
        "gradcam": GradCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        # "eigengradcam": EigenGradCAM,
    }

    # models.resnet50(pretrained=True)

    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4[-1]
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    # target_layer = model.layer4[-1]

    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")

    cam = methods[args.method](
        model=model, target_layer=target_layer, use_cuda=args.use_cuda
    )

    # rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    rgb_img = cv2_img[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(
        rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    # target_category = None

    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = 32

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    cam_image = show_cam_on_image(rgb_img, grayscale_cam)

    # gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    # gb = gb_model(input_tensor, target_category=target_category)

    # cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    # cam_gb = deprocess_image(cam_mask * gb)
    # gb = deprocess_image(gb)

    # cv2.imwrite(f"{args.method}_cam.jpg", cam_image)
    # cv2.imwrite(f"{args.method}_gb.jpg", gb)
    # cv2.imwrite(f"{args.method}_cam_gb.jpg", cam_gb)

    return cam_image


# ==============================================
# app.py
# ==============================================
import sys
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image


@st.cache(allow_output_mutation=True)
def load_model():
    # model = models.resnet101(pretrained=True)
    # model = models.resnet50(pretrained=True)
    model = models.resnet18(pretrained=True)
    # model = models.mobilenet_v2(pretrained=True)

    target_layer = model.layer4[-1]

    return model, target_layer


def load_file_up_image(file_up, size=224):
    pillow_img = Image.open(file_up)
    pillow_img = pillow_img.resize((size, size)) if size is not None else pillow_img
    cv2_img = pil2cv(pillow_img)
    return pillow_img, cv2_img


def pil2cv(pillow_img):
    """ PIL型 -> OpenCV型
    https://qiita.com/derodero24/items/f22c22b22451609908ee"""
    new_image = np.array(pillow_img, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image


def preprocessing_image(image_pil_array: "PIL.Image"):
    # https://pytorch.org/docs/stable/torchvision/models.html
    transform = transforms.Compose(
        [
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    batch_t = torch.unsqueeze(transform(image_pil_array), 0)
    return batch_t


def predict(args, pillow_img, cv2_img):
    batch_t = preprocessing_image(pillow_img)

    model, target_layer = load_model()
    model.eval()
    out = model(batch_t)

    with open("imagenet_classes.txt") as f:
        classes = [line.strip() for line in f.readlines()]

    prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
    _, indices = torch.sort(out, descending=True)
    top5 = [(classes[idx], prob[idx].item()) for idx in indices[0][:5]]

    top1_id = top5[0][0].split(",")[0]
    cam_image = cam_main(
        args, model, target_layer, cv2_img, target_category=int(top1_id)
    )
    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB)

    return top5, cam_image


def main():
    st.set_option("deprecation.showfileUploaderEncoding", False)
    st.title("Simple Image Classification App")
    st.write("")

    args = get_args()

    # ファイルupload
    file_up = st.file_uploader("Upload an image", type="jpg")

    # ラジオボタン
    now_st_method = ""
    st_method = st.radio(
        "Select Class Activation Map method",
        (
            "gradcam",
            "gradcam++",
            "xgradcam",
            "ablationcam",
            "scorecam",
            "eigencam",
            # "eigengradcam",
        ),
    )
    args.__setattr__("method", st_method)  # args.method

    def run():
        pillow_img, cv2_img = load_file_up_image(file_up)

        st.image(
            pillow_img,
            caption="Uploaded Image. Resize (224, 224).",
            use_column_width=True,
        )

        st.write("")
        st.write("Just a second...")
        labels, cam_image = predict(args, pillow_img, cv2_img)
        st.image(
            cam_image.transpose(0, 1, 2),
            caption="Class Activation Map method: " + st_method,
            use_column_width=True,
        )

        # print out the top 5 prediction labels with scores
        for i in labels[:1]:
            st.write("Prediction (index, name)", i[0], ",   Score: ", round(i[1], 2))

        now_st_method = st_method

    if file_up is not None and now_st_method != st_method:
        run()
    else:
        img_url = "https://github.com/riron1206/image_classification_streamlit_app/blob/master/image/dog.jpg?raw=true"
        st.image(
            img_url,
            caption="Sample Image. Please download and upload.",
            use_column_width=True,
        )


if __name__ == "__main__":
    main()
