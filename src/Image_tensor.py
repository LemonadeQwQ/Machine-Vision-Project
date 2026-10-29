from pathlib import Path

from PIL import Image, ImageOps
from torchvision import transforms

# 根据当前文件找到项目根目录，换电脑后不需要修改用户名和绝对路径。
root = Path(__file__).resolve().parents[1]
IMAGE_ROOT = root / "DATA" / "images"

# 图片处理流程def
def build_transform(training=False, augmentation='none'):
    """统一缩放到 320，中心裁剪到 224；默认无增强；训练时可选择镜像或旋转。"""
    # 操作：将图片的短边从原先统一短边450缩小为320
    if augmentation not in ('none', 'mirror', 'rotation'):
        raise ValueError(f'Unknown augmentation: {augmentation}')
    operations = [transforms.Resize(320)]
    if training and augmentation == 'mirror':
        operations.extend([transforms.RandomHorizontalFlip(0.5), transforms.RandomVerticalFlip(0.5)])
    elif training and augmentation == 'rotation':
        operations.append(transforms.RandomRotation(30, interpolation=transforms.InterpolationMode.BILINEAR))
    # Validation always uses deterministic preprocessing.
    operations.extend([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    return transforms.Compose(operations)

# 在每一次进行训练/验证的时候，运行一次build_transform
TRAIN_TRANSFORM = build_transform(training=True)
VAL_TRANSFORM = build_transform(training=False)


def preprocess_image(image, training=False, augmentation='none'):
    """
    1. 将图片短边缩放到 320。
    2. 训练和验证均中心裁剪到 224 × 224，仅训练按配置进行镜像或旋转增强。
    3. 转为 3 × 224 × 224 Tensor，并使用 ImageNet 均值和标准差归一化。
    """
    # 将对应输入的image以是否是测试集/训练集进行对应处理完图像的返回
    transform = build_transform(training=True, augmentation=augmentation) if training else VAL_TRANSFORM
    return transform(image)


def load_image_tensor(image_path, training=False, augmentation='none'):
    """按 CSV 已确定的路径读取图片，保留原来的方向校正和预处理。"""
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return preprocess_image(image, training=training, augmentation=augmentation)


def get_tensor(image_id, training=False, image_root=IMAGE_ROOT):
    """兼容按统一图片编号读取的旧调用；直接定位文件，不扫描目录。"""
    lesion_id, _, modality = image_id.rpartition("_")
    if (not lesion_id.startswith("L") or not lesion_id[1:].isascii()
            or not lesion_id[1:].isdigit() or modality not in ("clinical", "dermoscopic")):
        raise ValueError(f"无效统一图片编号: {image_id!r}")
    path = Path(image_root) / lesion_id / f"{image_id}.jpg"
    return load_image_tensor(path, training=training)
