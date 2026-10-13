from pathlib import Path

from PIL import Image, ImageOps
from torchvision import transforms

# 根据当前文件找到项目根目录，换电脑后不需要修改用户名和绝对路径。
root = Path(__file__).resolve().parents[1]
IMAGE_ROOT = root / "DATA" / "RAW_DATA" / "MILK10k_DATA"

# 第一次读取图片时建立索引(字典)；后续复用，不再为每张图片扫描整个文件夹。
# 这样就可以做到O(1)复杂度而不是O(N)
IMAGE_INDEX = None


def build_image_index(image_root):
    """建立 image_id -> 图片路径的字典。"""
    # 图片目录
    image_root = Path(image_root)
    # safty check if 图片目录无法查看
    if not image_root.is_dir():
        raise FileNotFoundError(f"图片目录不存在: {image_root}")
    # 创建dict
    image_index = {}
    # 对于每一个 jpg图，提取文件名字，如果存在相同名字则报错。否则将id录入dict中，指向对应路径
    for image_path in image_root.rglob("*.jpg"):

        image_id = image_path.stem
        if image_id in image_index:
            raise ValueError(f"图片 ID 重复，无法确定使用哪张图片: {image_id}")
        image_index[image_id] = image_path

    # safty check：如果当前路径没有任何jpg图报错！
    if not image_index:
        raise FileNotFoundError(f"图片目录中没有 JPG 图片: {image_root}")
    return image_index


# 图片处理流程def
def build_transform(training=False):
    """统一缩放到 320；训练时随机旋转和翻转，最后中心裁剪到 224。"""
    # 操作：将图片的短边从原先统一短边450缩小为320
    operations = [transforms.Resize(320)]
    # 对于training 的时候，进行旋转处理
    if training:
        # 将图片进行随机旋转，水平/垂直镜像，基于随机值进行，保证机器学习的时候对角度不敏感
        operations.extend([
            transforms.RandomRotation(
                degrees=180,
                interpolation=transforms.InterpolationMode.BILINEAR,
                expand=False,
                fill=0,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ])
        # 对旋转完的函数进行中心裁剪为224*224，然后对RGB颜色进行统一处理，使得他符合RESNET-50的舒适范围
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


def preprocess_image(image, training=False):
    """
    1. 将图片短边缩放到 320。
    2. 训练时随机旋转和水平、垂直翻转；随后中心裁剪到 224 × 224。
       验证时直接中心裁剪，不需要考虑旋转。
    3. 转为 3 × 224 × 224 Tensor，并使用 ImageNet 均值和标准差归一化。
    """
    # 将对应输入的image以是否是测试集/训练集进行对应处理完图像的返回
    transform = TRAIN_TRANSFORM if training else VAL_TRANSFORM
    return transform(image)


def get_tensor(image_id, training=False):
    """输入图片 ID（例如 ISIC_4671410），返回预处理后的 Tensor。"""
    global IMAGE_INDEX
    # 在第一次使用这个的时候，建立图片dict索引
    if IMAGE_INDEX is None:
        IMAGE_INDEX = build_image_index(IMAGE_ROOT)

    # 如果需要转为tensor的图片不在dict中，则报错，否则拿到这个图片所在的路径
    if image_id not in IMAGE_INDEX:
        raise FileNotFoundError(f"图片 ID 不在索引中: {image_id}")
    image_path = IMAGE_INDEX[image_id]

    with Image.open(image_path) as image:
        # 以RGB形式打开图片，病基于preprocess_image处理流程返回处理完成的图片tensor
        image = ImageOps.exif_transpose(image).convert("RGB")
        return preprocess_image(image, training=training)
