from pathlib import Path
from PIL import Image
from torchvision import transforms

# 这里需要根据自己路径进行修改！！！！！
root = Path("/Users/zhangxurui/PycharmProjects/Machine-Vision-Project")
IMAGE_ROOT = root / "DATA" / "RAW_DATA" / "MILK10k_DATA"


# DATA Preprocess

# 对每一个图片都建立一个可索引的dict，使得查图片路径从O(n)变为O(1)
# 这里在进行引用的时候会自动执行
# 其中为 IMAGE_INDEX[图片名字如ISIC_4671410] = 图片路径
IMAGE_INDEX = {}
for image_path in IMAGE_ROOT.rglob("*.jpg"):
    image_id = image_path.stem
    IMAGE_INDEX[image_id] = image_path


# 对输入图片进行预处理
def preprocess_image(image):
    """
    1. 对图片短边进行缩放，缩到256
    2. 将图片裁剪到224*224，基于图片中心进行裁剪
    3. 将图片转化为Tensor： 3 × 224 × 224，也就是每一个颜色在每一行每一列的具体数字
    4. 基于ImageNet pretrained ResNet 喜欢的均值与标准差将每一个像素的RGB数字进行标准化
    5. 返回这个可以直接使用的Tensor
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return transform(image)


def get_tensor(image_id):
    # 输入图片id。如ISIC_4671410。则可以得到这个图片相对应的tensor
    image_path = next(IMAGE_ROOT.rglob(f"{image_id}.jpg"))

    image = Image.open(image_path).convert("RGB")

    tensor = preprocess_image(image)

    return tensor
