import torch
import torch.nn as nn
from torchvision import transforms
import timm
import cv2
import os
from PIL import Image

# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 类别名称
class_names = ['Real', 'Fake']  # 0: Real, 1: Fake

# 初始化 Xception 模型
def initialize_xception():
    model = timm.create_model('xception', pretrained=False)
    # 修改第一层卷积层为单通道输入
    original_conv1 = model.conv1
    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=original_conv1.out_channels,
        kernel_size=original_conv1.kernel_size,
        stride=original_conv1.stride,
        padding=original_conv1.padding,
        bias=original_conv1.bias is not None
    )
    with torch.no_grad():
        model.conv1.weight = nn.Parameter(original_conv1.weight.sum(dim=1, keepdim=True))
    # 修改最后的全连接层为 2 类输出，并添加 Dropout 层
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_ftrs, 2)
    )
    return model.to(device)

# 加载模型
model = initialize_xception()
model.load_state_dict(torch.load('best_model_epoch10_xcept.pth', map_location=device))
model.eval()

# 定义预处理转换（与训练时一致）
preprocess_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 创建临时文件夹用于保存中间结果
temp_dir = 'temp_processed_images'
os.makedirs(temp_dir, exist_ok=True)

# 定义预处理函数，包括高斯模糊和噪声提取，然后保存为 JPG 格式
def preprocess_and_save(image_path):
    # 读取图像并转换为灰度图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像：{image_path}")
    # 应用高斯模糊
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    # 提取噪声
    noise = cv2.absdiff(img, blurred)
    # 将 NumPy 数组转换为 PIL 图像
    noise_image = Image.fromarray(noise)
    # 保存到临时目录，文件名与原文件名相同
    base_name = os.path.basename(image_path)
    temp_image_path = os.path.join(temp_dir, base_name)
    noise_image.save(temp_image_path, format='JPEG')
    return temp_image_path

# 预测函数
def predict(image_path):
    # 预处理图像并保存为 JPG
    temp_image_path = preprocess_and_save(image_path)
    # 读取保存的 JPG 图像
    image = Image.open(temp_image_path).convert('L')  # 单通道灰度图像
    # 应用预处理转换
    image = preprocess_transform(image)
    # 添加批量维度
    image = image.unsqueeze(0).to(device)
    # 禁用梯度计算
    with torch.no_grad():
        outputs = model(image)
        # 获取预测结果
        _, predicted = torch.max(outputs, 1)
    predicted_class = predicted.item()
    # 删除临时文件
    os.remove(temp_image_path)
    return predicted_class  # 返回类别索引（0 或 1）

if __name__ == "__main__":
    # 单张图像路径
    image_path = '5752.jpeg'  # 替换为实际的图像路径
    try:
        predicted_class = predict(image_path)
        class_name = class_names[predicted_class]
        print(f'图像 {image_path} 的预测结果: {class_name}')
    except Exception as e:
        print(f"处理图像 {image_path} 时出错：{e}")

    # 删除临时目录
    import shutil
    shutil.rmtree(temp_dir)
