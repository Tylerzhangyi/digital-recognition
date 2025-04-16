import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# 定义模型结构（必须和训练时一样）
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# 实例化模型并加载参数
model = Net()
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
model.eval()  # 进入推理模式

# 定义和训练时一样的图像预处理
transform = transforms.Compose([
    transforms.Resize((28, 28)),                # 缩放为28x28
    transforms.Grayscale(),                     # 灰度图
    transforms.ToTensor(),                      # 转换为Tensor
    transforms.Normalize((0.1307,), (0.3081,))  # 归一化
])

# 加载图像并处理
image_path = '00.png'  # 你想识别的图片路径
image = Image.open(image_path)
image = transform(image)
image = image.unsqueeze(0)  # 增加 batch 维度，变成 (1, 1, 28, 28)

# 推理预测
with torch.no_grad():
    output = model(image)
    pred = output.argmax(dim=1, keepdim=True)
    print("预测结果是：", pred.item())
