import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import boto3
import base64
import io
import os
from dotenv import load_dotenv

load_dotenv()

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
        print("开始前向传播...")
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        print("完成第一层卷积和池化...")
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        print("完成第二层卷积和池化...")
        x = x.view(-1, 320)
        print("完成展平操作...")
        x = F.relu(self.fc1(x))
        print("完成第一个全连接层...")
        x = F.dropout(x, training=self.training)
        print("完成 dropout 操作...")
        x = self.fc2(x)
        print("完成第二个全连接层...")
        return F.log_softmax(x, dim=1)

# 实例化模型并加载参数
model = Net()
print("加载模型参数...")
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
model.eval()  # 进入推理模式
print("模型已加载并进入推理模式...")

# 定义和训练时一样的图像预处理
transform = transforms.Compose([
    transforms.Resize((28, 28)),                # 缩放为28x28
    transforms.Grayscale(),                     # 灰度图
    transforms.ToTensor(),                      # 转换为Tensor
    transforms.Normalize((0.1307,), (0.3081,))  # 归一化
])
print("图像预处理函数已定义...")

# 初始化 SQS 客户端，使用环境变量获取 AWS 凭证
def get_sqs_client():
    try:
        print("正在初始化 SQS 客户端...")
        # 从环境变量中读取 AWS 配置
        aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        region_name = os.getenv('AWS_REGION')

        if not all([aws_access_key_id, aws_secret_access_key, region_name]):
            raise ValueError("缺少必要的环境变量: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION")

        # 创建 SQS 客户端
        sqs = boto3.client(
            'sqs',
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
        print("SQS 客户端初始化成功...")
        return sqs
    except Exception as e:
        print(f"初始化 SQS 客户端失败: {e}")
        raise

# 获取 SQS 客户端
sqs = get_sqs_client()

# SQS 队列 URL
queue_url = 'https://sqs.ap-northeast-2.amazonaws.com/961341521760/task.fifo'

# 从 SQS 获取消息并处理图片
def process_sqs_message():
    print("开始监听 SQS 消息...")
    while True:
        print("尝试从 SQS 接收消息...")
        # 从 SQS 接收消息
        response = sqs.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=1,  # 每次最多接收一条消息
            WaitTimeSeconds=20      # 长轮询等待时间
        )

        if 'Messages' not in response:
            print("没有新消息")
            continue

        print("接收到新消息...")
        # 提取消息内容
        message = response['Messages'][0]
        receipt_handle = message['ReceiptHandle']
        body = message['Body']

        try:
            print("解码 Base64 图片数据...")
            # 假设消息体是 Base64 编码的图片数据
            image_data = base64.b64decode(body)
            image = Image.open(io.BytesIO(image_data))

            print("对图片进行预处理...")
            # 图片预处理
            image = transform(image)
            image = image.unsqueeze(0)  # 增加 batch 维度，变成 (1, 1, 28, 28)

            print("开始模型推理...")
            # 推理预测
            with torch.no_grad():
                output = model(image)
                pred = output.argmax(dim=1, keepdim=True)
                print(f"预测结果是：{pred.item()}")

        except Exception as e:
            print(f"处理消息时出错: {e}")

        finally:
            print("删除已处理的消息...")
            # 删除已处理的消息
            sqs.delete_message(
                QueueUrl=queue_url,
                ReceiptHandle=receipt_handle
            )
            print("消息已删除")

# 启动消息处理
process_sqs_message()