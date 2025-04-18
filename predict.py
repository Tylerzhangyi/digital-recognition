import base64
import io
from PIL import Image
import boto3
import torch
import torch.nn as nn
import os
from dotenv import load_dotenv
from torchvision import transforms

load_dotenv()

# 模型定义
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = torch.relu(self.fc1(x))
        x = torch.dropout(x, training=self.training)
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

model = Net()
model.load_state_dict(torch.load(os.getenv('MODEL_PATH'), map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 初始化 S3 客户端
s3 = boto3.client('s3', region_name=os.getenv('AWS_REGION'))

def save_result_to_s3(image_name, result):
    """将推理结果保存到 S3"""
    bucket_name = os.getenv('S3_BUCKET_NAME')
    file_name = f'results/{image_name}.txt'  # 保存到 results 文件夹中
    s3.put_object(Bucket=bucket_name, Key=file_name, Body=result)

def process_sqs_message():
    sqs = boto3.client('sqs', region_name=os.getenv('AWS_REGION'))
    queue_url = os.getenv('SQS_QUEUE_URL')

    while True:
        response = sqs.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=10
        )

        if 'Messages' not in response:
            continue

        message = response['Messages'][0]
        receipt_handle = message['ReceiptHandle']
        body = message['Body']
        image_name = message['MessageAttributes']['ImageName']['StringValue']

        try:
            # 解码 Base64 数据
            image_data = base64.b64decode(body)
            image = Image.open(io.BytesIO(image_data))

            # 图片预处理
            image = transform(image)
            image = image.unsqueeze(0)

            # 推理预测
            with torch.no_grad():
                output = model(image)
                pred = output.argmax(dim=1, keepdim=True).item()

            # 保存结果到 S3
            save_result_to_s3(image_name, f'Predicted label: {pred}')

        except Exception as e:
            print(f"处理消息时出错: {e}")

        finally:
            # 删除已处理的消息
            sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)

if __name__ == "__main__":
    process_sqs_message()