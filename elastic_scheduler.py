import boto3
import os
from dotenv import load_dotenv
import time

# 加载环境变量
load_dotenv()

# 初始化客户端
ec2 = boto3.client('ec2', region_name=os.getenv('AWS_REGION'))
sqs = boto3.client('sqs', region_name=os.getenv('AWS_REGION'))

def get_queue_length():
    """获取 SQS 队列中的消息数量"""
    queue_url = os.getenv('SQS_QUEUE_URL')
    try:
        response = sqs.get_queue_attributes(
            QueueUrl=queue_url,
            AttributeNames=['ApproximateNumberOfMessages']
        )
        return int(response['Attributes']['ApproximateNumberOfMessages'])
    except Exception as e:
        print(f"获取队列长度时出错: {e}")
        return 0

def get_running_instances():
    """获取当前正在运行的实例 ID 列表"""
    try:
        response = ec2.describe_instances(
            Filters=[{'Name': 'instance-state-name', 'Values': ['running']}]
        )
        instances = []
        for reservation in response['Reservations']:
            for instance in reservation['Instances']:
                instances.append(instance['InstanceId'])
        return instances
    except Exception as e:
        print(f"获取运行实例时出错: {e}")
        return []

def start_instances(instance_count):
    """启动指定数量的 EC2 实例"""
    try:
        ami_id = os.getenv('AMI_ID')
        key_pair = os.getenv('EC2_KEY_PAIR')
        security_group = os.getenv('SECURITY_GROUP')
        instance_profile = os.getenv('INSTANCE_PROFILE')

        print(f"正在启动 {instance_count} 个新实例...")
        response = ec2.run_instances(
            ImageId=ami_id,
            MinCount=instance_count,
            MaxCount=instance_count,
            InstanceType='t2.micro',
            KeyName=key_pair,
            SecurityGroupIds=[security_group],
            IamInstanceProfile={'Name': instance_profile}
        )
        instance_ids = [instance['InstanceId'] for instance in response['Instances']]
        print(f"已启动实例: {instance_ids}")
        return instance_ids
    except Exception as e:
        print(f"启动实例时出错: {e}")
        return []

def terminate_instances(instance_ids):
    """终止指定的 EC2 实例"""
    try:
        print(f"正在终止实例: {instance_ids}")
        ec2.terminate_instances(InstanceIds=instance_ids)
        print("实例已终止")
    except Exception as e:
        print(f"终止实例时出错: {e}")

def elastic_scheduler():
    """弹性调度主循环"""
    while True:
        try:
            # 获取队列长度和当前运行的实例
            queue_length = get_queue_length()
            running_instances = get_running_instances()
            current_instance_count = len(running_instances)

            print(f"当前队列中有 {queue_length} 条消息")
            print(f"当前运行的实例数量: {current_instance_count}")

            # 计算期望的实例数量
            desired_instance_count = max(1, min(queue_length, 5))  # 最少 1 个实例，最多 5 个实例

            if queue_length == 0 and current_instance_count > 0:
                # 如果队列为空，终止所有实例
                print("队列为空，终止所有实例...")
                terminate_instances(running_instances)
            elif desired_instance_count > current_instance_count:
                # 如果需要更多实例，启动新实例
                instance_count = desired_instance_count - current_instance_count
                start_instances(instance_count)
            elif desired_instance_count < current_instance_count:
                # 如果实例过多，终止多余的实例
                extra_instances = running_instances[desired_instance_count:]
                terminate_instances(extra_instances)

        except Exception as e:
            print(f"调度器出错: {e}")

        # 每隔 60 秒检查一次
        time.sleep(60)

if __name__ == "__main__":
    print("启动弹性调度器...")
    elastic_scheduler()