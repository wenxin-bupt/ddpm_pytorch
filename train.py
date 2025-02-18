import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import os
import torchvision.utils as vutils
import argparse

import debugpy
try:
    debugpy.listen(("localhost", 9518))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
except Exception as e:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 添加命令行参数解析
parser = argparse.ArgumentParser()
parser.add_argument("--inference", action="store_true", help="直接加载 checkpoint 并进行推理")
parser.add_argument("--checkpoint", type=str, default="./checkpoints/diffusion_epoch_9.pt", help="模型 checkpoint 路径")
args = parser.parse_args()

# 数据预处理与加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)

# 模型初始化
model = Unet(
    dim=64,
    channels=3,
    dim_mults=(1, 2, 4, 8),
    flash_attn=False
).to(device)
diffusion = GaussianDiffusion(model, image_size=32, timesteps=1000).to(device)
opt = Adam(diffusion.parameters(), lr=1e-4)

# 创建保存检查点的目录
os.makedirs("./checkpoints", exist_ok=True)

# 检查是否处于推理模式
if args.inference:
    print(f"Loading checkpoint from {args.checkpoint} for inference...")
    diffusion.load_state_dict(torch.load(args.checkpoint, map_location=device))
    diffusion.eval()
    samples = diffusion.sample(batch_size=16)
    vutils.save_image(samples, "./checkpoints/inference_sample.png", nrow=4, normalize=True)
    print("Inference sample saved to ./checkpoints/inference_sample.png")
    exit(0)

# 训练循环
num_epochs = 10 # 可根据需要调整轮次
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for images, _ in train_loader:
        images = images.to(device)
        loss = diffusion(images)
        opt.zero_grad()
        loss.backward()
        opt.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch} loss: {epoch_loss / len(train_loader):.4f}")
    # 保存当前 epoch 的模型检查点
    torch.save(diffusion.state_dict(), f"./checkpoints/diffusion_epoch_{epoch}.pt")

    
    
# 训练完成后推理可视化
print("Generating sample images after training...")
samples = diffusion.sample(batch_size=16)
vutils.save_image(samples, "./checkpoints/sample.png", nrow=4, normalize=True)
print("Sample visualization saved to ./checkpoints/sample.png")

if __name__ == '__main__':
    pass