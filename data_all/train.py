import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
from generator_model import Generator
from discriminator_model import Discriminator
from dataset import MapDataset
import config
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
import torch.backends.cudnn as cudnn

# 启用cudnn基准测试
cudnn.benchmark = True

def train_fn(disc, gen, loader, opt_disc, opt_gen, l1, bce, g_scaler, d_scaler):
    loop = tqdm(loader, leave = True)

    for idx, (x, y) in enumerate(loop):
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)

        with torch.amp.autocast('cuda'):
            y_fake = gen(x)
            D_real = disc(x, y)
            D_fake = disc(x, y_fake.detach())
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        with torch.amp.autocast('cuda'):
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1(y_fake, y) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

def main():
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    disc = Discriminator(in_channels = 3).to(config.DEVICE)
    gen = Generator(in_channels = 3).to(config.DEVICE)
    
    # 使用更大的学习率
    opt_disc = optim.Adam(disc.parameters(), lr = config.LEARNING_RATE * 2, betas = (0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr = config.LEARNING_RATE * 2, betas = (0.5, 0.999))
    
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE)
        
    train_dataset = MapDataset(root_dir = "data/maps/maps/train")
    train_loader = DataLoader(
        dataset = train_dataset, 
        batch_size = config.BATCH_SIZE, 
        shuffle = True, 
        num_workers = config.NUM_WORKERS,
        pin_memory = True,  # 启用固定内存
        persistent_workers = True  # 保持工作进程
    )
    
    g_scaler = torch.amp.GradScaler('cuda')
    d_scaler = torch.amp.GradScaler('cuda')
    
    val_dataset = MapDataset(root_dir = "data/maps/maps/val")
    val_loader = DataLoader(
        dataset = val_dataset, 
        batch_size = 1, 
        shuffle = False,
        num_workers = config.NUM_WORKERS,
        pin_memory = True
    )
    
    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler)

        if config.SAVE_MODEL and epoch % 50 == 0:
            save_checkpoint(gen, opt_gen, filename = config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename = config.CHECKPOINT_DISC)

        save_some_examples(gen, val_loader, epoch, folder = "evaluation")
    

if __name__ == "__main__":
    main()
