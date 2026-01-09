import os
import tarfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import timm
import tqdm
from hydra import compose, initialize
from omegaconf import DictConfig
from datasets import load_dataset
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torchvision.ops import MLP

class SIGReg(torch.nn.Module):
    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), 256, device="cuda")
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


class ViTEncoder(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_small_patch8_224",
            pretrained=False,
            num_classes=512,
            drop_path_rate=0.1,
            img_size=128,
        )
        self.proj = MLP(512, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def forward(self, x):
        N, V = x.shape[:2]
        emb = self.backbone(x.flatten(0, 1))
        return emb, self.proj(emb).reshape(N, V, -1).transpose(0, 1)


class _DatasetSplit(torch.utils.data.Dataset):
    """
    Internal dataset class used by HFDataset manager.
    """
    def __init__(self, data_dir, split, V=1):
        self.V = V
        # Load using imagefolder (automatically maps 'train'/'validation' folders)
        self.ds = load_dataset("imagefolder", data_dir=data_dir, split=split)
        
        self.aug = v2.Compose([
            v2.RandomResizedCrop(128, scale=(0.08, 1.0)),
            v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            v2.RandomGrayscale(p=0.2),
            v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))], p=0.5),
            v2.RandomApply([v2.RandomSolarize(threshold=128)], p=0.2),
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.test = v2.Compose([
            v2.Resize(128),
            v2.CenterCrop(128),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, i):
        item = self.ds[i]
        img = item["image"].convert("RGB")
        label = item["label"]
        
        if self.V > 1:
            # Returns [V, C, H, W]
            return torch.stack([self.aug(img) for _ in range(self.V)]), label
        else:
            # FIX: Add unsqueeze(0) to make it [1, C, H, W]
            # This ensures ViTEncoder correctly sees '1' as the number of views
            return self.test(img).unsqueeze(0), label

    def __len__(self):
        return len(self.ds)


class HFDataset:
    """
    Manager class responsible for extraction and spawning dataset splits.
    """
    def __init__(self, archive_path, working_dir="/kaggle/working/imagenette-160"):
        self.working_dir = working_dir
        
        # Extraction logic
        if not os.path.exists(self.working_dir):
            if os.path.exists(archive_path):
                print(f"Extracting {archive_path} to {os.path.dirname(self.working_dir)}...")
                with tarfile.open(archive_path, "r:gz") as tar:
                    tar.extractall(path=os.path.dirname(self.working_dir))
                print("Extraction complete.")
            else:
                print(f"Warning: Archive not found at {archive_path}. Attempting to load from {self.working_dir} anyway.")
        else:
            print(f"Data already found at {self.working_dir}")

    def get_ds(self, split, V):
        """
        Factory method to return the actual Dataset object.
        """
        return _DatasetSplit(self.working_dir, split, V)


def main(cfg: DictConfig):
    torch.manual_seed(0)

    # Initialize Dataset Manager
    # Note: Ensure this path matches your Kaggle input path
    archive_path = "/kaggle/input/imagenette-160-px/imagenette-160.tgz"
    dataset_manager = HFDataset(archive_path)

    # Get PyTorch Datasets
    train_ds = dataset_manager.get_ds(split="train", V=cfg.V)
    test_ds = dataset_manager.get_ds(split="validation", V=1)
    
    # Adjusted num_workers for typical Kaggle environment (usually 2-4 is safer, but 8 might work)
    train_dl = DataLoader(
        train_ds, batch_size=cfg.bs, shuffle=True, drop_last=True, num_workers=4, pin_memory=True
    )
    test_dl = DataLoader(test_ds, batch_size=256, num_workers=4, pin_memory=True)

    # modules and loss
    net = ViTEncoder(proj_dim=cfg.proj_dim).to("cuda")
    probe = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 10)).to("cuda") # Changed 100 to 10 for Imagenette (10 classes)
    sigreg = SIGReg().to("cuda")
    
    # Optimizer and scheduler
    g1 = {"params": net.parameters(), "lr": cfg.lr, "weight_decay": 5e-2}
    g2 = {"params": probe.parameters(), "lr": 1e-3, "weight_decay": 1e-7}
    opt = torch.optim.AdamW([g1, g2])
    
    # --- Gradient Accumulation Adjustment for Scheduler ---
    # With gradient accumulation, the optimizer steps less frequently.
    # We must adjust the total steps and warmup steps to reflect the number of *updates*, not batches.
    steps_per_epoch = len(train_dl) // cfg.accum_steps
    warmup_steps = steps_per_epoch # warm up for 1 epoch
    total_steps = steps_per_epoch * cfg.epochs

    s1 = LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
    s2 = CosineAnnealingLR(opt, T_max=total_steps - warmup_steps, eta_min=1e-3)
    scheduler = SequentialLR(opt, schedulers=[s1, s2], milestones=[warmup_steps])

    scaler = GradScaler(enabled=torch.cuda.is_available())
    
    print(f"Starting training for {cfg.epochs} epochs with effective batch size {cfg.bs * cfg.accum_steps}...")
    
    best_acc = 0.0

    # Training
    for epoch in range(cfg.epochs):
        net.train()
        probe.train()
        pbar = tqdm.tqdm(train_dl, total=len(train_dl), desc=f"Epoch {epoch+1}")
        
        opt.zero_grad() # Ensure grads are zero before starting the epoch
        
        for batch_idx, (vs, y) in enumerate(pbar):
            # FIXME: adapt automaticly or put it in parameters
            # Changed from bfloat16 to float16 for P100 compatibility
            with autocast("cuda", dtype=torch.float16):
                vs = vs.to("cuda", non_blocking=True)
                y = y.to("cuda", non_blocking=True)
                
                emb, proj = net(vs)
                inv_loss = (proj.mean(0) - proj).square().mean()
                sigreg_loss = sigreg(proj)
                lejepa_loss = sigreg_loss * cfg.lamb + inv_loss * (1 - cfg.lamb)
                
                # Dynamic repeat based on the actual number of views in vs
                # This prevents shape mismatches if vs.shape[1] != cfg.V (e.g., config changes or edge cases)
                y_rep = y.repeat_interleave(vs.shape[1])
                
                # Using detach() as requested in previous steps for Linear Probing behavior
                yhat = probe(emb.detach())
                probe_loss = F.cross_entropy(yhat, y_rep)
                
                loss = lejepa_loss + probe_loss

                # Normalize loss for gradient accumulation
                loss = loss / cfg.accum_steps

            scaler.scale(loss).backward()

            # Perform optimizer step every accum_steps OR on the very last batch
            if (batch_idx + 1) % cfg.accum_steps == 0 or (batch_idx + 1) == len(train_dl):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                scheduler.step()
            
            # Show un-scaled loss for logging (approximate)
            pbar.set_postfix({'loss': loss.item() * cfg.accum_steps})

        # Evaluation
        net.eval()
        probe.eval()
        correct = 0
        total = 0
        with torch.inference_mode():
            for vs, y in test_dl:
                vs = vs.to("cuda", non_blocking=True)
                y = y.to("cuda", non_blocking=True)
                # FIXME: adapt automaticly or put it in parameters
                # Changed from bfloat16 to float16 for P100 compatibility
                with autocast("cuda", dtype=torch.float16):
                    # For eval, vs shape is [Batch, 1, Channels, Height, Width]
                    # We pass this 5D tensor directly to ViTEncoder.
                    # It will parse N, V correctly and flatten internally to [Batch*1, C, H, W]
                    emb, _ = net(vs) 
                    
                    preds = probe(emb).argmax(1)
                    correct += (preds == y).sum().item()
                    total += y.size(0)
        
        acc = correct / total
        print(f"Epoch {epoch+1} Accuracy: {acc:.4f}")

        # --- SAVE BEST MODEL ---
        if acc > best_acc:
            best_acc = acc
            print(f"New best accuracy: {best_acc:.4f}. Saving best model...")
            checkpoint = {
                'net_state_dict': net.state_dict(),
                'probe_state_dict': probe.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'config': dict(cfg),
                'epoch': epoch,
                'accuracy': acc
            }
            torch.save(checkpoint, "best_model.pth")

    # --- SAVING THE LAST MODEL ---
    print("Saving last model...")
    save_path = "last_model.pth" 
    
    checkpoint = {
        'net_state_dict': net.state_dict(),
        'probe_state_dict': probe.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'config': dict(cfg),
        'epoch': cfg.epochs,
        'accuracy': acc
    }
    torch.save(checkpoint, save_path)
    print(f"Last model saved to {save_path}")

if __name__ == "__main__":
    # Hydra composition for script/notebook usage
    with initialize(version_base=None, config_path=None):
        overrides = [
            "+lamb=0.02",
            "+V=4",
            "+proj_dim=16",
            "+lr=2e-3",
            "+bs=16",         # Reduced Batch Size
            "+accum_steps=16", # Added Gradient Accumulation Steps (Effective BS = 256)
            "+epochs=2" # was 800
        ]
        cfg = compose(config_name=None, overrides=overrides)
        
        print(f"Configuration:\n{cfg}")
        main(cfg)
