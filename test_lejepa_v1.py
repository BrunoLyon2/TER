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

# Setup device according to local hardware
if torch.cuda.is_available():
    device = "cuda"
    # Check GPU capability for mixed precision
    gpu_capability = torch.cuda.get_device_capability()
    use_amp = gpu_capability[0] >= 7  # Volta (V100) or newer
    amp_dtype = torch.bfloat16 if gpu_capability[0] >= 8 else torch.float16
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: {gpu_capability[0]}.{gpu_capability[1]}")
    print(f"Using AMP: {use_amp} with dtype: {amp_dtype if use_amp else 'N/A'}")
elif torch.backends.mps.is_available():
    device = "mps"
    use_amp = False
    amp_dtype = torch.float32
    print("Using MPS device")
else:
    device = "cpu"
    use_amp = False
    amp_dtype = torch.float32
    print("Using CPU")


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
        A = torch.randn(proj.size(-1), 256, device=device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


class ViTEncoder(nn.Module):
    def __init__(self, proj_dim=128, dropout=0.1):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_small_patch8_224",
            pretrained=False,
            num_classes=512,
            drop_path_rate=0.1,
            img_size=128,
        )
        self.proj = MLP(512, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d, dropout=dropout)

    def forward(self, x):
        N, V = x.shape[:2]
        emb = self.backbone(x.flatten(0, 1))
        return emb, self.proj(emb).reshape(N, V, -1).transpose(0, 1)


class _DatasetSplit(torch.utils.data.Dataset):
    """
    Internal dataset class used by DatasetManager.
    """
    def __init__(self, data_dir, split, V=1, img_size=128, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.V = V
        self.img_size = img_size
        # Load using imagefolder (automatically maps 'train'/'validation' folders)
        self.ds = load_dataset("imagefolder", data_dir=data_dir, split=split)
        
        self.aug = v2.Compose([
            v2.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
            v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            v2.RandomGrayscale(p=0.2),
            v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))], p=0.5),
            v2.RandomApply([v2.RandomSolarize(threshold=128)], p=0.2),
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=list(mean), std=list(std)),
        ])
        
        self.test = v2.Compose([
            v2.Resize(img_size),
            v2.CenterCrop(img_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=list(mean), std=list(std)),
        ])

    def __getitem__(self, i):
        item = self.ds[i]
        img = item["image"].convert("RGB")
        label = item["label"]
        
        if self.V > 1:
            # Returns [V, C, H, W]
            return torch.stack([self.aug(img) for _ in range(self.V)]), label
        else:
            # Returns [1, C, H, W] for consistency
            return self.test(img).unsqueeze(0), label

    def __len__(self):
        return len(self.ds)


class DatasetManager:
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

    def get_ds(self, split, V, img_size=128, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        """
        Factory method to return the actual Dataset object.
        """
        return _DatasetSplit(self.working_dir, split, V, img_size, mean, std)


class MetricsLogger:
    """Simple metrics logger for tracking training progress."""
    def __init__(self):
        self.metrics = {
            'epoch': [],
            'train_loss': [],
            'train_lejepa_loss': [],
            'train_probe_loss': [],
            'val_acc': [],
            'lr': []
        }
    
    def log(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.metrics:
                self.metrics[key].append(value)
    
    def print_summary(self, epoch):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch} Summary:")
        if self.metrics['train_loss']:
            print(f"  Train Loss: {self.metrics['train_loss'][-1]:.4f}")
            print(f"  LejEPA Loss: {self.metrics['train_lejepa_loss'][-1]:.4f}")
            print(f"  Probe Loss: {self.metrics['train_probe_loss'][-1]:.4f}")
        if self.metrics['val_acc']:
            print(f"  Val Accuracy: {self.metrics['val_acc'][-1]:.4f}")
        if self.metrics['lr']:
            print(f"  Learning Rate: {self.metrics['lr'][-1]:.6f}")
        print(f"{'='*60}\n")


def validate(net, probe, test_dl, device, use_amp, amp_dtype):
    """Run validation and return accuracy."""
    net.eval()
    probe.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for vs, y in test_dl:
            vs = vs.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            with autocast(device, dtype=amp_dtype, enabled=use_amp):
                emb, _ = net(vs)
                preds = probe(emb).argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)
    
    return correct / total


def main(cfg: DictConfig):
    # Configuration validation
    assert cfg.bs > 0, "Batch size must be positive"
    assert cfg.V >= 1, "Number of views must be at least 1"
    assert cfg.accum_steps > 0, "Accumulation steps must be positive"
    effective_bs = cfg.bs * cfg.accum_steps
    assert effective_bs <= 512, f"Effective batch size ({effective_bs}) too large, reduce bs or accum_steps"
    
    torch.manual_seed(0)
    
    print(f"\n{'='*60}")
    print("Training Configuration:")
    print(f"  Device: {device}")
    print(f"  Batch Size: {cfg.bs}")
    print(f"  Accumulation Steps: {cfg.accum_steps}")
    print(f"  Effective Batch Size: {effective_bs}")
    print(f"  Views per image: {cfg.V}")
    print(f"  Epochs: {cfg.epochs}")
    print(f"  Learning Rate: {cfg.lr}")
    print(f"  Lambda: {cfg.lamb}")
    print(f"  Projection Dim: {cfg.proj_dim}")
    print(f"{'='*60}\n")

    # Initialize Dataset Manager
    archive_path = "/kaggle/input/imagenette-160-px/imagenette-160.tgz"
    dataset_manager = DatasetManager(archive_path)

    # Get PyTorch Datasets
    train_ds = dataset_manager.get_ds(split="train", V=cfg.V)
    test_ds = dataset_manager.get_ds(split="validation", V=1)
    
    # DataLoaders with improved settings
    train_dl = DataLoader(
        train_ds, 
        batch_size=cfg.bs, 
        shuffle=True, 
        drop_last=True, 
        num_workers=4, 
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    test_dl = DataLoader(
        test_ds, 
        batch_size=256, 
        num_workers=4, 
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    # Modules and loss
    net = ViTEncoder(proj_dim=cfg.proj_dim, dropout=0.1).to(device)
    probe = nn.Sequential(
        nn.LayerNorm(512),
        nn.Dropout(0.1),
        nn.Linear(512, 10)
    ).to(device)
    sigreg = SIGReg().to(device)
    
    # Try torch.compile for PyTorch 2.0+
    try:
        if hasattr(torch, 'compile') and device == "cuda":
            print("Compiling model with torch.compile...")
            net = torch.compile(net)
            probe = torch.compile(probe)
            print("Model compilation successful!")
    except Exception as e:
        print(f"Model compilation not available or failed: {e}")
    
    # Optimizer and scheduler
    g1 = {"params": net.parameters(), "lr": cfg.lr, "weight_decay": 5e-2}
    g2 = {"params": probe.parameters(), "lr": 1e-3, "weight_decay": 1e-7}
    opt = torch.optim.AdamW([g1, g2])
    
    # Scheduler adjusted for gradient accumulation
    steps_per_epoch = len(train_dl) // cfg.accum_steps
    warmup_steps = steps_per_epoch  # warm up for 1 epoch
    total_steps = steps_per_epoch * cfg.epochs

    s1 = LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
    s2 = CosineAnnealingLR(opt, T_max=total_steps - warmup_steps, eta_min=1e-6)
    scheduler = SequentialLR(opt, schedulers=[s1, s2], milestones=[warmup_steps])

    scaler = GradScaler(enabled=use_amp)
    
    # Initialize logger
    logger = MetricsLogger()
    
    print(f"Starting training for {cfg.epochs} epochs...")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Total optimization steps: {total_steps}\n")
    
    best_acc = 0.0
    patience = 0
    max_patience = 20  # Early stopping patience
    
    # Training loop
    for epoch in range(cfg.epochs):
        net.train()
        probe.train()
        pbar = tqdm.tqdm(train_dl, total=len(train_dl), desc=f"Epoch {epoch+1}/{cfg.epochs}")
        
        opt.zero_grad()
        epoch_loss = 0.0
        epoch_lejepa_loss = 0.0
        epoch_probe_loss = 0.0
        n_batches = 0
        
        for batch_idx, (vs, y) in enumerate(pbar):
            with autocast(device, dtype=amp_dtype, enabled=use_amp):
                vs = vs.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                
                emb, proj = net(vs)
                inv_loss = (proj.mean(0) - proj).square().mean()
                sigreg_loss = sigreg(proj)
                lejepa_loss = sigreg_loss * cfg.lamb + inv_loss * (1 - cfg.lamb)
                
                # Dynamic repeat based on actual views
                y_rep = y.repeat_interleave(vs.shape[1])
                
                # Linear probing with detached embeddings
                yhat = probe(emb.detach())
                probe_loss = F.cross_entropy(yhat, y_rep, label_smoothing=0.1)
                
                loss = lejepa_loss + probe_loss
                
                # Normalize loss for gradient accumulation
                loss = loss / cfg.accum_steps

            scaler.scale(loss).backward()
            
            # Accumulate metrics
            epoch_loss += loss.item() * cfg.accum_steps
            epoch_lejepa_loss += lejepa_loss.item()
            epoch_probe_loss += probe_loss.item()
            n_batches += 1

            # Optimizer step with gradient clipping
            if (batch_idx + 1) % cfg.accum_steps == 0 or (batch_idx + 1) == len(train_dl):
                # Gradient clipping
                scaler.unscale_(opt)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(net.parameters()) + list(probe.parameters()), 
                    max_norm=1.0
                )
                
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                scheduler.step()  # Step scheduler after optimizer step
                
                # Update progress bar
                current_lr = scheduler.get_last_lr()[0]
                pbar.set_postfix({
                    'loss': f'{loss.item() * cfg.accum_steps:.4f}',
                    'lr': f'{current_lr:.6f}',
                    'grad': f'{grad_norm:.2f}'
                })

        # Calculate epoch averages
        avg_loss = epoch_loss / n_batches
        avg_lejepa = epoch_lejepa_loss / n_batches
        avg_probe = epoch_probe_loss / n_batches
        current_lr = scheduler.get_last_lr()[0]

        # Validation
        print(f"\nRunning validation...")
        acc = validate(net, probe, test_dl, device, use_amp, amp_dtype)
        
        # Log metrics
        logger.log(
            epoch=epoch+1,
            train_loss=avg_loss,
            train_lejepa_loss=avg_lejepa,
            train_probe_loss=avg_probe,
            val_acc=acc,
            lr=current_lr
        )
        logger.print_summary(epoch+1)

        # Save best model
        if acc > best_acc:
            best_acc = acc
            patience = 0
            print(f"✓ New best accuracy: {best_acc:.4f}. Saving best model...")
            checkpoint = {
                'net_state_dict': net.state_dict() if not hasattr(net, '_orig_mod') else net._orig_mod.state_dict(),
                'probe_state_dict': probe.state_dict() if not hasattr(probe, '_orig_mod') else probe._orig_mod.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'config': dict(cfg),
                'epoch': epoch,
                'best_accuracy': best_acc,
                'metrics': logger.metrics
            }
            torch.save(checkpoint, "best_model.pth")
        else:
            patience += 1
            print(f"No improvement. Patience: {patience}/{max_patience}")
        
        # Early stopping
        if patience >= max_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
        
        # Clear cache periodically
        if device == "cuda" and (epoch + 1) % 5 == 0:
            torch.cuda.empty_cache()

    # Save final model
    print("\nSaving final model...")
    final_checkpoint = {
        'net_state_dict': net.state_dict() if not hasattr(net, '_orig_mod') else net._orig_mod.state_dict(),
        'probe_state_dict': probe.state_dict() if not hasattr(probe, '_orig_mod') else probe._orig_mod.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'config': dict(cfg),
        'epoch': epoch,
        'final_accuracy': acc,
        'best_accuracy': best_acc,
        'metrics': logger.metrics
    }
    torch.save(final_checkpoint, f"final_model_epoch_{epoch+1}_acc_{acc:.4f}.pth")
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"Best Validation Accuracy: {best_acc:.4f}")
    print(f"Final Validation Accuracy: {acc:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Hydra composition for script/notebook usage
    with initialize(version_base=None, config_path=None):
        overrides = [
            "+lamb=0.02",
            "+V=4",
            "+proj_dim=16",
            "+lr=2e-3",
            "+bs=16",
            "+accum_steps=16",
            "+epochs=2"
        ]
        cfg = compose(config_name=None, overrides=overrides)
        
        print(f"Configuration:\n{cfg}")
        main(cfg)