import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

@torch.no_grad()
def generate_consistency_data(reference_model, num_samples=20000, spatial=False, H=16, W=16):
    """
    生成一致性训练数据，固定 batch=1。
    - spatial=False: 产生输入 [1,10]，输出 [1,3]
    - spatial=True : 产生输入 [1,10,H,W]，输出 [1,3,H,W]
    """
    device = next(reference_model.parameters()).device
    reference_model.eval()

    inputs, targets = [], []
    for _ in tqdm(range(num_samples)):
        if spatial:
            inp = torch.rand(1, 10, H, W, device=device)
        else:
            inp = torch.rand(1, 10, device=device)

        out = reference_model(inp)
        inputs.append(inp.squeeze(0).cpu())    # 存成 [10] 或 [10,H,W]
        targets.append(out.squeeze(0).cpu())   # 存成 [3] 或 [3,H,W]

    return torch.stack(inputs), torch.stack(targets)


def simple_fine_tune(model, train_inputs, train_targets, val_inputs, val_targets,
                     best_path='best_consistency_model.pth', spatial=False):
    """
    一致性微调，固定 batch=1。
    - 输入:  [10] 或 [10,H,W]
    - 输出:  [3]  或 [3,H,W]
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 还原 batch=1 维度
    train_inputs = train_inputs.unsqueeze(1).to(device)   # [N,1,...]
    train_targets = train_targets.unsqueeze(1).to(device)
    val_inputs = val_inputs.unsqueeze(1).to(device)
    val_targets = val_targets.unsqueeze(1).to(device)

    config = {
        'lr': 1e-4,
        'epochs': 50,
        'patience': 10
    }

    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5)

    # DataLoader 在 batch=1 情况下只是打乱顺序
    train_loader = DataLoader(TensorDataset(train_inputs, train_targets), shuffle=True)
    val_loader = DataLoader(TensorDataset(val_inputs, val_targets), shuffle=False)

    best_val = float('inf')
    patience_counter = 0

    for epoch in range(config['epochs']):
        # ---- train ----
        model.train()
        train_loss = 0
        for data, target in train_loader:
            data, target = data.squeeze(1), target.squeeze(1)  # 去掉 fake batch=1
            optimizer.zero_grad()
            out = model(data)
            loss = F.mse_loss(out, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # ---- val ----
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.squeeze(1), target.squeeze(1)
                out = model(data)
                val_loss += F.mse_loss(out, target).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1

        if epoch % 5 == 0:
            logger.info(f"[{epoch}] train={train_loss:.6f} val={val_loss:.6f} best={best_val:.6f}")

        if patience_counter >= config['patience']:
            logger.info(f"Early stopping at {epoch}")
            break

    model.load_state_dict(torch.load(best_path, map_location=device))


@torch.no_grad()
def test_rgba_consistency(original_model, finetuned_model,
                          num_tests=1000, spatial=False, H=16, W=16):
    """一致性测试，固定 batch=1"""
    device = next(original_model.parameters()).device
    original_model.eval().to(device)
    finetuned_model.eval().to(device)

    total_diff, max_diff, hits = 0, 0, 0
    for _ in range(num_tests):
        if spatial:
            x = torch.rand(1, 10, H, W, device=device)
        else:
            x = torch.rand(1, 10, device=device)

        y0 = original_model(x)
        y1 = finetuned_model(x)
        diff = torch.mean(torch.abs(y0 - y1)).item()

        total_diff += diff
        max_diff = max(max_diff, diff)
        if diff < 0.01:  # 1% 容差
            hits += 1

    avg_diff = total_diff / num_tests
    rate = hits / num_tests
    logger.info(f"Consistency: avg={avg_diff:.6f}, max={max_diff:.6f}, hit={rate:.2%}")

    return avg_diff < 0.005 and rate > 0.95
