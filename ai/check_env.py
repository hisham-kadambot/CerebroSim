import time, torch, physicsnemo, torch_geometric, warp as wp
from torch_geometric.data import Data
from physicsnemo.models.meshgraphnet import MeshGraphNet
print("physicsnemo", physicsnemo.__version__, "torch", torch.__version__, "pyg", torch_geometric.__version__, "warp", wp.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
dev = "cuda"
n = 20000
src = torch.arange(n - 1); ei = torch.stack([torch.cat([src, src + 1]), torch.cat([src + 1, src])]).to(dev)
g = Data(edge_index=ei, num_nodes=n)
m = MeshGraphNet(input_dim_nodes=8, input_dim_edges=4, output_dim=2, processor_size=10, hidden_dim_processor=128).to(dev)
x, e = torch.randn(n, 8, device=dev), torch.randn(ei.shape[1], 4, device=dev)
opt = torch.optim.Adam(m.parameters(), 1e-3)
for i in range(30):
    if i == 10: torch.cuda.synchronize(); t = time.time()
    loss = (m(x, e, g) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
torch.cuda.synchronize(); print(f"train step {1000 * (time.time() - t) / 20:.1f} ms, loss {loss.item():.4f}, mem {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
