import torch
import torch.nn as nn
from torch.distributed.fsdp import fully_shard

from olmo_core.distributed.parallel import (
    DataParallelConfig,
    DataParallelType,
    build_world_mesh,
)
from olmo_core.distributed.utils import get_rank, init_distributed
from olmo_core.utils import get_default_device

init_distributed()
mesh = build_world_mesh(
    dp=DataParallelConfig(name=DataParallelType.hsdp, shard_degree=2, num_replicas=2)
)
device = get_default_device()

seed = 0 + get_rank()
generator = torch.Generator(device).manual_seed(seed)

model = nn.Linear(4, 4, device="meta")
fully_shard(model, mesh=mesh)
model.to_empty(device=device)
nn.init.trunc_normal_(model.weight, mean=0.0, std=0.02, a=-0.06, b=0.06, generator=generator)

print(model)
