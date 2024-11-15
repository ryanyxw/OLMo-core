"""
Convert an unsharded peteish7 model checkpoint from the old codebase to the right format for this
codebase.
"""

import logging
import sys

import rich
import torch
from cached_path import cached_path

from olmo_core.aliases import PathOrStr
from olmo_core.utils import prepare_cli_environment

log = logging.getLogger(__name__)


def main(input_path: PathOrStr, output_path: PathOrStr):
    input_path = cached_path(input_path)

    log.info("Loading old model checkpoint...")
    old_sd = torch.load(input_path, map_location="cpu")

    new_sd = {
        "embeddings.weight": old_sd.pop("transformer.wte.weight"),
        "norm.weight": old_sd.pop("transformer.ln_f.weight"),
        "w_out.weight": old_sd.pop("transformer.ff_out.weight"),
    }

    n_blocks = (
        max([int(k.split("blocks.")[1].split(".")[0]) for k in old_sd.keys() if "blocks." in k]) + 1
    )
    for block_idx in range(n_blocks):
        # Split up fused QKV projection.
        w_q, w_k, w_v = old_sd[f"transformer.blocks.{block_idx}.att_proj.weight"].chunk(3, dim=-1)
        new_sd[f"blocks.{block_idx}.attention.w_q.weight"] = w_q
        new_sd[f"blocks.{block_idx}.attention.w_k.weight"] = w_k
        new_sd[f"blocks.{block_idx}.attention.w_v.weight"] = w_v

        new_sd[f"blocks.{block_idx}.attention.w_out.weight"] = old_sd[
            f"transformer.blocks.{block_idx}.att_out.weight"
        ]
        new_sd[f"blocks.{block_idx}.attention.q_norm.weight"] = old_sd[
            f"transformer.blocks.{block_idx}.q_norm.weight"
        ]
        new_sd[f"blocks.{block_idx}.attention.k_norm.weight"] = old_sd[
            f"transformer.blocks.{block_idx}.k_norm.weight"
        ]
        new_sd[f"blocks.{block_idx}.attention_norm.weight"] = old_sd[
            f"transformer.blocks.{block_idx}.attn_norm.weight"
        ]

        # Split up fused feed-forward projection.
        w3, w1 = old_sd[f"transformer.blocks.{block_idx}.ff_proj.weight"].chunk(2, dim=-1)
        new_sd[f"blocks.{block_idx}.feed_forward.w1.weight"] = w1
        new_sd[f"blocks.{block_idx}.feed_forward.w3.weight"] = w3

        new_sd[f"blocks.{block_idx}.feed_forward.w2.weight"] = old_sd[
            f"transformer.blocks.{block_idx}.ff_out.weight"
        ]
        new_sd[f"blocks.{block_idx}.feed_forward_norm.weight"] = old_sd[
            f"transformer.blocks.{block_idx}.ff_norm.weight"
        ]

    assert len(old_sd) == 0

    log.info("Saving new model checkpoint...")
    torch.save(new_sd, output_path)
    log.info(f"Done, checkpoint saved to '{output_path}'")


if __name__ == "__main__":
    prepare_cli_environment()

    if len(sys.argv) != 3:
        rich.get_console().print(
            f"[yellow]Usage:[/] python {sys.argv[0]} INPUT_PATH OUTPUT_PATH", highlight=False
        )
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
