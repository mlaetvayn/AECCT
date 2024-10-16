from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class Code():
    n: int
    k: int
    code_type: str
    pc_matrix: Any = None
    generator_matrix: Any = None


@dataclass
class Config():

    # training param
    epochs: int = 1000
    workers: int = 4
    lr: float = 1e-4
    gpus: str = '-1'
    batch_size: int = 128
    test_batch_size: int = 512
    seed: int = 42
    eta_min=1e-6

    # code params
    standardize: bool = True

    # dimensions
    N_dec: int = 10
    d_model: int = 128
    h: int = 8
    code: Code = None

    # other
    path: str = None

    # lpe
    lpe_dim: int = 8
    lpe_num_heads: int = 8

    # head partitioning
    num_heads_for_one_ring: int = 4

    # quantization
    use_aap_linear_training: bool = False
    use_aap_linear_inference: bool = False

    act_bits: int = 8
    initial_percentile: Optional[float] = 0.45
