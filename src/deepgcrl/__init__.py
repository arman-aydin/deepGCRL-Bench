"""
deepgcrl: Disaggregated Multi-GPU Post-Training & Rollout Engine.
Patterned after verl, Ray-Train, and OpenRLHF.
"""

from .diffusion_policy import (
    GoalConditionedDiffusionTransformer,
    DiTBlock,
    SinusoidalPosEmb,
)
from .fsdp2_trainer import build_sharded_policy, setup_distributed
from .gpu_rollout_engine import DisaggregatedGPURolloutEngine
from .ipc_weight_sync import (
    CUDABroadcastPayload,
    CUDAPeerWeightBroadcaster,
    CUDAPeerWeightReceiver,
)
from .sharded_replay_buffer import StreamingShardedReplayBuffer

__all__ = [
    "GoalConditionedDiffusionTransformer",
    "DiTBlock",
    "SinusoidalPosEmb",
    "DisaggregatedGPURolloutEngine",
    "CUDAPeerWeightBroadcaster",
    "CUDAPeerWeightReceiver",
    "CUDABroadcastPayload",
    "StreamingShardedReplayBuffer",
    "build_sharded_policy",
    "setup_distributed",
]