import os
import sys
import torch
import numpy as np

# Ensure root & src are on path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from deepgcrl import (
    GoalConditionedDiffusionTransformer,
    StreamingShardedReplayBuffer,
    CUDAPeerWeightBroadcaster,
    CUDAPeerWeightReceiver
)


def test_diffusion_transformer_parameters_and_shapes():
    """Verify 150M+ parameter scaling and denoising score network shapes."""
    state_dim = 69
    action_dim = 8
    
    # 1. Full 150M+ scale verification
    full_model = GoalConditionedDiffusionTransformer(
        state_dim=state_dim,
        action_dim=action_dim,
        dim=1024,
        depth=12,
        num_heads=16
    )
    total_params = full_model.count_parameters()
    print(f"Total Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    assert total_params >= 150_000_000, f"Expected >= 150M params, got {total_params}"

    # 2. Functional shape verification on smaller model
    model = GoalConditionedDiffusionTransformer(
        state_dim=state_dim,
        action_dim=action_dim,
        action_horizon=1,
        dim=128,
        depth=2,
        num_heads=4
    )
    
    batch_size = 4
    s = torch.randn(batch_size, state_dim)
    g = torch.randn(batch_size, state_dim)
    a = torch.randn(batch_size, action_dim)
    
    loss = model.compute_loss(a, s, g)
    assert loss.ndim == 0
    assert torch.isfinite(loss).item()

    # DDIM fast action sample with action_horizon=1
    sampled_actions = model.sample_actions(s, g, num_inference_steps=3, horizon=1)
    assert sampled_actions.shape == (batch_size, action_dim)
    assert torch.all(sampled_actions >= -1.0) and torch.all(sampled_actions <= 1.0)

    # DDIM sample with chunked horizon=4
    sampled_chunk = model.sample_actions(s, g, num_inference_steps=3, horizon=4)
    assert sampled_chunk.shape == (batch_size, 4, action_dim)

    print("test_diffusion_transformer_parameters_and_shapes PASSED")


def test_sharded_replay_buffer():
    """Verify high-throughput vectorized circular batch insertion and sampling."""
    max_size = 100
    state_dim = 10
    action_dim = 2
    buffer = StreamingShardedReplayBuffer(max_size=max_size, state_dim=state_dim, action_dim=action_dim)

    # Insert batch 1 (60 elements)
    b1 = 60
    obs1 = np.ones((b1, state_dim), dtype=np.float32)
    act1 = np.ones((b1, action_dim), dtype=np.float32)
    next_obs1 = np.ones((b1, state_dim), dtype=np.float32)
    r1 = np.ones((b1, 1), dtype=np.float32)
    d1 = np.zeros((b1, 1), dtype=np.float32)
    g1 = np.ones((b1, state_dim), dtype=np.float32)
    
    buffer.push_batch(obs1, act1, next_obs1, r1, d1, g1)
    assert len(buffer) == 60
    assert buffer.ptr == 60

    # Insert batch 2 (70 elements - tests circular wraparound)
    b2 = 70
    obs2 = np.full((b2, state_dim), 2.0, dtype=np.float32)
    act2 = np.full((b2, action_dim), 2.0, dtype=np.float32)
    next_obs2 = np.full((b2, state_dim), 2.0, dtype=np.float32)
    r2 = np.ones((b2, 1), dtype=np.float32)
    d2 = np.zeros((b2, 1), dtype=np.float32)
    g2 = np.full((b2, state_dim), 2.0, dtype=np.float32)
    buffer.push_batch(obs2, act2, next_obs2, r2, d2, g2)
    
    assert len(buffer) == 100 # Capped at max_size
    assert buffer.ptr == 30   # (60 + 70) % 100 = 30

    # Sample tensors
    s_batch, a_batch, _, _, _, _ = buffer.sample(32)
    assert s_batch.shape == (32, state_dim)
    assert a_batch.shape == (32, action_dim)
    assert isinstance(s_batch, torch.Tensor)
    print("test_sharded_replay_buffer PASSED")


def test_cuda_peer_weight_sync():
    """Verify zero-copy parameter packing, transfer, and unpacking equality."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src_model = GoalConditionedDiffusionTransformer(
        state_dim=32, action_dim=4, dim=128, depth=2, num_heads=4
    ).to(device)
    dst_model = GoalConditionedDiffusionTransformer(
        state_dim=32, action_dim=4, dim=128, depth=2, num_heads=4
    ).to(device)

    # Scramble source model weights
    with torch.no_grad():
        for p in src_model.parameters():
            p.add_(torch.randn_like(p))

    broadcaster = CUDAPeerWeightBroadcaster(src_model, device)
    receiver = CUDAPeerWeightReceiver(dst_model, device)

    handle, lat_ms = broadcaster.pack_and_broadcast_async(src_model)
    receiver.unpack_weights_async(handle)

    # Verify all parameters in dst_model match src_model exactly
    for (name1, p1), (name2, p2) in zip(src_model.named_parameters(), dst_model.named_parameters()):
        assert torch.allclose(p1, p2, atol=1e-5), f"Mismatch in weight tensor {name1}"

    print(f"test_cuda_peer_weight_sync PASSED (Sync Latency: {lat_ms:.3f} ms)")


if __name__ == '__main__':
    test_diffusion_transformer_parameters_and_shapes()
    test_sharded_replay_buffer()
    test_cuda_peer_weight_sync()
    print("\nAll unit and integration tests passed successfully.")