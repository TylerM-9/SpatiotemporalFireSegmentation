"""
Debug test script for ST-DeepLabV3+
Tests forward pass and checks for NaN/Inf values
"""

import torch
import torch.nn as nn
from network.deeplabs_temporal import create_stdeeplabv3plus
from network.joint_pred_seg import FramePredDecoder, FramePredEncoder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("ST-DeepLabV3+ Debug Test")
print("=" * 60)

# Create temporal encoder/decoder
print("\n1. Creating temporal components...")
pred_enc = FramePredEncoder(frame_nums=4)
pred_dec = FramePredDecoder()
print("✓ Temporal components created")

# Create ST-DeepLabV3+ model
print("\n2. Creating ST-DeepLabV3+ model...")
net = create_stdeeplabv3plus(
    pred_enc=pred_enc,
    pred_dec=pred_dec,
    num_frame=4,
    num_classes=1,
    backbone='resnet50',
    output_stride=16,
    input_size=256
)
net.to(device)
net.freeze_temporal_branch()
net.freeze_bn()
print("✓ Model created and moved to device")

# Count parameters
total_params = sum(p.numel() for p in net.parameters())
trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f"\nModel Statistics:")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  Frozen parameters: {total_params - trainable_params:,}")

# Test forward pass
print("\n3. Testing forward pass with detailed debugging...")
batch_size = 2
seq = torch.randn(batch_size, 12, 256, 256).to(device)  # 4 frames * 3 channels
frame = torch.randn(batch_size, 3, 256, 256).to(device)

print(f"  Input seq shape: {seq.shape}")
print(f"  Input frame shape: {frame.shape}")

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

try:
    net.eval()

    # Get temporal ASPP features first
    print("\n  Testing temporal branch...")
    with torch.no_grad():
        temporal_aspp = net.seg_model.get_aspp_features(frame)
        print(f"  ✓ Temporal ASPP shape: {temporal_aspp.shape}")
        print(f"    Range: [{temporal_aspp.min():.4f}, {temporal_aspp.max():.4f}]")
        if torch.isnan(temporal_aspp).any():
            print("  ✗ NaN in temporal ASPP!")

    # Test spatial encoder
    print("\n  Testing spatial encoder...")
    with torch.no_grad():
        low_level, high_level = net.seg_model.backbone(frame)
        print(f"  ✓ Low-level features: {low_level.shape}")
        print(f"    Range: [{low_level.min():.4f}, {low_level.max():.4f}]")
        if torch.isnan(low_level).any():
            print("  ✗ NaN in low-level features!")

        print(f"  ✓ High-level features: {high_level.shape}")
        print(f"    Range: [{high_level.min():.4f}, {high_level.max():.4f}]")
        if torch.isnan(high_level).any():
            print("  ✗ NaN in high-level features!")

    # Test ASPP
    print("\n  Testing ASPP...")
    with torch.no_grad():
        aspp_out = net.seg_model.aspp(high_level)
        print(f"  ✓ ASPP output: {aspp_out.shape}")
        print(f"    Range: [{aspp_out.min():.4f}, {aspp_out.max():.4f}]")
        if torch.isnan(aspp_out).any():
            print("  ✗ NaN in ASPP output!")

    # Test attention separately
    print("\n  Testing attention module...")
    with torch.no_grad():
        if net.seg_model.decoder.use_attention:
            print("  Testing attention inputs...")
            print(f"    x (aspp_out): [{aspp_out.min():.4f}, {aspp_out.max():.4f}]")
            print(f"    prev (aspp_out.detach): [{aspp_out.detach().min():.4f}, {aspp_out.detach().max():.4f}]")
            print(f"    temporal: [{temporal_aspp.min():.4f}, {temporal_aspp.max():.4f}]")
            print(f"    context_high: [{temporal_aspp.min():.4f}, {temporal_aspp.max():.4f}]")

            out_down, out_attended = net.seg_model.decoder.attention(
                x=aspp_out,
                prev=aspp_out.detach(),
                temporal=temporal_aspp,
                context_high=temporal_aspp
            )

            print(f"  ✓ Attention out_down: {out_down.shape}")
            print(f"    Range: [{out_down.min():.4f}, {out_down.max():.4f}]")
            if torch.isnan(out_down).any():
                print("  ✗ NaN in out_down!")

            print(f"  ✓ Attention output: {out_attended.shape}")
            print(f"    Range: [{out_attended.min():.4f}, {out_attended.max():.4f}]")
            if torch.isnan(out_attended).any():
                print("  ✗ NaN in attention output!")

    # Full forward pass
    print("\n  Testing full forward pass...")
    with torch.no_grad():
        seg_output, pred, attention = net(seq, frame)

    print(f"\n✓ Forward pass successful!")
    print(f"  Segmentation output shape: {seg_output.shape}")
    print(f"  Prediction shape: {pred.shape}")
    print(f"  Attention shape: {attention.shape if attention is not None else None}")

    # Check for NaN/Inf
    print("\n4. Checking for NaN/Inf values...")
    has_nan = False

    if torch.isnan(seg_output).any():
        print("  ✗ NaN detected in segmentation output!")
        has_nan = True
    else:
        print("  ✓ Segmentation output: No NaN")

    if torch.isinf(seg_output).any():
        print("  ✗ Inf detected in segmentation output!")
        has_nan = True
    else:
        print("  ✓ Segmentation output: No Inf")

    if torch.isnan(pred).any():
        print("  ✗ NaN detected in prediction!")
        has_nan = True
    else:
        print("  ✓ Prediction: No NaN")

    if attention is not None:
        if torch.isnan(attention).any():
            print("  ✗ NaN detected in attention!")
            has_nan = True
        else:
            print("  ✓ Attention: No NaN")

    # Print value ranges
    print("\n5. Value ranges:")
    print(f"  Segmentation output: [{seg_output.min():.4f}, {seg_output.max():.4f}]")
    print(f"  Prediction: [{pred.min():.4f}, {pred.max():.4f}]")
    if attention is not None:
        print(f"  Attention: [{attention.min():.4f}, {attention.max():.4f}]")

    if not has_nan:
        print("\n✓ All checks passed! Model is healthy.")
    else:
        print("\n✗ NaN/Inf detected! Model has issues.")

except Exception as e:
    print(f"\n✗ Forward pass failed with error:")
    print(f"  {str(e)}")
    import traceback
    traceback.print_exc()

# Test backward pass
print("\n6. Testing backward pass...")
try:
    net.train()
    net.freeze_bn()

    seg_output, pred, attention = net(seq, frame)

    # Create dummy ground truth
    gt = torch.rand(batch_size, 1, 256, 256).to(device)

    # Compute loss
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(seg_output, gt)

    print(f"  Loss value: {loss.item():.6f}")

    if torch.isnan(loss) or torch.isinf(loss):
        print("  ✗ NaN/Inf in loss!")
    else:
        print("  ✓ Loss is valid")

        # Backward pass
        loss.backward()

        # Check gradients
        print("\n7. Checking gradients...")
        has_bad_grad = False
        for name, param in net.named_parameters():
            if param.requires_grad and param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"  ✗ NaN gradient in: {name}")
                    has_bad_grad = True
                if torch.isinf(param.grad).any():
                    print(f"  ✗ Inf gradient in: {name}")
                    has_bad_grad = True

        if not has_bad_grad:
            print("  ✓ All gradients are valid")
        else:
            print("  ✗ Bad gradients detected!")

except Exception as e:
    print(f"\n✗ Backward pass failed with error:")
    print(f"  {str(e)}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Debug test completed")
print("=" * 60)