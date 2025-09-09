
from network.joint_pred_seg import STCNN,SegDecoderCBAM,JointSegDecoderCBAM,FramePredEncoder,SegEncoder,JointSegDecoder,SegDecoder
import torch

seg_decoder = JointSegDecoder()  # or pass appropriate args if needed
joint_decoder = JointSegDecoderCBAM()
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_parameter_summary(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name:50} {param.numel():10} parameters, shape: {tuple(param.shape)}")

print("== SegDecoderNoCBAM ==")
print_parameter_summary(seg_decoder)

print("\n== JointSegDecoderCBAM ==")
print_parameter_summary(joint_decoder)

print("SegDecoderNoPPM parameters:", count_parameters(seg_decoder))
print("JointSegDecoderCBAM parameters:", count_parameters(joint_decoder))

"""
from collections import OrderedDict

def get_param_dict(model_or_state_dict):
    if isinstance(model_or_state_dict, dict):
        return model_or_state_dict
    else:
        return dict(model_or_state_dict.named_parameters())

def compare_models(model_a, model_b, name_a="Model A", name_b="Model B"):
    params_a = get_param_dict(model_a)
    params_b = get_param_dict(model_b)

    extra_in_b = {k: v for k, v in params_b.items() if k not in params_a}
    total_extra_params = sum(p.numel() for p in extra_in_b.values() if isinstance(p, torch.Tensor))

    print(f"\n== Parameters in {name_b} but not in {name_a} ==")
    for name, param in extra_in_b.items():
        if isinstance(param, torch.Tensor):
            print(f"{name:50} {param.numel():10} parameters, shape: {tuple(param.shape)}")

    print(f"\nTotal extra parameters in {name_b}: {total_extra_params:,}")

# Models
seg_decoder = SegDecoderCBAM()
joint_decoder = JointSegDecoderCBAM()

pretrained_SegBranch_dict = torch.load("/home/r56x196/STCNN/output/Seg_Branch_CBAM/Seg_Branch_CBAM_epoch-11999.pth", map_location=torch.device('cpu'))
pretrained_SegBranch_dict = {
    (k[8:] if k.startswith("decoder.") else k): v
    for k, v in pretrained_SegBranch_dict.items()
}
# Run comparison
compare_models(pretrained_SegBranch_dict, joint_decoder, name_a="SegDecoder", name_b="JointSegDecoder")
compare_models(joint_decoder, pretrained_SegBranch_dict, name_a="SegDecoder", name_b="JointSegDecoder")
"""