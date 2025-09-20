import torch
import onnxsim
import onnx
import json

# From https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
from infer.lib.infer_pack.models_onnx import SynthesizerTrnMsNSFsidM

def export_onnx(model_path, exported_path, use_fp16=True):
    cpt = torch.load(model_path, map_location="cpu")
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    vec_channels = 256 if cpt.get("version", "v1") == "v1" else 768

    test_phone = torch.rand(1, 200, vec_channels) 
    test_phone_lengths = torch.tensor([200]).long() 
    test_pitch = torch.randint(size=(1, 200), low=5, high=255)  
    test_pitchf = torch.rand(1, 200)
    test_ds = torch.LongTensor([0])  
    test_rnd = torch.rand(1, 192, 200)  

    device = "cpu" 

    net_g = SynthesizerTrnMsNSFsidM(
        *cpt["config"], 
        is_half=use_fp16, 
        version=cpt.get("version", "v1")
    )  
    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g.eval()

    if use_fp16:
        net_g.half()
        test_phone = test_phone.half()
        test_pitchf = test_pitchf.half()
        test_rnd = test_rnd.half()

    test_phone = test_phone.to(device)
    test_phone_lengths = test_phone_lengths.to(device)
    test_pitch = test_pitch.to(device)
    test_pitchf = test_pitchf.to(device)
    test_ds = test_ds.to(device)
    test_rnd = test_rnd.to(device)

    input_names = ["phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd"]
    output_names = ["audio"]

    dyn_axes = {
        "phone": [1],
        "pitch": [1],
        "pitchf": [1],
        "rnd": [2],
    }

    torch.onnx.export(
        net_g,
        (test_phone, test_phone_lengths, test_pitch, test_pitchf, test_ds, test_rnd),
        exported_path,
        dynamic_axes=dyn_axes,
        do_constant_folding=True,
        opset_version=17,
        verbose=True,
        input_names=input_names,
        output_names=output_names,
    )
    model, check  = onnxsim.simplify(exported_path)
    if check:
        for k, v in cpt.items():
            if k == "weight":
                continue
            meta = model.metadata_props.add()
            print(k, v)
            try:
                meta.key = k
                meta.value = json.dumps(v)
            except Exception:
                meta.key = k
                meta.value = str(v)
        onnx.save(model, exported_path)


model_pth_path = "model.pth"
model_onnx_path = "model.onnx"
export_onnx(model_pth_path, model_onnx_path)