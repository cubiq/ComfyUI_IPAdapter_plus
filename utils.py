import re
import torch
import os
import folder_paths
from comfy.clip_vision import clip_preprocess, Output
import comfy.utils
import comfy.model_management as model_management
try:
    import torchvision.transforms.v2 as T
except ImportError:
    import torchvision.transforms as T

def get_clipvision_file(preset):
    preset = preset.lower()
    clipvision_list = folder_paths.get_filename_list("clip_vision")

    if preset.startswith("vit-g"):
        pattern = '(ViT.bigG.14.*39B.b160k|ipadapter.*sdxl|sdxl.*model\.(bin|safetensors))'
    else:
        pattern = '(ViT.H.14.*s32B.b79K|ipadapter.*sd15|sd1.?5.*model\.(bin|safetensors))'
    clipvision_file = [e for e in clipvision_list if re.search(pattern, e, re.IGNORECASE)]

    clipvision_file = folder_paths.get_full_path("clip_vision", clipvision_file[0]) if clipvision_file else None

    return clipvision_file

def get_ipadapter_file(preset, is_sdxl):
    preset = preset.lower()
    ipadapter_list = folder_paths.get_filename_list("ipadapter")
    is_insightface = False
    lora_pattern = None

    if preset.startswith("light"):
        if is_sdxl:
            raise Exception("light model is not supported for SDXL")
        pattern = 'sd15.light.v11\.(safetensors|bin)$'
        # if light model v11 is not found, try with the old version
        if not [e for e in ipadapter_list if re.search(pattern, e, re.IGNORECASE)]:
            pattern = 'sd15.light\.(safetensors|bin)$'
    elif preset.startswith("standard"):
        if is_sdxl:
            pattern = 'ip.adapter.sdxl.vit.h\.(safetensors|bin)$'
        else:
            pattern = 'ip.adapter.sd15\.(safetensors|bin)$'
    elif preset.startswith("vit-g"):
        if is_sdxl:
            pattern = 'ip.adapter.sdxl\.(safetensors|bin)$'
        else:
            pattern = 'sd15.vit.g\.(safetensors|bin)$'
    elif preset.startswith("plus ("):
        if is_sdxl:
            pattern = 'plus.sdxl.vit.h\.(safetensors|bin)$'
        else:
            pattern = 'ip.adapter.plus.sd15\.(safetensors|bin)$'
    elif preset.startswith("plus face"):
        if is_sdxl:
            pattern = 'plus.face.sdxl.vit.h\.(safetensors|bin)$'
        else:
            pattern = 'plus.face.sd15\.(safetensors|bin)$'
    elif preset.startswith("full"):
        if is_sdxl:
            raise Exception("full face model is not supported for SDXL")
        pattern = 'full.face.sd15\.(safetensors|bin)$'
    elif preset.startswith("faceid portrait"):
        if is_sdxl:
            raise Exception("portrait model is not supported for SDXL")
        pattern = 'portrait.sd15\.(safetensors|bin)$'
        is_insightface = True
    elif preset == "faceid":
        if is_sdxl:
            pattern = 'faceid.sdxl\.(safetensors|bin)$'
            lora_pattern = 'faceid.sdxl.lora\.safetensors$'
        else:
            pattern = 'faceid.sd15\.(safetensors|bin)$'
            lora_pattern = 'faceid.sd15.lora\.safetensors$'
        is_insightface = True
    elif preset.startswith("faceid plus -"):
        if is_sdxl:
            raise Exception("faceid plus model is not supported for SDXL")
        pattern = 'faceid.plus.sd15\.(safetensors|bin)$'
        lora_pattern = 'faceid.plus.sd15.lora\.safetensors$'
        is_insightface = True
    elif preset.startswith("faceid plus v2"):
        if is_sdxl:
            pattern = 'faceid.plusv2.sdxl\.(safetensors|bin)$'
            lora_pattern = 'faceid.plusv2.sdxl.lora\.safetensors$'
        else:
            pattern = 'faceid.plusv2.sd15\.(safetensors|bin)$'
            lora_pattern = 'faceid.plusv2.sd15.lora\.safetensors$'
        is_insightface = True
    else:
        raise Exception(f"invalid type '{preset}'")

    ipadapter_file = [e for e in ipadapter_list if re.search(pattern, e, re.IGNORECASE)]
    ipadapter_file = folder_paths.get_full_path("ipadapter", ipadapter_file[0]) if ipadapter_file else None

    return ipadapter_file, is_insightface, lora_pattern

def get_lora_file(pattern):
    lora_list = folder_paths.get_filename_list("loras")
    lora_file = [e for e in lora_list if re.search(pattern, e, re.IGNORECASE)]
    lora_file = folder_paths.get_full_path("loras", lora_file[0]) if lora_file else None

    return lora_file

def ipadapter_model_loader(file):
    model = comfy.utils.load_torch_file(file, safe_load=True)

    if file.lower().endswith(".safetensors"):
        st_model = {"image_proj": {}, "ip_adapter": {}}
        for key in model.keys():
            if key.startswith("image_proj."):
                st_model["image_proj"][key.replace("image_proj.", "")] = model[key]
            elif key.startswith("ip_adapter."):
                st_model["ip_adapter"][key.replace("ip_adapter.", "")] = model[key]
        model = st_model
        del st_model

    if not "ip_adapter" in model.keys() or not model["ip_adapter"]:
        raise Exception("invalid IPAdapter model {}".format(file))

    if 'plusv2' in file.lower():
        model["faceidplusv2"] = True

    return model

def insightface_loader(provider):
    try:
        from insightface.app import FaceAnalysis
    except ImportError as e:
        raise Exception(e)

    path = os.path.join(folder_paths.models_dir, "insightface")
    model = FaceAnalysis(name="buffalo_l", root=path, providers=[provider + 'ExecutionProvider',])
    model.prepare(ctx_id=0, det_size=(640, 640))
    return model

def encode_image_masked(clip_vision, image, mask=None):
    model_management.load_model_gpu(clip_vision.patcher)
    image = image.to(clip_vision.load_device)

    pixel_values = clip_preprocess(image.to(clip_vision.load_device)).float()

    if mask is not None:
        pixel_values = pixel_values * mask.to(clip_vision.load_device)

    out = clip_vision.model(pixel_values=pixel_values, intermediate_output=-2)

    outputs = Output()
    outputs["last_hidden_state"] = out[0].to(model_management.intermediate_device())
    outputs["image_embeds"] = out[2].to(model_management.intermediate_device())
    outputs["penultimate_hidden_states"] = out[1].to(model_management.intermediate_device())
    return outputs

def tensor_to_size(source, dest_size):
    if isinstance(dest_size, torch.Tensor):
        dest_size = dest_size.shape[0]
    source_size = source.shape[0]

    if source_size < dest_size:
        shape = [dest_size - source_size] + [1]*(source.dim()-1)
        source = torch.cat((source, source[-1:].repeat(shape)), dim=0)
    elif source_size > dest_size:
        source = source[:dest_size]

    return source

def min_(tensor_list):
    # return the element-wise min of the tensor list.
    x = torch.stack(tensor_list)
    mn = x.min(axis=0)[0]
    return torch.clamp(mn, min=0)

def max_(tensor_list):
    # return the element-wise max of the tensor list.
    x = torch.stack(tensor_list)
    mx = x.max(axis=0)[0]
    return torch.clamp(mx, max=1)

# From https://github.com/Jamy-L/Pytorch-Contrast-Adaptive-Sharpening/
def contrast_adaptive_sharpening(image, amount):
    img = T.functional.pad(image, (1, 1, 1, 1)).cpu()

    a = img[..., :-2, :-2]
    b = img[..., :-2, 1:-1]
    c = img[..., :-2, 2:]
    d = img[..., 1:-1, :-2]
    e = img[..., 1:-1, 1:-1]
    f = img[..., 1:-1, 2:]
    g = img[..., 2:, :-2]
    h = img[..., 2:, 1:-1]
    i = img[..., 2:, 2:]

    # Computing contrast
    cross = (b, d, e, f, h)
    mn = min_(cross)
    mx = max_(cross)

    diag = (a, c, g, i)
    mn2 = min_(diag)
    mx2 = max_(diag)
    mx = mx + mx2
    mn = mn + mn2

    # Computing local weight
    inv_mx = torch.reciprocal(mx)
    amp = inv_mx * torch.minimum(mn, (2 - mx))

    # scaling
    amp = torch.sqrt(amp)
    w = - amp * (amount * (1/5 - 1/8) + 1/8)
    div = torch.reciprocal(1 + 4*w)

    output = ((b + d + f + h)*w + e) * div
    output = torch.nan_to_num(output)
    output = output.clamp(0, 1)

    return output

def tensor_to_image(tensor):
    image = tensor.mul(255).clamp(0, 255).byte().cpu()
    image = image[..., [2, 1, 0]].numpy()
    return image

def image_to_tensor(image):
    tensor = torch.clamp(torch.from_numpy(image).float() / 255., 0, 1)
    tensor = tensor[..., [2, 1, 0]]
    return tensor