import torch
import contextlib
import os
import inspect

import comfy.utils
import comfy.model_management
import folder_paths

from torch import nn
import torch.nn.functional as F
import torchvision.transforms as TT

from .resampler import Resampler

MODELS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")

# attention_channels
SD_V12_CHANNELS = [320] * 4 + [640] * 4 + [1280] * 4 + [1280] * 6 + [640] * 6 + [320] * 6 + [1280] * 2
SD_XL_CHANNELS = [640] * 8 + [1280] * 40 + [1280] * 60 + [640] * 12 + [1280] * 20

def get_filename_list(path):
    return [f for f in os.listdir(path) if f.endswith('.bin')]

class ImageProjModel(nn.Module):
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()
        
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = nn.LayerNorm(cross_attention_dim)
        
    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(-1, self.clip_extra_context_tokens, self.cross_attention_dim)
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens

class To_KV(nn.Module):
    def __init__(self, cross_attention_dim):
        super().__init__()

        channels = SD_XL_CHANNELS if cross_attention_dim == 2048 else SD_V12_CHANNELS
        self.to_kvs = nn.ModuleList([nn.Linear(cross_attention_dim, channel, bias=False) for channel in channels])
        
    def load_state_dict(self, state_dict):
        for i, key in enumerate(state_dict.keys()):
            self.to_kvs[i].weight.data = state_dict[key]

def set_model_patch_replace(model, patch_kwargs, key):
    to = model.model_options["transformer_options"]
    if "patches_replace" not in to:
        to["patches_replace"] = {}
    if "attn2" not in to["patches_replace"]:
        to["patches_replace"]["attn2"] = {}
    if key not in to["patches_replace"]["attn2"]:
        patch = CrossAttentionPatch(**patch_kwargs)
        to["patches_replace"]["attn2"][key] = patch
    else:
        to["patches_replace"]["attn2"][key].set_new_condition(**patch_kwargs)

def attention(q, k, v, extra_options):
    if not hasattr(F, "multi_head_attention_forward"):
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=extra_options["n_heads"]), (q, k, v))
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * (extra_options["dim_head"] ** -0.5)
        sim = F.softmax(sim, dim=-1)
        out = torch.einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=extra_options["n_heads"])
    else:
        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.view(b, -1, extra_options["n_heads"], extra_options["dim_head"]).transpose(1, 2),
            (q, k, v),
        )
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(b, -1, extra_options["n_heads"] * extra_options["dim_head"])
    return out

# TODO: still have to find the best way to add noise to the uncond image
def image_add_noise(image, noise):
    image = image.permute([0,3,1,2])
    torch.manual_seed(0) # use a fixed random for reproducible results
    transforms = TT.Compose([
        TT.CenterCrop(min(image.shape[2], image.shape[3])),
        TT.Resize((224, 224), interpolation=TT.InterpolationMode.BICUBIC, antialias=True),
        TT.ElasticTransform(alpha=75.0, sigma=noise*3.5), # shuffle the image
        #TT.GaussianBlur(5, sigma=1.5),              # by adding blur in the negative image we get sharper results
        #TT.RandomSolarize(threshold=.75, p=1),       # add color aberration to prevent sending the same colors in the negative image
        TT.RandomVerticalFlip(p=1.0),                # flip the image to change the geometry even more
        TT.RandomHorizontalFlip(p=1.0),
    ])
    image = transforms(image.cpu())
    image = image.permute([0,2,3,1])
    image = image + ((0.25*(1-noise)+0.05) * torch.randn_like(image) )   # add random noise
    return image

def zeroed_hidden_states(clip_vision, batch_size):
    image = torch.zeros([batch_size, 224, 224, 3])
    img = torch.clip((255. * image), 0, 255).round().int()
    img = list(map(lambda a: a, img))
    inputs = clip_vision.processor(images=img, return_tensors="pt")
    comfy.model_management.load_model_gpu(clip_vision.patcher)
    pixel_values = torch.zeros_like(inputs['pixel_values']).to(clip_vision.load_device)

    if clip_vision.dtype != torch.float32:
        precision_scope = torch.autocast
    else:
        precision_scope = lambda a, b: contextlib.nullcontext(a)

    with precision_scope(comfy.model_management.get_autocast_device(clip_vision.load_device), torch.float32):
        outputs = clip_vision.model(pixel_values, output_hidden_states=True)

    # we only need the penultimate hidden states
    outputs = outputs['hidden_states'][-2].cpu() if 'hidden_states' in outputs else None

    return outputs

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
    img = F.pad(image, pad=(1, 1, 1, 1)).cpu()

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
    output = output.clamp(0, 1)
    output = torch.nan_to_num(output)

    return (output)

class IPAdapter(nn.Module):
    def __init__(self, ipadapter_model, cross_attention_dim=1024, output_cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4, is_sdxl=False):
        super().__init__()

        self.clip_embeddings_dim = clip_embeddings_dim
        self.cross_attention_dim = cross_attention_dim
        self.output_cross_attention_dim = output_cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.is_sdxl = is_sdxl

        self.image_proj_model = self.init_proj()
        self.image_proj_model.load_state_dict(ipadapter_model["image_proj"])
        self.ip_layers = To_KV(self.output_cross_attention_dim)
        self.ip_layers.load_state_dict(ipadapter_model["ip_adapter"])

    def init_proj(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.cross_attention_dim,
            clip_embeddings_dim=self.clip_embeddings_dim,
            clip_extra_context_tokens=self.clip_extra_context_tokens
        )
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, clip_embed, clip_embed_zeroed):
        image_prompt_embeds = self.image_proj_model(clip_embed)
        uncond_image_prompt_embeds = self.image_proj_model(clip_embed_zeroed)
        return image_prompt_embeds, uncond_image_prompt_embeds

class IPAdapterPlus(IPAdapter):
    def init_proj(self):
        image_proj_model = Resampler(
            dim=self.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=20 if self.is_sdxl else 12,
            num_queries=self.clip_extra_context_tokens,
            embedding_dim=self.clip_embeddings_dim,
            output_dim=self.output_cross_attention_dim,
            ff_mult=4
        )
        return image_proj_model

class CrossAttentionPatch:
    # forward for patching
    def __init__(self, weight, ipadapter, dtype, number, cond, uncond, mask=None):
        self.weights = [weight]
        self.ipadapters = [ipadapter]
        self.conds = [cond]
        self.unconds = [uncond]
        self.dtype = dtype
        self.device = 'cuda'
        self.number = number
        self.masks = [mask]
    
    def set_new_condition(self, weight, ipadapter, cond, uncond, dtype, number, mask=None):
        self.weights.append(weight)
        self.ipadapters.append(ipadapter)
        self.conds.append(cond)
        self.unconds.append(uncond)
        self.masks.append(mask)
        self.dtype = dtype
        self.device = 'cuda'

    def __call__(self, n, context_attn2, value_attn2, extra_options):
        org_dtype = n.dtype
        frame = inspect.currentframe()
        outer_frame = frame.f_back
        cond_or_uncond = outer_frame.f_locals["transformer_options"]["cond_or_uncond"] if "cond_or_uncond" in outer_frame.f_locals["transformer_options"] else None
        with torch.autocast(device_type=self.device, dtype=self.dtype):
            q = n
            k = context_attn2
            v = value_attn2
            b, _, _ = q.shape
            batch_prompt = b // len(cond_or_uncond) if cond_or_uncond is not None else None
            out = attention(q, k, v, extra_options)

            for weight, cond, uncond, ipadapter, mask in zip(self.weights, self.conds, self.unconds, self.ipadapters, self.masks):
                if cond_or_uncond is not None:
                    uncond_cond = torch.cat([(cond.repeat(batch_prompt, 1, 1), uncond.repeat(batch_prompt, 1, 1))[i] for i in cond_or_uncond], dim=0)
                else:
                    uncond_cond = torch.cat([uncond.repeat(b//2, 1, 1), cond.repeat(b//2, 1, 1)], dim=0)

                # k, v for ip_adapter
                ip_k = ipadapter.ip_layers.to_kvs[self.number*2](uncond_cond)
                ip_v = ipadapter.ip_layers.to_kvs[self.number*2+1](uncond_cond)

                ip_out = attention(q, ip_k, ip_v, extra_options)

                out = out + ip_out * weight

        return out.to(dtype=org_dtype)

class IPAdapterModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "ipadapter_file": (get_filename_list(MODELS_DIR), )}}

    RETURN_TYPES = ("IPADAPTER",)
    FUNCTION = "load_ipadapter_model"

    CATEGORY = "ipadapter"

    def load_ipadapter_model(self, ipadapter_file):
        ckpt_path = os.path.join(MODELS_DIR, ipadapter_file)

        model = comfy.utils.load_torch_file(ckpt_path, safe_load=True)
        keys = model.keys()

        if not "ip_adapter" in keys:
            raise Exception("invalid IPAdapter model {}".format(ckpt_path))

        return (model,)

class IPAdapterApply:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ipadapter": ("IPADAPTER", ),
                "clip_vision": ("CLIP_VISION",),
                "image": ("IMAGE",),
                "model": ("MODEL", ),
                "weight": ("FLOAT", { "default": 1.0, "min": -1, "max": 3, "step": 0.05 }),
                "noise": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01 })
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_ipadapter"
    CATEGORY = "ipadapter"

    def apply_ipadapter(self, ipadapter, model, weight, clip_vision=None, image=None, noise=None, embeds=None):
        self.dtype = model.model.diffusion_model.dtype
        self.device = comfy.model_management.get_torch_device()
        self.weight = weight
        self.is_plus = "latents" in ipadapter["image_proj"]

        output_cross_attention_dim = ipadapter["ip_adapter"]["1.to_k_ip.weight"].shape[1]
        self.is_sdxl = output_cross_attention_dim == 2048
        cross_attention_dim = 1280 if self.is_plus and self.is_sdxl else output_cross_attention_dim
        clip_extra_context_tokens = 16 if self.is_plus else 4

        if embeds is not None:
            embeds = torch.unbind(embeds)
            clip_embed = embeds[0].cpu()
            clip_embed_zeroed = embeds[1].cpu()
        else:
            if image.shape[1] != image.shape[2]:
                print("\033[33mINFO: the IPAdapter reference image is not a square, CLIPImageProcessor will resize and crop it at the center. If the main focus of the picture is not in the middle the result might not be what you are expecting.\033[0m")

            clip_embed = clip_vision.encode_image(image)
            neg_image = image_add_noise(image, noise) if noise > 0 else None
            
            if self.is_plus:
                clip_embed = clip_embed.penultimate_hidden_states
                if noise > 0:
                    clip_embed_zeroed = clip_vision.encode_image(neg_image).penultimate_hidden_states
                else:
                    clip_embed_zeroed = zeroed_hidden_states(clip_vision, image.shape[0])
            else:
                clip_embed = clip_embed.image_embeds
                if noise > 0:
                    clip_embed_zeroed = clip_vision.encode_image(neg_image).image_embeds
                else:
                    clip_embed_zeroed = torch.zeros_like(clip_embed)

        clip_embeddings_dim = clip_embed.shape[-1]

        IPA = IPAdapterPlus if self.is_plus else IPAdapter

        self.ipadapter = IPA(
            ipadapter,
            cross_attention_dim=cross_attention_dim,
            output_cross_attention_dim=output_cross_attention_dim,
            clip_embeddings_dim=clip_embeddings_dim,
            clip_extra_context_tokens=clip_extra_context_tokens,
            is_sdxl=self.is_sdxl,
        )
        
        self.ipadapter.to(self.device, dtype=self.dtype)

        image_prompt_embeds, uncond_image_prompt_embeds = self.ipadapter.get_image_embeds(clip_embed.to(self.device, self.dtype), clip_embed_zeroed.to(self.device, self.dtype))
        image_prompt_embeds = image_prompt_embeds.to(self.device, dtype=self.dtype)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.to(self.device, dtype=self.dtype)

        work_model = model.clone()

        patch_kwargs = {
            "number": 0,
            "weight": self.weight,
            "ipadapter": self.ipadapter,
            "dtype": self.dtype,
            "cond": image_prompt_embeds,
            "uncond": uncond_image_prompt_embeds,
        }

        if not self.is_sdxl:
            for id in [1,2,4,5,7,8]: # id of input_blocks that have cross attention
                set_model_patch_replace(work_model, patch_kwargs, ("input", id))
                patch_kwargs["number"] += 1
            for id in [3,4,5,6,7,8,9,10,11]: # id of output_blocks that have cross attention
                set_model_patch_replace(work_model, patch_kwargs, ("output", id))
                patch_kwargs["number"] += 1
            set_model_patch_replace(work_model, patch_kwargs, ("middle", 0))
        else:
            for id in [4,5,7,8]: # id of input_blocks that have cross attention
                block_indices = range(2) if id in [4, 5] else range(10) # transformer_depth
                for index in block_indices:
                    set_model_patch_replace(work_model, patch_kwargs, ("input", id, index))
                    patch_kwargs["number"] += 1
            for id in range(6): # id of output_blocks that have cross attention
                block_indices = range(2) if id in [3, 4, 5] else range(10) # transformer_depth
                for index in block_indices:
                    set_model_patch_replace(work_model, patch_kwargs, ("output", id, index))
                    patch_kwargs["number"] += 1
            for index in range(10):
                set_model_patch_replace(work_model, patch_kwargs, ("midlle", 0, index))
                patch_kwargs["number"] += 1

        return (work_model, )

class PrepImageForClipVision:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "interpolation": (["bicubic", "nearest", "bilinear", "area", "nearest-exact", "lanczos"],),
            "crop_position": (["top", "bottom", "left", "right", "center"],),
            "sharpening": ("FLOAT", {"default": 0.0, "min": 0, "max": 1, "step": 0.05}),
            "add_weight": ("BOOLEAN", {"default": False}),
        },
    }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "prep_image"

    CATEGORY = "ipadapter"

    def prep_image(self, image, interpolation="lanczos", crop_position="center", sharpening=0.0, add_weight=False):
        _, oh, ow, _ = image.shape

        crop_size = min(oh, ow)
        x = (ow-crop_size) // 2
        y = (oh-crop_size) // 2
        if "top" in crop_position:
            y = 0
        elif "bottom" in crop_position:
            y = oh-crop_size
        elif "left" in crop_position:
            x = 0
        elif "right" in crop_position:
            x = ow-crop_size
        
        x2 = x+crop_size
        y2 = y+crop_size

        # crop
        output = image[:, y:y2, x:x2, :]

        output = output.permute([0,3,1,2])

        # resize
        if interpolation == "lanczos":
            output = comfy.utils.lanczos(output, 224, 224)
        else:
            output = F.interpolate(output, size=(224, 224), mode=interpolation)
       
        if sharpening > 0:
            output = contrast_adaptive_sharpening(output, sharpening)
        
        output = output.permute([0,2,3,1])

        if add_weight is True:
            output = torch.stack((output,output)).squeeze()

        return (output,)


class IPAdapterEncoder:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip_vision": ("CLIP_VISION",),
            "image": ("IMAGE",),
            "plus": ("BOOLEAN", { "default": False }),
            "noise": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01 }),
        },
    }

    RETURN_TYPES = ("EMBEDS",)
    FUNCTION = "preprocess"
    CATEGORY = "ipadapter"

    def preprocess(self, clip_vision, image, plus, noise):
        clip_embed = clip_vision.encode_image(image)
        neg_image = image_add_noise(image, noise) if noise > 0 else None
        
        if plus:
            clip_embed = clip_embed.penultimate_hidden_states
            if noise > 0:
                clip_embed_zeroed = clip_vision.encode_image(neg_image).penultimate_hidden_states
            else:
                clip_embed_zeroed = zeroed_hidden_states(clip_vision, image.shape[0])
        else:
            clip_embed = clip_embed.image_embeds
            if noise > 0:
                clip_embed_zeroed = clip_vision.encode_image(neg_image).image_embeds
            else:
                clip_embed_zeroed = torch.zeros_like(clip_embed)
        
        output = torch.stack((clip_embed, clip_embed_zeroed))

        return( output, )

class IPAdapterApplyEncoded(IPAdapterApply):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ipadapter": ("IPADAPTER", ),
                "embeds": ("EMBEDS",),
                "model": ("MODEL", ),
                "weight": ("FLOAT", { "default": 1.0, "min": -1, "max": 3, "step": 0.05 }),
            },
        }

class IPAdapterSaveEmbeds:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "embeds": ("EMBEDS",),
            "filename_prefix": ("STRING", {"default": "embeds/IPAdapter"})
        },
    }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "ipadapter"

    def save(self, embeds, filename_prefix):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        file = f"{filename}_{counter:05}_.ipadpt"
        file = os.path.join(full_output_folder, file)

        torch.save(embeds, file)
        return (None, )


class IPAdapterLoadEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.endswith(".ipadpt")]
        return {"required": {"embeds": [sorted(files), ]}, }

    RETURN_TYPES = ("EMBEDS", )
    FUNCTION = "load"
    CATEGORY = "ipadapter"

    def load(self, embeds):
        path = folder_paths.get_annotated_filepath(embeds)
        output = torch.load(path).cpu()

        return (output, )

NODE_CLASS_MAPPINGS = {
    "IPAdapterModelLoader": IPAdapterModelLoader,
    "IPAdapterApply": IPAdapterApply,
    "IPAdapterApplyEncoded": IPAdapterApplyEncoded,
    "PrepImageForClipVision": PrepImageForClipVision,
    "IPAdapterEncoder": IPAdapterEncoder,
    "IPAdapterSaveEmbeds": IPAdapterSaveEmbeds,
    "IPAdapterLoadEmbeds": IPAdapterLoadEmbeds,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IPAdapterModelLoader": "Load IPAdapter Model",
    "IPAdapterApply": "Apply IPAdapter",
    "IPAdapterApplyEncoded": "Apply IPAdapter from Encoded",
    "PrepImageForClipVision": "Prepare Image For Clip Vision",
    "IPAdapterEncoder": "Encode IPAdapter Image",
    "IPAdapterSaveEmbeds": "Save IPAdapter Embeds",
    "IPAdapterLoadEmbeds": "Load IPAdapter Embeds",
}
