import torch
import os
import math
import folder_paths

import comfy.model_management as model_management
from comfy.clip_vision import load as load_clip_vision
from comfy.sd import load_lora_for_models
import comfy.utils

import torch.nn as nn
from PIL import Image
try:
    import torchvision.transforms.v2 as T
except ImportError:
    import torchvision.transforms as T

from .image_proj_models import MLPProjModel, MLPProjModelFaceId, ProjModelFaceIdPlus, Resampler, ImageProjModel
from .CrossAttentionPatch import CrossAttentionPatch
from .utils import (
    encode_image_masked,
    tensor_to_size,
    contrast_adaptive_sharpening,
    tensor_to_image,
    image_to_tensor,
    ipadapter_model_loader,
    insightface_loader,
    get_clipvision_file,
    get_ipadapter_file,
    get_lora_file,
)

# set the models directory
if "ipadapter" not in folder_paths.folder_names_and_paths:
    current_paths = [os.path.join(folder_paths.models_dir, "ipadapter")]
else:
    current_paths, _ = folder_paths.folder_names_and_paths["ipadapter"]
folder_paths.folder_names_and_paths["ipadapter"] = (current_paths, folder_paths.supported_pt_extensions)

WEIGHT_TYPES = ["linear", "ease in", "ease out", 'ease in-out', 'reverse in-out', 'weak input', 'weak output', 'weak middle', 'strong middle', 'style transfer (SDXL)']

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Main IPAdapter Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
class IPAdapter(nn.Module):
    def __init__(self, ipadapter_model, cross_attention_dim=1024, output_cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4, is_sdxl=False, is_plus=False, is_full=False, is_faceid=False):
        super().__init__()

        self.clip_embeddings_dim = clip_embeddings_dim
        self.cross_attention_dim = cross_attention_dim
        self.output_cross_attention_dim = output_cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.is_sdxl = is_sdxl
        self.is_full = is_full
        self.is_plus = is_plus

        if is_faceid:
            self.image_proj_model = self.init_proj_faceid()
        elif is_full:
            self.image_proj_model = self.init_proj_full()
        elif is_plus:
            self.image_proj_model = self.init_proj_plus()
        else:
            self.image_proj_model = self.init_proj()

        self.image_proj_model.load_state_dict(ipadapter_model["image_proj"])
        self.ip_layers = To_KV(ipadapter_model["ip_adapter"])

    def init_proj(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.cross_attention_dim,
            clip_embeddings_dim=self.clip_embeddings_dim,
            clip_extra_context_tokens=self.clip_extra_context_tokens
        )
        return image_proj_model

    def init_proj_plus(self):
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

    def init_proj_full(self):
        image_proj_model = MLPProjModel(
            cross_attention_dim=self.cross_attention_dim,
            clip_embeddings_dim=self.clip_embeddings_dim
        )
        return image_proj_model

    def init_proj_faceid(self):
        if self.is_plus:
            image_proj_model = ProjModelFaceIdPlus(
                cross_attention_dim=self.cross_attention_dim,
                id_embeddings_dim=512,
                clip_embeddings_dim=self.clip_embeddings_dim, # 1280,
                num_tokens=self.clip_extra_context_tokens, # 4,
            )
        else:
            image_proj_model = MLPProjModelFaceId(
                cross_attention_dim=self.cross_attention_dim,
                id_embeddings_dim=512,
                num_tokens=self.clip_extra_context_tokens,
            )
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, clip_embed, clip_embed_zeroed):
        image_prompt_embeds = self.image_proj_model(clip_embed)
        uncond_image_prompt_embeds = self.image_proj_model(clip_embed_zeroed)
        return image_prompt_embeds, uncond_image_prompt_embeds

    @torch.inference_mode()
    def get_image_embeds_faceid_plus(self, face_embed, clip_embed, s_scale, shortcut):
        embeds = self.image_proj_model(face_embed, clip_embed, scale=s_scale, shortcut=shortcut)
        return embeds

class To_KV(nn.Module):
    def __init__(self, state_dict):
        super().__init__()

        self.to_kvs = nn.ModuleDict()
        for key, value in state_dict.items():
            self.to_kvs[key.replace(".weight", "").replace(".", "_")] = nn.Linear(value.shape[1], value.shape[0], bias=False)
            self.to_kvs[key.replace(".weight", "").replace(".", "_")].weight.data = value

def set_model_patch_replace(model, patch_kwargs, key):
    to = model.model_options["transformer_options"]
    if "patches_replace" not in to:
        to["patches_replace"] = {}
    if "attn2" not in to["patches_replace"]:
        to["patches_replace"]["attn2"] = {}
    if key not in to["patches_replace"]["attn2"]:
        to["patches_replace"]["attn2"][key] = CrossAttentionPatch(**patch_kwargs)
    else:
        to["patches_replace"]["attn2"][key].set_new_condition(**patch_kwargs)

def ipadapter_execute(model,
                      ipadapter,
                      clipvision,
                      insightface=None,
                      image=None,
                      image_negative=None,
                      weight=1.0,
                      weight_faceidv2=None,
                      weight_type="linear",
                      combine_embeds="concat",
                      start_at=0.0,
                      end_at=1.0,
                      attn_mask=None,
                      pos_embed=None,
                      neg_embed=None,
                      unfold_batch=False,
                      embeds_scaling='V only'):
    dtype = torch.float16 if model_management.should_use_fp16() else torch.bfloat16 if model_management.should_use_bf16() else torch.float32
    device = model_management.get_torch_device()

    is_full = "proj.3.weight" in ipadapter["image_proj"]
    is_portrait = "proj.2.weight" in ipadapter["image_proj"] and not "proj.3.weight" in ipadapter["image_proj"] and not "0.to_q_lora.down.weight" in ipadapter["ip_adapter"]
    is_faceid = is_portrait or "0.to_q_lora.down.weight" in ipadapter["ip_adapter"]
    is_plus = is_full or "latents" in ipadapter["image_proj"] or "perceiver_resampler.proj_in.weight" in ipadapter["image_proj"]
    is_faceidv2 = "faceidplusv2" in ipadapter
    output_cross_attention_dim = ipadapter["ip_adapter"]["1.to_k_ip.weight"].shape[1]
    is_sdxl = output_cross_attention_dim == 2048

    if weight_type == "style transfer (SDXL)" and not is_sdxl:
        weight_type = "linear"
        print("\033[33mINFO: 'Style Transfer' weight type is only available for SDXL models, falling back to 'linear'.\033[0m")

    if is_faceid and not insightface:
        raise Exception("insightface model is required for FaceID models")

    if is_faceidv2:
        weight_faceidv2 = weight_faceidv2 if weight_faceidv2 is not None else weight*2

    cross_attention_dim = 1280 if is_plus and is_sdxl and not is_faceid else output_cross_attention_dim
    clip_extra_context_tokens = 16 if (is_plus and not is_faceid) or is_portrait else 4

    if image is not None and image.shape[1] != image.shape[2]:
        print("\033[33mINFO: the IPAdapter reference image is not a square, CLIPImageProcessor will resize and crop it at the center. If the main focus of the picture is not in the middle the result might not be what you are expecting.\033[0m")

    face_cond_embeds = None
    if is_faceid:
        if insightface is None:
            raise Exception("Insightface model is required for FaceID models")

        from insightface.utils import face_align

        insightface.det_model.input_size = (640,640) # reset the detection size
        image_iface = tensor_to_image(image)
        face_cond_embeds = []
        image = []

        for i in range(image_iface.shape[0]):
            for size in [(size, size) for size in range(640, 256, -64)]:
                insightface.det_model.input_size = size # TODO: hacky but seems to be working
                face = insightface.get(image_iface[i])
                if face:
                    face_cond_embeds.append(torch.from_numpy(face[0].normed_embedding).unsqueeze(0))
                    image.append(image_to_tensor(face_align.norm_crop(image_iface[i], landmark=face[0].kps, image_size=256)))

                    if 640 not in size:
                        print(f"\033[33mINFO: InsightFace detection resolution lowered to {size}.\033[0m")
                    break
            else:
                raise Exception('InsightFace: No face detected.')
        face_cond_embeds = torch.stack(face_cond_embeds).to(device, dtype=dtype)
        image = torch.stack(image)
        del image_iface, face

    if image is not None:
        img_cond_embeds = encode_image_masked(clipvision, image)

        if is_plus:
            img_cond_embeds = img_cond_embeds.penultimate_hidden_states
            image_negative = image_negative if image_negative is not None else torch.zeros([1, 224, 224, 3])
            img_uncond_embeds = encode_image_masked(clipvision, image_negative).penultimate_hidden_states
        else:
            img_cond_embeds = img_cond_embeds.image_embeds if not is_faceid else face_cond_embeds
            if image_negative is not None:
                img_uncond_embeds = encode_image_masked(clipvision, image_negative).image_embeds
            else:
                img_uncond_embeds = torch.zeros_like(img_cond_embeds)
    elif pos_embed is not None:
        img_cond_embeds = pos_embed

        if neg_embed is not None:
            img_uncond_embeds = neg_embed
        else:
            if is_plus:
                img_uncond_embeds = encode_image_masked(clipvision, torch.zeros([1, 224, 224, 3])).penultimate_hidden_states
            else:
                img_uncond_embeds = torch.zeros_like(img_cond_embeds)
    else:
        raise Exception("Images or Embeds are required")

    # ensure that cond and uncond have the same batch size
    img_uncond_embeds = tensor_to_size(img_uncond_embeds, img_cond_embeds.shape[0])

    img_cond_embeds = img_cond_embeds.to(device, dtype=dtype)
    img_uncond_embeds = img_uncond_embeds.to(device, dtype=dtype)

    # combine the embeddings if needed
    if combine_embeds != "concat" and img_cond_embeds.shape[0] > 1 and not unfold_batch:
        if combine_embeds == "add":
            img_cond_embeds = torch.sum(img_cond_embeds, dim=0).unsqueeze(0)
            if face_cond_embeds is not None:
                face_cond_embeds = torch.sum(face_cond_embeds, dim=0).unsqueeze(0)
        elif combine_embeds == "subtract":
            img_cond_embeds = img_cond_embeds[0] - torch.mean(img_cond_embeds[1:], dim=0)
            img_cond_embeds = img_cond_embeds.unsqueeze(0)
            if face_cond_embeds is not None:
                face_cond_embeds = face_cond_embeds[0] - torch.mean(face_cond_embeds[1:], dim=0)
                face_cond_embeds = face_cond_embeds.unsqueeze(0)
        elif combine_embeds == "average":
            img_cond_embeds = torch.mean(img_cond_embeds, dim=0).unsqueeze(0)
            if face_cond_embeds is not None:
                face_cond_embeds = torch.mean(face_cond_embeds, dim=0).unsqueeze(0)
        elif combine_embeds == "norm average":
            img_cond_embeds = torch.mean(img_cond_embeds / torch.norm(img_cond_embeds, dim=0, keepdim=True), dim=0).unsqueeze(0)
            if face_cond_embeds is not None:
                face_cond_embeds = torch.mean(face_cond_embeds / torch.norm(face_cond_embeds, dim=0, keepdim=True), dim=0).unsqueeze(0)
        img_uncond_embeds = img_uncond_embeds[0].unsqueeze(0) # TODO: better strategy for uncond could be to average them

    if attn_mask is not None:
        attn_mask = attn_mask.to(device, dtype=dtype)

    ipa = IPAdapter(
        ipadapter,
        cross_attention_dim=cross_attention_dim,
        output_cross_attention_dim=output_cross_attention_dim,
        clip_embeddings_dim=img_cond_embeds.shape[-1],
        clip_extra_context_tokens=clip_extra_context_tokens,
        is_sdxl=is_sdxl,
        is_plus=is_plus,
        is_full=is_full,
        is_faceid=is_faceid
    ).to(device, dtype=dtype)

    if is_faceid and is_plus:
        cond = ipa.get_image_embeds_faceid_plus(face_cond_embeds, img_cond_embeds, weight_faceidv2, is_faceidv2)
        # TODO: check if noise helps with the uncod face embeds
        uncod = ipa.get_image_embeds_faceid_plus(torch.zeros_like(face_cond_embeds), img_uncond_embeds, weight_faceidv2, is_faceidv2)
    else:
        cond, uncod = ipa.get_image_embeds(img_cond_embeds, img_uncond_embeds)

    cond = cond.to(device, dtype=dtype)
    uncod = uncod.to(device, dtype=dtype)

    del img_cond_embeds, img_uncond_embeds

    sigma_start = model.model.model_sampling.percent_to_sigma(start_at)
    sigma_end = model.model.model_sampling.percent_to_sigma(end_at)

    patch_kwargs = {
        "ipadapter": ipa,
        "number": 0,
        "weight": weight,
        "cond": cond,
        "uncond": uncod,
        "weight_type": weight_type,
        "mask": attn_mask,
        "sigma_start": sigma_start,
        "sigma_end": sigma_end,
        "unfold_batch": unfold_batch,
        "embeds_scaling": embeds_scaling,
    }

    if not is_sdxl:
        for id in [1,2,4,5,7,8]: # id of input_blocks that have cross attention
            set_model_patch_replace(model, patch_kwargs, ("input", id))
            patch_kwargs["number"] += 1
        for id in [3,4,5,6,7,8,9,10,11]: # id of output_blocks that have cross attention
            set_model_patch_replace(model, patch_kwargs, ("output", id))
            patch_kwargs["number"] += 1
        set_model_patch_replace(model, patch_kwargs, ("middle", 0))
    else:
        for id in [4,5,7,8]: # id of input_blocks that have cross attention
            block_indices = range(2) if id in [4, 5] else range(10) # transformer_depth
            for index in block_indices:
                set_model_patch_replace(model, patch_kwargs, ("input", id, index))
                patch_kwargs["number"] += 1
        for id in range(6): # id of output_blocks that have cross attention
            block_indices = range(2) if id in [3, 4, 5] else range(10) # transformer_depth
            for index in block_indices:
                set_model_patch_replace(model, patch_kwargs, ("output", id, index))
                patch_kwargs["number"] += 1
        for index in range(10):
            set_model_patch_replace(model, patch_kwargs, ("middle", 0, index))
            patch_kwargs["number"] += 1

    return model


"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Loaders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
class IPAdapterUnifiedLoader:
    def __init__(self):
        self.lora = None
        self.clipvision = { "file": None, "model": None }
        self.ipadapter = { "file": None, "model": None }
        self.insightface = { "provider": None, "model": None }

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL", ),
            "preset": (['LIGHT - SD1.5 only (low strength)', 'STANDARD (medium strength)', 'VIT-G (medium strength)', 'PLUS (high strength)', 'PLUS FACE (portraits)', 'FULL FACE - SD1.5 only (portraits stronger)'], ),
        },
        "optional": {
            "ipadapter": ("IPADAPTER", ),
        }}

    RETURN_TYPES = ("MODEL", "IPADAPTER", )
    RETURN_NAMES = ("model", "ipadapter", )
    FUNCTION = "load_models"
    CATEGORY = "ipadapter"

    def load_models(self, model, preset, lora_strength=0.0, provider="CPU", ipadapter=None):
        pipeline = { "clipvision": { 'file': None, 'model': None }, "ipadapter": { 'file': None, 'model': None }, "insightface": { 'provider': None, 'model': None } }
        if ipadapter is not None:
            pipeline = ipadapter

        # 1. Load the clipvision model
        clipvision_file = get_clipvision_file(preset)
        if clipvision_file is None:
            raise Exception("ClipVision model not found.")

        if clipvision_file != self.clipvision['file']:
            if clipvision_file != pipeline['clipvision']['file']:
                self.clipvision['file'] = clipvision_file
                self.clipvision['model'] = load_clip_vision(clipvision_file)
                print(f"\033[33mINFO: Clip Vision model loaded from {clipvision_file}\033[0m")
            else:
                self.clipvision = pipeline['clipvision']

        # 2. Load the ipadapter model
        is_sdxl = isinstance(model.model, comfy.model_base.SDXL)
        ipadapter_file, is_insightface, lora_pattern = get_ipadapter_file(preset, is_sdxl)
        if ipadapter_file is None:
            raise Exception("IPAdapter model not found.")

        if ipadapter_file != self.ipadapter['file']:
            if pipeline['ipadapter']['file'] != ipadapter_file:
                self.ipadapter['file'] = ipadapter_file
                self.ipadapter['model'] = ipadapter_model_loader(ipadapter_file)
                print(f"\033[33mINFO: IPAdapter model loaded from {ipadapter_file}\033[0m")
            else:
                self.ipadapter = pipeline['ipadapter']

        # 3. Load the lora model if needed
        if lora_pattern is not None:
            lora_file = get_lora_file(lora_pattern)
            lora_model = None
            if lora_file is None:
                raise Exception("LoRA model not found.")

            if self.lora is not None:
                if lora_file == self.lora['file']:
                    lora_model = self.lora['model']
                else:
                    self.lora = None
                    torch.cuda.empty_cache()

            if lora_model is None:
                lora_model = comfy.utils.load_torch_file(lora_file, safe_load=True)
                self.lora = { 'file': lora_file, 'model': lora_model }
                print(f"\033[33mINFO: LoRA model loaded from {lora_file}\033[0m")

            if lora_strength > 0:
                model, _ = load_lora_for_models(model, None, lora_model, lora_strength, 0)

        # 4. Load the insightface model if needed
        if is_insightface:
            if provider != self.insightface['provider']:
                if pipeline['insightface']['provider'] != provider:
                    self.insightface['provider'] = provider
                    self.insightface['model'] = insightface_loader(provider)
                    print(f"\033[33mINFO: InsightFace model loaded with {provider} provider\033[0m")
                else:
                    self.insightface = pipeline['insightface']

        return (model, { 'clipvision': self.clipvision, 'ipadapter': self.ipadapter, 'insightface': self.insightface }, )

class IPAdapterUnifiedLoaderFaceID(IPAdapterUnifiedLoader):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL", ),
            "preset": (['FACEID', 'FACEID PLUS - SD1.5 only', 'FACEID PLUS V2', 'FACEID PORTRAIT (style transfer)'], ),
            "lora_strength": ("FLOAT", { "default": 0.6, "min": 0, "max": 1, "step": 0.01 }),
            "provider": (["CPU", "CUDA", "ROCM", "DirectML", "OpenVINO", "CoreML"], ),
        },
        "optional": {
            "ipadapter": ("IPADAPTER", ),
        }}

    RETURN_NAMES = ("MODEL", "ipadapter", )

class IPAdapterModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "ipadapter_file": (folder_paths.get_filename_list("ipadapter"), )}}

    RETURN_TYPES = ("IPADAPTER",)
    FUNCTION = "load_ipadapter_model"
    CATEGORY = "ipadapter"

    def load_ipadapter_model(self, ipadapter_file):
        ipadapter_file = folder_paths.get_full_path("ipadapter", ipadapter_file)
        return (ipadapter_model_loader(ipadapter_file),)

class IPAdapterInsightFaceLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "provider": (["CPU", "CUDA", "ROCM"], ),
            },
        }

    RETURN_TYPES = ("INSIGHTFACE",)
    FUNCTION = "load_insightface"
    CATEGORY = "ipadapter"

    def load_insightface(self, provider):
        return (insightface_loader(provider),)

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Main Apply Nodes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
class IPAdapterSimple:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "ipadapter": ("IPADAPTER", ),
                "image": ("IMAGE",),
                "weight": ("FLOAT", { "default": 1.0, "min": -1, "max": 3, "step": 0.05 }),
                "start_at": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
            },
            "optional": {
                "attn_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_ipadapter"
    CATEGORY = "ipadapter"

    def apply_ipadapter(self, model, ipadapter, image, weight, start_at, end_at, attn_mask=None):
        ipa_args = {
            "image": image,
            "weight": weight,
            "start_at": start_at,
            "end_at": end_at,
            "attn_mask": attn_mask,
            "insightface": ipadapter['insightface']['model'] if 'insightface' in ipadapter else None,
        }

        if 'ipadapter' not in ipadapter:
            raise Exception("IPAdapter model not present in the pipeline. Please load the models with the IPAdapterUnifiedLoader node.")
        if 'clipvision' not in ipadapter:
            raise Exception("CLIPVision model not present in the pipeline. Please load the models with the IPAdapterUnifiedLoader node.")

        return (ipadapter_execute(model.clone(), ipadapter['ipadapter']['model'], ipadapter['clipvision']['model'], **ipa_args), )

class IPAdapterAdvanced:
    def __init__(self):
        self.unfold_batch = False

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "ipadapter": ("IPADAPTER", ),
                "image": ("IMAGE",),
                "weight": ("FLOAT", { "default": 1.0, "min": -1, "max": 3, "step": 0.05 }),
                "weight_type": (WEIGHT_TYPES, ),
                "combine_embeds": (["concat", "add", "subtract", "average", "norm average"],),
                "start_at": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'], ),
            },
            "optional": {
                "image_negative": ("IMAGE",),
                "attn_mask": ("MASK",),
                "clip_vision": ("CLIP_VISION",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_ipadapter"
    CATEGORY = "ipadapter"

    def apply_ipadapter(self, model, ipadapter, image, weight, weight_type, start_at, end_at, combine_embeds="concat", weight_faceidv2=None, image_negative=None, clip_vision=None, attn_mask=None, insightface=None, embeds_scaling='V only'):
        ipa_args = {
            "image": image,
            "image_negative": image_negative,
            "weight": weight,
            "weight_faceidv2": weight_faceidv2,
            "weight_type": weight_type,
            "combine_embeds": combine_embeds,
            "start_at": start_at,
            "end_at": end_at,
            "attn_mask": attn_mask,
            "unfold_batch": self.unfold_batch,
            "embeds_scaling": embeds_scaling,
            "insightface": insightface if insightface is not None else ipadapter['insightface']['model'] if 'insightface' in ipadapter else None
        }

        if 'ipadapter' in ipadapter:
            ipadapter_model = ipadapter['ipadapter']['model']
            clip_vision = clip_vision if clip_vision is not None else ipadapter['clipvision']['model']
        else:
            ipadapter_model = ipadapter
            clip_vision = clip_vision

        if clip_vision is None:
            raise Exception("Missing CLIPVision model.")

        del ipadapter

        return (ipadapter_execute(model.clone(), ipadapter_model, clip_vision, **ipa_args), )

class IPAdapterBatch(IPAdapterAdvanced):
    def __init__(self):
        self.unfold_batch = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "ipadapter": ("IPADAPTER", ),
                "image": ("IMAGE",),
                "weight": ("FLOAT", { "default": 1.0, "min": -1, "max": 3, "step": 0.05 }),
                "weight_type": (WEIGHT_TYPES, ),
                "start_at": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'], ),
            },
            "optional": {
                "image_negative": ("IMAGE",),
                "attn_mask": ("MASK",),
                "clip_vision": ("CLIP_VISION",),
            }
        }

class IPAdapterFaceID(IPAdapterAdvanced):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "ipadapter": ("IPADAPTER", ),
                "image": ("IMAGE",),
                "weight": ("FLOAT", { "default": 1.0, "min": -1, "max": 3, "step": 0.05 }),
                "weight_faceidv2": ("FLOAT", { "default": 1.0, "min": -1, "max": 5.0, "step": 0.05 }),
                "weight_type": (WEIGHT_TYPES, ),
                "combine_embeds": (["concat", "add", "subtract", "average", "norm average"],),
                "start_at": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'], ),
            },
            "optional": {
                "image_negative": ("IMAGE",),
                "attn_mask": ("MASK",),
                "clip_vision": ("CLIP_VISION",),
                "insightface": ("INSIGHTFACE",),
            }
        }

class IPAAdapterFaceIDBatch(IPAdapterFaceID):
    def __init__(self):
        self.unfold_batch = True

class IPAdapterTiled:
    def __init__(self):
        self.unfold_batch = False

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "ipadapter": ("IPADAPTER", ),
                "image": ("IMAGE",),
                "weight": ("FLOAT", { "default": 1.0, "min": -1, "max": 3, "step": 0.05 }),
                "weight_type": (WEIGHT_TYPES, ),
                "combine_embeds": (["concat", "add", "subtract", "average", "norm average"],),
                "start_at": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "sharpening": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05 }),
                "embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'], ),
            },
            "optional": {
                "image_negative": ("IMAGE",),
                "attn_mask": ("MASK",),
                "clip_vision": ("CLIP_VISION",),
            }
        }

    RETURN_TYPES = ("MODEL", "IMAGE", "MASK", )
    RETURN_NAMES = ("MODEL", "tiles", "masks", )
    FUNCTION = "apply_tiled"
    CATEGORY = "ipadapter"

    def apply_tiled(self, model, ipadapter, image, weight, weight_type, start_at, end_at, sharpening, combine_embeds="concat", image_negative=None, attn_mask=None, clip_vision=None, embeds_scaling='V only'):
        # 1. Select the models
        if 'ipadapter' in ipadapter:
            ipadapter_model = ipadapter['ipadapter']['model']
            clip_vision = clip_vision if clip_vision is not None else ipadapter['clipvision']['model']
        else:
            ipadapter_model = ipadapter
            clip_vision = clip_vision

        if clip_vision is None:
            raise Exception("Missing CLIPVision model.")

        del ipadapter

        # 2. Extract the tiles
        tile_size = 256     # I'm using 256 instead of 224 as it is more likely divisible by the latent size, it will be downscaled to 224 by the clip vision encoder
        _, oh, ow, _ = image.shape
        if attn_mask is None:
            attn_mask = torch.ones([1, oh, ow], dtype=image.dtype, device=image.device)

        image = image.permute([0,3,1,2])
        attn_mask = attn_mask.unsqueeze(1)
        # the mask should have the same proportions as the reference image and the latent
        attn_mask = T.Resize((oh, ow), interpolation=T.InterpolationMode.BICUBIC, antialias=True)(attn_mask)

        # if the image is almost a square, we crop it to a square
        if oh / ow > 0.75 and oh / ow < 1.33:
            # crop the image to a square
            image = T.CenterCrop(min(oh, ow))(image)
            resize = (tile_size*2, tile_size*2)

            attn_mask = T.CenterCrop(min(oh, ow))(attn_mask)
        # otherwise resize the smallest side and the other proportionally
        else:
            resize = (int(tile_size * ow / oh), tile_size) if oh < ow else (tile_size, int(tile_size * oh / ow))

         # using PIL for better results
        imgs = []
        for img in image:
            img = T.ToPILImage()(img)
            img = img.resize(resize, resample=Image.Resampling['LANCZOS'])
            imgs.append(T.ToTensor()(img))
        image = torch.stack(imgs)
        del imgs, img

        # we don't need a high quality resize for the mask
        attn_mask = T.Resize(resize[::-1], interpolation=T.InterpolationMode.BICUBIC, antialias=True)(attn_mask)

        # we allow a maximum of 4 tiles
        if oh / ow > 4 or oh / ow < 0.25:
            crop = (tile_size, tile_size*4) if oh < ow else (tile_size*4, tile_size)
            image = T.CenterCrop(crop)(image)
            attn_mask = T.CenterCrop(crop)(attn_mask)

        attn_mask = attn_mask.squeeze(1)

        if sharpening > 0:
            image = contrast_adaptive_sharpening(image, sharpening)

        image = image.permute([0,2,3,1])

        _, oh, ow, _ = image.shape

        # find the number of tiles for each side
        tiles_x = math.ceil(ow / tile_size)
        tiles_y = math.ceil(oh / tile_size)
        overlap_x = max(0, (tiles_x * tile_size - ow) / (tiles_x - 1 if tiles_x > 1 else 1))
        overlap_y = max(0, (tiles_y * tile_size - oh) / (tiles_y - 1 if tiles_y > 1 else 1))

        base_mask = torch.zeros([attn_mask.shape[0], oh, ow], dtype=image.dtype, device=image.device)

        # extract all the tiles from the image and create the masks
        tiles = []
        masks = []
        for y in range(tiles_y):
            for x in range(tiles_x):
                start_x = int(x * (tile_size - overlap_x))
                start_y = int(y * (tile_size - overlap_y))
                tiles.append(image[:, start_y:start_y+tile_size, start_x:start_x+tile_size, :])
                mask = base_mask.clone()
                mask[:, start_y:start_y+tile_size, start_x:start_x+tile_size] = attn_mask[:, start_y:start_y+tile_size, start_x:start_x+tile_size]
                masks.append(mask)
        del mask

        # 3. Apply the ipadapter to each group of tiles
        model = model.clone()
        for i in range(len(tiles)):
            ipa_args = {
                "image": tiles[i],
                "image_negative": image_negative,
                "weight": weight,
                "weight_type": weight_type,
                "combine_embeds": combine_embeds,
                "start_at": start_at,
                "end_at": end_at,
                "attn_mask": masks[i],
                "unfold_batch": self.unfold_batch,
                "embeds_scaling": embeds_scaling,
            }
            # apply the ipadapter to the model without cloning it
            model = ipadapter_execute(model, ipadapter_model, clip_vision, **ipa_args)

        return (model, torch.cat(tiles), torch.cat(masks), )

class IPAdapterTiledBatch(IPAdapterTiled):
    def __init__(self):
        self.unfold_batch = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "ipadapter": ("IPADAPTER", ),
                "image": ("IMAGE",),
                "weight": ("FLOAT", { "default": 1.0, "min": -1, "max": 3, "step": 0.05 }),
                "weight_type": (WEIGHT_TYPES, ),
                "start_at": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "sharpening": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05 }),
                "embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'], ),
            },
            "optional": {
                "image_negative": ("IMAGE",),
                "attn_mask": ("MASK",),
                "clip_vision": ("CLIP_VISION",),
            }
        }

class IPAdapterEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "ipadapter": ("IPADAPTER", ),
                "pos_embed": ("EMBEDS",),
                "weight": ("FLOAT", { "default": 1.0, "min": -1, "max": 3, "step": 0.05 }),
                "weight_type": (WEIGHT_TYPES, ),
                "start_at": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'], ),
            },
            "optional": {
                "neg_embed": ("EMBEDS",),
                "attn_mask": ("MASK",),
                "clip_vision": ("CLIP_VISION",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_ipadapter"
    CATEGORY = "ipadapter"

    def apply_ipadapter(self, model, ipadapter, pos_embed, weight, weight_type, start_at, end_at, neg_embed=None, attn_mask=None, clip_vision=None, embeds_scaling='V only'):
        ipa_args = {
            "pos_embed": pos_embed,
            "neg_embed": neg_embed,
            "weight": weight,
            "weight_type": weight_type,
            "start_at": start_at,
            "end_at": end_at,
            "attn_mask": attn_mask,
            "embeds_scaling": embeds_scaling,
        }

        if 'ipadapter' in ipadapter:
            ipadapter_model = ipadapter['ipadapter']['model']
            clip_vision = clip_vision if clip_vision is not None else ipadapter['clipvision']['model']
        else:
            ipadapter_model = ipadapter
            clip_vision = clip_vision

        if clip_vision is None and neg_embed is None:
            raise Exception("Missing CLIPVision model.")

        del ipadapter

        return (ipadapter_execute(model.clone(), ipadapter_model, clip_vision, **ipa_args), )

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Helpers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
class IPAdapterEncoder:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "ipadapter": ("IPADAPTER",),
            "image": ("IMAGE",),
            "weight": ("FLOAT", { "default": 1.0, "min": -1.0, "max": 3.0, "step": 0.01 }),
            },
            "optional": {
                "mask": ("MASK",),
                "clip_vision": ("CLIP_VISION",),
            }
        }

    RETURN_TYPES = ("EMBEDS", "EMBEDS",)
    RETURN_NAMES = ("pos_embed", "neg_embed",)
    FUNCTION = "encode"
    CATEGORY = "ipadapter"

    def encode(self, ipadapter, image, weight, mask=None, clip_vision=None):
        if 'ipadapter' in ipadapter:
            ipadapter_model = ipadapter['ipadapter']['model']
            clip_vision = clip_vision if clip_vision is not None else ipadapter['clipvision']['model']
        else:
            ipadapter_model = ipadapter
            clip_vision = clip_vision

        if clip_vision is None:
            raise Exception("Missing CLIPVision model.")

        is_plus = "proj.3.weight" in ipadapter_model["image_proj"] or "latents" in ipadapter_model["image_proj"] or "perceiver_resampler.proj_in.weight" in ipadapter_model["image_proj"]

        # resize and crop the mask to 224x224
        if mask is not None and mask.shape[1:3] != torch.Size([224, 224]):
            mask = mask.unsqueeze(1)
            transforms = T.Compose([
                T.CenterCrop(min(mask.shape[2], mask.shape[3])),
                T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            ])
            mask = transforms(mask).squeeze(1)
            #mask = T.Resize((image.shape[1], image.shape[2]), interpolation=T.InterpolationMode.BICUBIC, antialias=True)(mask.unsqueeze(1)).squeeze(1)

        img_cond_embeds = encode_image_masked(clip_vision, image, mask)

        if is_plus:
            img_cond_embeds = img_cond_embeds.penultimate_hidden_states
            img_uncond_embeds = encode_image_masked(clip_vision, torch.zeros([1, 224, 224, 3])).penultimate_hidden_states
        else:
            img_cond_embeds = img_cond_embeds.image_embeds
            img_uncond_embeds = torch.zeros_like(img_cond_embeds)

        if weight != 1:
            img_cond_embeds = img_cond_embeds * weight

        return (img_cond_embeds, img_uncond_embeds, )

class IPAdapterCombineEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "embed1": ("EMBEDS",),
            "method": (["concat", "add", "subtract", "average", "norm average", "max", "min"], ),
        },
        "optional": {
            "embed2": ("EMBEDS",),
            "embed3": ("EMBEDS",),
            "embed4": ("EMBEDS",),
            "embed5": ("EMBEDS",),
        }}

    RETURN_TYPES = ("EMBEDS",)
    FUNCTION = "batch"
    CATEGORY = "ipadapter"

    def batch(self, embed1, method, embed2=None, embed3=None, embed4=None, embed5=None):
        if method=='concat' and embed2 is None and embed3 is None and embed4 is None and embed5 is None:
            return (embed1, )

        embeds = [embed1, embed2, embed3, embed4, embed5]
        embeds = [embed for embed in embeds if embed is not None]
        embeds = torch.cat(embeds, dim=0)

        if method == "add":
            embeds = torch.sum(embeds, dim=0).unsqueeze(0)
        elif method == "subtract":
            embeds = embeds[0] - torch.mean(embeds[1:], dim=0)
            embeds = embeds.unsqueeze(0)
        elif method == "average":
            embeds = torch.mean(embeds, dim=0).unsqueeze(0)
        elif method == "norm average":
            embeds = torch.mean(embeds / torch.norm(embeds, dim=0, keepdim=True), dim=0).unsqueeze(0)
        elif method == "max":
            embeds = torch.max(embeds, dim=0).values.unsqueeze(0)
        elif method == "min":
            embeds = torch.min(embeds, dim=0).values.unsqueeze(0)

        return (embeds, )

class IPAdapterNoise:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "type": (["fade", "dissolve", "gaussian", "shuffle"], ),
                "strength": ("FLOAT", { "default": 1.0, "min": 0, "max": 1, "step": 0.05 }),
                "blur": ("INT", { "default": 0, "min": 0, "max": 32, "step": 1 }),
            },
            "optional": {
                "image_optional": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "make_noise"
    CATEGORY = "ipadapter"

    def make_noise(self, type, strength, blur, image_optional=None):
        if image_optional is None:
            image = torch.zeros([1, 224, 224, 3])
        else:
            transforms = T.Compose([
                T.CenterCrop(min(image_optional.shape[1], image_optional.shape[2])),
                T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            ])
            image = transforms(image_optional.permute([0,3,1,2])).permute([0,2,3,1])

        seed = int(torch.sum(image).item()) % 1000000007 # hash the image to get a seed, grants predictability
        torch.manual_seed(seed)

        if type == "fade":
            noise = torch.rand_like(image)
            noise = image * (1 - strength) + noise * strength
        elif type == "dissolve":
            mask = (torch.rand_like(image) < strength).float()
            noise = torch.rand_like(image)
            noise = image * (1-mask) + noise * mask
        elif type == "gaussian":
            noise = torch.randn_like(image) * strength
            noise = image + noise
        elif type == "shuffle":
            transforms = T.Compose([
                T.ElasticTransform(alpha=75.0, sigma=(1-strength)*3.5),
                T.RandomVerticalFlip(p=1.0),
                T.RandomHorizontalFlip(p=1.0),
            ])
            image = transforms(image.permute([0,3,1,2])).permute([0,2,3,1])
            noise = torch.randn_like(image) * (strength*0.75)
            noise = image * (1-noise) + noise

        del image
        noise = torch.clamp(noise, 0, 1)

        if blur > 0:
            if blur % 2 == 0:
                blur += 1
            noise = T.functional.gaussian_blur(noise.permute([0,3,1,2]), blur).permute([0,2,3,1])

        return (noise, )

class PrepImageForClipVision:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "interpolation": (["LANCZOS", "BICUBIC", "HAMMING", "BILINEAR", "BOX", "NEAREST"],),
            "crop_position": (["top", "bottom", "left", "right", "center", "pad"],),
            "sharpening": ("FLOAT", {"default": 0.0, "min": 0, "max": 1, "step": 0.05}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "prep_image"

    CATEGORY = "ipadapter"

    def prep_image(self, image, interpolation="LANCZOS", crop_position="center", sharpening=0.0, mask=None):
        size = (224, 224)
        _, oh, ow, _ = image.shape
        output = image.permute([0,3,1,2])

        if crop_position == "pad":
            if oh != ow:
                if oh > ow:
                    pad = (oh - ow) // 2
                    pad = (pad, 0, pad, 0)
                elif ow > oh:
                    pad = (ow - oh) // 2
                    pad = (0, pad, 0, pad)
                output = T.functional.pad(output, pad, fill=0)
                if mask is not None:
                    mask = T.functional.pad(mask.unsqueeze(1), pad, fill=0).squeeze(1)
        else:
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

            output = output[:, :, y:y2, x:x2]
            if mask is not None:
                mask = mask[:, y:y2, x:x2]

        imgs = []
        for img in output:
            img = T.ToPILImage()(img) # using PIL for better results
            img = img.resize(size, resample=Image.Resampling[interpolation])
            imgs.append(T.ToTensor()(img))
        output = torch.stack(imgs)
        del imgs, img

        if mask is not None:
            mask = T.Resize(size, interpolation=T.InterpolationMode.BICUBIC, antialias=True)(mask.unsqueeze(1)).squeeze(1)

        if sharpening > 0:
            output = contrast_adaptive_sharpening(output, sharpening)

        output = output.permute([0,2,3,1])

        if mask is None:
            mask = torch.ones((output.shape[0], *size), dtype=output.dtype, device=output.device)

        return (output, mask)
        

class IPAdapterSaveEmbeds:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "embeds": ("EMBEDS",),
            "filename_prefix": ("STRING", {"default": "IP_embeds"})
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "ipadapter"

    def save(self, embeds, filename_prefix):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        file = f"{filename}_{counter:05}.ipadpt"
        file = os.path.join(full_output_folder, file)

        torch.save(embeds, file)
        return (None, )

class IPAdapterLoadEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [os.path.relpath(os.path.join(root, file), input_dir) for root, dirs, files in os.walk(input_dir) for file in files if file.endswith('.ipadpt')]
        return {"required": {"embeds": [sorted(files), ]}, }

    RETURN_TYPES = ("EMBEDS", )
    FUNCTION = "load"
    CATEGORY = "ipadapter"

    def load(self, embeds):
        path = folder_paths.get_annotated_filepath(embeds)
        return (torch.load(path).cpu(), )

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Register
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
NODE_CLASS_MAPPINGS = {
    # Main Apply Nodes
    "IPAdapter": IPAdapterSimple,
    "IPAdapterAdvanced": IPAdapterAdvanced,
    "IPAdapterBatch": IPAdapterBatch,
    "IPAdapterFaceID": IPAdapterFaceID,
    "IPAAdapterFaceIDBatch": IPAAdapterFaceIDBatch,
    "IPAdapterTiled": IPAdapterTiled,
    "IPAdapterTiledBatch": IPAdapterTiledBatch,
    "IPAdapterEmbeds": IPAdapterEmbeds,

    # Loaders
    "IPAdapterUnifiedLoader": IPAdapterUnifiedLoader,
    "IPAdapterUnifiedLoaderFaceID": IPAdapterUnifiedLoaderFaceID,
    "IPAdapterModelLoader": IPAdapterModelLoader,
    "IPAdapterInsightFaceLoader": IPAdapterInsightFaceLoader,

    # Helpers
    "IPAdapterEncoder": IPAdapterEncoder,
    "IPAdapterCombineEmbeds": IPAdapterCombineEmbeds,
    "IPAdapterNoise": IPAdapterNoise,
    "PrepImageForClipVision": PrepImageForClipVision,
    "IPAdapterSaveEmbeds": IPAdapterSaveEmbeds,
    "IPAdapterLoadEmbeds": IPAdapterLoadEmbeds,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Main Apply Nodes
    "IPAdapter": "IPAdapter",
    "IPAdapterAdvanced": "IPAdapter Advanced",
    "IPAdapterBatch": "IPAdapter Batch (Adv.)",
    "IPAdapterFaceID": "IPAdapter FaceID",
    "IPAAdapterFaceIDBatch": "IPAdapter FaceID Batch",
    "IPAdapterTiled": "IPAdapter Tiled",
    "IPAdapterTiledBatch": "IPAdapter Tiled Batch",
    "IPAdapterEmbeds": "IPAdapter Embeds",

    # Loaders
    "IPAdapterUnifiedLoader": "IPAdapter Unified Loader",
    "IPAdapterUnifiedLoaderFaceID": "IPAdapter Unified Loader FaceID",
    "IPAdapterModelLoader": "IPAdapter Model Loader",
    "IPAdapterInsightFaceLoader": "IPAdapter InsightFace Loader",

    # Helpers
    "IPAdapterEncoder": "IPAdapter Encoder",
    "IPAdapterCombineEmbeds": "IPAdapter Combine Embeds",
    "IPAdapterNoise": "IPAdapter Noise",
    "PrepImageForClipVision": "Prep Image For ClipVision",
    "IPAdapterSaveEmbeds": "IPAdapter Save Embeds",
    "IPAdapterLoadEmbeds": "IPAdapter Load Embeds",
}
