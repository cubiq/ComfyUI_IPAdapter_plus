# ComfyUI IPAdapter plus
[ComfyUI](https://github.com/comfyanonymous/ComfyUI) reference implementation for [IPAdapter](https://github.com/tencent-ailab/IP-Adapter/) models.

IPAdapter implementation that follows the ComfyUI way of doing things. The code is memory efficient, fast, and shouldn't break with Comfy updates.

# Sponsorship

If you like my work and wish to see updates and new features please consider sponsoring my projects.

- [ComfyUI IPAdapter Plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus)
- [ComfyUI InstantID (Native)](https://github.com/cubiq/ComfyUI_InstantID)
- [ComgyUI Essentials](https://github.com/cubiq/ComfyUI_essentials)
- [ComfyUI FaceAnalysis](https://github.com/cubiq/Comfy_Dungeon)
- [Comfy Dungeon](https://github.com/cubiq/Comfy_Dungeon)

Not to mention the documentation and videos tutorials. Check my **ComfyUI Advanced Understanding** videos on YouTube for example.

- [ComfyUI Advanced Understanding part 1](https://www.youtube.com/watch?v=_C7kR2TFIX0)
- [ComfyUI Advanced Understanding part 2](https://www.youtube.com/watch?v=ijqXnW_9gzc)

I'm talking especially to companies here, **if you are making a profit out of Open Source code the only way to keep getting updates, bug fixes and documentation is by giving something back to those projects.**

Please contact me if you are interested in a sponsorship at _matt3o@gmail_ or consider a contribution via [PayPal](https://paypal.me/matt3o) (Matteo "matt3o" Spinelli, Firenze, IT). For "donations" of $50+, add a comment if you'd like to be mentioned in this readme file.

# Current sponsors

I really need to thank [Nathan Shipley](https://www.nathanshipley.com/) for his generous donation. Go check his website, he's terribly talented.

## :warning: IPAdapter V2: complete Code rewrite warning :warning:

The new code is not compatible with the previous version of IPAdapter. There is no `IPAdapter Apply` node anymore but the `IPAdapter Advanced` node is a drop in replacement. Delete the old node, add the new one and connect the pipelines as they were before. Everything should work.

Check the `example` directory for most of the old and new features.

## Important updates

**2024/03/23**: Complete code rewrite!. **This is a breaking update!** Your previous workflows won't work and you'll need to recreate them. You've been warned! After the update, refresh your browser, delete the old IPAdapter nodes and create the new ones.

*(I removed all previous updates because they were about the previous version of the extension)*

## What is it?

The IPAdapter are very powerful models for image-to-image conditioning. The subject or even just the style of the reference image can be applied to a generation. Think of it as a 1-image lora.

## Example workflow

The [example directory](./examples/) has many workflows that cover all IPAdapter functionalities.

![IPAdapter Example workflow](./examples/demo_workflow.jpg)

## Video Tutorials

<a href="https://youtu.be/_JzDcgKgghY" target="_blank">
 <img src="https://img.youtube.com/vi/_JzDcgKgghY/hqdefault.jpg" alt="Watch the video" />
</a>

**:star: [New IPAdapter features](https://youtu.be/_JzDcgKgghY)**

The following videos are about the previous version of IPAdapter, but they still contain valuable information.

**:nerd_face: [Basic usage video](https://youtu.be/7m9ZZFU3HWo)**

**:rocket: [Advanced features video](https://www.youtube.com/watch?v=mJQ62ly7jrg)**

**:japanese_goblin: [Attention Masking video](https://www.youtube.com/watch?v=vqG1VXKteQg)**

**:movie_camera: [Animation Features video](https://www.youtube.com/watch?v=ddYbhv3WgWw)**

## Installation

Download or git clone this repository inside `ComfyUI/custom_nodes/` directory or use the Manager. IPAdapter always requires the latest version of ComfyUI. If something doesn't work be sure to upgrade. Beware that the automatic update of the manager sometimes doesn't work and you may need to upgrade manually.

There's now a *Unified Model Loader*, for it to work you need to name the files exactly as described below. The legacy loaders work with any file name but you have to select them manually. The models can be placed into sub-directories.

Remember you can also use any custom location setting an `ipadapter` entry in the `extra_model_paths.yaml` file.

- `/ComfyUI/models/clip_vision`
    - [CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors](https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors), download and rename
    - [CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors](https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/model.safetensors), download and rename
- `/ComfyUI/models/ipadapter`, create it if not present
    - [ip-adapter_sd15.safetensors](https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.safetensors), Basic model, average strength
    - [ip-adapter_sd15_light_v11.bin](https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15_light_v11.bin), Light impact model
    - [ip-adapter-plus_sd15.safetensors](https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.safetensors), Plus model, very strong
    - [ip-adapter-plus-face_sd15.safetensors](https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus-face_sd15.safetensors), Face model, portraits
    - [ip-adapter-full-face_sd15.safetensors](https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-full-face_sd15.safetensors), Stronger face model, not necessarily better
    - [ip-adapter_sd15_vit-G.safetensors](https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15_vit-G.safetensors), Base model, **requires bigG clip vision encoder**
    - [ip-adapter_sdxl_vit-h.safetensors](https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl_vit-h.safetensors), SDXL model
    - [ip-adapter-plus_sdxl_vit-h.safetensors](https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors), SDXL plus model
    - [ip-adapter-plus-face_sdxl_vit-h.safetensors](https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors), SDXL face model
    - [ip-adapter_sd15_light.safetensors](https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15_light.safetensors)
    - [ip-adapter_sdxl.safetensors](https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl.safetensors), vit-G SDXL model, **requires bigG clip vision encoder**
    - **Deprecated** [ip-adapter_sd15_light.bin](https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15_light_v11.bin), v1.0 Light impact model

**FaceID** models require `insightface`, you need to install it in your ComfyUI environment. Check [this issue](https://github.com/cubiq/ComfyUI_IPAdapter_plus/issues/162) for help. Remember that most FaceID models also need a LoRA.

For the Unified Loader to work the files need to be named exactly as shown in the table below.

- `/ComfyUI/models/ipadapter`
    - [ip-adapter-faceid_sd15.bin](https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sd15.bin), base FaceID model
    - [ip-adapter-faceid-plusv2_sd15.bin](https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15.bin), FaceID plus v2
    - [ip-adapter-faceid-portrait-v11_sd15.bin](https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-portrait-v11_sd15.bin), text prompt style transfer
    - [ip-adapter-faceid_sdxl.bin](https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sdxl.bin), SDXL base FaceID
    - [ip-adapter-faceid-plusv2_sdxl.bin](https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sdxl.bin), SDXL plus v2
    - [ip-adapter-faceid-portrait_sdxl.bin](https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-portrait_sdxl.bin), SDXL text prompt style transfer
    - **Deprecated** [ip-adapter-faceid-plus_sd15.bin](https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plus_sd15.bin), FaceID plus v1 
    - **Deprecated** [ip-adapter-faceid-portrait_sd15.bin](https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-portrait-v11_sd15.bin), v1 of the portrait model

Most FaceID models require a LoRA. If you use the `IPAdapter Unified Loader FaceID` it will be loaded automatically if you follow the naming convention. Otherwise you have to load them manually, be careful each FaceID model has to be paired with its own specific LoRA.

- `/ComfyUI/models/loras`
    - [ip-adapter-faceid_sd15_lora.safetensors](https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sd15_lora.safetensors)
    - [ip-adapter-faceid-plusv2_sd15_lora.safetensors](https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15_lora.safetensors)
    - [ip-adapter-faceid_sdxl_lora.safetensors](https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sdxl_lora.safetensors), SDXL FaceID LoRA
    - [ip-adapter-faceid-plusv2_sdxl_lora.safetensors](https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sdxl_lora.safetensors), SDXL plus v2 LoRA
    - **Deprecated** [ip-adapter-faceid-plus_sd15_lora.safetensors](https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plus_sd15_lora.safetensors), LoRA for the deprecated FaceID plus v1 model

All models can be found on [huggingface](https://huggingface.co/h94).

## Generic suggestions

There are many workflows included in the [examples](./examples/) directory. Please check them before asking for support.

Usually it's a good idea to lower the `weight` to at least `0.8` and increase the number steps. To increase adherece to the prompt you may try to change the **weight type** in the `IPAdapter Advanced` node.

## Nodes reference

Below I'm trying to document all the nodes. It's still very incomplete, be sure to check back later.

### :knot: IPAdapter Unified Loader

Loads the full stack of models needed for IPAdapter to function. The returned object will contain information regarding the **ipadapter** and **clip vision models**.

Multiple unified loaders should always be daisy chained through the `ipadapter` in/out. **Failing to do so will cause all models to be loaded twice.** For **the first** unified loader the `ipadapter` input **should never be connected**.

#### Inputs
- **model**, main ComfyUI model pipeline

#### Optional Inputs
- **ipadapter**, it's important to note that this is optional and used exclusively to daisy chain unified loaders. **The `ipadapter` input is never connected in the first `IPAdapter Unified Loader` of the chain.**

#### Outputs
- **model**, the model pipeline is used exclusively for configuration, the model comes out of this node untouched and it can be considered a reroute. Note that this is different  from the Unified Loader FaceID that actually alters the model with a LoRA.
- **ipadapter**, connect this to any ipadater node. Each node will automatically detect if the `ipadapter` object contains the full stack of models or just one (like in the case [IPAdapter Model Loader](#ipadapter-model-loader)).

### :knot: IPAdapter Model Loader

Loads the IPAdapter model only. The returned object will be the IPAdapter model contrary to the [Unified loader](#ipadapter-unified-loader) that contains the full stack of models.

#### Configuration parameters
- **ipadapter_file**, the main IPAdapter model. It must be located into `ComfyUI/models/ipadapter` or in any path specified in the `extra_model_paths.yaml` configuration file.

#### Outputs
- **IPADAPTER**, contains the loaded model only. Note that `IPADAPTER` will have a different structure when loaded by the [Unified Loader](#ipadapter-unified-loader).

### :knot: IPAdapter Advanced

This node contains all the options to fine tune the IPAdapter models. It is a drop in replacement for the old `IPAdapter Apply` that is no longer available. If you have an old workflow, delete the existing `IPadapter Apply` node, add `IPAdapter Advanced` and connect all the pipes as before.

#### Inputs
- **model**, main model pipeline.
- **ipadapter**, the IPAdapter model. It can be connected to the [IPAdapter Model Loader](#ipadapter-model-loader) or any of the Unified Loaders. If a Unified loader is used anywhere in the workflow and you don't need a different model, it's always adviced to reuse the previous `ipadapter` pipeline.
- **image**, the reference image used to generate the positive conditioning. It should be a square image, other aspect ratios are automatically cropped in the center.

#### Optional inputs
- **image_negative**, image used to generate the negative conditioning. This is optional and normally handled by the code. It is possible to send noise or actually any image to instruct the model about what we don't want to see in the composition.
- **attn_mask**, a mask that will be applied during the image generation. **The mask should have the same size or at least the same aspect ratio of the latent**. The mask will define the area of influence of the IPAdapter models on the final image. Black zones won't be affected, white zones will get maximum influence. It can be a grayscale mask.
- **clip_vision**, this is optional if using any of the Unified loaders. If using the [IPAdapter Model Loader](#knot-ipadapter-model-loader) you also have to provide the clip vision model with a `Load CLIP Vision` node.

#### Configuration parameters
- **weight**, weight of the IPAdapter model. For `linear` `weight_type` (the default), a good starting point is 0.8. If you use other weight types you can experiment with higher values.
- **weight_type**, this is how the IPAdapter is applied to the UNet block. For example `ease-in` means that the input blocks have higher weight than the output ones. `week input` means that the whole input block has lower weight. `style transfer (SDXL)` only works with SDXL and it's a very powerful tool to tranfer only the style of an image but not its content. This parameter hugely impacts how the composition reacts to the text prompting.
- **combine_embeds**, when sending more than one reference image the embeddings can be sent one after the other (`concat`) or combined in various ways. For low spec GPUs it is adviced to `average` the embeds if you send multiple images. `subtract` subtracts the embeddings of the second image to the first; in case of 3 or more images they are averaged and subtracted to the first.
- **start_at/end_at**, this is the timestepping. Defines at what percentage point of the generation to start applying the IPAdapter model. The initial steps are the most important so if you start later (eg: `start_at=0.3`) the generated image will have a very light conditioning.
- **embeds_scaling**, the way the IPAdapter models are applied to the K,V. This parameter has a small impact on how the model reacts to text prompting. `K+mean(V) w/ C penalty` grants good quality at high weights (>1.0) without burning the image.

## Troubleshooting

Please check the [troubleshooting](https://github.com/cubiq/ComfyUI_IPAdapter_plus/issues/108) before posting a new issue. Also remember to check the previous closed issues.

## Credits

- [IPAdapter](https://github.com/tencent-ailab/IP-Adapter/)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [laksjdjf](https://github.com/laksjdjf/IPAdapter-ComfyUI/)
