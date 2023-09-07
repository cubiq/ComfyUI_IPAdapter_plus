# ComfyUI_IPAdapter_Plus
ComfyUI reference implementation for [IPAdapter](https://github.com/tencent-ailab/IP-Adapter/tree/6fb9d3554a5c774f41e187e8fdbc7b9a1db8c2e3) models.

The code is mostly taken from the original IPAdapter repository and [laksjdjf's](https://github.com/laksjdjf/IPAdapter-ComfyUI/tree/main) implementation, all credit goes to them. I just made the extension closer to ComfyUI philosophy.

Example workflow

![IPAdapter Example workflow](./ipadapter_workflow.png)

## Installation

Download or git clone this repository inside `ComfyUI/custom_nodes/` directory.

The pre-trained models are available on [huggingface](https://huggingface.co/h94/IP-Adapter), download and place them in the `ComfyUI/custom_nodes/ComfyUI_IPAdapter_plus/models` directory.

For SD1.5 you need:

- [ip-adapter_sd15.bin](https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.bin)
- [ip-adapter-plus_sd15.bin](https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.bin)
- [ip-adapter-plus-face_sd15.bin](https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus-face_sd15.bin)

For SDXL you need:
- [ip-adapter_sdxl.bin](https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl.bin)

Additionally you need the clip vision models:

- SD 1.5: [pytorch_model.bin](https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/pytorch_model.bin)
- SDXL: [pytorch_model.bin](https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/pytorch_model.bin)

You can rename them to something easier to remember (eg: `ip-adapter_sd15-image-encoder.bin`) and place them under `ComfyUI/models/clip_vision/`.

## How to use

There are two workflows included in this repo.

**IMPORTANT:** To use the *IPAdapter Plus* model you must use the new `CLIP Vision Encode (IPAdapter)` node (the workflow is [Plus_workflow.json](./Plus_workflow.json)). The non-plus version works with both the standard `CLIP Vision Encode` and the new one.