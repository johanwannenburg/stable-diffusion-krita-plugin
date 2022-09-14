# Imports
from PIL import Image
from typing import Optional
from pydantic import BaseModel
from enum import Enum

# Txt2Img Request model
class Txt2ImgRequest(BaseModel):
    orig_width: int
    orig_height: int

    prompt: Optional[str]
    negative_prompt: Optional[str]
    prompt_style: Optional[str]
    sampler_name: Optional[str]
    steps: Optional[int]
    cfg_scale: Optional[float]

    batch_count: Optional[int]
    batch_size: Optional[int]
    base_size: Optional[int]
    max_size: Optional[int]
    seed: Optional[str]
    subseed: Optional[float]
    subseed_strength: Optional[float]
    seed_resize_from_h: Optional[int]
    seed_resize_from_w: Optional[int]
    tiling: Optional[bool]

    restore_faces: Optional[bool]  # Should be able to handle GFPGAN or CodeFormer

# Img2Img Request model
class Img2ImgRequest(BaseModel):
    mode: Optional[int]

    src_path: str
    mask_path: Optional[str]
    mask_mode: Optional[int]

    prompt: Optional[str]
    negative_prompt: Optional[str]
    prompt_style: Optional[str]
    sampler_name: Optional[str]
    steps: Optional[int]
    cfg_scale: Optional[float]
    denoising_strength: Optional[float]
    denoising_strength_change_factor: Optional[float]

    batch_count: Optional[int]
    batch_size: Optional[int]
    base_size: Optional[int]
    max_size: Optional[int]
    seed: Optional[str]
    subseed: Optional[float]
    subseed_strength: Optional[float]
    seed_resize_from_h: Optional[int]
    seed_resize_from_w: Optional[int]
    tiling: Optional[bool]

    restore_faces: Optional[bool]

    upscale_overlap: Optional[int]
    upscaler_name: Optional[str]

    inpainting_fill: Optional[int]
    inpaint_full_res: Optional[bool]
    mask_blur: Optional[int]

# Upscaler Request model
class UpscaleRequest(BaseModel):
    src_path: str
    upscaler_name: Optional[str]
    downscale_first: Optional[bool]

#####################################################
# Models
# 1. text to image request
# 2. image to image request
# 3. upscaler request
# 4. Config/settings request
#####################################################

# Data types
class inpainting_fill_type(Enum):
    fill = 0
    original = 1
    latnet_noise = 2
    latent_nothing = 3

# Text2Img
# txt2img(
#     prompt: str, 
#     negative_prompt: str, 
#     prompt_style: str, 
#     steps: int, 
#     sampler_index: int, 
#     restore_faces: bool, 
#     tiling: bool, 
#     n_iter: int, 
#     batch_size: int, 
#     cfg_scale: float, 
#     seed: int, 
#     subseed: int, 
#     subseed_strength: float, 
#     seed_resize_from_h: int, 
#     seed_resize_from_w: int, 
#     height: int, 
#     width: int, 
#     *args):

# SD Image class
class sdImage(BaseModel):
    src_path: str
    mask_path: Optional[str]

    # image: Image
    # mask: Image

    def __getitem__(self, item):
        return getattr(self, item)

# SD Upscaler class
class sdUpscalerApi(BaseModel):
    src_path: str
    upscaler_name: Optional[str]
    downscale_first: Optional[bool]
    upscale_overlap: Optional[int]
    upscaler_index: Optional[str]    

    def __getitem__(self, item):
        return getattr(self, item)

# SD Base class 
class sdBaseApi(BaseModel):
    prompt: Optional[str]
    negative_prompt: Optional[str]
    prompt_style: Optional[str]
    steps: Optional[int]
    sampler_name: Optional[str]
    cfg_scale: Optional[float]

    restore_faces: Optional[bool]           # Should be able to handle GFPGAN or CodeFormer
    tiling: Optional[bool]
    batch_count: Optional[int]
    batch_size: Optional[int]
    
    seed: Optional[str]
    subseed: Optional[float]
    subseed_strength: Optional[float]
    seed_resize_from_h: Optional[int]
    seed_resize_from_w: Optional[int]

    orig_width: int
    orig_height: int

    base_size: Optional[int]
    max_size: Optional[int]

    def __getitem__(self, item):
        return getattr(self, item)

# SD Base and image class
class sdImageApi(sdBaseApi):
    initImage: Optional[sdImage]

    mask_mode: Optional[int]
    mask_blur: Optional[int]

    inpainting_fill: Optional[int]
    inpaint_full_res: Optional[bool]
    mode: Optional[int]
    denoising_strength: Optional[float]
    denoising_strength_change_factor: Optional[float]
    resize_mode: Optional[int]
    inpainting_mask_invert: Optional[int]

    upscaler: Optional[sdUpscalerApi]

    def __getitem__(self, item):
        return getattr(self, item)


# Img2img
# img2img(
#     prompt: str, 
#     negative_prompt: str, 
#     prompt_style: str, 
#     init_img, 
#     init_img_with_mask, 
#     init_mask, 
#     mask_mode, 
#     steps: int, 
#     sampler_index: int, 
#     mask_blur: int, 
#     inpainting_fill: int, 
#     restore_faces: bool, 
#     tiling: bool, 
#     mode: int, 
#     n_iter: int, 
#     batch_size: int, 
#     cfg_scale: float,
#     denoising_strength: float, 
#     denoising_strength_change_factor: float, 
#     seed: int, 
#     subseed: int, 
#     subseed_strength: float, 
#     seed_resize_from_h: int, 
#     seed_resize_from_w: int, 
#     height: int, 
#     width: int, 
#     resize_mode: int, 
#     upscaler_index: str, 
#     upscale_overlap: int, 
#     inpaint_full_res: bool, 
#     inpainting_mask_invert: int, 
#     *args):