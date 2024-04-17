import os
import sys
import folder_paths

import argparse
import numpy as np
import torch
import rembg
from PIL import Image, ImageOps, ImageSequence
from torchvision.transforms import v2
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from einops import rearrange, repeat
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
import datetime

comfy_path = os.path.dirname(folder_paths.__file__)

instant_mesh_path = f'{comfy_path}/custom_nodes/ComfyUI-InstantMesh'

sys.path.append(instant_mesh_path)

from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import (
    FOV_to_intrinsics,
    get_zero123plus_input_cameras,
    get_circular_camera_poses,
)
from src.utils.mesh_util import save_obj, save_obj_with_mtl
from src.utils.infer_util import remove_background, resize_foreground
from src.utils.infer_util import save_video

def get_render_cameras(batch_size=1, M=120, radius=4.0, elevation=20.0, is_flexicubes=False):
    """
    Get the rendering camera parameters.
    """
    c2ws = get_circular_camera_poses(M=M, radius=radius, elevation=elevation)
    if is_flexicubes:
        cameras = torch.linalg.inv(c2ws)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    else:
        extrinsics = c2ws.flatten(-2)
        intrinsics = FOV_to_intrinsics(30.0).unsqueeze(0).repeat(M, 1, 1).float().flatten(-2)
        cameras = torch.cat([extrinsics, intrinsics], dim=-1)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1)
    return cameras


def render_frames(model, planes, render_cameras, render_size=512, chunk_size=1, is_flexicubes=False):
    """
    Render frames from triplanes.
    """
    frames = []
    for i in tqdm(range(0, render_cameras.shape[1], chunk_size)):
        if is_flexicubes:
            frame = model.forward_geometry(
                planes,
                render_cameras[:, i:i + chunk_size],
                render_size=render_size,
            )['img']
        else:
            frame = model.forward_synthesizer(
                planes,
                render_cameras[:, i:i + chunk_size],
                render_size=render_size,
            )['images_rgb']
        frames.append(frame)

    frames = torch.cat(frames, dim=1)[0]  # we suppose batch size is always 1
    return frames

import pkg_resources

from pathlib import Path

def load_InstantMeshModel(config_from_node):
    original_directory = os.getcwd()

    python_path = sys.executable

    python_dir = os.path.dirname(python_path)

    scripts_dir = os.path.join(python_dir, 'Scripts')

    os.environ['PATH'] += os.pathsep + scripts_dir

    os.chdir(instant_mesh_path)

    try:
        ###############################################################################
        # Stage 0: Configuration.
        ###############################################################################

        config = OmegaConf.load(config_from_node)
        config_name = os.path.basename(config_from_node).replace('.yaml', '')
        model_config = config.model_config
        infer_config = config.infer_config

        IS_FLEXICUBES = True if config_name.startswith('instant-mesh') else False

        device = torch.device('cuda')

        # load reconstruction model
        print('Loading reconstruction model ...')
        model = instantiate_from_config(model_config)
        if os.path.exists(infer_config.model_path):
            model_ckpt_path = infer_config.model_path
        else:
            model_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh",
                                              filename=f"{config_name.replace('-', '_')}.ckpt", repo_type="model")
        state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
        state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.')}
        model.load_state_dict(state_dict, strict=True)

        model = model.to(device)
        if IS_FLEXICUBES:
            model.init_flexicubes_geometry(device, fovy=30.0)
        model = model.eval()
    finally:
        os.chdir(original_directory)

    return model,config_from_node

def run_InstantMesh(model, config_from_node, input_path_from_node, diffusion_steps=75, view=6, export_texmap=True,
                    store_video=False, rem_bg=True, output_path='outputs/', seed=42, scale=1.0, distance=4.5):

    no_rembg = not rem_bg

    original_directory = os.getcwd()

    os.chdir(instant_mesh_path)

    try:
        #seed_everything(seed)

        ###############################################################################
        # Stage 0: Configuration.
        ###############################################################################

        config = OmegaConf.load(config_from_node)
        config_name = os.path.basename(config_from_node).replace('.yaml', '')
        infer_config = config.infer_config

        IS_FLEXICUBES = True if config_name.startswith('instant-mesh') else False

        device = torch.device('cuda')

        # load diffusion model
        print('Loading diffusion model ...')
        pipeline = DiffusionPipeline.from_pretrained(
            "sudo-ai/zero123plus-v1.2",
            custom_pipeline="zero123plus",
            torch_dtype=torch.float16,
        )
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipeline.scheduler.config, timestep_spacing='trailing'
        )

        # load custom white-background UNet
        print('Loading custom white-background unet ...')
        if os.path.exists(infer_config.unet_path):
            unet_ckpt_path = infer_config.unet_path
        else:
            unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin",
                                             repo_type="model")
        state_dict = torch.load(unet_ckpt_path, map_location='cpu')
        pipeline.unet.load_state_dict(state_dict, strict=True)

        pipeline = pipeline.to(device)

        # make output directories
        image_path = os.path.join(output_path, config_name, 'images')
        mesh_path = os.path.join(output_path, config_name, 'meshes')
        video_path = os.path.join(output_path, config_name, 'videos')
        os.makedirs(image_path, exist_ok=True)
        os.makedirs(mesh_path, exist_ok=True)
        os.makedirs(video_path, exist_ok=True)

        # process input files
        if os.path.isdir(input_path_from_node):
            input_files = [
                os.path.join(input_path_from_node, file)
                for file in os.listdir(input_path_from_node)
                if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.webp')
            ]
        else:
            input_files = [input_path_from_node]
        print(f'Total number of input images: {len(input_files)}')

        ###############################################################################
        # Stage 1: Multiview generation.
        ###############################################################################

        rembg_session = None if no_rembg else rembg.new_session()

        outputs = []
        for idx, image_file in enumerate(input_files):
            name = os.path.basename(image_file).split('.')[0]
            print(f'[{idx + 1}/{len(input_files)}] Imagining {name} ...')

            # remove background optionally
            input_image = Image.open(image_file)
            if not no_rembg:
                input_image = remove_background(input_image, rembg_session)
                input_image = resize_foreground(input_image, 0.85)

            # sampling
            output_image = pipeline(
                input_image,
                num_inference_steps=diffusion_steps,
            ).images[0]

            output_image.save(os.path.join(image_path, f'{name}.png'))
            print(f"Image saved to {os.path.join(image_path, f'{name}.png')}")

            preview_img = os.path.join(image_path, f'{name}.png')

            images = np.asarray(output_image, dtype=np.float32) / 255.0
            images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()  # (3, 960, 640)
            images = rearrange(images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)  # (6, 3, 320, 320)

            outputs.append({'name': name, 'images': images})

        # delete pipeline to save memory
        del pipeline

        ###############################################################################
        # Stage 2: Reconstruction.
        ###############################################################################

        input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0 * scale).to(device)
        chunk_size = 20 if IS_FLEXICUBES else 1

        video_path_idx = None

        for idx, sample in enumerate(outputs):
            name = sample['name']
            print(f'[{idx + 1}/{len(outputs)}] Creating {name} ...')

            images = sample['images'].unsqueeze(0).to(device)
            images = v2.functional.resize(images, 320, interpolation=3, antialias=True).clamp(0, 1)

            if view == 4:
                indices = torch.tensor([0, 2, 4, 5]).long().to(device)
                images = images[:, indices]
                input_cameras = input_cameras[:, indices]

            with torch.no_grad():
                # get triplane
                planes = model.forward_planes(images, input_cameras)

                # get mesh
                mesh_path_idx = os.path.join(mesh_path, f'{name}.obj')

                mesh_out = model.extract_mesh(
                    planes,
                    use_texture_map=export_texmap,
                    **infer_config,
                )
                if export_texmap:
                    vertices, faces, uvs, mesh_tex_idx, tex_map = mesh_out
                    save_obj_with_mtl(
                        vertices.data.cpu().numpy(),
                        uvs.data.cpu().numpy(),
                        faces.data.cpu().numpy(),
                        mesh_tex_idx.data.cpu().numpy(),
                        tex_map.permute(1, 2, 0).data.cpu().numpy(),
                        mesh_path_idx,
                    )
                else:
                    vertices, faces, vertex_colors = mesh_out
                    save_obj(vertices, faces, vertex_colors, mesh_path_idx)
                print(f"Mesh saved to {mesh_path_idx}")

                # get video
                if store_video:
                    video_path_idx = os.path.join(video_path, f'{name}.mp4')
                    render_size = infer_config.render_resolution
                    render_cameras = get_render_cameras(
                        batch_size=1,
                        M=120,
                        radius=distance,
                        elevation=20.0,
                        is_flexicubes=IS_FLEXICUBES,
                    ).to(device)

                    frames = render_frames(
                        model,
                        planes,
                        render_cameras=render_cameras,
                        render_size=render_size,
                        chunk_size=chunk_size,
                        is_flexicubes=IS_FLEXICUBES,
                    )

                    save_video(
                        frames,
                        video_path_idx,
                        fps=30,
                    )
                    print(f"Video saved to {video_path_idx}")
    finally:
        os.chdir(original_directory)

    return preview_img, mesh_path_idx, video_path_idx

class InstantMeshLoader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "config_name": (["instant-mesh-base", "instant-mesh-large", "instant-nerf-base", "instant-nerf-large"],),
            },
        }

    RETURN_TYPES = ("InstantMeshModel","InstantMeshConfig")
    RETURN_NAMES = ("model","config")

    FUNCTION = "run"
    CATEGORY = "InstantMesh"

    def run(self, config_name):
        config_path = "configs/" + config_name + ".yaml"

        model,config = load_InstantMeshModel(config_path)

        return (model, config)

class InstantMeshRun:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("InstantMeshModel",),
                "config": ("InstantMeshConfig",),
                "images": ("IMAGE",),
                "diffusion_steps": ("INT", {
                    "default": 75,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "number"
                }),
                "view": ("INT", {
                    "default": 6,
                    "min": 4,
                    "max": 6,
                    "step": 1,
                    "display": "number"
                }),
                "export_texmap": ([True, False],),
                "save_video": ([True, False],),
                "remove_bg": ([True, False],),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "mesh_path", "video_path")

    FUNCTION = "run"

    CATEGORY = "InstantMesh"

    def run(self, model, config, images, diffusion_steps, view, export_texmap, save_video, remove_bg):
        img_full_path = None

        for (batch_number, ii) in enumerate(images):
            i = 255. * ii.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            now = datetime.datetime.now()

            timestamp = now.strftime("%Y%m%d_%H%M%S")

            img_file_name = f'InstantMesh_{timestamp}.png'

            tmp_path = f'{instant_mesh_path}/tmp'

            if not os.path.exists(tmp_path):
                os.makedirs(tmp_path)

            img_full_path = f'{tmp_path}/{img_file_name}'

            img.save(img_full_path)

        preview_img_path, mesh_path_idx, video_path_idx = run_InstantMesh(
            model, config, img_full_path, diffusion_steps, view, export_texmap, save_video, remove_bg)

        my_path = os.path.dirname(__file__)

        preview_img_path = os.path.join(my_path, preview_img_path)
        mesh_path_idx = os.path.join(my_path, mesh_path_idx)

        if video_path_idx is not None:
            video_path_idx = os.path.join(my_path, video_path_idx)

        print(preview_img_path)
        print(mesh_path_idx)
        print(video_path_idx)

        preview_img = Image.open(preview_img_path)

        image = None

        for i in ImageSequence.Iterator(preview_img):
            i = ImageOps.exif_transpose(i)
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]

        return (image, mesh_path_idx, video_path_idx)

NODE_CLASS_MAPPINGS = {
    "InstantMeshLoader": InstantMeshLoader,
    "InstantMeshRun": InstantMeshRun
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InstantMeshLoader": "InstantMeshLoader",
    "InstantMeshRun": InstantMeshRun
}
