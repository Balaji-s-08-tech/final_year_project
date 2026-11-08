import os


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    from runtime_policy import require_local_asset
    vision_tower = require_local_asset(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 'use_s2', False)

    if 'sig' in vision_tower.lower():
        from .siglip.siglip_encoder import SiglipVisionTower, SiglipVisionTowerS2
        if use_s2:
            return SiglipVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return SiglipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif 'eva' in vision_tower.lower():
        # EVA imports xformers.  Import it only when an EVA checkpoint is
        # actually selected so Phi-2/SigLIP/CLIP Bunny imports do not acquire
        # an unrelated optional CUDA-extension dependency.
        from .eva_clip.eva_clip_encoder import EvaClipVisionTower
        if use_s2:
            raise ValueError(f'Currently not supporting S2 for EVA-CLIP')
        else:
            return EvaClipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    elif 'clip' in vision_tower.lower():
        from .clip.clip_encoder import CLIPVisionTower
        if use_s2:
            raise ValueError(f'Currently not supporting S2 for CLIP')
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    else:
        raise ValueError(f'Unknown vision tower: {vision_tower}')
