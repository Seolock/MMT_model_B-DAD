import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import cv2
import os


def _load_attention_map(attention_map):
    if isinstance(attention_map, str):
        attention_map = os.path.expanduser(attention_map)
        if attention_map.endswith(".pth") or attention_map.endswith(".pt"):
            try:
                import torch

                attn = torch.load(attention_map, map_location="cpu")
                if hasattr(attn, "detach"):
                    attn = attn.detach().cpu().numpy()
            except Exception as exc:
                raise RuntimeError(f"attention_map 파일을 읽지 못했습니다: {attention_map}") from exc
        else:
            attn = np.load(attention_map)
    else:
        attn = attention_map

    return np.array(attn, dtype=np.float32)


def _build_attention_overlay(image, attention_1d, num_patches, alpha, colormap):
    img_array = np.array(image)
    H, W = img_array.shape[:2]

    if attention_1d.shape[0] == num_patches + 1:
        attention_1d = attention_1d[1:]

    assert attention_1d.shape[0] == num_patches, \
        f"attention_map 길이가 {num_patches}여야 합니다. 현재: {attention_1d.shape[0]}"

    grid_size = int(np.sqrt(num_patches))
    assert grid_size * grid_size == num_patches, \
        "num_patches는 완전제곱수여야 합니다."

    attn_2d = attention_1d.reshape(grid_size, grid_size)
    attn_resized = cv2.resize(attn_2d, (W, H), interpolation=cv2.INTER_CUBIC)

    attn_min, attn_max = attn_resized.min(), attn_resized.max()
    if attn_max - attn_min > 1e-8:
        attn_norm = (attn_resized - attn_min) / (attn_max - attn_min)
    else:
        attn_norm = np.zeros_like(attn_resized)

    cmap = cm.get_cmap(colormap)
    heatmap_rgba = cmap(attn_norm)
    heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)

    blended = (
        (1 - alpha) * img_array.astype(np.float32) +
        alpha * heatmap_rgb.astype(np.float32)
    ).astype(np.uint8)

    return img_array, heatmap_rgb, blended, attn_min, attn_max, attn_norm

def visualize_vit_attention(
    image_path: str,
    attention_map: np.ndarray,
    output_path: str = "attention_visualization.png",
    patch_size: int = 16,
    num_patches: int = 576,
    alpha: float = 0.5,
    colormap: str = "jet"
):
    """
    ViT attention map을 이미지 위에 히트맵으로 시각화합니다.
    
    Args:
        image_path: 입력 이미지 경로
        attention_map: shape [576] 또는 [576+1] (CLS 토큰 포함 가능)
        output_path: 저장할 이미지 경로
        patch_size: 패치 크기 (기본값 16 → 224/16 = 14, 14x14=196 / 384/16=24, 24x24=576)
        num_patches: 패치 수 (기본값 576 = 24x24)
        alpha: 히트맵 투명도 (0~1)
        colormap: 컬러맵 이름
    """
    image_path = os.path.expanduser(image_path)
    output_path = os.path.expanduser(output_path)

    # ── 1. 이미지 로드 ──────────────────────────────────────────────
    image = Image.open(image_path).convert("RGB")

    # ── 2. Attention map 전처리 ─────────────────────────────────────
    attn = _load_attention_map(attention_map)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    base, ext = os.path.splitext(output_path)
    if not ext:
        ext = ".png"

    if attn.ndim == 1:
        attn = attn[None, :]

    assert attn.ndim == 2, f"attention_map은 [576] 또는 [T, 576] 형태여야 합니다. 현재: {attn.shape}"

    saved_paths = []
    total_steps = attn.shape[0]

    for time_index, attn_step in enumerate(attn):
        img_array, heatmap_rgb, blended, attn_min, attn_max, _ = _build_attention_overlay(
            image=image,
            attention_1d=attn_step,
            num_patches=num_patches,
            alpha=alpha,
            colormap=colormap,
        )

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.patch.set_facecolor("#1a1a2e")

        titles = ["Original Image", "Overlay", f"Attention Heatmap (t={time_index + 1}/{total_steps})"]
        imgs = [img_array, blended, heatmap_rgb]

        for ax, title, img in zip(axes, titles, imgs):
            ax.imshow(img)
            ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=10)
            ax.axis("off")

        sm = plt.cm.ScalarMappable(
            cmap=colormap,
            norm=plt.Normalize(vmin=attn_min, vmax=attn_max)
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=axes[2], shrink=0.7, pad=0.02)
        cbar.set_label("Attention Score", color="white", fontsize=12)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

        plt.suptitle("ViT Attention Map Visualization", color="white",
                     fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout()

        if total_steps == 1:
            save_path = f"{base}{ext}"
        else:
            save_path = f"{base}_t{time_index:03d}{ext}"

        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        saved_paths.append(save_path)

    print(f"✅ 저장 완료: {len(saved_paths)}개")
    return saved_paths


# ── 사용 예시 ─────────────────────────────────────────────────────────
if __name__ == "__main__":

    path = "checkpoints/multi30k-en2de-vit_tiny_patch16_384-entropy_test"

    visualize_vit_attention(
        image_path   = "../../flickr30k/testcoco-images/COCO_train2014_000000423537.jpg",   # 이미지 경로
        attention_map= path+"/visualization/68map.pth",     # shape: [576] or [577]
        output_path  = path+"/map/coco68_layer4.png",
        patch_size   = 16,
        num_patches  = 576,
        alpha        = 0.4,                # 히트맵 투명도
        colormap     = "jet"               # jet / viridis / hot / plasma
    )

    # 68 COCO_train2014_000000423537
    # 460 COCO_val2014_000000310227