import argparse
import torch
import numpy as np
import sys
import os
import logging
from pathlib import Path
from PIL import Image

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, SamModel, SamProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def get_bounding_box(image_pil, text_prompt, processor, model, device):
    inputs = processor(images=image_pil, text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-process results
    target_sizes = torch.tensor([image_pil.size[::-1]])
    try:
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.3, # User code constant
            text_threshold=0.25, # User code constant
            target_sizes=target_sizes
        )[0]
    except TypeError:
         # Fallback for older transformers
         results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            target_sizes=target_sizes
        )[0]
    
    return results["boxes"], results["scores"], results["labels"]

def segment_with_sam(image_pil, boxes, processor, model, device):
    if len(boxes) == 0:
        return None
    
    input_boxes = [boxes.tolist()] # SAM processor expects list of lists
    
    inputs = processor(image_pil, input_boxes=input_boxes, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks, 
        inputs["original_sizes"], 
        inputs["reshaped_input_sizes"]
    )[0]
    
    return masks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, required=True, help="HuggingFace dataset repo id")
    parser.add_argument("--root", type=str, default=None, help="Dataset root directory")
    parser.add_argument("--num-episodes", type=int, default=5, help="Number of episodes to visualize")
    parser.add_argument("--output-dir", type=str, default="debug_sam_masks", help="Output directory")
    args = parser.parse_args()

    # Constants from user code adapted
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    logger.info(f"Using device: {DEVICE}")

    # Load Models (User specified models)
    logger.info("Loading models...")
    gd_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
    gd_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to(DEVICE)
    
    sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(DEVICE)
    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    # Load Dataset
    logger.info(f"Loading dataset: {args.repo_id}")
    dataset = LeRobotDataset(args.repo_id, root=args.root)
    
    # Processing
    total_eps = len(dataset.meta.episodes)
    num_episodes = min(total_eps, args.num_episodes)
    
    for ep_idx in range(num_episodes):
        ep_info = dataset.meta.episodes[ep_idx]
        from_idx = ep_info["dataset_from_index"]
        to_idx = ep_info["dataset_to_index"]
        
        # Visualize first and last frame
        frames_to_process = {"first": from_idx, "last": to_idx - 1}
        
        for label, frame_idx in frames_to_process.items():
            try:
                item = dataset[frame_idx]
            except IndexError:
                continue
                
            for key in dataset.meta.camera_keys:
                if key not in item: continue
                
                img_tensor = item[key]
                # Convert to PIL
                if isinstance(img_tensor, torch.Tensor):
                    img_np = img_tensor.permute(1, 2, 0).numpy()
                    if img_np.max() <= 1.0:
                        img_np = (img_np * 255).astype(np.uint8)
                    else:
                        img_np = img_np.astype(np.uint8)
                    image_pil = Image.fromarray(img_np)
                else:
                    # Assume PIL or compatible
                    image_pil = img_tensor
                    img_np = np.array(image_pil)
                
                # 1. Detect "black tape" vs "robotic arm"
                boxes, scores, labels_text = get_bounding_box(image_pil, "black tape . robotic arm .", gd_processor, gd_model, DEVICE)
                
                # Filter for tape
                target_indices = [i for i, l in enumerate(labels_text) if "tape" in l]
                
                if len(target_indices) > 0:
                    boxes = boxes[target_indices]
                    scores = scores[target_indices]
                    
                    logger.info(f"Ep {ep_idx} ({label}): Found black tape (Score: {scores.max():.2f})")
                    
                    # 2. Segment
                    masks = segment_with_sam(image_pil, boxes, sam_processor, sam_model, DEVICE)
                    
                    # 3. Visualize and Save
                    # Combine masks
                    if masks is not None:
                        combined_mask = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=bool)
                        # masks shape: (N_boxes, 1, H, W) -> we take best mask per box (index 0)
                        for i in range(len(masks)):
                            m = masks[i][0].cpu().numpy()
                            combined_mask = np.logical_or(combined_mask, m)
                        
                        # Save Binary Mask for Training Reference
                        # (White tape, Black background)
                        mask_img = Image.fromarray((combined_mask * 255).astype(np.uint8))
                        mask_path = os.path.join(OUTPUT_DIR, f"ep{ep_idx}_{key}_{label}_mask.png")
                        mask_img.save(mask_path)

                        # Save Overlay for Debugging (User requested visual)
                        # Create green overlay
                        overlay = img_np.copy()
                        # Green channel boost where mask is true
                        # Simple alpha blend
                        color_mask = np.zeros_like(img_np)
                        color_mask[combined_mask] = [0, 255, 0]
                        
                        # Blend using PIL for simplicity/no-cv2 dependency
                        # Or manual numpy
                        alpha = 0.5
                        bs = combined_mask
                        overlay[bs] = (overlay[bs] * (1-alpha) + color_mask[bs] * alpha).astype(np.uint8)
                        
                        overlay_path = os.path.join(OUTPUT_DIR, f"ep{ep_idx}_{key}_{label}_overlay.png")
                        Image.fromarray(overlay).save(overlay_path)
                        
                        # Save Original
                        orig_path = os.path.join(OUTPUT_DIR, f"ep{ep_idx}_{key}_{label}_orig.png")
                        Image.fromarray(img_np).save(orig_path)
                        
                        logger.info(f"Saved: {mask_path}")
                else:
                    logger.info(f"Ep {ep_idx} ({label}): No tape detected.")

if __name__ == "__main__":
    main()
