# Project Progress: ManiSkill + RoboEXP Integration

This document tracks the progress of integrating RoboEXP perception with ManiSkill simulation.

## Goal

Use ManiSkill to simulate a robot arm with a wrist camera (`panda_wristcam` in `PickCube-v1` environment) capturing RGBD images from different viewpoints. Run two perception pipelines on these observations:

1.  **Original RoboEXP:** GroundingDINO + SAM (currently implemented with EfficientViT-SAM variant via RoboEXP's config).
2.  **New Pipeline:** YOLO-UniOW + EfficientViT-SAM.

The ultimate aim is to compare the effect of these pipelines on scene graph construction using RoboEXP's `RoboMemory` and `ActionSceneGraph`.

## Setup

*   **Environment:** Python 3.10 virtual environment (`venv310`).
*   **Core Libraries:** ManiSkill, SAPIEN, PyTorch, GroundingDINO, EfficientViT-SAM, OpenCV, Open3D.
*   **Key Local Dependencies:** `RoboEXP`, `efficientvit`, `GroundingDINO` (integrated within RoboEXP's perception module).
*   **Main Script:** `interactive_wristcam_viewer.py`

## Progress & Key Milestones

1.  **Environment & Basic Simulation (Done):**
    *   Set up ManiSkill with `PickCube-v1` and `panda_wristcam`.
    *   Implemented keyboard teleoperation for robot arm and gripper.
    *   Enabled GUI rendering (`render_mode="human"`).

2.  **RoboEXP Integration - Perception & Memory (Largely Working - Refinements Ongoing):**
    *   Successfully integrated `RoboPercept` to get object detections (bounding boxes, phrases, masks) from RGB images.
    *   Successfully integrated `RoboMemory` to process these detections and `obs` data (RGB, depth, point cloud, camera parameters).
    *   **CUDA Out of Memory Resolved (08-May-2025):** Resolved by implementing conditional/lazy loading for DenseCLIP.
    *   **GroundingDINO & SAM Tuning - MAJOR BREAKTHROUGH (08-May-2025):**
        *   Tuned GroundingDINO's thresholds, leading to higher-quality bounding box proposals for SAM.
        *   SAM (EfficientViT-SAM) now generates much more accurate and tighter segmentation masks.
    *   **Accurate Scene Graph Generation (Single & Multiple Objects - VALIDATED 08-May-2025, timestamp fix 08-May-2025):**
        *   The system now correctly generates scene graphs for both single-object and multi-object scenes.
        *   Fixed `AttributeError: 'myInstance' object has no attribute 'timestamp'` by initializing `self.timestamp = 0.0` in `myInstance.__init__`. This unblocked consistent scene graph generation.
        *   The high-level ACSG (2D PNG) is now reliably produced after perception runs.
    *   Implemented visualization of GroundingDINO/SAM outputs (boxes, masks) using OpenCV.
    *   Implemented display of the 2D scene graph PNG.
    *   Corrected various `TypeError`, `NameError`, and `AttributeError` issues during RoboEXP component initialization and operation by ensuring correct argument names, providing necessary default paths, and fixing logic errors in data handling (e.g., tensor vs. numpy, return value unpacking).
    *   **Low-Level Memory Visualization (Open3D - In Progress 08-May-2025):**
        *   Added a feature to `interactive_wristcam_viewer.py` (triggered by 'v' key) to visualize the accumulated `RoboMemory.memory_scene` (voxel point cloud) using Open3D.
        *   Currently debugging why the Open3D window appears white despite reporting points being visualized. Added point size adjustments and camera reset calls.

## Current Status & Next Steps

The core perception and scene graph generation pipeline is now functioning much more reliably. The main focus is shifting to refining the low-level visualization and then more systematic testing.

1.  **Low-Level Memory Visualization (Open3D - Immediate Focus):**
    *   Resolve the white screen issue with the Open3D point cloud viewer. Ensure points are visible and correctly represent the `memory_scene`.
2.  **Systematic Testing & Robustness:**
    *   Test with a wider variety of objects and more complex arrangements.
    *   Evaluate robustness to different camera viewpoints, lighting (if configurable), and partial occlusions.
    *   Observe behavior when objects are removed or moved significantly between perception cycles.
3.  **Threshold Refinement (Ongoing):**
    *   The GroundingDINO thresholds (currently `box_threshold=0.20`, `text_threshold=0.20`) and `RoboMemory`'s `iou_thres=0.4` are producing reasonable results. Continue to monitor and adjust if false positives/negatives or incorrect merging/splitting of instances appear in more complex scenes.
4.  **Multi-Cycle Scene Graph Updates (Further Investigation):**
    *   The current memory clearing logic is good for isolated tests. For persistent understanding, further work on how `RoboMemory` accumulates and updates its state across many views will be needed.
5.  **Mouse Teleoperation:** Re-integrate mouse-based teleoperation for finer control.

## Open Questions/Future Work

*   Integration of the YOLO-UniOW + EfficientViT-SAM pipeline for comparison.
*   Quantitative evaluation of scene graph quality.
*   Handling of "part-level" concepts.
*   Exploration of RoboEXP's LLM-based features.

This progress is very encouraging! The critical issues have been resolved, and the system is producing the expected outputs. 