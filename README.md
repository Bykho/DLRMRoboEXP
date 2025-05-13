# Project Component: Perception Pipeline Enhancement (Branch: `kri`)

This part of the project focuses on enhancing the perception capabilities of the RoboEXP framework, specifically by integrating and validating more efficient vision models.

Key contributions include:
*   **EfficientViT-SAM Integration:** Successfully integrated the EfficientViT-SAM model (L0 variant) as a replacement for SAM-HQ, including developing a custom wrapper and modifying the RoboEXP codebase.
*   **Debugging and Integration Challenges:** Addressed model/library incompatibilities, managed memory/dimensionality issues, resolved checkpoint loading problems, and ensured model interface compatibility, including adapting data types passed between components (e.g., PIL Images vs. Tensors).
*   **Validation and Testing:** Utilized an interactive script (`interactive_wristcam_viewer.py`) and the ManiSkill 3 simulator for debugging, validation, and addressing specific integration errors.

The primary tools and environments used were RoboEXP, ManiSkill 3, EfficientViT-SAM, and GroundingDINO.

*Note: This README describes the work primarily associated with the `kri` branch. See the main project README for overall context and contributions from other branches.* 