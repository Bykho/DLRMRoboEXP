# Project: Enhancing RoboEXP's Perception Pipeline and Extending Action-Conditioned Scene Graphs

This project focuses on enhancing the perception capabilities of the RoboEXP framework and proposing extensions to its Action-Conditioned Scene Graphs (ACSGs). The work is primarily divided into two main components, developed in separate branches.

## Kheri's Contributions (Branch: `kri`)

Kheri's work centered on the **Perception Pipeline Enhancement** within the RoboEXP framework, primarily focusing on the integration and validation of more efficient vision models.

Key contributions include:
*   **EfficientViT-SAM Integration:** Successfully integrated the EfficientViT-SAM model (L0 variant) as a replacement for SAM-HQ, including developing a custom wrapper and modifying the RoboEXP codebase.
*   **Debugging and Integration Challenges:** Addressed model/library incompatibilities, managed memory/dimensionality issues, resolved checkpoint loading problems, and ensured model interface compatibility, including adapting data types passed between components (e.g., PIL Images vs. Tensors).
*   **Validation and Testing:** Utilized an interactive script (`interactive_wristcam_viewer.py`) and the ManiSkill 3 simulator for debugging, validation, and addressing specific integration errors (e.g., type mismatches, tensor processing, memory access).

The primary tools and environments for this part of the project were RoboEXP, ManiSkill 3, EfficientViT-SAM, and GroundingDINO.

## Nico's Contributions (Branch: `scene_setup`)

Nico's work focused on the **Conceptual Extensions to Action-Conditioned Scene Graphs (ACSGs)**, aiming to enrich the robot's environmental understanding by inferring information about unobserved areas.

Key contributions include:
*   **Augmenting ACSGs with Expected Content:** Proposed and implemented a system to insert "expected-content" nodes into the ACSG for unexplored container objects, based on their context.
*   **LLM-based Inference for Latent Objects:** Used a Large Language Model (Claude 3.5 Sonnet) with a custom prompting regime, utilizing both visual observations and existing graph structure context, to infer likely contents of unexplored spaces and inject these hypotheses (`expected_inside` relations) into the ACSG.
*   **Methodology and Implementation:** Developed and visualized the process for augmenting ACSGs, including outlining how downstream modules could leverage this information for exploration and planning.
*   **Experimental Validation:** Conducted experiments in various simulated environments (bedroom, kitchen, office), using video clips and GroundingDINO outputs to incrementally build and update the augmented ACSG, analyzing the "Expectation Correctness" and "Generalization" of the approach.

This part of the project aimed to create a richer scene graph to aid downstream tasks by leveraging LLM capabilities for reasoning about partially observable environments. 