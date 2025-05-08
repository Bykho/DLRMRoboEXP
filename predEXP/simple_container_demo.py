import argparse
from simple_container_predictor import SimpleContainerPredictor
from scene_visualizer import SceneVisualizer
import os

def analyze_scene(scene_objects, container_type, scene_name=None, api_key=None, visualize=True):
    """
    Analyze a scene and predict container contents
    
    Args:
        scene_objects: List of objects in the scene
        container_type: Type of container to predict contents for
        scene_name: Optional name for the scene (for visualization)
        api_key: Optional API key for Claude
        visualize: Whether to create a visualization
        
    Returns:
        Tuple of (predicted_contents, visualization_path)
    """
    # Create predictor
    predictor = SimpleContainerPredictor(api_key=api_key)
    
    # Get predictions
    predicted_contents = predictor.predict_container_contents(container_type, scene_objects)
    
    # Print results
    print(f"\nScene Analysis Results:")
    print(f"=====================")
    print(f"Scene contains: {', '.join(scene_objects)}")
    print(f"Container type: {container_type}")
    print(f"\nPredicted contents:")
    for i, item in enumerate(predicted_contents, 1):
        print(f"  {i}. {item}")
    
    # Create visualization if requested
    viz_path = None
    if visualize:
        # If no scene name provided, create one from the container type
        if scene_name is None:
            scene_name = f"{container_type}_scene"
        
        # Create visualizer and generate visualization
        visualizer = SceneVisualizer()
        viz_path = visualizer.visualize_scene(
            scene_name,
            scene_objects,
            container_type,
            predicted_contents
        )
        
        print(f"\nVisualization created: {viz_path}")
    
    return predicted_contents, viz_path

def main():
    """Main function to run the demo"""
    parser = argparse.ArgumentParser(description="Predict container contents based on scene context")
    parser.add_argument("--container", "-c", type=str, default="cabinet", help="Container type (cabinet, drawer, box, etc.)")
    parser.add_argument("--objects", "-o", type=str, nargs="+", required=True, help="Objects in the scene")
    parser.add_argument("--scene-name", "-n", type=str, help="Scene name for visualization (optional)")
    parser.add_argument("--api-key", type=str, help="Claude API key (optional, will use environment variable if not provided)")
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization")
    
    args = parser.parse_args()
    
    # Create a scene name if not provided
    scene_name = args.scene_name
    if scene_name is None:
        # Create scene name from the first few objects
        obj_names = "_".join(args.objects[:2])  # Use first two objects
        scene_name = f"{obj_names}_{args.container}_scene"
    
    # Run analysis with visualization
    analyze_scene(
        args.objects, 
        args.container, 
        scene_name=scene_name,
        api_key=args.api_key,
        visualize=not args.no_viz
    )

# Predefined scenes for quick testing
SAMPLE_SCENES = {
    "kitchen": {
        "objects": ["stove", "refrigerator", "microwave", "sink", "toaster"],
        "container": "cabinet"
    },
    "office": {
        "objects": ["desk", "chair", "computer", "printer", "monitor", "keyboard"],
        "container": "drawer"
    },
    "bathroom": {
        "objects": ["sink", "toilet", "shower", "bathtub", "towel"],
        "container": "cabinet"
    },
    "living_room": {
        "objects": ["sofa", "tv", "coffee_table", "bookshelf", "lamp"],
        "container": "box"
    },
    "bedroom": {
        "objects": ["bed", "nightstand", "dresser", "mirror", "lamp"],
        "container": "closet"
    },
    "garage": {
        "objects": ["car", "workbench", "tools", "bicycles", "lawnmower"],
        "container": "toolbox"
    }
}

if __name__ == "__main__":
    # If no arguments provided, run a quick demo of all sample scenes
    import sys
    if len(sys.argv) == 1:
        print("Running demo with sample scenes...\n")
        
        visualizations = []
        for scene_name, scene_data in SAMPLE_SCENES.items():
            print(f"\n--- {scene_name.upper()} SCENE ---")
            _, viz_path = analyze_scene(
                scene_data["objects"], 
                scene_data["container"],
                scene_name=scene_name
            )
            if viz_path:
                visualizations.append(viz_path)
            print("\n" + "-" * 50)
        
        if visualizations:
            print("\nAll visualizations created:")
            for viz in visualizations:
                print(f"- {viz}")
    else:
        main()