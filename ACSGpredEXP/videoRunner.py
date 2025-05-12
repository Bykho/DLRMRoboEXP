import os
import argparse
import sys
from scene_image_analyzer import SceneImageAnalyzer
from simple_container_predictor import SimpleContainerPredictor
from scene_visualizer import SceneVisualizer, EnhancedActionSceneGraph, DummyInstance

# Helper to add RoboEXP to path if needed (similar to previous debug code)
def add_roboexp_to_path():
    roboexp_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # Assumes script is in RoboEXP/ACSGpredEXP
    if roboexp_parent_dir not in sys.path:
        print(f"--- Adding {roboexp_parent_dir} to sys.path for imports ---")
        sys.path.insert(0, roboexp_parent_dir)

# --- Main Logic ---
def main():
    add_roboexp_to_path() # Ensure necessary modules can be found

    parser = argparse.ArgumentParser(description="Process a sequence of images to build an accumulated ACSG.")
    parser.add_argument('--input-dir', type=str, default='./allJPEG', help='Directory containing input image sequence')
    parser.add_argument('--output-dir', type=str, default='./ExpectedVidRun', help='Output directory for the final graph')
    parser.add_argument('--openai-key', type=str, help='OpenAI API key (optional)')
    parser.add_argument('--claude-key', type=str, help='Claude API key (optional)')
    args = parser.parse_args()

    # --- Initialization ---
    print("Initializing components...")
    predictor = SimpleContainerPredictor(api_key=args.claude_key)
    visualizer = SceneVisualizer(output_dir=args.output_dir) # Needed for base_dir
    analyzer = SceneImageAnalyzer(api_key=args.openai_key)

    # Initialize the persistent scene graph
    # Use a generic name for the root node
    scene_graph = EnhancedActionSceneGraph(
        node_id="accumulated_scene",
        instance_label="accumulated_scene",
        instance=DummyInstance("accumulated_scene"),
        base_dir=visualizer.output_dir
    )
    print("Scene graph initialized.")

    # --- Image Processing Loop ---
    image_files = sorted([f for f in os.listdir(args.input_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))])

    if not image_files:
        print(f"No image files found in {args.input_dir}")
        return

    print(f"Found {len(image_files)} images to process.")

    processed_object_ids = set() # Keep track of added object node IDs
    container_nodes = {} # Keep track of container nodes added to the graph

    for i, fname in enumerate(image_files):
        image_path = os.path.join(args.input_dir, fname)
        scene_name = os.path.splitext(fname)[0] # Use image name as scene name for this frame
        print(f"\n--- Processing Image {i+1}/{len(image_files)}: {fname} ---")

        try:
            # Analyze image to get detections and predictions
            objects, containers, relationships, predicted_contents_by_container_id = \
                analyzer.analyze_image(image_path, predictor, scene_name)

            # --- Update Persistent Scene Graph ---
            print("Updating scene graph...")
            
            # Build relationship lookup for this frame
            relationship_map = {(rel["object1"], rel["object2"]): rel["relation"] for rel in relationships}

            # Add/Update Objects
            current_object_nodes = {} # Nodes relevant to *this* frame for relationship context
            for obj_label in objects:
                # Simple ID generation - assume first instance if not seen
                # More robust ID/matching needed for real tracking
                node_id = f"{obj_label}_0" 
                instance = DummyInstance(obj_label)
                # Default parent to root unless relationship found (simple approach)
                parent_node = scene_graph.root
                parent_relation = "in_scene"
                
                # Check relationships involving this object (basic check)
                # This part needs improvement for robust relationship handling across frames
                # For now, mainly connects to root
                
                obj_node = scene_graph.add_object(parent_node, node_id, obj_label, instance, parent_relation)
                current_object_nodes[obj_label] = obj_node # Store for container parenting
                processed_object_ids.add(node_id)

            # Add/Update Containers
            for container in containers:
                container_type = container["type"]
                container_id = container["id"] # Use the ID from GPT-4V
                container_location = container.get("location", "")
                container_purpose = container.get("purpose", "")
                node_label = f"{container_type} ({container_location})" if container_location else container_type
                instance = DummyInstance(container_type)

                # Determine parent based on relationships in *this frame*
                parent_node = scene_graph.root
                parent_relation = "in_scene"
                for rel in relationships:
                     # If this container is object1 in a relationship
                    if rel["object1"] == container_type:
                        # And object2 was detected in this frame
                        if rel["object2"] in current_object_nodes:
                            parent_node = current_object_nodes[rel["object2"]]
                            parent_relation = rel["relation"]
                            break 
                        # Maybe check existing graph nodes if object2 wasn't seen now?
                        elif f"{rel['object2']}_0" in scene_graph.object_nodes:
                             parent_node = scene_graph.object_nodes[f"{rel['object2']}_0"]
                             parent_relation = rel["relation"]
                             break

                container_node = scene_graph.add_object(parent_node, container_id, node_label, instance, parent_relation)
                container_nodes[container_id] = container_node # Store reference
                processed_object_ids.add(container_id)
                
                # Add purpose if available
                if container_purpose and hasattr(container_node, 'purpose'): # Check hasattr for safety
                     container_node.purpose = container_purpose

            # Update Expected Contents for detected containers
            for container_id, predicted_contents in predicted_contents_by_container_id.items():
                if container_id in container_nodes:
                    container_node = container_nodes[container_id]
                    
                    # --- Explicitly clear previous frame's predictions for this container ---
                    if hasattr(container_node, 'expected_contents'): 
                        container_node.expected_contents = []
                    else:
                         # Initialize if it somehow doesn't exist yet (shouldn't happen often)
                         container_node.expected_contents = [] 
                    # --- End clearing ---
                         
                    # Add all predictions from the current frame
                    for label, confidence in predicted_contents:
                         scene_graph.add_expected_object(container_node, label, confidence) # This method now just appends
                else:
                     print(f"Warning: Predicted contents for unknown container ID '{container_id}'")

            # --- Visualize intermediate graph after this frame's updates ---
            intermediate_filename = f"accumulated_graph_frame_{i+1}_{scene_name}"
            print(f"Visualizing intermediate graph: {intermediate_filename}.png")
            scene_graph.visualize(filename=intermediate_filename)
            # --- End of intermediate visualization ---

        except Exception as e:
            print(f"ERROR processing {fname}: {e}")
            # Optionally continue to next image or stop
            continue 

    # --- Final Visualization --- 
    print("\n--- All images processed. Generating final accumulated graph visualization... ---")
    try:
        scene_graph.visualize(filename="accumulated_scene_graph_FINAL") # Renamed to avoid overwrite if only 1 frame
        print("Final graph saved.")
    except Exception as e:
        print(f"Error visualizing final graph: {e}")

if __name__ == "__main__":
    main() 