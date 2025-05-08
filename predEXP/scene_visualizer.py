import graphviz
import os
from typing import List, Dict, Any

class SceneVisualizer:
    """
    Simple class to visualize a scene with predicted container contents.
    """
    def __init__(self, output_dir="./output"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualization files
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")
    
    def visualize_scene(
        self, 
        scene_name: str, 
        scene_objects: List[str], 
        container_type: str, 
        predicted_contents: List[str]
    ):
        """
        Create a visual graph of the scene and predictions.
        
        Args:
            scene_name: Name of the scene (e.g., "kitchen", "office")
            scene_objects: List of objects in the scene
            container_type: Type of container
            predicted_contents: Predicted contents of the container
            
        Returns:
            Path to the rendered graph image
        """
        # Create a new directed graph
        dot = graphviz.Digraph(
            name=f"{scene_name}_scene",
            comment=f"Scene graph for {scene_name}",
            directory=self.output_dir,
            format="png"
        )
        
        # Add a scene node
        dot.node(
            "scene", 
            f"{scene_name.capitalize()} Scene", 
            shape="box", 
            style="filled", 
            fillcolor="lightblue"
        )
        
        # Add container node
        dot.node(
            "container", 
            f"{container_type.capitalize()}", 
            shape="box", 
            style="filled", 
            fillcolor="lightsalmon"
        )
        
        # Add edge from scene to container
        dot.edge("scene", "container", label="contains")
        
        # Add visible object nodes
        for i, obj in enumerate(scene_objects):
            node_id = f"obj_{i}"
            dot.node(
                node_id,
                obj.capitalize(),
                shape="ellipse",
                style="filled",
                fillcolor="lightgreen"
            )
            dot.edge("scene", node_id, label="contains")
        
        # Add predicted content nodes
        for i, item in enumerate(predicted_contents):
            node_id = f"pred_{i}"
            dot.node(
                node_id,
                item.capitalize(),
                shape="ellipse",
                style="filled",
                fillcolor="lightpink",
            )
            dot.edge("container", node_id, label="expected_inside", style="dotted")
        
        # Add a legend
        with dot.subgraph(name="cluster_legend") as legend:
            legend.attr(label="Legend", style="filled", fillcolor="white")
            legend.node("l_scene", "Scene", shape="box", style="filled", fillcolor="lightblue")
            legend.node("l_container", "Container", shape="box", style="filled", fillcolor="lightsalmon")
            legend.node("l_visible", "Visible Object", shape="ellipse", style="filled", fillcolor="lightgreen")
            legend.node("l_predicted", "Predicted Content", shape="ellipse", style="dotted,filled", fillcolor="lightpink")
        
        # Render the graph
        output_path = dot.render(filename=f"{scene_name}_graph", cleanup=True)
        print(f"Scene graph visualization saved to {output_path}")
        
        return output_path

# Example usage
if __name__ == "__main__":
    from simple_container_predictor import SimpleContainerPredictor
    
    # Create predictor
    predictor = SimpleContainerPredictor()
    
    # Define a scene
    scene_name = "kitchen"
    scene_objects = ["stove", "refrigerator", "microwave", "sink"]
    container_type = "cabinet"
    
    # Get predictions
    predicted_contents = predictor.predict_container_contents(container_type, scene_objects)
    
    # Visualize
    visualizer = SceneVisualizer()
    visualizer.visualize_scene(scene_name, scene_objects, container_type, predicted_contents)