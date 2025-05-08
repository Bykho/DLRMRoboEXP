"""
Simple example of integrating the container predictor with a robot system.
This is just a skeleton - you'll need to adapt it to your specific robot framework.
"""
from simple_container_predictor import SimpleContainerPredictor
from scene_visualizer import SceneVisualizer

class RobotSceneAnalyzer:
    """
    A bridge between your robot perception system and the container predictor.
    """
    def __init__(self, api_key=None):
        """
        Initialize the scene analyzer.
        
        Args:
            api_key: Optional API key for Claude
        """
        self.predictor = SimpleContainerPredictor(api_key=api_key)
        self.visualizer = SceneVisualizer()
        
    def analyze_perception_results(self, perception_results, visualize=True):
        """
        Analyze perception results to predict container contents.
        
        Args:
            perception_results: Results from your robot's perception system
            visualize: Whether to generate a visualization
            
        Returns:
            Dictionary mapping container IDs to their predicted contents
        """
        # Extract container objects from perception results
        containers = self._extract_containers(perception_results)
        
        # Extract visible non-container objects
        visible_objects = self._extract_visible_objects(perception_results)
        
        # Generate predictions for each container
        predictions = {}
        for container_id, container_info in containers.items():
            container_type = container_info["type"]
            predicted_contents = self.predictor.predict_container_contents(
                container_type, visible_objects
            )
            
            predictions[container_id] = {
                "type": container_type,
                "predicted_contents": predicted_contents
            }
            
            # Visualize if requested
            if visualize:
                scene_name = f"robot_scene_{container_id}"
                self.visualizer.visualize_scene(
                    scene_name,
                    visible_objects,
                    container_type,
                    predicted_contents
                )
        
        return predictions
    
    def _extract_containers(self, perception_results):
        """
        Extract container objects from perception results.
        Adapt this to your specific perception system format.
        
        Args:
            perception_results: Results from your robot's perception system
            
        Returns:
            Dictionary mapping container IDs to container info
        """
        # This is a placeholder implementation - adapt to your system
        containers = {}
        
        # Example format assuming perception_results has a list of objects
        if hasattr(perception_results, "objects"):
            for obj in perception_results.objects:
                if self._is_container(obj.label):
                    containers[obj.id] = {
                        "type": obj.label,
                        "position": obj.position,
                        "size": obj.size
                    }
        
        return containers
    
    def _extract_visible_objects(self, perception_results):
        """
        Extract visible non-container objects from perception results.
        Adapt this to your specific perception system format.
        
        Args:
            perception_results: Results from your robot's perception system
            
        Returns:
            List of visible object labels
        """
        # This is a placeholder implementation - adapt to your system
        visible_objects = []
        
        # Example format assuming perception_results has a list of objects
        if hasattr(perception_results, "objects"):
            for obj in perception_results.objects:
                if not self._is_container(obj.label):
                    visible_objects.append(obj.label)
        
        return visible_objects
    
    def _is_container(self, label):
        """
        Check if an object label represents a container.
        
        Args:
            label: Object label
            
        Returns:
            True if the object is a container, False otherwise
        """
        container_types = [
            "cabinet", "drawer", "box", "refrigerator", "closet", 
            "container", "shelf", "cupboard", "chest", "wardrobe"
        ]
        return label.lower() in container_types
    
    def update_scene_graph(self, scene_graph, container_id, predicted_contents):
        """
        Update a scene graph with predicted container contents.
        Adapt this to your specific scene graph format.
        
        Args:
            scene_graph: Your system's scene graph
            container_id: ID of the container to update
            predicted_contents: Predicted contents of the container
            
        Returns:
            Updated scene graph
        """
        # This is a placeholder implementation - adapt to your system
        
        # Example assuming scene_graph has a way to add expected relations
        if hasattr(scene_graph, "add_expected_relation"):
            for item in predicted_contents:
                scene_graph.add_expected_relation(
                    container_id, 
                    "expected_inside", 
                    item
                )
        
        return scene_graph


# Mock classes to demonstrate usage
class MockPerceptionResults:
    def __init__(self):
        self.objects = [
            MockObject("cabinet_1", "cabinet", [0, 0, 0], [0.5, 0.5, 1.0]),
            MockObject("drawer_1", "drawer", [1, 0, 0], [0.3, 0.3, 0.3]),
            MockObject("table_1", "table", [0, 1, 0], [1.0, 1.0, 0.5]),
            MockObject("chair_1", "chair", [1, 1, 0], [0.5, 0.5, 1.0]),
            MockObject("printer_1", "printer", [2, 0, 0], [0.4, 0.4, 0.4]),
            MockObject("monitor_1", "monitor", [2, 1, 0], [0.6, 0.4, 0.1])
        ]

class MockObject:
    def __init__(self, id, label, position, size):
        self.id = id
        self.label = label
        self.position = position
        self.size = size

class MockSceneGraph:
    def __init__(self):
        self.nodes = {}
        self.relations = []
    
    def add_expected_relation(self, from_id, relation, to_label):
        self.relations.append({
            "from": from_id,
            "relation": relation,
            "to": to_label
        })
        print(f"Added relation: {from_id} --{relation}--> {to_label}")


# Example usage
if __name__ == "__main__":
    # Create the scene analyzer
    analyzer = RobotSceneAnalyzer()
    
    # Create mock perception results
    perception_results = MockPerceptionResults()
    
    # Create mock scene graph
    scene_graph = MockSceneGraph()
    
    # Analyze perception results
    predictions = analyzer.analyze_perception_results(perception_results)
    
    print("\nPredictions:")
    for container_id, prediction in predictions.items():
        print(f"\n{container_id} ({prediction['type']}):")
        for item in prediction['predicted_contents']:
            print(f"  - {item}")
            
        # Update scene graph
        scene_graph = analyzer.update_scene_graph(
            scene_graph, 
            container_id, 
            prediction['predicted_contents']
        )