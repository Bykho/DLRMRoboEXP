#simple_robot_integration.py


from simple_container_predictor import SimpleContainerPredictor
from scene_visualizer import SceneVisualizer, DummyInstance, EnhancedActionSceneGraph

class RobotSceneAnalyzer:
    """
    A bridge between your robot perception system and the container predictor.
    """
    def __init__(self, api_key=None, output_dir="./output"):
        """
        Initialize the scene analyzer.
        
        Args:
            api_key: Optional API key for Claude
            output_dir: Directory to save output files
        """
        self.predictor = SimpleContainerPredictor(api_key=api_key)
        self.visualizer = SceneVisualizer(output_dir=output_dir)
        self.output_dir = output_dir
        
        # Define container types for detection
        self.container_types = [
            "cabinet", "drawer", "box", "refrigerator", "closet", 
            "container", "shelf", "cupboard", "chest", "wardrobe"
        ]
        
    def analyze_perception_results(self, perception_results, visualize=True):
        """
        Analyze perception results to predict container contents.
        
        Args:
            perception_results: Results from your robot's perception system
            visualize: Whether to generate a visualization
            
        Returns:
            Dictionary mapping container IDs to their predicted contents and a scene graph
        """
        # Extract container objects from perception results
        containers = self._extract_containers(perception_results)
        
        # Extract visible non-container objects
        visible_objects = self._extract_visible_objects(perception_results)
        
        # Create scene graph
        scene_graph = EnhancedActionSceneGraph(
            node_id="robot_scene_0", 
            instance_label="robot_scene",
            instance=self._create_dummy_instance("scene"),
            base_dir=self.output_dir
        )
        
        # Add visible objects to the scene graph
        for label in visible_objects:
            scene_graph.add_object(
                parent=scene_graph.root,
                node_id=f"{label}_node",
                instance_label=label,
                instance=self._create_dummy_instance(label),
                parent_relation="on"
            )
        
        # Generate predictions for each container
        predictions = {}
        for container_id, container_info in containers.items():
            container_type = container_info["type"]
            
            # Predict container contents
            predicted_contents = self.predictor.predict_container_contents(
                container_type, visible_objects
            )
            
            # Add container to scene graph
            container_node = scene_graph.add_object(
                parent=scene_graph.root,
                node_id=container_id,
                instance_label=container_type,
                instance=self._create_dummy_instance(container_type),
                parent_relation="on"
            )
            
            # Add predicted contents to container in scene graph
            for label, confidence in predicted_contents:
                scene_graph.add_expected_object(container_node, label, confidence)
            
            # Store predictions
            predictions[container_id] = {
                "type": container_type,
                "predicted_contents": predicted_contents
            }
            
        # Visualize if requested
        if visualize:
            scene_graph.visualize()
        
        return scene_graph, predictions
    
    def update_scene_graph_after_opening(self, scene_graph, container_id, observed_objects):
        """
        Update scene graph after opening a container and observing its contents.
        
        Args:
            scene_graph: The scene graph to update
            container_id: ID of the container that was opened
            observed_objects: List of objects observed inside the container
            
        Returns:
            Updated scene graph
        """
        # Find the container node
        container_node = None
        for node_id, node in scene_graph.object_nodes.items():
            if node_id == container_id:
                container_node = node
                break
                
        if container_node is None:
            print(f"Container node {container_id} not found in scene graph")
            return scene_graph
            
        # Mark container as explored
        container_node.explored = True
        
        # Get expected contents
        expected_contents = getattr(container_node, 'expected_contents', [])
        
        # Match observed objects against expected contents
        matched = set()
        for expected in expected_contents:
            expected_label = expected['label']
            
            # Check if this expected object was observed
            for observed in observed_objects:
                observed_label = observed['label']
                
                # Check for exact match or similarity
                if (expected_label == observed_label or 
                    expected_label in observed_label or 
                    observed_label in expected_label):
                    matched.add(expected_label)
                    # Boost confidence for confirmed predictions
                    expected['confidence'] = min(1.0, expected['confidence'] + 0.2)
                    break
        
        # Decrease confidence for unmatched predictions
        for expected in expected_contents:
            if expected['label'] not in matched:
                expected['confidence'] = max(0.1, expected['confidence'] - 0.3)
        
        # Add observed objects to the scene graph
        for observed in observed_objects:
            observed_label = observed['label']
            
            # Create instance and add to scene graph
            instance = self._create_dummy_instance(observed_label)
            scene_graph.add_object(
                parent=container_node,
                node_id=f"{observed_label}_inside_{container_id}",
                instance_label=observed_label,
                instance=instance,
                parent_relation="inside"
            )
        
        # Visualize the updated scene graph
        scene_graph.visualize()
        
        return scene_graph
    
    def bias_detection(self, container_id, detection_results):
        """
        Bias detection results based on expected objects in a container.
        
        Args:
            container_id: ID of the container being observed
            detection_results: Detection results to adjust
            
        Returns:
            Biased detection results
        """
        # Check if we have predictions for this container
        container_node = None
        for node_id, node in self.scene_graph.object_nodes.items():
            if node_id == container_id:
                container_node = node
                break
                
        if container_node is None or not hasattr(container_node, 'expected_contents'):
            return detection_results
        
        # Get expected objects
        expected_objects = container_node.expected_contents
        
        # Bias detection confidences for expected objects
        biased_results = detection_results.copy()
        
        # Loop through detections
        for i, detection in enumerate(biased_results['detections']):
            detected_label = detection['label']
            original_confidence = detection['confidence']
            
            # Check if this matches any expected object
            for expected in expected_objects:
                expected_label = expected['label']
                expected_confidence = expected['confidence']
                
                # Check for match or similarity
                if (expected_label == detected_label or 
                    expected_label in detected_label or 
                    detected_label in expected_label):
                    # Bias confidence based on expectation
                    # Small boost (10%) to avoid hallucination but help with ambiguous detections
                    boost = expected_confidence * 0.1
                    biased_results['detections'][i]['confidence'] = min(0.95, original_confidence + boost)
                    break
        
        return biased_results
    
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
        return label.lower() in self.container_types
    
    def _create_dummy_instance(self, label):
        """
        Create a dummy instance for the scene graph.
        
        Args:
            label: Label for the instance
            
        Returns:
            DummyInstance object
        """
        return DummyInstance(label)


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

class MockDetectionResults:
    def __init__(self):
        self.detections = [
            {"label": "plate", "confidence": 0.65, "bbox": [100, 100, 200, 200]},
            {"label": "unknown_object", "confidence": 0.55, "bbox": [300, 300, 400, 400]},
            {"label": "cup", "confidence": 0.70, "bbox": [500, 500, 600, 600]}
        ]


# Example usage
if __name__ == "__main__":
    # Create the scene analyzer
    analyzer = RobotSceneAnalyzer()
    
    # Create mock perception results
    perception_results = MockPerceptionResults()
    
    # Analyze perception results
    scene_graph, predictions = analyzer.analyze_perception_results(perception_results)
    
    print("\nPredictions:")
    for container_id, prediction in predictions.items():
        print(f"\n{container_id} ({prediction['type']}):")
        for item, confidence in prediction['predicted_contents']:
            print(f"  - {item} (confidence: {confidence:.2f})")
    
    # Example of updating the scene graph after opening a container
    observed_objects = [
        {"label": "plate", "position": [0.1, 0.1, 0.1]},
        {"label": "cup", "position": [0.2, 0.2, 0.2]}
    ]
    scene_graph = analyzer.update_scene_graph_after_opening(scene_graph, "cabinet_1", observed_objects)
    
    # Example of biasing detection based on predictions
    detection_results = MockDetectionResults()
    biased_results = analyzer.bias_detection("cabinet_1", detection_results.__dict__)
    
    print("\nBiased detection results:")
    for detection in biased_results['detections']:
        print(f"  - {detection['label']}: {detection['confidence']:.2f}")