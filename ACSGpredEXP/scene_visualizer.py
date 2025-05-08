#scene_visualizer.py

import os
import graphviz
from typing import List, Dict, Tuple, Any, Optional
import sys
import importlib.util

# Check if the ActionSceneGraph module exists in the path
try:
    from RoboEXP.roboexp.memory.scene_graph.graph import ActionSceneGraph
except ImportError:
    # Use our own implementation of DummyActionSceneGraph
    class DummyInstance:
        def __init__(self, label):
            self.label = label
            self.instance_id = f"{label}_instance"
            self.voxel_indexes = []  # Empty for visualization purposes
            self.deleted = False

    class DummyNode:
        def __init__(self, parent, node_id, node_label):
            self.parent = parent
            self.node_id = node_id
            self.node_label = node_label
            self.children = {}
            self.actions = {}
            self.expected_contents = []
        
        def add_child(self, child):
            self.children[child.node_id] = child

    class DummyObjectNode(DummyNode):
        def __init__(self, parent, node_id, node_label, instance, parent_relation=None, is_part=False):
            super().__init__(parent, node_id, node_label)
            self.instance = instance
            self.parent_relation = parent_relation
            self.is_part = is_part
            self.explored = False
            
        def is_object(self):
            return True
            
        def is_action(self):
            return False

    class DummyActionNode(DummyNode):
        def __init__(self, parent, node_id, node_label, preconditions=None):
            super().__init__(parent, node_id, node_label)
            self.preconditions = preconditions or []
            
        def is_object(self):
            return False
            
        def is_action(self):
            return True

    class EnhancedActionSceneGraph:
        """Implementation of ActionSceneGraph that supports expected_inside relations"""
        def __init__(self, node_id, instance_label, instance, base_dir):
            self.root = DummyObjectNode(None, node_id, instance_label, instance)
            self.object_nodes = {node_id: self.root}
            self.base_dir = base_dir
            os.makedirs(os.path.join(base_dir, "scene_graphs"), exist_ok=True)
            
        def add_object(self, parent, node_id, instance_label, instance, parent_relation, is_part=False):
            node = DummyObjectNode(
                parent,
                node_id,
                instance_label,
                instance,
                parent_relation=parent_relation,
                is_part=is_part,
            )
            parent.add_child(node)
            self.object_nodes[node_id] = node
            return node
            
        def add_action(self, parent, node_id, node_label, preconditions=None):
            node = DummyActionNode(parent, node_id, node_label, preconditions or [])
            parent.add_action(node)
            return node
            
        def add_expected_object(self, container_node, expected_label, confidence):
            """Add an expected object to the container node"""
            if not hasattr(container_node, 'expected_contents'):
                container_node.expected_contents = []
                
            container_node.expected_contents.append({
                'label': expected_label,
                'confidence': confidence
            })
            
        def visualize(self):
            """Visualize the scene graph with expected_inside relations"""
            dag = graphviz.Digraph(
                directory=f"{self.base_dir}/scene_graphs", 
                filename="scene_graph",
                format="png"
            )
            
            # Add root node
            dag.node(
                self.root.node_id,
                label=self.root.node_label,
                shape="egg",
                color="lightblue2",
                style="filled",
            )
            
            # Process normal nodes
            queue = [self.root]
            while queue:
                node = queue.pop(0)
                
                # Process regular children
                for child in list(node.children.values()) + list(node.actions.values()):
                    if child.is_object():
                        if getattr(child, 'is_part', False):
                            color = "lightpink"
                        else:
                            color = "lightblue2"
                        shape = "egg"
                    else:
                        color = "lightsalmon"
                        shape = "diamond"
                        
                    dag.node(
                        child.node_id,
                        label=child.node_label,
                        shape=shape,
                        color=color,
                        style="filled",
                    )
                    
                    if child.is_object():
                        dag.edge(node.node_id, child.node_id, label=child.parent_relation)
                    else:
                        dag.edge(node.node_id, child.node_id)
                        for precondition in child.preconditions:
                            dag.edge(
                                precondition.node_id,
                                child.node_id,
                            )
                    queue.append(child)
            
            # Process expected objects
            for node_id, node in self.object_nodes.items():
                if hasattr(node, 'expected_contents') and node.expected_contents:
                    for i, expected in enumerate(node.expected_contents):
                        expected_id = f"{node_id}_expected_{i}"
                        dag.node(
                            expected_id,
                            label=f"{expected['label']} ({expected['confidence']:.2f})",
                            shape="egg",
                            color="lightyellow",
                            style="dotted,filled",
                        )
                        dag.edge(
                            node_id,
                            expected_id,
                            label="expected_inside",
                            style="dotted",
                        )
            
            # Add a legend
            with dag.subgraph(name="cluster_legend") as legend:
                legend.attr(label="Legend", style="filled", fillcolor="white")
                legend.node("l_scene", "Scene", shape="egg", style="filled", fillcolor="lightblue2")
                legend.node("l_container", "Container", shape="egg", style="filled", fillcolor="lightblue2")
                legend.node("l_part", "Part", shape="egg", style="filled", fillcolor="lightpink")
                legend.node("l_action", "Action", shape="diamond", style="filled", fillcolor="lightsalmon")
                legend.node("l_expected", "Expected Content", shape="egg", style="dotted,filled", fillcolor="lightyellow")
            
            # Render and return the path
            path = dag.render(cleanup=True)
            print(f"Scene graph visualization saved to {path}")
            return path


class SceneVisualizer:
    """
    Class to visualize a scene with predicted container contents using ActionSceneGraph.
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
            
        # Create scene_graphs subdirectory
        if not os.path.exists(os.path.join(self.output_dir, "scene_graphs")):
            os.makedirs(os.path.join(self.output_dir, "scene_graphs"))
    
    def visualize_scene(
        self, 
        scene_name: str, 
        scene_objects: List[str], 
        container_type: str, 
        predicted_contents: List[Tuple[str, float]]
    ) -> str:
        """
        Create a visual graph of the scene and predictions using ActionSceneGraph.
        
        Args:
            scene_name: Name of the scene (e.g., "kitchen", "office")
            scene_objects: List of objects in the scene
            container_type: Type of container
            predicted_contents: Predicted contents of the container
            
        Returns:
            Path to the rendered graph image
        """
        # Create the enhanced scene graph
        scene_graph = self._create_scene_graph(scene_name, scene_objects, container_type, predicted_contents)
        
        # Visualize the graph
        path = scene_graph.visualize()
        
        return path
    
    def _create_scene_graph(
        self, 
        scene_name: str, 
        scene_objects: List[str], 
        container_type: str, 
        predicted_contents: List[Tuple[str, float]]
    ) -> EnhancedActionSceneGraph:
        """
        Create an ActionSceneGraph with the scene objects and predicted contents.
        
        Args:
            scene_name: Name of the scene
            scene_objects: List of objects in the scene
            container_type: Type of container
            predicted_contents: Predicted contents of the container
            
        Returns:
            Enhanced ActionSceneGraph with predicted contents
        """
        # Create a root instance
        root_instance = DummyInstance(scene_name)
        
        # Create the scene graph
        scene_graph = EnhancedActionSceneGraph(
            node_id=f"{scene_name}_0",
            instance_label=scene_name,
            instance=root_instance,
            base_dir=self.output_dir
        )
        
        # Add scene objects
        object_nodes = {}
        for i, obj in enumerate(scene_objects):
            object_instance = DummyInstance(obj)
            object_node = scene_graph.add_object(
                parent=scene_graph.root,
                node_id=f"{obj}_{i}",
                instance_label=obj,
                instance=object_instance,
                parent_relation="on"
            )
            object_nodes[obj] = object_node
        
        # Add container
        container_instance = DummyInstance(container_type)
        container_node = scene_graph.add_object(
            parent=scene_graph.root,
            node_id=f"{container_type}_0",
            instance_label=container_type,
            instance=container_instance,
            parent_relation="on"
        )
        
        # Add predicted contents to the container
        for label, confidence in predicted_contents:
            scene_graph.add_expected_object(container_node, label, confidence)
        
        return scene_graph

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