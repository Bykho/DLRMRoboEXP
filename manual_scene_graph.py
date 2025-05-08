import sys
import os

# Add the RoboEXP directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Bypass the main roboexp import to avoid loading perception
from roboexp.memory.scene_graph.graph import ActionSceneGraph
from roboexp.memory.instance import myInstance
import numpy as np
import graphviz  # Required for visualization


def create_manual_scene_graph():
    # Create base directory for scene graph visualization
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create a dummy instance for the root (table)
    table_instance = myInstance(
        label="table",
        confidence=1.0,
        voxel_indexes=[],  # Empty since this is just for demonstration
        feature=None,
        index_to_pcd=lambda x: []  # Dummy function since we don't need point cloud conversion
    )
    
    # Initialize the scene graph with table as root
    scene_graph = ActionSceneGraph(
        node_id="table_0",
        instance_label="table",
        instance=table_instance,
        base_dir=base_dir
    )
    
    # Create a cabinet instance
    cabinet_instance = myInstance(
        label="cabinet",
        confidence=1.0,
        voxel_indexes=[],
        feature=None,
        index_to_pcd=lambda x: []
    )
    
    # Add cabinet to the scene graph
    cabinet_node = scene_graph.add_object(
        parent=scene_graph.root,
        node_id="cabinet_0",
        instance_label="cabinet",
        instance=cabinet_instance,
        parent_relation="on"
    )
    
    # Create a handle instance
    handle_instance = myInstance(
        label="handle",
        confidence=1.0,
        voxel_indexes=[],
        feature=None,
        index_to_pcd=lambda x: []
    )
    
    # Add handle to the cabinet
    handle_node = scene_graph.add_object(
        parent=cabinet_node,
        node_id="handle_0",
        instance_label="door_handle",  # Using door_handle type
        instance=handle_instance,
        parent_relation="belong",
        is_part=True
    )
    
    # Add handle attributes
    handle_node.handle_center = np.array([0.75, 0.0, 0.3])
    handle_node.handle_direction = np.array([0, 0, 1])
    handle_node.open_direction = np.array([1, 0, 0])
    handle_node.joint_type = "revolute"
    handle_node.joint_axis = np.array([0, 0, 1])
    handle_node.joint_origin = np.array([0.75, 0.0, 0.3])
    handle_node.side_direction = np.array([0, 1, 0])
    
    # Add an action node for opening the handle
    action_node = scene_graph.add_action(
        parent=handle_node,
        node_id="open_0",
        node_label="open"
    )
    
    # Add an object inside the cabinet
    stapler_instance = myInstance(
        label="stapler",
        confidence=1.0,
        voxel_indexes=[],
        feature=None,
        index_to_pcd=lambda x: []
    )
    
    # Add stapler as a child of the open action (indicating it's revealed after opening)
    stapler_node = scene_graph.add_object(
        parent=action_node,
        node_id="stapler_0",
        instance_label="stapler",
        instance=stapler_instance,
        parent_relation="inside"
    )
    
    # Visualize the scene graph
    scene_graph.visualize()
    
    return scene_graph

if __name__ == "__main__":
    scene_graph = create_manual_scene_graph()
    print("Scene graph created and visualized. Check the scene_graphs directory for the output.") 