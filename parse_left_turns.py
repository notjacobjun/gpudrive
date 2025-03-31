import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.utils import transform_utils

def load_scenario(filename):
    """Load a scenario from a TFRecord file."""
    dataset = tf.data.TFRecordDataset(filename, compression_type='')
    for data in dataset:
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(data.numpy())
        yield scenario

def is_unprotected_left_turn(scenario, ego_vehicle_id=None):
    """
    Determine if a scenario contains an unprotected left turn for the ego vehicle.
    
    Args:
        scenario: A Waymo Scenario proto
        ego_vehicle_id: ID of ego vehicle. If None, will use the first vehicle found
    
    Returns:
        Boolean indicating if this is an unprotected left turn and the ego vehicle ID
    """
    # Find ego vehicle if not specified
    if ego_vehicle_id is None and len(scenario.tracks) > 0:
        # Typically, the SDC (self-driving car) is the first track
        ego_vehicle_id = scenario.tracks[0].id
    
    if ego_vehicle_id is None:
        return False, None
    
    # Find the ego vehicle track
    ego_track = None
    for track in scenario.tracks:
        if track.id == ego_vehicle_id:
            ego_track = track
            break
    
    if ego_track is None:
        return False, None
    
    # Check for map features indicating an intersection
    has_intersection = any(
        feature.lane.type == scenario_pb2.MapFeature.Lane.TYPE_INTERSECTION
        for feature in scenario.map_features
    )
    
    if not has_intersection:
        return False, ego_vehicle_id
    
    # Check for a significant left turn in the ego vehicle's trajectory
    # by analyzing heading changes
    headings = []
    for state in ego_track.states:
        if state.valid:
            heading = np.arctan2(state.velocity_y, state.velocity_x)
            headings.append(heading)
    
    if len(headings) < 10:  # Need enough trajectory points
        return False, ego_vehicle_id
    
    # Calculate heading change
    heading_changes = np.diff(np.unwrap(headings))
    
    # A significant left turn would have negative heading changes summing to about -π/2 (-90°)
    # We look for a cumulative change of at least 60 degrees
    significant_left_turn = np.sum(np.minimum(heading_changes, 0)) < -np.pi/3
    
    # Check if there are traffic lights in the scenario
    has_traffic_signals = any(
        dynamic_state.object_type == scenario_pb2.DynamicMapState.TYPE_TRAFFIC_LIGHT
        for map_feature in scenario.map_features 
        for dynamic_state in map_feature.dynamic_map_states
    )
    
    # If it's a significant left turn at an intersection without traffic signals,
    # or with red traffic signals, it's likely an unprotected left turn
    return significant_left_turn and has_intersection, ego_vehicle_id

def extract_unprotected_left_turns(dataset_dir, output_dir, max_scenarios=100):
    """
    Extract unprotected left turn scenarios from the Waymo dataset.
    
    Args:
        dataset_dir: Directory containing TFRecord files
        output_dir: Directory to save extracted scenarios
        max_scenarios: Maximum number of scenarios to extract
    """
    os.makedirs(output_dir, exist_ok=True)
    
    scenario_count = 0
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.tfrecord'):
                file_path = os.path.join(root, file)
                print(f"Processing {file_path}")
                
                for scenario in load_scenario(file_path):
                    is_left_turn, ego_id = is_unprotected_left_turn(scenario)
                    
                    if is_left_turn:
                        # Save the scenario
                        output_path = os.path.join(
                            output_dir, 
                            f"unprotected_left_turn_{scenario_count}.tfrecord"
                        )
                        
                        with tf.io.TFRecordWriter(output_path) as writer:
                            writer.write(scenario.SerializeToString())
                        
                        print(f"Saved unprotected left turn scenario to {output_path}")
                        
                        scenario_count += 1
                        if scenario_count >= max_scenarios:
                            print(f"Reached maximum number of scenarios ({max_scenarios})")
                            return

def visualize_scenario(scenario, ego_vehicle_id=None):
    """Create a simple visualization of the scenario."""
    plt.figure(figsize=(10, 10))
    
    # Find ego vehicle if not specified
    if ego_vehicle_id is None and len(scenario.tracks) > 0:
        ego_vehicle_id = scenario.tracks[0].id
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(scenario.tracks)))
    
    # Plot all vehicle trajectories
    for i, track in enumerate(scenario.tracks):
        if track.object_type == scenario_pb2.Track.TYPE_VEHICLE:
            states = []
            for state in track.states:
                if state.valid:
                    states.append((state.center_x, state.center_y))
            
            if states:
                x, y = zip(*states)
                if track.id == ego_vehicle_id:
                    plt.plot(x, y, 'r-', linewidth=3, label='Ego Vehicle')
                    plt.scatter(x[0], y[0], c='r', s=100, marker='o')  # Starting point
                    plt.scatter(x[-1], y[-1], c='r', s=100, marker='x')  # Ending point
                else:
                    plt.plot(x, y, color=colors[i % len(colors)], linestyle='-', alpha=0.7)
    
    # Plot road features (simplified)
    for feature in scenario.map_features:
        if feature.HasField('lane'):
            lane_points = []
            for polyline in feature.lane.polyline:
                lane_points.append((polyline.x, polyline.y))
            
            if lane_points:
                x, y = zip(*lane_points)
                if feature.lane.type == scenario_pb2.MapFeature.Lane.TYPE_INTERSECTION:
                    plt.plot(x, y, 'k--', alpha=0.5)
                else:
                    plt.plot(x, y, 'k-', alpha=0.3)
    
    plt.axis('equal')
    plt.grid(True)
    plt.title('Scenario Visualization')
    plt.legend()
    
    return plt

# Example usage
if __name__ == "__main__":
    dataset_dir = "/path/to/waymo/dataset"
    output_dir = "/path/to/output/unprotected_left_turns"
    
    # Extract scenarios
    extract_unprotected_left_turns(dataset_dir, output_dir)
    
    # Visualize a sample extracted scenario
    sample_file = os.path.join(output_dir, "unprotected_left_turn_0.tfrecord")
    for scenario in load_scenario(sample_file):
        fig = visualize_scenario(scenario)
        fig.savefig(os.path.join(output_dir, "scenario_visualization.png"))
        break