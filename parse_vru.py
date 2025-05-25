import json
import os
from gpudrive.env.dataset import SceneDataLoader
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.utils import transform_utils
import math
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set

class InteractionType(Enum):
    CROSSING = 1
    PARALLEL_MOVEMENT = 2
    WAITING_PEDESTRIAN = 3
    CYCLIST_OVERTAKING = 4
    CLOSE_PROXIMITY = 5

@dataclass
class VulnerableRoadUser:
    track_id: int
    object_type: str  # "PEDESTRIAN" or "CYCLIST"
    min_distance: float
    interaction_type: InteractionType
    crossing_angle: Optional[float] = None
    time_to_collision: Optional[float] = None

def load_scenario(filename):
    """Load a scenario from a TFRecord file."""
    dataset = tf.data.TFRecordDataset(filename, compression_type='')
    for data in dataset:
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(data.numpy())
        yield scenario

def compute_distance(x1, y1, x2, y2):
    """Compute Euclidean distance between two points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def compute_heading(vx, vy):
    """Compute heading from velocity components."""
    return math.atan2(vy, vx)

def angular_difference(angle1, angle2):
    """Compute the minimal angular difference between two angles."""
    diff = (angle1 - angle2) % (2 * math.pi)
    if diff > math.pi:
        diff = 2 * math.pi - diff
    return diff

def estimate_ttc(ego_x, ego_y, ego_vx, ego_vy, vru_x, vru_y, vru_vx, vru_vy):
    """
    Estimate time-to-collision (TTC) between ego vehicle and VRU (Vulnerable Road User).
    Returns infinity if trajectories don't intersect.
    
    Parameters:
    - ego_x, ego_y: Position of ego vehicle
    - ego_vx, ego_vy: Velocity of ego vehicle
    - vru_x, vru_y: Position of VRU
    - vru_vx, vru_vy: Velocity of VRU
    """
    # Vector from ego to VRU
    dx = vru_x - ego_x
    dy = vru_y - ego_y
    
    # Relative velocity
    dvx = vru_vx - ego_vx
    dvy = vru_vy - ego_vy
    
    # Check if relative speed is near zero
    rel_speed_sq = dvx**2 + dvy**2
    if rel_speed_sq < 1e-6:  # Threshold for near-zero relative speed
        return float('inf')
    
    # Compute closest approach time
    # This is when the distance between the two objects will be minimized
    t = -(dx*dvx + dy*dvy) / rel_speed_sq
    
    # If t is negative, closest approach is in the past (i.e. their distance is increasing)
    if t < 0:
        return float('inf')
    
    # Calculate minimum distance at closest approach
    closest_x = dx + dvx * t
    closest_y = dy + dvy * t
    min_dist_sq = closest_x**2 + closest_y**2
    
    # Define collision threshold (this depends on vehicle and VRU sizes)
    # TODO Using a placeholder value - should be calibrated to actual sizes
    collision_threshold_sq = 4.0  # e.g., 2m radius squared
    
    # Check if collision occurs
    if min_dist_sq <= collision_threshold_sq:
        return t
    else:
        return float('inf')

def is_vru_scenario(scenario):
    """
    Detect whether the scenario passed in is a VRU scenario or not
    
    Args:
        scenario: A GPU drive processed scenario
        ego_vehicle_id: ID of ego vehicle. If None, will use the first vehicle found
    
    Returns:
        Boolean indicating whether this scenario contains a VRU (vulerable road user) interaction or not
    """
    return len(detect_processed_vru_interactions(scenario).items()) > 0
    

def detect_processed_vru_interactions(scenario, ego_vehicle_id=None, distance_threshold=15.0, ttc_threshold=5.0):
    """
    Detect interactions between ego vehicle and vulnerable road users (pedestrians/cyclists) for GPUDrive processed scenarios.
    We are doing this b/c the simulator can only use the processed scenarios.
    
    Args:
        scenario: A GPU drive processed scenario
        ego_vehicle_id: ID of ego vehicle. If None, will use the first vehicle found
        distance_threshold: Maximum distance to consider an interaction (meters)
        ttc_threshold: TTC threshold for identifying potential collisions (seconds)
    
    Returns:
        Dictionary mapping timesteps to lists of VRU interactions
    """
    # Find ego vehicle since it should be specified in the metadata
    if ego_vehicle_id is None:
        if 'sdc_track_index' not in scenario['metadata']:
            print("There is no ego vehicle ID in this scenario")
            return {}

        ego_vehicle_id = scenario['metadata']['sdc_track_index']
    
    # Find the ego vehicle track
    ego_track = scenario['objects'][ego_vehicle_id]
    
    if ego_track is None:
        return {}
    
    # Find all pedestrian and cyclist (Vulnerable road users) tracks
    vru_tracks = []
    for track in scenario['objects']:
        if track['type'] in ['vehicle', 'cyclist']:
            vru_tracks.append(track)
    
    if not vru_tracks:
        print("Didn't find any VRU in this scenario")
        return {}
    
    # For each timestep, compute interactions
    timestep_interactions = {}
    total_timesteps = len(ego_track['position'])
    
    for t in range(total_timesteps):
        if not ego_track['valid'][t]:
            continue
        
        ego_x, ego_y = ego_track['position'][t]['x'], ego_track['position'][t]['y']
        ego_vx, ego_vy = ego_track['velocity'][t]['x'], ego_track['velocity'][t]['y']
        ego_heading = ego_track['heading'][t]
        
        interactions = []
        
        for vru_track in vru_tracks:
            if not vru_track['valid'][t]:
                continue
            
            vru_x, vru_y = vru_track['position'][t]['x'], vru_track['position'][t]['y']
            vru_vx, vru_vy = vru_track['velocity'][t]['x'], vru_track['velocity'][t]['y']
            vru_heading = vru_track['heading'][t]
            
            # TODO double check that the GPU drive env is using meters for the distance measurement or not (or else the doc string above is going to be wrong)
            # Compute distance
            distance = compute_distance(ego_x, ego_y, vru_x, vru_y)
            
            if distance <= distance_threshold:
                vru_speed = math.sqrt(vru_vx**2 + vru_vy**2)
                
                # Detect interaction type
                interaction_type = InteractionType.CLOSE_PROXIMITY
                crossing_angle = None
                
                if vru_speed > 0.2:  # Moving VRU
                    # TODO double check if this angular difference calculation is correct or not because this will be hard to debug later on
                    heading_diff = angular_difference(ego_heading, vru_heading)
                    
                    # Crossing path (roughly perpendicular)
                    if 1.0 < heading_diff < 2.0:  # ~60-120 degrees
                        interaction_type = InteractionType.CROSSING
                        crossing_angle = heading_diff
                    # Parallel movement (roughly same or opposite direction)
                    elif heading_diff < 0.5 or heading_diff > 2.6:  # <30 degrees or >150 degrees
                        # TODO reconsider this entire interaction logic because seems too naive
                        if vru_track['type'] == 'cyclist':
                            interaction_type = InteractionType.CYCLIST_OVERTAKING
                        else:
                            interaction_type = InteractionType.PARALLEL_MOVEMENT
                    else:
                        interaction_type = InteractionType.CLOSE_PROXIMITY
                else:
                    # Stationary VRU
                    # TODO get what the type would be from the GPU drive env for pedestrians
                    if vru_track['type'] == 'pedestrian':
                        interaction_type = InteractionType.WAITING_PEDESTRIAN
                
                # Compute time-to-collision
                time_to_collision = estimate_ttc(ego_x, ego_y, ego_vx, ego_vy, vru_x, vru_y, vru_vx, vru_vy)

                # Create VRU interaction object
                vru_interaction = VulnerableRoadUser(
                    track_id=vru_track.id,
                    # TODO double check if the interaction type logic is correct or not
                    object_type=interaction_type,
                    min_distance=distance,
                    interaction_type=interaction_type,
                    crossing_angle=crossing_angle,
                    time_to_collision=time_to_collision
                )
                
                interactions.append(vru_interaction)
        
        if interactions:
            timestep_interactions[t] = interactions
    
    return timestep_interactions

def detect_vru_interactions(scenario, ego_vehicle_id=None, distance_threshold=15.0, ttc_threshold=5.0):
    """
    Detect interactions between ego vehicle and vulnerable road users (pedestrians/cyclists).
    
    Args:
        scenario: A Waymo Scenario proto
        ego_vehicle_id: ID of ego vehicle. If None, will use the first vehicle found
        distance_threshold: Maximum distance to consider an interaction (meters)
        ttc_threshold: TTC threshold for identifying potential collisions (seconds)
    
    Returns:
        Dictionary mapping timesteps to lists of VRU interactions
    """
    # Find ego vehicle if not specified
    if ego_vehicle_id is None and len(scenario.tracks) > 0:
        # Typically, the SDC (self-driving car) is the first track
        ego_vehicle_id = scenario.tracks[0].id
    
    # Find the ego vehicle track
    ego_track = None
    for track in scenario.tracks:
        if track.id == ego_vehicle_id:
            ego_track = track
            break
    
    if ego_track is None:
        return {}
    
    # Find all pedestrian and cyclist tracks
    vru_tracks = []
    for track in scenario.tracks:
        if track.object_type in [scenario_pb2.Track.TYPE_PEDESTRIAN, scenario_pb2.Track.TYPE_CYCLIST]:
            vru_tracks.append(track)
    
    if not vru_tracks:
        return {}
    
    # For each timestep, compute interactions
    timestep_interactions = {}
    
    for t in range(len(ego_track.states)):
        if not ego_track.states[t].valid:
            continue
        
        ego_state = ego_track.states[t]
        ego_x, ego_y = ego_state.center_x, ego_state.center_y
        ego_vx, ego_vy = ego_state.velocity_x, ego_state.velocity_y
        ego_heading = compute_heading(ego_vx, ego_vy)
        
        interactions = []
        
        for vru_track in vru_tracks:
            if t >= len(vru_track.states) or not vru_track.states[t].valid:
                continue
            
            vru_state = vru_track.states[t]
            vru_x, vru_y = vru_state.center_x, vru_state.center_y
            vru_vx, vru_vy = vru_state.velocity_x, vru_state.velocity_y
            
            # Compute distance
            distance = compute_distance(ego_x, ego_y, vru_x, vru_y)
            
            if distance <= distance_threshold:
                # Determine VRU heading
                vru_speed = math.sqrt(vru_vx**2 + vru_vy**2)
                
                # Detect interaction type
                interaction_type = InteractionType.CLOSE_PROXIMITY
                crossing_angle = None
                
                if vru_speed > 0.2:  # Moving VRU
                    vru_heading = compute_heading(vru_vx, vru_vy)
                    heading_diff = angular_difference(ego_heading, vru_heading)
                    
                    # Crossing path (roughly perpendicular)
                    if 1.0 < heading_diff < 2.0:  # ~60-120 degrees
                        interaction_type = InteractionType.CROSSING
                        crossing_angle = heading_diff
                    # Parallel movement (roughly same or opposite direction)
                    elif heading_diff < 0.5 or heading_diff > 2.6:  # <30 degrees or >150 degrees
                        if vru_track.object_type == scenario_pb2.Track.TYPE_CYCLIST:
                            interaction_type = InteractionType.CYCLIST_OVERTAKING
                        else:
                            interaction_type = InteractionType.PARALLEL_MOVEMENT
                    else:
                        interaction_type = InteractionType.CLOSE_PROXIMITY
                else:
                    # Stationary VRU
                    if vru_track.object_type == scenario_pb2.Track.TYPE_PEDESTRIAN:
                        interaction_type = InteractionType.WAITING_PEDESTRIAN
                
                # Compute time-to-collision
                ttc = estimate_ttc(ego_x, ego_y, ego_vx, ego_vy, vru_x, vru_y, vru_vx, vru_vy)
                
                # Only set time_to_collision if it's below threshold
                time_to_collision = ttc if ttc < ttc_threshold else None
                
                # Create VRU interaction object
                vru_interaction = VulnerableRoadUser(
                    track_id=vru_track.id,
                    object_type="PEDESTRIAN" if vru_track.object_type == scenario_pb2.Track.TYPE_PEDESTRIAN else "CYCLIST",
                    min_distance=distance,
                    interaction_type=interaction_type,
                    crossing_angle=crossing_angle,
                    time_to_collision=time_to_collision
                )
                
                interactions.append(vru_interaction)
        
        if interactions:
            timestep_interactions[t] = interactions
    
    return timestep_interactions

def extract_vru_interaction_scenarios(dataset_dir, output_dir, min_interaction_duration=5, max_scenarios=500):
    """
    Extract scenarios with significant pedestrian/cyclist interactions.
    
    Args:
        dataset_dir: Directory containing TFRecord files
        output_dir: Directory to save extracted scenarios
        min_interaction_duration: Minimum number of timesteps with VRU interactions
        max_scenarios: Maximum number of scenarios to extract
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create separate directories for different interaction types
    for interaction_type in InteractionType:
        os.makedirs(os.path.join(output_dir, interaction_type.name.lower()), exist_ok=True)
    
    # Track counts by type
    scenario_counts = {interaction_type: 0 for interaction_type in InteractionType}
    total_count = 0

    # TODO use this data loader once we are using this for training in the env
    data_loader = SceneDataLoader(
        # TODO replace this with the dataset_dir parameter from above
        root="data/processed/examples", # Path to the dataset
        batch_size=10, # Batch size, you want this to be equal to the number of worlds (envs) so that every world receives a different scene
        dataset_size=50, # Total number of different scenes we want to use
        sample_with_replacement=True,
        shuffle=True,
    )

    for root, _, files in os.walk(dataset_dir):
        for file in files:
            # using json file type because we are using the processed scenarios from GPU drive
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                print(f"Processing {file_path}")
                
                with open(file_path) as file:
                    scenario = json.load(file)
                    # Detect VRU interactions
                    timestep_interactions = detect_vru_interactions(scenario)
                    
                    if not timestep_interactions:
                        continue
                    
                    # Count consecutive frames with interactions
                    interaction_durations = {}
                    current_durations = {interaction_type: 0 for interaction_type in InteractionType}
                    
                    for t in sorted(timestep_interactions.keys()):
                        # Reset counters if there's a gap
                        if t > 0 and t-1 not in timestep_interactions:
                            for itype in InteractionType:
                                if current_durations[itype] > 0:
                                    if itype not in interaction_durations:
                                        interaction_durations[itype] = 0
                                    interaction_durations[itype] = max(interaction_durations[itype], current_durations[itype])
                                current_durations[itype] = 0
                        
                        # Count by interaction type
                        for vru in timestep_interactions[t]:
                            current_durations[vru.interaction_type] += 1
                    
                    # Final check after all timesteps
                    for itype in InteractionType:
                        if current_durations[itype] > 0:
                            if itype not in interaction_durations:
                                interaction_durations[itype] = 0
                            interaction_durations[itype] = max(interaction_durations[itype], current_durations[itype])
                    
                    # Find the dominant interaction type
                    dominant_type = None
                    max_duration = 0
                    
                    for itype, duration in interaction_durations.items():
                        if duration > max_duration:
                            max_duration = duration
                            dominant_type = itype
                    
                    # Skip if no significant interaction
                    if dominant_type is None or max_duration < min_interaction_duration:
                        continue
                    
                    scenario_counts[dominant_type] += 1
                    total_count += 1
                    
                    if total_count >= max_scenarios:
                        print(f"Reached maximum number of scenarios ({max_scenarios})")
                        return

# TODO continue the migration from here
def visualize_vru_scenario(scenario, timestep=None):
    """Create a visualization of the VRU interaction scenario at a specific timestep."""
    plt.figure(figsize=(12, 12))
    
    # Default to a timestep in the middle of the scenario if not specified
    if timestep is None:
        max_states = max(len(track.states) for track in scenario.tracks)
        timestep = max_states // 2
    
    # Find the ego vehicle (typically the first track)
    ego_track = scenario.tracks[0]
    ego_vehicle_id = ego_track.id
    
    # Separate tracks by type
    vehicle_tracks = []
    pedestrian_tracks = []
    cyclist_tracks = []
    
    for track in scenario.tracks:
        if track.object_type == scenario_pb2.Track.TYPE_VEHICLE:
            vehicle_tracks.append(track)
        elif track.object_type == scenario_pb2.Track.TYPE_PEDESTRIAN:
            pedestrian_tracks.append(track)
        elif track.object_type == scenario_pb2.Track.TYPE_CYCLIST:
            cyclist_tracks.append(track)
    
    # Plot road features
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
        
        elif feature.HasField('crosswalk'):
            crosswalk_points = []
            for polyline in feature.crosswalk.polygon:
                crosswalk_points.append((polyline.x, polyline.y))
            
            if crosswalk_points:
                x, y = zip(*crosswalk_points)
                plt.fill(x, y, color='lightgray', alpha=0.5)
    
    # Plot vehicle positions and trajectories
    for track in vehicle_tracks:
        if timestep < len(track.states) and track.states[timestep].valid:
            state = track.states[timestep]
            
            # Plot vehicle position as rectangle
            if track.id == ego_vehicle_id:
                rect_color = 'red'
                label = 'Ego Vehicle'
            else:
                rect_color = 'blue'
                label = 'Other Vehicle'
            
            # Get vehicle dimensions
            length = track.metadata.dimensions.length
            width = track.metadata.dimensions.width
            
            # Create rectangle at vehicle position and orientation
            heading = math.atan2(state.velocity_y, state.velocity_x) if abs(state.velocity_x) > 0.01 or abs(state.velocity_y) > 0.01 else 0
            
            # Create a rectangle centered at the vehicle position
            rect = plt.Rectangle(
                (state.center_x - length/2, state.center_y - width/2),
                length, width,
                angle=math.degrees(heading),
                color=rect_color, alpha=0.7,
                label=label if track.id == ego_vehicle_id else None
            )
            plt.gca().add_patch(rect)
            
            # Plot trajectory for ego vehicle
            if track.id == ego_vehicle_id:
                traj_x = []
                traj_y = []
                for t_state in track.states[:timestep+1]:
                    if t_state.valid:
                        traj_x.append(t_state.center_x)
                        traj_y.append(t_state.center_y)
                plt.plot(traj_x, traj_y, 'r-', alpha=0.5)
    
    # Plot pedestrians
    for track in pedestrian_tracks:
        if timestep < len(track.states) and track.states[timestep].valid:
            state = track.states[timestep]
            plt.scatter(state.center_x, state.center_y, s=50, color='green', marker='o', label='Pedestrian')
            
            # Show recent trajectory
            traj_x = []
            traj_y = []
            for t_state in track.states[max(0, timestep-10):timestep+1]:
                if t_state.valid:
                    traj_x.append(t_state.center_x)
                    traj_y.append(t_state.center_y)
            plt.plot(traj_x, traj_y, 'g-', alpha=0.5)
    
    # Plot cyclists
    for track in cyclist_tracks:
        if timestep < len(track.states) and track.states[timestep].valid:
            state = track.states[timestep]
            plt.scatter(state.center_x, state.center_y, s=80, color='purple', marker='^', label='Cyclist')
            
            # Show recent trajectory
            traj_x = []
            traj_y = []
            for t_state in track.states[max(0, timestep-10):timestep+1]:
                if t_state.valid:
                    traj_x.append(t_state.center_x)
                    traj_y.append(t_state.center_y)
            plt.plot(traj_x, traj_y, 'm-', alpha=0.5)
    
    # Detect and annotate interactions at this timestep
    interactions = detect_vru_interactions(scenario, ego_vehicle_id)[timestep] if timestep in detect_vru_interactions(scenario, ego_vehicle_id) else []
    
    # Add annotations for interactions
    ego_state = ego_track.states[timestep] if timestep < len(ego_track.states) else None
    
    if ego_state and ego_state.valid:
        ego_x, ego_y = ego_state.center_x, ego_state.center_y
        
        for interaction in interactions:
            # Find the VRU track
            vru_track = None
            for track in scenario.tracks:
                if track.id == interaction.track_id:
                    vru_track = track
                    break
            
            if vru_track and timestep < len(vru_track.states) and vru_track.states[timestep].valid:
                vru_state = vru_track.states[timestep]
                vru_x, vru_y = vru_state.center_x, vru_state.center_y
                
                # Draw line connecting ego and VRU
                plt.plot([ego_x, vru_x], [ego_y, vru_y], 'y--', alpha=0.7)
                
                # Annotate interaction type
                mid_x = (ego_x + vru_x) / 2
                mid_y = (ego_y + vru_y) / 2
                plt.annotate(
                    f"{interaction.interaction_type.name}\nDist: {interaction.min_distance:.1f}m", 
                    (mid_x, mid_y),
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5)
                )
                
                # Annotate TTC if available
                if interaction.time_to_collision:
                    plt.annotate(
                        f"TTC: {interaction.time_to_collision:.1f}s", 
                        (mid_x, mid_y - 2),
                        fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3", fc="orange", alpha=0.5)
                    )
    
    plt.axis('equal')
    plt.grid(True)
    plt.title(f'VRU Interaction Scenario (Timestep {timestep})')
    
    # Remove duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    return plt

# Example usage
if __name__ == "__main__":
    dataset_dir = "/path/to/waymo/dataset"
    output_dir = "/path/to/output/vru_interactions"
    
    # Extract scenarios
    extract_vru_interaction_scenarios(dataset_dir, output_dir)
    
    # Visualize a sample extracted scenario (e.g., a crossing scenario)
    sample_file = os.path.join(output_dir, "crossing/crossing_0.tfrecord")
    for scenario in load_scenario(sample_file):
        # Find a good timestep with interactions
        interactions = detect_vru_interactions(scenario)
        sample_timestep = list(interactions.keys())[len(interactions)//2] if interactions else None
        
        fig = visualize_vru_scenario(scenario, timestep=sample_timestep)
        fig.savefig(os.path.join(output_dir, "vru_interaction_visualization.png"))
        break