import json
import matplotlib.pyplot as plt
import os
from pathlib import Path
import seaborn as sns
import pandas as pd

from gpudrive.env.dataset import SceneDataLoader
from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
# from waymo_open_dataset.protos import scenario_pb2
# from waymo_open_dataset.utils import transform_utils

DATASET_PATH = "data/GPUDrive_mini/testing" 

# def is_unprotected_left_turn(scenario, ego_vehicle_id=None):
#     """
#     Determine if a scenario contains an unprotected left turn for the ego vehicle.
    
#     Args:
#         scenario: A Waymo Scenario proto
#         ego_vehicle_id: ID of ego vehicle. If None, will use the first vehicle found
    
#     Returns:
#         Boolean indicating if this is an unprotected left turn and the ego vehicle ID
#     """
#     # Find ego vehicle if not specified
#     if ego_vehicle_id is None and len(scenario.tracks) > 0:
#         # Typically, the SDC (self-driving car) is the first track
#         ego_vehicle_id = scenario.tracks[0].id
    
#     if ego_vehicle_id is None:
#         return False, None
    
#     # Find the ego vehicle track
#     ego_track = None
#     for track in scenario.tracks:
#         if track.id == ego_vehicle_id:
#             ego_track = track
#             break
    
#     if ego_track is None:
#         return False, None
    
#     # Check for map features indicating an intersection
#     has_intersection = any(
#         feature.lane.type == scenario_pb2.MapFeature.Lane.TYPE_INTERSECTION
#         for feature in scenario.map_features
#     )
    
#     if not has_intersection:
#         return False, ego_vehicle_id
    
#     # Check for a significant left turn in the ego vehicle's trajectory
#     # by analyzing heading changes
#     headings = []
#     for state in ego_track.states:
#         if state.valid:
#             heading = np.arctan2(state.velocity_y, state.velocity_x)
#             headings.append(heading)
    
#     if len(headings) < 10:  # Need enough trajectory points
#         return False, ego_vehicle_id
    
#     # Calculate heading change
#     heading_changes = np.diff(np.unwrap(headings))
    
#     # A significant left turn would have negative heading changes summing to about -π/2 (-90°)
#     # We look for a cumulative change of at least 60 degrees
#     significant_left_turn = np.sum(np.minimum(heading_changes, 0)) < -np.pi/3
    
#     # Check if there are traffic lights in the scenario
#     has_traffic_signals = any(
#         dynamic_state.object_type == scenario_pb2.DynamicMapState.TYPE_TRAFFIC_LIGHT
#         for map_feature in scenario.map_features 
#         for dynamic_state in map_feature.dynamic_map_states
#     )
    
#     # If it's a significant left turn at an intersection without traffic signals,
#     # or with red traffic signals, it's likely an unprotected left turn
#     return significant_left_turn and has_intersection, ego_vehicle_id

# Set working directory to the base directory 'gpudrive'
working_dir = Path.cwd()
while working_dir.name != 'gpudrive':
    working_dir = working_dir.parent
    if working_dir == Path.home():
        raise FileNotFoundError("Base directory 'gpudrive' not found")
os.chdir(working_dir)

cmap = ["r", "g", "b", "y", "c"]
sns.set("notebook", font_scale=1.1, rc={"figure.figsize": (8, 3)})
sns.set_style("ticks", rc={"figure.facecolor": "none", "axes.facecolor": "none"})


data_loader = SceneDataLoader(
    root=DATASET_PATH, # Path to the dataset
    batch_size=50, # Batch size, you want this to be equal to the number of worlds (envs) so that every world receives a different scene
    dataset_size=400, # Total number of different scenes we want to use
    sample_with_replacement=True,
    seed=99,
    shuffle=True,
)

# Test if we are loading the dataset properly
print("Testing if we loaded the dataset properly", data_loader.dataset)

# Pass the data_loader to the environment 
env = GPUDriveTorchEnv(
    config=EnvConfig(),
    data_loader=data_loader,
    max_cont_agents=64,  # Maximum number of agents to control per scene
)

# Take an example scene
data_path = data_loader.dataset[0]

with open(data_path) as file:
    traffic_scene = json.load(file)

print("traffic keys: ", traffic_scene.keys())

dataset = data_loader.dataset
for scenario_path in dataset:
    with open(scenario_path) as file:
        traffic_scene = json.load(file)
    print("traffic light states: ", traffic_scene["tl_states"])