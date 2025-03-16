# parsedata.py 

import pandas as pd 
import numpy as np
import matplotlib as plot
import json 

# convert NumPy types to Python types 
def convert_numpy(obj):
    if isinstance(obj, np.ndarray):  
        return obj.tolist()  # Convert NumPy arrays to lists
    elif isinstance(obj, np.integer):  
        return int(obj)  # Convert NumPy int to Python int
    elif isinstance(obj, np.floating):  
        return float(obj)  # Convert NumPy float to Python float
    else:
        return obj  

# .csv file name in same folder 
filename = "data/RECONSTRUCTED trajectories-400-0415_NO MOTORCYCLES" + ".csv"
data = pd.read_csv(filename); 

# List of all the individial cars 
unique_vehicle_ids = data["Vehicle_ID"].unique()

# Initialize dictionary 
velocityProfiles = {}

# Store frame_id and velocity data from each vehicle_id
for vehicle_id in unique_vehicle_ids:
    vehicle_id_str = str(vehicle_id)
    vehicle_data = data[data["Vehicle_ID"] == vehicle_id].sort_values(by="Frame_ID")
    
    # Store both velocity and acceleration as NumPy arrays
    if vehicle_data["Mean_Speed"].size > 180 and vehicle_data["Mean_Speed"].min() >= 3: 
        velocityProfiles[vehicle_id_str] = {
            "frame_id": vehicle_data["Frame_ID"].to_numpy(),
            "velocity": vehicle_data["Mean_Speed"].to_numpy(),
        }

# Test dictionary 
example_id = str(515)

if example_id in velocityProfiles:
    print(f"Frame ID for Vehicle {example_id}: {velocityProfiles[example_id]['frame_id']}")
    print(f"Velocity profile for Vehicle {example_id}: {velocityProfiles[example_id]['velocity']}")
else:
    print(f"Vehicle {example_id} not found.")

# Take each value in the dictionary and convert them from NumPy to Python types 
velocityProfiles = {key: {subkey: convert_numpy(value) for subkey, value in subdict.items()} 
                    for key, subdict in velocityProfiles.items()}

# Save dictionary as .json file 
with open("data/velocityProfiles.json","w") as f: 
    json.dump(velocityProfiles,f,indent=4)

# Test .json file 
with open("data/velocityProfiles.json","r") as f: 
    newVelocityProfiles = json.load(f) 

example_id = str(15); 

if example_id in newVelocityProfiles:
    print(f"Frame ID for Vehicle {example_id}: {velocityProfiles[example_id]['frame_id']}")
else: 
    print(f"Vehicle {example_id} not found.")