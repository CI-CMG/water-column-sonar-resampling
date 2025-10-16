# Developing the data tree method seperately before I lose my mind
# This is just a WIP :D

# TODO:
# REMINDER: THIS METHOD IS MEANT TO 1) DETERMINE # OF ZOOM LEVELS AND 2) MAKE A DATA TREE BASED ON THAT
# [X] Actually pull out the time dimension from the zarr store
# [] Determine number of zoom levels based on the time dimension (work in powers of 2)
#      - Use the time dimension as the base dimension
#      - Get the lowest level as close to 4096 as possible (define some kind of acceptable range)
#      - Each zoom level would be half the previous level (use a for or until loop to do this and count the number of levels)
# [] Create a data tree based on the number of zoom levels
#      - Just open empty zarr arrays for each zoom level
#      - Attatch each array to a node in the data tree

import xarray as xr
import s3fs
import json
import numpy as np
import tqdm
import zarr

# Write a new method to get dimensions!!!
def get_dimension(self, variable=None):
    # It's similar to return_shape but only for time dimension/coord
    if self.data_set is None:
        self.open_store() # Opens the store if it hasn't been opened yet
    
    if variable: # Processes a specific variable if one is given
        if variable in self.data_set.coords:
                var_dims = dict(zip(self.data_set[variable].dims, self.data_set[variable].shape))
                return json.dumps({f"{variable}_dimensions": var_dims}, indent=2)
        else:
                return json.dumps({"error": f"Variable '{variable}' not found in dataset"}, indent=2)

    else: # Returns default dimensions of the dataset
            return json.dumps(dict(self.data_set.sizes), indent=2) # Prints the shape of the data
    
    # The actually working method has been added to water_column_resampling.py
    pass

def determine_zoom_levels(time_dim, base_size=4096):
    zoom_levels = 0
    current_size = time_dim

    while current_size > base_size:
        current_size = current_size // 2
        zoom_levels += 1

    return zoom_levels, current_size