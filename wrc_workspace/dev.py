# Developing the data tree method seperately before I lose my mind
# This is just a WIP :D

# TODO:
# REMINDER: THIS METHOD IS MEANT TO 1) DETERMINE # OF ZOOM LEVELS AND 2) MAKE A DATA TREE BASED ON THAT
# [X] Actually pull out the time dimension from the zarr store
# [X] Determine number of zoom levels based on the time dimension (work in powers of 2)
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

# Might want to define a range above and below 4096 since it just has to be close, not exact
# Alter as needed when placed inside the class
def determine_zoom_levels(time_dim, base_size=4096):
    zoom_levels = 0
    current_size = time_dim

    while current_size > base_size:
        current_size = current_size // 2
        zoom_levels += 1

    return zoom_levels, current_size
        # zoom levels = # of branches/nodes in the tree

def make_tree(levels):
    tree = xr.DataTree(name="level_0")
    current_node = tree

    # For loop to continuously add levels
    for level in range(1, levels + 1):
        name = f"level_{level}"
        current_node[name] = xr.DataTree(name=name)
        current_node = current_node[name]
        
    for node in tree.subtree:
        print(f"  {node.path}")
    
    return tree

tree = make_tree(4)

# make_tree(4)
# print(xr.__version__)
