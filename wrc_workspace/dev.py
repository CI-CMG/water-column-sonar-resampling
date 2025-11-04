import xarray as xr

# Open the DataTree
tree = xr.open_datatree('empty_tree.zarr', engine='zarr')

# Access individual levels
level_0 = tree['level_0'].dataset
level_1 = tree['level_1'].dataset

# Printing out the information for each level
print(level_0)
print('\n')
print(level_1)