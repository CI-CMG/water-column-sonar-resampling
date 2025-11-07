import xarray as xr
from water_column_resampling import water_column_resample
"""
# Open the DataTree
tree = xr.open_datatree('empty_tree.zarr', engine='zarr')

# Access individual levels
level_0 = tree['level_0'].dataset
level_1 = tree['level_1'].dataset

# Printing out the information for each level
print(level_0)
print('\n')
print(level_1)
"""

x = water_column_resample("s3://noaa-wcsd-zarr-pds/level_2/Henry_B._Bigelow/HB0707/EK60/HB0707.zarr", 0.075)
# print(x.get_dimension("time"))
# print(x.determine_zoom_levels())
# print(x.make_tree())
tree = x.resample_tree()
# print(x.new_dataarray())

# Printing out some information about each level to help "visualize" the resampling
for level in range(x.zoom_levels + 1):
    level_name = f'level_{level}'
    ds = tree[level_name].dataset
        
    print(f'\n---{level_name}---')
    print(f'shape: {ds['Sv'].shape}')
    print(f'data (first 10x10):')
    print(ds['Sv'].values[:10, :10])