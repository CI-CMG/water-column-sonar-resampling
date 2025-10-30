import xarray as xr
import s3fs
import json
import numpy as np
import tqdm
import zarr

# Can change method name later on
class water_column_resample:
    def __init__(self, store_link, fraction):
        self.store_link = store_link
        self.file_system = s3fs.S3FileSystem(anon=True)
        self.store = None
        self.data_set = None
        self.attributes = None
        self.fraction = fraction # This is strictly for testing purposes as it will slice the time dimension to x% of the original

    # Actually opens the zarr store based on the link given
    def open_store(self):
        if "s3://" in str(self.store_link):
            self.data_set = xr.open_dataset(
                self.store_link, 
                engine='zarr',
                chunks='auto',
                storage_options={'anon': True}
                )
        else:
            self.data_set = xr.open_dataset(
                self.store_link, 
                engine='zarr', 
                chunks='auto'
                )
            
        if self.fraction < 1.0:
            max_index = int(len(self.data_set.time) * self.fraction)
            self.data_set = self.data_set.isel(time=slice(0, max_index))
    
    # Returns the default dimensions of the data set, or the dimensions of a specified variable
    def get_dimension(self, dimension=None):
        self.open_store()
        ds = self.data_set

        if dimension in ds.dims:
            return ds.sizes[dimension][0]
            
        else:
            return "Error: Dimension not found in dataset."

    # Given the time dimension, determines the number of zoom levels    
    def determine_zoom_levels(self):
        time_dim = self.get_dimension('time')['time']
        zoom_levels = 0

        while time_dim >= 4096: # Can define some kind of acceptable range later
            time_dim = time_dim // 2
            zoom_levels += 1

        self.zoom_levels = zoom_levels

    # Makes an empty datatree and writes it to the disk
    def make_tree(self):
        # Initializing an empty data tree
        empty_tree = xr.DataTree()

        # The empty tree is written to the disk
        empty_tree.to_zarr("empty_tree.zarr", mode='w')

    def resample_tree(self):
        tree = self.make_tree()
        zoom_levels = self.determine_zoom_levels()

        for level in range(1, zoom_levels + 1):

            # Getting the last level of the tree
            last_level = f'level_{level - 1}'
            last_ds = tree[last_level].dataset

            # Uses the coarsen method to downsample by a factor of 2 along the time dimension
            resampled_data = last_ds.coarsen(time=2, boundary='trim').mean()

            # Assigns the resampled data to the appropriate level in the tree
            tree[f'level_{level}'].dataset = resampled_data

        return tree

    # Creates a new dataarray with just depth and time   
    def new_dataarray(self):
        # This opens the store from the cloud servers
        cloud_store = self.data_set
        masked_store = cloud_store.Sv.where(cloud_store.depth < cloud_store.bottom)

        # Pulling specific data from the cloud store
        depth = masked_store['depth'].values
        time = masked_store['time'].values

        # Initializing the local data array
        dt_array = xr.DataArray(
            data=np.empty((len(depth), len(time)), dtype='int8'),
            dims=('depth', 'time')
        )

        dt_array = dt_array.chunk({'time': 1024, 'depth': 1024})

        # Initializing the local store with the data array in it
        local_store = xr.Dataset(
            data_vars={
                'Sv': dt_array
            }
        )
        
        # Copies the data in 1024 chunks across the time axis (for loops)
        depth_chunk = 1024
        time_chunk = 1024
        for time_start in tqdm.tqdm(range(0, len(time), time_chunk), desc="Processing time chunks"):
            time_end = min(time_start + time_chunk, len(time))
            for depth_start in tqdm.tqdm(range(0, len(depth), depth_chunk), desc="Processing depth chunks", leave=False):
                depth_end = min(depth_start + depth_chunk, len(depth))
                
                # Extract the chunk from the masked_store
                chunk = masked_store.isel(depth=slice(depth_start, depth_end), time=slice(time_start, time_end), frequency=0)

                # Add/Replace all needed zeros
                chunk_clean = np.nan_to_num(chunk.values, nan=0.0, posinf=0.0, neginf=0.0)

                # Recast
                chunk_clean = chunk_clean.astype('int8')
                
                # Assign the chunked data to the corresponding location in the local_store
                local_store['Sv'][depth_start:depth_end, time_start:time_end] = chunk_clean

        return local_store

# A test to see if it works-- use as needed
if __name__ == "__main__":
    x = water_column_resample("s3://noaa-wcsd-zarr-pds/level_2/Henry_B._Bigelow/HB0707/EK60/HB0707.zarr", 1)
    print(x.get_dimension("time"))
    # print(x.determine_zoom_levels())
    # print(x.make_tree())
    # print(x.resample_tree())
    # print(x.new_dataarray())