import xarray as xr
import s3fs
import json
import numpy as np
import tqdm
import zarr

# Can change method name later on
class water_column_resample:
    def __init__(self, store_link):
        self.store_link = store_link
        self.file_system = s3fs.S3FileSystem(anon=True)
        self.store = None
        self.data_set = None
        self.attributes = None

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

    # Returns default attributes of the dataset
    def return_attributes(self):
        if self.data_set is None:
            self.open_store() # Opens the store if it hasn't been opened yet

        self.attributes = dict(self.data_set.attrs) 
        return json.dumps(self.attributes, indent=2) 
    
    # Returns the default dimensions of the data set, or the dimensions of a specified variable
    def get_dimension(self, variable=None):
        if self.data_set is None:
            self.open_store() # Opens the store if it hasn't been opened yet

        if variable: # Processes a specific variable if one is given
            if variable in self.data_set.data_vars or self.data_set.coords:
                var_dims = dict(zip(self.data_set[variable].dims, self.data_set[variable].shape))
                return var_dims # json.dumps({f"{variable}_dimensions": var_dims}, indent=2)
            else:
                return json.dumps({"error": f"Variable '{variable}' not found in dataset"}, indent=2)

        else: # Returns default dimensions of the dataset
            return json.dumps(dict(self.data_set.sizes), indent=2) # Prints the shape of the data

    # Given the time dimension, determines the number of zoom levels    
    def determine_zoom_levels(self):
        time_dim = self.get_dimension('time')['time']
        zoom_levels = 0

        while time_dim >= 4096: # Can define some kind of acceptable range later
            time_dim = time_dim // 2
            zoom_levels += 1

        return zoom_levels
        
    # Creates a local copy of the sv data (complete sv, depth, time and frequency)
    def copy_sv_data(self):
        if self.data_set is None:
            self.open_store() # Opens the store if it hasn't been opened yet
        
        # This opens the store from the cloud servers
        cloud_store = self.data_set

        # This opens a local zarr store to write to
        local_store = xr.Dataset()
        local_store['Sv'] = xr.DataArray()

        # Pulling the sv data from the cloud store
        sv_data = cloud_store[['Sv']]

        # Writing the sv data to the local store (copies the following data arrays: Sv, frequency, time, depth)
        local_store = sv_data.to_zarr('local_sv_data.zarr', mode='w', compute=True, zarr_format=2)

    # Makes an empty datatree based on the number of zoom levels
    def make_tree(self):
        levels = self.determine_zoom_levels()
        level_0 = self.new_dataarray() # This calls the new_dataarray method to create level 0 (base data)

        tree = xr.DataTree(name='root')
        tree['level_0'] = xr.DataTree(name='level_0', dataset=level_0)

        # For loop to continuously add levels
        for level in range(1, levels + 1):
            name = f"level_{level}"
            tree[name] = xr.DataTree(name=name)
        
        return tree

    def resample_tree(self):
        if self.data_set is None:
            self.open_store() # Opens the store if it hasn't been opened yet
        
        tree = self.make_tree()
        root_ds = self.data_set
        current_ds = root_ds
        zoom_levels = self.determine_zoom_levels()

        for level in range(1, zoom_levels + 1):
            
            # TODO: make this resample lol
            # - open level_0 with new_datarray
            # - use corsen method to downsample by 2x on the previous level

            name = level

            masked = current_ds['Sv'].where(current_ds['Sv'] != 0)

            # Coarsen while skipping NaNs (formerly 0s)
            downsampled_Sv = masked.coarsen(time=2, boundary='trim').mean(skipna=True)

            # Fill NaNs back with 0 after averaging
            downsampled_Sv = downsampled_Sv.fillna(0).astype('int8')

            resampled_ds = xr.Dataset({'Sv': downsampled_Sv})

            tree[f'level_{name}'] = xr.DataTree(name=f'level_{name}', dataset=resampled_ds)
            current_ds = resampled_ds

        return tree

    # Creates a new dataarray with just depth and time-- copies it locally   
    def new_dataarray(self):
        if self.data_set is None:
            self.open_store() # Opens the store if it hasn't been opened yet
        
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
    x = water_column_resample("s3://noaa-wcsd-zarr-pds/level_2/Henry_B._Bigelow/HB0707/EK60/HB0707.zarr")
    print(x.get_dimension("time"))
    print(x.determine_zoom_levels())
    print(x.make_tree())
    # print(x.resample_tree())
    # print(x.new_dataarray())
