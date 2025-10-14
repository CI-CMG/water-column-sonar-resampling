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
        if "s3://" in self.store_link:
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
    def return_shape(self, variable=None):
        if self.data_set is None:
            self.open_store() # Opens the store if it hasn't been opened yet

        if variable: # Processes a specific variable if one is given
            if variable in self.data_set.data_vars:
                var_dims = dict(zip(self.data_set[variable].dims, self.data_set[variable].shape))
                return json.dumps({f"{variable}_dimensions": var_dims}, indent=2)
            else:
                return json.dumps({"error": f"Variable '{variable}' not found in dataset"}, indent=2)

        else: # Returns default dimensions of the dataset
            return json.dumps(dict(self.data_set.sizes), indent=2) # Prints the shape of the data
        
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

    # Creates a new dataarray with just depth and time-- copies it locally   
    def new_dataarray(self, output_path='local_dataarray.zarr'):
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

        local_store.to_zarr(output_path, mode='w', compute=False, zarr_format=2)

        local_store = zarr.open(output_path, mode='a')
        
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

    # TODO: Make it all close cleanly-- later goal
    def close(self):
        pass
"""
# A test to see if it works-- use as needed
if __name__ == "__main__":
    x = water_column_resample("s3://noaa-wcsd-zarr-pds/level_2/Henry_B._Bigelow/HB0707/EK60/HB0707.zarr")
    x.new_dataarray()
"""