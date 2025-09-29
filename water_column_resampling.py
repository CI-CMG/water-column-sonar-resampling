import xarray as xr
import s3fs
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

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
        self.store = s3fs.S3Map(root=self.store_link, s3=self.file_system)
        self.data_set = xr.open_zarr(store=self.store, consolidated=True)

    # Returns default attributes of the dataset
    def return_attributes(self):
        if self.store is None:
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
        cloud_store = cloud_store.chunk({'depth': 512, 'time': 512, 'frequency': 1})

        # This opens a local zarr store to write to
        local_store = xr.Dataset()
        local_store['Sv'] = xr.DataArray()

        # Pulling the sv data from the cloud store
        sv_data = cloud_store[['Sv']]
        sv_data = sv_data.chunk({'depth': 512, 'time': 512, 'frequency': 1})

        # Writing the sv data to the local store (copies the following data arrays: Sv, frequency, time, depth)
        local_store = sv_data.to_zarr('local_sv_data.zarr', mode='w', compute=True, zarr_format=2)

    # Creates a new dataarray with just depth and time-- copies is locally   
    def new_dataarray(self):
        if self.data_set is None:
            self.open_store() # Opens the store if it hasn't been opened yet
        
        # This opens the store from the cloud servers
        cloud_store = self.data_set
        cloud_store = cloud_store.chunk({'depth': 512, 'time': 512})

        # Pulling specific data from the cloud store
        depth = cloud_store['depth'].values
        time = cloud_store['time'].values

        # This opens a local zarr store to write to
        local_store = xr.Dataset(
            {
                "Sv": xr.DataArray(
                    data=np.empty((len(depth), len(time)), dtype='int16'),
                    dims=("depth", "time"),
                    coords={"depth": depth, "time": time}
                )
            }
        )

        local_store = local_store.chunk({'depth': 512, 'time': 512}) # Re-chunking to optimize performance

        local_store['Sv'].plot() # Quick plot to see if it works-- can be removed later

        plt.show()
        
        # Writing the sv data to the local store (copies the following data arrays: Sv, frequency, time, depth)
        # local_store = local_store.to_zarr('local_dataarray.zarr', mode='w', compute=True, zarr_format=2)

    # TODO: Make it all close cleanly-- later goal
    def close(self):
        pass

# A test to see if it works-- use as needed
if __name__ == "__main__":
    x = water_column_resample("noaa-wcsd-zarr-pds/level_2/Henry_B._Bigelow/HB0707/EK60/HB0707.zarr/")
    x.new_dataarray()
