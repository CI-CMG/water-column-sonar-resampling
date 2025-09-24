import xarray as xr
import s3fs
import json

# Can change method name later on
class water_column_resample:
    def __init__(self, store_link):
        self.store_link = store_link
        self.file_system = s3fs.S3FileSystem(anon=True)
        self.store = None
        self.data_set = None
        self.attributes = None

    # Actually opens the zarr store based in the link given
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

    # TODO: Make it all close cleanly-- later goal
    def close(self):
        pass

"""
# A test to see if it works-- use as needed
if __name__ == "__main__":
    x = water_comlumn_resample("noaa-wcsd-zarr-pds/level_2/Henry_B._Bigelow/HB1906/EK60/HB1906.zarr/")
    print(x.return_attributes())
    print(x.return_shape()) # Shows the Sv dimensions by default
    print(x.return_shape(variable="Sv")) # You can pass additonal variables like: speed, bottom, longitude, latitude, etc.
"""