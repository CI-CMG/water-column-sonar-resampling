import xarray as xr
import s3fs
import json
import asyncio

# Can change method name later on
class OpenStore:
    def __init__(self, store_link):
        self.store_link = store_link
        self.file_system = s3fs.S3FileSystem(anon=True)
        self.store = None
        self.data_set = None
        self.attributes = None

    def open_store(self):
        self.store = s3fs.S3Map(root=self.store_link, s3=self.file_system)
        self.data_set = xr.open_zarr(store=self.store, consolidated=True)

    def return_data(self):
        if self.store is None:
            self.open_store()

        self.attributes = self.data_set.attrs # Prints all meta data
        print(self.data_set['Sv'].shape)
        print(json.dumps(self.attributes, indent=4))

    # TODO: Make it all close cleanly
    def close(self):
        pass

# Tests to make sure it still works
if __name__ == "__main__":
    x = OpenStore("noaa-wcsd-zarr-pds/level_2/Henry_B._Bigelow/HB1906/EK60/HB1906.zarr/")
    x.return_data()