import wrc_workspace.water_column_resampling as wcr
import numpy as np
import xarray as xr

def test_open(tmp_path):
    # Opening a temporary zarr store to test with
    dt_array = xr.DataArray(
        data=np.empty((1024, 1024), dtype='int8'),
        dims=('depth', 'time')
    )

    # Chucking it into a baby store
    dt_array = dt_array.chunk({'time': 1024, 'depth': 1024})

    # Adding it to a local store
    local_store = xr.Dataset(data_vars={'Sv': dt_array})

    # Defining a temporary store path
    temp_store = f'{tmp_path}/TMP_STORE.zarr'

    # Writing the local store to a temporary zarr file
    local_store.to_zarr(temp_store, mode='w', compute=True, zarr_format=2)
    
    # Opening it and running tests
    x = wcr.water_column_resample(temp_store)
    x.open_store()
    assert x.return_attributes() is not None
    assert x.return_shape() is not None

