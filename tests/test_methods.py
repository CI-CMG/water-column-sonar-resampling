import wrc_workspace.water_column_resampling as wcr
import numpy as np
import xarray as xr
import zarr

def test_open(tmp_path):
    # Opening a temporary zarr array to test with
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

def test_new_array(tmp_path):
    depth = np.arange(0, 4)
    time = np.arange(0, 6)
    freq = np.array([18])

    # Make synthetic data deterministic for testing
    np.random.seed(0)
    sv_data = np.random.randint(-70, -20, size=(len(freq), len(depth), len(time))).astype(np.float32)
    bottom = np.array([1, 2, 2, 3, 3, 4])
    
    # Opening a temporary zarr array to test with
    dt_array = xr.Dataset(
        {
            'Sv': (('frequency', 'depth', 'time'), sv_data),
            'bottom': (('time',), bottom)
        },

        coords= {
            'frequency': freq,
            'depth': depth,
            'time': time
        },
    )

    # Chucking it into a baby store
    dt_array = dt_array.chunk({'frequency': 1, 'time': 2, 'depth': 2})

    # Defining a temporary store path
    temp_store = f'{tmp_path}/TMP_STORE.zarr'

    # Writing to the local store to a temporary zarr file
    dt_array.to_zarr(temp_store, mode='w', compute=True, zarr_format=2)

    # Opening it and running tests
    x = wcr.water_column_resample(temp_store)
    local_store_path = tmp_path/'local_dataarray.zarr'
    x.new_dataarray(output_path=local_store_path)

    local_store = zarr.open(local_store_path, mode='r')

    assert 'Sv' in local_store
    assert local_store['Sv'].dtype == np.int8

    # Build expected array from the original sv_data (frequency, depth, time)
    expected = sv_data[0].copy()
    for t_idx in range(expected.shape[1]):
        mask = np.arange(expected.shape[0]) >= bottom[t_idx]
        expected[mask, t_idx] = 0.0

    expected = expected.astype(np.int8)

    stored = local_store['Sv'][:]

    # Ensure shapes match and values are equal after masking and casting
    assert stored.shape == expected.shape
    assert np.array_equal(stored, expected)