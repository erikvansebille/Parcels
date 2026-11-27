from datetime import timedelta

import numpy as np
import pytest

from parcels import Field, FieldSet
from parcels._core.model import StructuredModelData
from parcels._datasets.structured.generated import simple_UV_dataset, simple_UV_sigma_dataset
from parcels._datasets.structured.generic import T as T_structured
from parcels._datasets.structured.generic import datasets as datasets_structured
from parcels._datasets.unstructured.generated import sigma_coordinate_lattice_dataset
from parcels._python import NOTSET


@pytest.mark.parametrize("time_varying", [False, True])
def test_structured_model_requires_depth_positive_attribute(time_varying):
    if time_varying:
        ds = simple_UV_sigma_dataset(n_times=2, nz=3, ny=3, nx=3)
    else:
        ds = simple_UV_dataset(n_times=2, nz=3, ny=3, nx=3, mesh_str="flat")
    ds["depth"].attrs.pop("positive")

    with pytest.raises(ValueError, match="Vertical coordinate 'depth'.*'positive'"):
        FieldSet.from_sgrid_conventions(ds)


@pytest.mark.parametrize("time_varying", [False, True])
def test_unstructured_model_requires_depth_positive_attribute(time_varying):
    bottom_depth = np.full((3, 3), 20.0) if time_varying else 20.0
    ds = sigma_coordinate_lattice_dataset(3, (0.0, 1.0), (0.0, 1.0), 3, bottom_depth)
    ds.variables["zf"].attrs.pop("positive")

    with pytest.raises(ValueError, match="Vertical coordinate 'zf'.*'positive'"):
        FieldSet.from_ugrid_conventions(ds)


def test_model_init_fail_on_float_time_dim():
    """Test that accessing time_interval fails when dataset has float time dimension.

    (users are expected to use timedelta64 or datetime).
    Time validation has moved from Field.__init__ to ModelData.time_interval.
    """
    ds = datasets_structured["ds_2d_left"].copy()
    ds["time"] = (
        ds["time"].dims,
        np.arange(0, T_structured, dtype="float64"),
        ds["time"].attrs,
    )

    model = StructuredModelData.from_sgrid_conventions(ds, vector_fields=NOTSET)
    with pytest.raises(
        ValueError,
        match=r"Are you sure that the time dimension on the xarray dataset is stored as timedelta, datetime or cftime datetime objects\?",
    ):
        _ = model.time_interval


def test_time_not_increasing_monotonic():
    ds = simple_UV_dataset(mesh_str="flat", n_times=3).copy()
    ds["time"] = np.array([np.timedelta64(0, "h"), np.timedelta64(1, "h"), np.timedelta64(1, "h")])

    with pytest.raises(ValueError):
        FieldSet.from_sgrid_conventions(ds)


def test_model_time_interval():
    """Test that field.time_interval delegates correctly to model.time_interval."""
    data = datasets_structured["ds_2d_left"]
    model = StructuredModelData.from_sgrid_conventions(data, vector_fields=NOTSET)
    field = Field(name="data_g", model=model)
    assert field.time_interval.left == np.datetime64("2000-01-01")
    assert field.time_interval.right == np.datetime64("2001-01-01")


def test_multi_model_time_interval():
    ds1 = datasets_structured["ds_2d_left"][["U_A_grid", "V_A_grid", "grid"]]
    fieldset = FieldSet.from_sgrid_conventions(ds1)

    ds2 = ds1.copy().rename({"U_A_grid": "U2", "V_A_grid": "V2"})
    ds2["time"] = (ds2["time"].dims, ds2["time"].data + np.timedelta64(timedelta(days=1)), ds2["time"].attrs)
    fieldset += FieldSet.from_sgrid_conventions(ds2)

    ds3 = ds1.copy().rename({"U_A_grid": "U3", "V_A_grid": "V3"})
    ds3["time"] = (ds3["time"].dims, ds3["time"].data + np.timedelta64(timedelta(days=2)), ds3["time"].attrs)
    fieldset += FieldSet.from_sgrid_conventions(ds3)

    fieldset.add_constant_field("constant_field", 1.0)

    assert len(fieldset.models) == 3
    assert fieldset.constant_model is not None
    assert fieldset.time_interval.left == np.datetime64("2000-01-03")
    assert fieldset.time_interval.right == np.datetime64("2001-01-01")


def test_multi_model_nonoverlapping_time_interval():
    ds1 = datasets_structured["ds_2d_left"][["U_A_grid", "V_A_grid", "grid"]]
    fieldset = FieldSet.from_sgrid_conventions(ds1)

    ds2 = ds1.copy().rename({"U_A_grid": "U2", "V_A_grid": "V2"})
    ds2["time"] = (ds2["time"].dims, ds2["time"].data + np.timedelta64(timedelta(days=1000)), ds2["time"].attrs)
    fieldset += FieldSet.from_sgrid_conventions(ds2)

    ds3 = ds1.copy().rename({"U_A_grid": "U3", "V_A_grid": "V3"})
    ds3["time"] = (ds3["time"].dims, ds3["time"].data + np.timedelta64(timedelta(days=2000)), ds3["time"].attrs)
    fieldset += FieldSet.from_sgrid_conventions(ds3)

    fieldset.add_constant_field("constant_field", 1.0)

    assert len(fieldset.models) == 3
    assert fieldset.constant_model is not None
    assert fieldset.time_interval is None
