from unittest.mock import MagicMock, patch

import pytest

from nessai.reparameterisations import (
    KnownReparameterisation,
    ReparameterisationDict,
)


def test_known_reparameterisation():
    known = KnownReparameterisation("test", "class", {"key": "value"})
    assert known.name == "test"
    assert known.class_fn == "class"
    assert known.keyword_arguments == {"key": "value"}


def test_known_reparameterisation_no_kwargs():
    known = KnownReparameterisation("test", "class")
    assert known.name == "test"
    assert known.class_fn == "class"
    assert known.keyword_arguments == {}


def test_reparameterisation_dict_add_reparam():
    reparam_dict = ReparameterisationDict()
    reparam_dict.add_reparameterisation("test", "class", {"key": "value"})
    assert reparam_dict["test"].name == "test"
    assert reparam_dict["test"].class_fn == "class"
    assert reparam_dict["test"].keyword_arguments == {"key": "value"}


def test_reparameterisation_dict_add_reparam_existing_entry():
    reparam_dict = ReparameterisationDict(
        {"test": KnownReparameterisation("test", "class", {"key": "value"})}
    )
    with pytest.raises(
        ValueError, match="Reparameterisation test already exists"
    ):
        reparam_dict.add_reparameterisation("test", "class", {"key": "value"})


def test_reparameterisation_dict_add_external_reparam():
    reparam_dict = ReparameterisationDict()
    # Mock class
    external_reparam = KnownReparameterisation(
        "external_reparam", "class", {"key": "value"}
    )

    # Mock what is normally returned by the entry point before they are loaded
    EntryPointClass = MagicMock(spec=["load"])
    EntryPointClass.load = MagicMock(return_value=external_reparam)

    # Always return the version that needs to be loaded
    with patch(
        "nessai.reparameterisations.utils.get_entry_points",
        return_value={"external_class": EntryPointClass},
    ) as mock_get_entry_points:
        reparam_dict.add_external_reparameterisations(
            "nessai.reparameterisations"
        )
    mock_get_entry_points.assert_called_once_with("nessai.reparameterisations")

    assert external_reparam == reparam_dict["external_reparam"]


def test_reparameterisation_dict_add_external_reparam_invalid_type():
    reparam_dict = ReparameterisationDict()
    # Mock class
    external_reparam = ("external_reparam", "class", {"key": "value"})

    # Mock what is normally returned by the entry point before they are loaded
    EntryPointClass = MagicMock(spec=["load"])
    EntryPointClass.load = MagicMock(return_value=external_reparam)

    # Always return the version that needs to be loaded
    with (
        patch(
            "nessai.reparameterisations.utils.get_entry_points",
            return_value={"external_class": EntryPointClass},
        ),
        pytest.raises(
            RuntimeError, match="Invalid external reparameterisation"
        ),
    ):
        reparam_dict.add_external_reparameterisations(
            "nessai.reparameterisations"
        )


def test_reparameterisation_dict_add_external_reparam_name_conflict():
    reparam_dict = ReparameterisationDict()
    reparam_dict.add_reparameterisation("default", "class", {"key": "value"})
    # Mock class
    external_reparam = KnownReparameterisation(
        "default", "class", {"key": "value"}
    )

    # Mock what is normally returned by the entry point before they are loaded
    EntryPointClass = MagicMock(spec=["load"])
    EntryPointClass.load = MagicMock(return_value=external_reparam)

    # Always return the version that needs to be loaded
    with (
        patch(
            "nessai.reparameterisations.utils.get_entry_points",
            return_value={"external_class": EntryPointClass},
        ),
        pytest.raises(
            ValueError, match="Reparameterisation default already exists"
        ),
    ):
        reparam_dict.add_external_reparameterisations(
            "nessai.reparameterisations"
        )
