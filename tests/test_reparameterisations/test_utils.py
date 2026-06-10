import pytest

from nessai.reparameterisations.utils import (
    build_reparameterisation_spec,
    normalise_reparameterisation_spec,
    parse_reparameterisations,
    resolve_reparameterisation_parameters,
)


@pytest.mark.parametrize(
    "spec_cfg",
    [
        {"reparameterisation": "scale", "parameters": ["y"], "foo": 1},
        {"reparameterisation": "scale", "foo": 1},
    ],
)
def test_build_reparameterisation_spec_model_key(spec_cfg):
    key = "y"
    spec_index = 0
    model_names = ["x", "y", "z"]
    spec = build_reparameterisation_spec(
        key, spec_cfg, spec_index, model_names
    )
    assert spec.source_key == key
    assert spec.reparameterisation == "scale"
    assert spec.parameters == ["y"]
    assert spec.kwargs == {"foo": 1}


def test_build_reparameterisation_spec_reparam_key():
    key = "scale"
    spec_cfg = {"parameters": ["y"], "foo": 1}
    spec_index = 0
    model_names = ["x", "y", "z"]
    spec = build_reparameterisation_spec(
        key, spec_cfg, spec_index, model_names
    )
    assert spec.source_key == key
    assert spec.reparameterisation == "scale"
    assert spec.parameters == ["y"]
    assert spec.kwargs == {"foo": 1}


def test_build_reparameterisation_spec_model_key_missing_reparameterisation():
    with pytest.raises(
        RuntimeError, match="No reparameterisation found for x"
    ):
        build_reparameterisation_spec("x", {"scale": 2.0}, 0, ["x"])


@pytest.mark.parametrize(
    "parameters, expected",
    [("y", ["x", "y"]), (None, [])],
)
def test_build_reparameterisation_spec_model_key_parameter_variants(
    parameters, expected
):
    spec = build_reparameterisation_spec(
        "x",
        {"reparameterisation": "scale", "parameters": parameters},
        0,
        ["x"],
    )
    assert spec.parameters == expected


def test_build_reparameterisation_spec_reparam_key_list():
    spec = build_reparameterisation_spec("scale", ["x", "y"], 0, ["x", "y"])
    assert spec.parameters == ["x", "y"]


def test_build_reparameterisation_spec_reparam_key_invalid():
    with pytest.raises(TypeError, match="Unknown config type for: scale"):
        build_reparameterisation_spec("scale", 1, 0, ["x"])


def test_normalise_reparameterisation_spec_str():
    key = "x"
    spec_cfg = "scale"
    spec_list = normalise_reparameterisation_spec(key, spec_cfg, [key])
    assert spec_list == [spec_cfg]


def test_normalise_reparameterisation_spec_dict():
    key = "x"
    spec_cfg = {"reparameterisation": "scale", "parameters": ["y"], "foo": 1}
    spec = normalise_reparameterisation_spec(key, spec_cfg, [key])
    assert spec == [spec_cfg]


def test_normalise_reparameterisation_spec_list():
    key = "x"
    spec_cfg = ["y", "z"]
    spec_list = normalise_reparameterisation_spec(key, spec_cfg, [key])
    assert spec_list == spec_cfg


def test_normalise_reparameterisation_spec_invalid():
    key = "x"
    spec_cfg = 1
    with pytest.raises(
        TypeError,
        match="Unknown config type for: x. Expected str, dict or list, received instance of <class 'int'>.",
    ):
        normalise_reparameterisation_spec(key, spec_cfg, [key])


def test_parse_reparameterisations_dict():
    reparameterisations = {
        "scale": {"parameters": ["w"]},
        "x": "scale",
        "y": {
            "reparameterisation": "log",
            "parameters": ["y_prime"],
            "foo": 1,
        },
        "log": "z",
    }
    model_names = ["w", "x", "y", "z"]
    specs = parse_reparameterisations(reparameterisations, model_names)

    assert len(specs) == 4

    spec_w = specs[0]
    assert spec_w.source_key == "scale"
    assert spec_w.reparameterisation == "scale"
    assert spec_w.parameters == ["w"]
    assert spec_w.kwargs == {}

    spec_x = specs[1]
    assert spec_x.source_key == "x"
    assert spec_x.reparameterisation == "scale"
    assert spec_x.parameters == ["x"]
    assert spec_x.kwargs == {}

    spec_y = specs[2]
    assert spec_y.source_key == "y"
    assert spec_y.reparameterisation == "log"
    assert spec_y.parameters == ["y", "y_prime"]
    assert spec_y.kwargs == {"foo": 1}

    spec_z = specs[3]
    assert spec_z.source_key == "log"
    assert spec_z.reparameterisation == "log"
    assert spec_z.parameters == ["z"]
    assert spec_z.kwargs == {}


def test_parse_reparameterisations_dict_reparam_list():
    reparameterisations = {"scale": ["x", "y", "z"]}
    model_names = ["x", "y", "z"]
    specs = parse_reparameterisations(reparameterisations, model_names)

    assert len(specs) == 1

    spec_x = specs[0]
    assert spec_x.source_key == "scale"
    assert spec_x.reparameterisation == "scale"
    assert spec_x.parameters == ["x", "y", "z"]
    assert spec_x.kwargs == {}


def test_parse_reparameterisations_str():
    reparameterisations = "scale"
    model_names = ["x", "y", "z"]
    specs = parse_reparameterisations(reparameterisations, model_names)

    assert len(specs) == 1

    spec_x = specs[0]
    assert spec_x.source_key == "scale"
    assert spec_x.reparameterisation == "scale"
    assert spec_x.parameters == ["x", "y", "z"]
    assert spec_x.kwargs == {}


def test_parse_reparameterisations_none():
    reparameterisations = None
    model_names = ["x", "y", "z"]
    specs = parse_reparameterisations(reparameterisations, model_names)
    assert len(specs) == 0


def test_parse_reparameterisations_regex():
    reparameterisations = {"scale": {"parameters": ["x.*"]}}
    model_names = ["x_0", "x_1", "y"]
    specs = parse_reparameterisations(reparameterisations, model_names)

    assert len(specs) == 1

    spec_x = specs[0]
    assert spec_x.source_key == "scale"
    assert spec_x.reparameterisation == "scale"
    # Match happens later in resolve_reparameterisation_parameters
    assert spec_x.parameters == ["x.*"]
    assert spec_x.kwargs == {}


def test_parse_reparameterisations_chained():
    reparameterisations = {
        "x": [
            {
                "reparameterisation": "rescaletobounds",
                "prime_parameters": ["x_01"],
            },
            {"reparameterisation": "log", "prime_requires": ["x_01"]},
        ]
    }
    model_names = ["x"]
    specs = parse_reparameterisations(reparameterisations, model_names)
    assert len(specs) == 2
    assert specs[0].reparameterisation == "rescaletobounds"
    assert specs[1].reparameterisation == "log"
    assert specs[0].parameters == ["x"]
    assert specs[1].parameters == ["x"]
    assert specs[0].spec_index == 0
    assert specs[1].spec_index == 1


def test_resolve_reparameterisation_parameters():
    parameters = ["x.*"]
    available_parameters = ["x_0", "x_1", "y"]
    resolved = resolve_reparameterisation_parameters(
        parameters, available_parameters
    )
    assert resolved == ["x_0", "x_1"]


def test_resolve_reparameterisation_parameters_no_match():
    parameters = ["z.*"]
    available_parameters = ["x_0", "x_1", "y"]
    resolved = resolve_reparameterisation_parameters(
        parameters, available_parameters
    )
    assert resolved == []


def test_resolve_reparameterisation_parameters_list():
    parameters = ["x_0", "x_1"]
    available_parameters = ["x_0", "x_1", "y"]
    resolved = resolve_reparameterisation_parameters(
        parameters, available_parameters
    )
    assert resolved == ["x_0", "x_1"]


def test_resolve_reparameterisation_parameters_str():
    parameters = "x_0"
    available_parameters = ["x_0", "x_1", "y"]
    resolved = resolve_reparameterisation_parameters(
        parameters, available_parameters
    )
    assert resolved == ["x_0"]
