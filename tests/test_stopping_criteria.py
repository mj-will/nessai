import pytest

from nessai.stopping_criteria import (
    CriterionGroup,
    StoppingCriterion,
    StoppingCriterionRegistry,
)


@pytest.mark.parametrize(
    "comparison, value, tolerance, expected",
    [
        ("<", 4, 5, True),
        ("<=", 5, 5, True),
        (">", 6, 5, True),
        (">=", 5, 5, True),
        ("==", 5, 5, True),
        ("!=", 4, 5, True),
        ("<", 6, 5, False),
        ("<=", 6, 5, False),
        (">", 5, 5, False),
        (">=", 4, 5, False),
        ("==", 4, 5, False),
        ("!=", 5, 5, False),
    ],
)
def test_stopping_criterion_is_met(comparison, value, tolerance, expected):
    crit = StoppingCriterion("test", tolerance, comparison)
    assert crit.is_met(value) == expected


def test_invalid_comparison_operator_raises():
    with pytest.raises(ValueError, match="Invalid comparison operator"):
        StoppingCriterion("bad", 1.0, "===")


def test_criterion_group_and_logic():
    c1 = StoppingCriterion("c1", 5, "<=")
    c2 = StoppingCriterion("c2", 10, ">")
    group = CriterionGroup([c1, c2], mode="and")
    values = {"c1": 5, "c2": 11}
    assert group.is_met(values)


def test_criterion_group_or_logic():
    c1 = StoppingCriterion("c1", 5, "<=")
    c2 = StoppingCriterion("c2", 10, ">")
    group = CriterionGroup([c1, c2], mode="or")
    values = {"c1": 6, "c2": 11}
    assert group.is_met(values)


def test_group_names_and_tolerances():
    c1 = StoppingCriterion("c1", 0.5, "<")
    c2 = StoppingCriterion("c2", 1.5, ">")
    group = CriterionGroup([c1, c2])
    assert group.names == ["c1", "c2"]
    assert group.tolerances == {"c1": 0.5, "c2": 1.5}


def test_group_composition_and_or():
    c1 = StoppingCriterion("a", 1.0, "<")
    c2 = StoppingCriterion("b", 2.0, "!=")
    g1 = CriterionGroup([c1])
    g2 = CriterionGroup([c2])

    combined_and = g1 & g2
    assert isinstance(combined_and, CriterionGroup)
    assert len(combined_and.criteria) == 2
    assert combined_and.mode == "and"

    combined_or = g1 | g2
    assert isinstance(combined_or, CriterionGroup)
    assert len(combined_or.criteria)


def test_composition_and_or():
    c1 = StoppingCriterion("a", 1.0, "<")
    c2 = StoppingCriterion("b", 2.0, "!=")

    combined = c1 & c2
    assert isinstance(combined, CriterionGroup)
    assert len(combined.criteria) == 2
    assert combined.mode == "and"

    combined = c1 | c2
    assert isinstance(combined, CriterionGroup)
    assert len(combined.criteria) == 2
    assert combined.mode == "or"


class DummyCriterion(StoppingCriterion):
    def __init__(self, name="dummy", tolerance=1.0, comparison="<"):
        super().__init__(name, tolerance, comparison)


@pytest.mark.parametrize(
    "names",
    [
        ("single_name",),
        ("name_one", "name_two"),
        ("MixedCase", "lowercase", "UPPERCASE"),
    ],
)
def test_register_and_get_criterion(names):
    # Register dummy criterion under multiple names
    StoppingCriterionRegistry.register(*names)(DummyCriterion)

    for name in names:
        retrieved = StoppingCriterionRegistry.get(name)
        assert isinstance(retrieved, DummyCriterion)
        assert retrieved.name == "dummy"


@pytest.mark.parametrize(
    "lookup_name, expected_cls",
    [
        ("ess", "ESS"),
        ("log_evidence_ratio", "LogEvidenceRatio"),
        ("log_evidence_ratio_nested_samples", "LogEvidenceRatioNestedSamples"),
        ("ratio", "LogEvidenceRatio"),
        ("Z_err", "EvidenceError"),
        ("difference_log_evidence", "DifferenceLogEvidence"),
        ("fractional_error", "FractionalError"),
    ],
)
def test_get_registered_criteria_classes(lookup_name, expected_cls):
    retrieved = StoppingCriterionRegistry.get(lookup_name)
    assert retrieved.__class__.__name__ == expected_cls


def test_registry_case_insensitivity():
    StoppingCriterionRegistry.register("CaseTest")(DummyCriterion)
    assert isinstance(
        StoppingCriterionRegistry.get("casetest"), DummyCriterion
    )
    assert isinstance(
        StoppingCriterionRegistry.get("CASETEST"), DummyCriterion
    )
    assert isinstance(
        StoppingCriterionRegistry.get("CaseTest"), DummyCriterion
    )


def test_list_available_includes_defaults():
    available = StoppingCriterionRegistry.list_available()
    # These are defaults from your submodule
    expected_keys = {
        "ess",
        "log_evidence_ratio",
        "ratio",
        "log_evidence_ratio_nested_samples",
        "ratio_ns",
        "z_err",
        "evidence_error",
        "dlogz",
        "difference_log_evidence",
        "fractional_error",
    }
    assert expected_keys.issubset(set(available))


def test_get_unregistered_name_raises():
    with pytest.raises(
        ValueError, match="No registered criterion with name 'unknown_name'"
    ):
        StoppingCriterionRegistry.get("unknown_name")
