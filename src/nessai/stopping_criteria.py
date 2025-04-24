"""Stopping criteria for nested sampling.

This module defines various stopping criteria for nested sampling algorithms.
"""

import operator
from dataclasses import dataclass
from typing import Dict, Literal, Union

COMPARISON_OPERATORS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}


class StoppingCriterionRegistry:
    """
    Registry for stopping criteria.

    This class is used to register and retrieve stopping criteria classes.
    It allows for easy extension and management of different stopping criteria.
    """

    _registry = {}

    @classmethod
    def register(cls, *names: str):
        def decorator(criterion_cls):
            for name in names:
                cls._registry[name.lower()] = criterion_cls
            return criterion_cls

        return decorator

    @classmethod
    def get(cls, name: str, **kwargs):
        key = name.lower()
        if key not in cls._registry:
            raise ValueError(f"No registered criterion with name '{name}'")
        return cls._registry[key](**kwargs)

    @classmethod
    def list_available(cls):
        return list(cls._registry.keys())


@dataclass
class StoppingCriterion:
    """Class for defining stopping criteria.

    Parameters
    ----------
    name : str
        Name of the stopping criterion.
    tolerance : float
        Tolerance value for the stopping criterion.
    comparison : str
        Comparison operator for the stopping criterion.
            Valid options are: '<', '>', '<=', '>=', '==', '!='.
    """

    name: str
    tolerance: float
    comparison: Literal["<", ">", "<=", ">=", "==", "!="]

    def __post_init__(self):
        if self.comparison not in COMPARISON_OPERATORS:
            raise ValueError(f"Invalid comparison operator: {self.comparison}")
        self._operator = COMPARISON_OPERATORS[self.comparison]

    def is_met(self, value: Union[float, int]) -> bool:
        """
        Check if the stopping criterion is met.

        Parameters
        ----------
        value : float or int
            The value to compare against the stopping criterion.

        Returns
        -------
        bool
            True if the stopping criterion is met, False otherwise.
        """
        return self._operator(value, self.tolerance)

    def __and__(self, other: "CriterionGroup") -> "CriterionGroup":
        return CriterionGroup([self]) & other

    def __or__(self, other: "CriterionGroup") -> "CriterionGroup":
        return CriterionGroup([self]) | other


@dataclass
class CriterionGroup:
    criteria: list
    mode: Literal["and", "or"] = "and"

    def is_met(self, values: Dict[str, float]) -> bool:
        results = [c.is_met(values[c.name]) for c in self.criteria]
        return all(results) if self.mode == "and" else any(results)

    def __and__(
        self, other: Union[StoppingCriterion, "CriterionGroup"]
    ) -> "CriterionGroup":
        return CriterionGroup(
            self.criteria
            + (
                [other]
                if isinstance(other, StoppingCriterion)
                else other.criteria
            ),
            mode="and",
        )

    def __or__(
        self, other: Union[StoppingCriterion, "CriterionGroup"]
    ) -> "CriterionGroup":
        return CriterionGroup(
            self.criteria
            + (
                [other]
                if isinstance(other, StoppingCriterion)
                else other.criteria
            ),
            mode="or",
        )

    @property
    def names(self) -> list[str]:
        """
        Get the names of the stopping criteria in the group.

        Returns
        -------
        list[str]
            List of names of the stopping criteria.
        """
        return [criterion.name for criterion in self.criteria]

    @property
    def tolerances(self) -> Dict[str, float]:
        """
        Get the tolerances of the stopping criteria in the group.

        Returns
        -------
        Dict[str, float]
            Dictionary mapping criterion names to their tolerances.
        """
        return {
            criterion.name: criterion.tolerance for criterion in self.criteria
        }


@StoppingCriterionRegistry.register("ess")
class ESS(StoppingCriterion):
    """
    Effective sample size (ESS) stopping criterion.

    Parameters
    ----------
    tolerance : float
        Tolerance value for the stopping criterion.
    """

    def __init__(self, tolerance: float = 5000.0):
        super().__init__("ess", tolerance, comparison=">=")


@StoppingCriterionRegistry.register("log_evidence_ratio", "ratio")
class LogEvidenceRatio(StoppingCriterion):
    """
    Log ratio of evidence between live points and all samples.

    Parameters
    ----------
    tolerance : float
        Tolerance value for the stopping criterion.
    """

    def __init__(self, tolerance: float = 0.0):
        super().__init__("log_evidence_ratio", tolerance, comparison="<=")


@StoppingCriterionRegistry.register(
    "log_evidence_ratio_nested_samples", "ratio_ns"
)
class LogEvidenceRatioNestedSamples(StoppingCriterion):
    """
    Log ratio of evidence between live point and nested samples.

    Parameters
    ----------
    tolerance : float
        Tolerance value for the stopping criterion.
    """

    def __init__(self, tolerance: float = 0.0):
        super().__init__(
            "log_evidence_ratio_nested_samples", tolerance, comparison="<="
        )


@StoppingCriterionRegistry.register("Z_err", "evidence_error")
class EvidenceError(StoppingCriterion):
    """
    Evidence error stopping criterion.

    Parameters
    ----------
    name : str
        Name of the stopping criterion.
    tolerance : float
        Tolerance value for the stopping criterion.
    comparison : str
        Comparison operator for the stopping criterion.
    """

    def __init__(self, tolerance: float = 0.1):
        super().__init__("evidence_error", tolerance, comparison="<=")


@StoppingCriterionRegistry.register("dlogZ", "difference_log_evidence")
class DifferenceLogEvidence(StoppingCriterion):
    """Difference in log evidence stopping criterion.

    This is the standard nested sampling stopping criterion.

    Parameters
    ----------
    tolerance : float
        Tolerance value for the stopping criterion.
    """

    def __init__(self, tolerance: float = 0.1):
        super().__init__("difference_log_evidence", tolerance, comparison="<=")


@StoppingCriterionRegistry.register("fractional_error")
class FractionalError(StoppingCriterion):
    """
    Fractional error stopping criterion.

    Parameters
    ----------
    tolerance : float
        Tolerance value for the stopping criterion.
    """

    def __init__(self, tolerance: float = 0.1):
        super().__init__("fractional_error", tolerance, comparison="<=")
