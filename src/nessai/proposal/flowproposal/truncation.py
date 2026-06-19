"""Truncation rules for flow proposals."""

from __future__ import annotations

import logging

import numpy as np

from ...utils import compute_radius
from ...utils.structures import get_subset_arrays

logger = logging.getLogger(__name__)

DEFAULT_TRUNCATION_METHODS = ["latent_radius"]
DEFAULT_TRUNCATION_KWARGS = {
    "latent_radius": {
        "radius_mode": "constant_volume",
        "volume_fraction": 0.95,
    }
}

LEGACY_LATENT_RADIUS_ARGUMENTS = (
    "constant_volume_mode",
    "volume_fraction",
    "fuzz",
    "expansion_fraction",
    "fixed_radius",
    "radius_mode",
    "min_radius",
    "max_radius",
    "compute_radius_with_all",
)


def get_deprecated_latent_radius_arguments(**kwargs) -> list[str]:
    """Return deprecated latent-radius arguments that were explicitly set."""
    return [
        name
        for name in LEGACY_LATENT_RADIUS_ARGUMENTS
        if kwargs[name] is not None
    ]


def get_deprecated_latent_radius_kwargs(**kwargs) -> dict:
    """Build sparse latent-radius kwargs from deprecated proposal arguments."""
    return {
        name: kwargs[name]
        for name in LEGACY_LATENT_RADIUS_ARGUMENTS
        if kwargs[name] is not None
    }


def normalise_truncation_methods(
    truncation_method=None, truncation_methods=None
) -> list[str]:
    """Normalise truncation-method input into an ordered unique list."""
    methods = (
        truncation_methods
        if truncation_methods is not None
        else truncation_method
    )
    if methods is None:
        return []
    if isinstance(methods, str):
        methods = [methods]
    return list(dict.fromkeys(methods))


def should_enable_latent_radius(latent_radius_kwargs=None) -> bool:
    """Check if latent-radius truncation should be enabled from kwargs."""
    return bool(latent_radius_kwargs)


def build_truncation_methods(
    truncation_method=None,
    truncation_methods=None,
    truncate_log_q=False,
    enforce_likelihood_threshold=False,
    latent_radius_kwargs=None,
    default_latent_radius: bool = False,
) -> list[str]:
    """Build the effective truncation-method list from legacy and new inputs."""
    if truncation_method is not None and truncation_methods is not None:
        raise ValueError(
            "Specify only one of truncation_method or truncation_methods"
        )

    methods = normalise_truncation_methods(
        truncation_method, truncation_methods
    )

    if default_latent_radius and "latent_radius" not in methods:
        methods.insert(0, "latent_radius")
    elif (
        should_enable_latent_radius(latent_radius_kwargs)
        and "latent_radius" not in methods
    ):
        methods.insert(0, "latent_radius")
    if truncate_log_q and "min_log_q" not in methods:
        methods.append("min_log_q")
    if enforce_likelihood_threshold and "likelihood_threshold" not in methods:
        methods.append("likelihood_threshold")
    return methods


def apply_default_truncation_config(
    methods,
    truncation_kwargs=None,
    *,
    default_latent_radius: bool = False,
):
    """Apply canonical default truncation configuration."""
    if default_latent_radius and not methods:
        methods = list(DEFAULT_TRUNCATION_METHODS)
    else:
        methods = list(methods)

    kwargs = {
        name: dict(value) for name, value in (truncation_kwargs or {}).items()
    }

    for name, default_kwargs in DEFAULT_TRUNCATION_KWARGS.items():
        if name not in methods:
            continue
        kwargs.setdefault(name, {})
        for key, value in default_kwargs.items():
            kwargs[name].setdefault(key, value)

    return methods, kwargs


class BaseTruncationRule:
    """Base class for truncation rules."""

    name = "base"
    _transient_defaults = {}

    def __init__(self) -> None:
        self.reset()

    @property
    def requires_log_likelihood(self) -> bool:
        """Indicate if the rule needs log-likelihood values."""
        return False

    def configure(self, proposal) -> None:
        """Apply any proposal-level configuration needed by the rule."""
        return None

    def prepare(self, proposal, worst_point, radius=None):
        """Prepare per-population data for the rule."""
        return None

    def apply_latent(self, proposal, z):
        """Apply truncation in latent space before the inverse pass."""
        return z

    def apply_after_backward(self, proposal, x, log_q, z):
        """Apply truncation after the inverse pass and rescaling."""
        return x, log_q, z

    def apply_after_likelihood(self, proposal, x, log_q, z):
        """Apply truncation after likelihood evaluation."""
        return x, log_q, z

    def reset(self) -> None:
        """Reset transient state."""
        for key, value in self._transient_defaults.items():
            setattr(self, key, value)

    def __getstate__(self):
        state = self.__dict__.copy()
        for key, value in self._transient_defaults.items():
            state[key] = value
        return state


class LatentRadiusTruncation(BaseTruncationRule):
    """Filter latent samples using a radial threshold."""

    name = "latent_radius"
    _transient_defaults = {"_radius": np.nan, "_threshold": np.nan}

    def __init__(
        self,
        radius_mode: str | None = None,
        fixed_radius: float | bool = False,
        min_radius: float | bool = False,
        max_radius: float | bool = 50.0,
        compute_radius_with_all: bool = False,
        constant_volume_mode: bool = False,
        volume_fraction: float = 0.95,
        fuzz: float = 1.0,
        expansion_fraction: float | None = 4.0,
    ) -> None:
        super().__init__()
        self.fixed_radius = self._coerce_radius(
            fixed_radius, name="fixed_radius"
        )
        self.min_radius = self._coerce_radius(min_radius, name="min_radius")
        self.max_radius = self._coerce_radius(max_radius, name="max_radius")
        self.compute_radius_with_all = compute_radius_with_all
        self.volume_fraction = float(volume_fraction)
        self.fuzz = float(fuzz)
        self.expansion_fraction = expansion_fraction
        self.radius_mode = self._resolve_radius_mode(
            radius_mode,
            fixed_radius=self.fixed_radius,
            constant_volume_mode=constant_volume_mode,
        )

    @property
    def radius(self) -> float:
        return self._radius

    @property
    def threshold(self) -> float:
        return self._threshold

    @staticmethod
    def _coerce_radius(value, *, name):
        if value in (False, None):
            return False
        if not isinstance(value, (int, float)):
            raise RuntimeError(f"{name} must be an int or float")
        return float(value)

    @staticmethod
    def _resolve_radius_mode(
        radius_mode, *, fixed_radius, constant_volume_mode
    ):
        if radius_mode is None:
            if constant_volume_mode:
                radius_mode = "constant_volume"
            elif fixed_radius is not False:
                radius_mode = "fixed"
            else:
                radius_mode = "adaptive"

        if radius_mode not in {"adaptive", "fixed", "constant_volume"}:
            raise ValueError(
                "Unknown radius mode: "
                f"{radius_mode}. Choose from: adaptive, fixed, constant_volume"
            )
        if radius_mode == "fixed" and fixed_radius is False:
            raise ValueError(
                "Fixed radius mode requires `fixed_radius` to be specified"
            )
        return radius_mode

    @property
    def constant_volume_mode(self) -> bool:
        return self.radius_mode == "constant_volume"

    def to_kwargs(self) -> dict:
        """Return keyword arguments that reconstruct the rule."""
        return {
            "radius_mode": self.radius_mode,
            "fixed_radius": self.fixed_radius,
            "min_radius": self.min_radius,
            "max_radius": self.max_radius,
            "compute_radius_with_all": self.compute_radius_with_all,
            "constant_volume_mode": self.constant_volume_mode,
            "volume_fraction": self.volume_fraction,
            "fuzz": self.fuzz,
            "expansion_fraction": self.expansion_fraction,
        }

    def configure(self, proposal) -> None:
        if self.expansion_fraction and self.expansion_fraction is not None:
            logger.info(
                "Overwriting latent-radius fuzz factor with expansion fraction"
            )
            self.fuzz = (1 + self.expansion_fraction) ** (
                1 / proposal.prime_dims
            )
            logger.info(f"New latent-radius fuzz factor: {self.fuzz}")

        if not self.constant_volume_mode:
            return

        self.fixed_radius = compute_radius(
            proposal.prime_dims, self.volume_fraction
        )
        self.fuzz = 1.0

        if self.max_radius and self.max_radius < self.fixed_radius:
            logger.warning(
                "Max radius is less than the constant-volume radius. "
                "Disabling max radius."
            )
            self.max_radius = False
        if self.min_radius and self.min_radius > self.fixed_radius:
            logger.warning(
                "Min radius is greater than the constant-volume radius. "
                "Disabling min radius."
            )
            self.min_radius = False

    def _compute_radius(self, proposal, worst_point, radius=None) -> float:
        if radius is not None:
            return float(radius)

        if self.radius_mode in {"fixed", "constant_volume"}:
            if self.fixed_radius is False:
                raise RuntimeError(
                    "Fixed radius mode requires `fixed_radius` to be specified"
                )
            return float(self.fixed_radius)

        if self.compute_radius_with_all:
            worst_point = proposal.training_data

        worst_z = proposal.forward_pass(
            worst_point, rescale=True, compute_radius=True
        )[0]
        radius = float(np.sqrt(np.sum(worst_z**2.0, axis=-1)).max())
        if self.max_radius and radius > self.max_radius:
            radius = self.max_radius
        if self.min_radius and radius < self.min_radius:
            radius = self.min_radius
        return float(radius)

    def prepare(self, proposal, worst_point, radius=None):
        radius = self._compute_radius(proposal, worst_point, radius=radius)
        self._radius = radius
        self._threshold = self.fuzz * radius

    def apply_latent(self, proposal, z):
        radius = np.sqrt(np.sum(z**2.0, axis=-1))
        keep = radius <= self.threshold
        logger.debug(
            "Discarding %s latent samples outside radius threshold",
            len(z) - int(keep.sum()),
        )
        return z[keep]


class MinLogQTruncation(BaseTruncationRule):
    """Truncate samples using the minimum live-point log q."""

    name = "min_log_q"
    _transient_defaults = {"_min_log_q": np.nan}

    @property
    def min_log_q(self) -> float:
        return self._min_log_q

    def prepare(self, proposal, worst_point, radius=None):
        self._min_log_q = float(
            proposal.forward_pass(proposal.training_data)[1].min()
        )
        logger.debug("Truncating with log_q=%0.3f", self.min_log_q)

    def apply_after_backward(self, proposal, x, log_q, z):
        keep = log_q > self.min_log_q
        logger.debug(
            "Discarding %s samples below log_q_min",
            len(log_q) - int(keep.sum()),
        )
        return get_subset_arrays(keep, x, log_q, z)


class LikelihoodThresholdTruncation(BaseTruncationRule):
    """Truncate samples using the current likelihood threshold."""

    name = "likelihood_threshold"
    _transient_defaults = {"_threshold": np.nan}

    @property
    def requires_log_likelihood(self) -> bool:
        return True

    @property
    def threshold(self) -> float:
        return self._threshold

    def prepare(self, proposal, worst_point, radius=None):
        self._threshold = float(worst_point["logL"])

    def apply_after_likelihood(self, proposal, x, log_q, z):
        keep = x["logL"] > self.threshold
        logger.debug(
            "Accepting %s / %s samples above logL threshold",
            int(keep.sum()),
            len(x),
        )
        return get_subset_arrays(keep, x, log_q, z)


TRUNCATION_REGISTRY = {
    "latent_radius": LatentRadiusTruncation,
    "min_log_q": MinLogQTruncation,
    "likelihood_threshold": LikelihoodThresholdTruncation,
}


def get_truncation_rule_class(name: str):
    """Get the truncation rule class for a configured method name."""
    try:
        return TRUNCATION_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown truncation method: {name}") from exc


class TruncationScheme:
    """Apply an ordered set of truncation rules."""

    def __init__(self, rules: list[BaseTruncationRule] | None = None) -> None:
        self.rules = []
        for rule in rules or []:
            self.add_rule(rule)

    @property
    def rule_names(self) -> list[str]:
        return [rule.name for rule in self.rules]

    @property
    def requires_log_likelihood(self) -> bool:
        return any(rule.requires_log_likelihood for rule in self.rules)

    def has_rule(self, name: str) -> bool:
        return any(rule.name == name for rule in self.rules)

    def get_rule(self, name: str):
        for rule in self.rules:
            if rule.name == name:
                return rule
        return None

    def add_rule(
        self, rule: BaseTruncationRule, index: int | None = None
    ) -> None:
        if self.has_rule(rule.name):
            raise ValueError(f"Duplicate truncation rule: {rule.name}")
        if index is None:
            self.rules.append(rule)
        else:
            self.rules.insert(index, rule)

    def configure(self, proposal) -> None:
        for rule in self.rules:
            rule.configure(proposal)

    def prepare(self, proposal, worst_point, radius=None):
        self.reset()
        for rule in self.rules:
            rule.prepare(proposal, worst_point, radius=radius)

    def apply_latent(self, proposal, z):
        for rule in self.rules:
            z = rule.apply_latent(proposal, z)
        return z

    def apply_after_backward(self, proposal, x, log_q, z):
        for rule in self.rules:
            x, log_q, z = rule.apply_after_backward(proposal, x, log_q, z)
        return x, log_q, z

    def apply_after_likelihood(self, proposal, x, log_q, z):
        for rule in self.rules:
            x, log_q, z = rule.apply_after_likelihood(proposal, x, log_q, z)
        return x, log_q, z

    def reset(self) -> None:
        for rule in self.rules:
            rule.reset()
