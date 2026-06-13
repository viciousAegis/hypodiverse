from __future__ import annotations

import math
from dataclasses import dataclass
from random import Random


@dataclass(frozen=True)
class EvidenceSummary:
    key: str
    samples: int
    latest_signal: float | None
    mean_signal: float | None
    log_odds: float
    posterior: float
    status: str


class GaussianEvidenceModel:
    def __init__(
        self,
        true_mean: float,
        false_mean: float,
        sigma: float,
        accept_threshold: float,
        reject_threshold: float,
    ) -> None:
        self.true_mean = true_mean
        self.false_mean = false_mean
        self.sigma = sigma
        self.accept_threshold = accept_threshold
        self.reject_threshold = reject_threshold

    def sample(self, rng: Random, is_true: bool) -> float:
        mean = self.true_mean if is_true else self.false_mean
        return rng.gauss(mean, self.sigma)

    def log_likelihood_ratio(self, signal: float) -> float:
        sigma2 = self.sigma * self.sigma
        true_term = -((signal - self.true_mean) ** 2) / (2 * sigma2)
        false_term = -((signal - self.false_mean) ** 2) / (2 * sigma2)
        return true_term - false_term

    def status(self, posterior: float) -> str:
        if posterior >= self.accept_threshold:
            return "accepted"
        if posterior <= self.reject_threshold:
            return "rejected"
        return "unresolved"


class EvidenceStore:
    def __init__(self, model: GaussianEvidenceModel) -> None:
        self.model = model
        self._log_odds: dict[str, float] = {}
        self._samples: dict[str, int] = {}
        self._signals: dict[str, list[float]] = {}

    def update(self, key: str, signal: float) -> EvidenceSummary:
        self._log_odds[key] = self._log_odds.get(
            key, 0.0
        ) + self.model.log_likelihood_ratio(signal)
        self._samples[key] = self._samples.get(key, 0) + 1
        self._signals.setdefault(key, []).append(signal)
        return self.summary(key)

    def posterior(self, key: str) -> float:
        log_odds = self._log_odds.get(key, 0.0)
        if log_odds >= 0:
            return 1.0 / (1.0 + math.exp(-log_odds))
        exp_value = math.exp(log_odds)
        return exp_value / (1.0 + exp_value)

    def summary(self, key: str) -> EvidenceSummary:
        posterior = self.posterior(key)
        signals = self._signals.get(key, [])
        latest_signal = signals[-1] if signals else None
        mean_signal = sum(signals) / len(signals) if signals else None
        return EvidenceSummary(
            key=key,
            samples=self._samples.get(key, 0),
            latest_signal=latest_signal,
            mean_signal=mean_signal,
            log_odds=self._log_odds.get(key, 0.0),
            posterior=posterior,
            status=self.model.status(posterior),
        )

    def summaries(self) -> list[EvidenceSummary]:
        return [self.summary(key) for key in sorted(self._samples)]

    def is_accepted(self, key: str) -> bool:
        return self.summary(key).status == "accepted"
