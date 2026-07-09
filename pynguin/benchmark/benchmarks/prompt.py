#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2026 Pynguin Contributors
#
#  SPDX-License-Identifier: MIT
#
"""LLM model benchmarking."""

import logging

from benchmark.benchmark import (
    BenchmarkExperiment,
    BenchmarkExperimentResult,
    Sample,
    measure_exec_time,
)

_LOGGER = logging.getLogger(__name__)


class PromptBenchmarkExperiment(BenchmarkExperiment):
    """Benchmarks the performance of a prompt."""

    def __init__(
        self,
    ) -> None:
        """Benchmark the performance of a prompt.

        Args:
            ...
        """
        raise NotImplementedError("PromptBenchmarkExperiment is not implemented yet.")

    def setup(self):  # noqa: D102
        raise NotImplementedError("Setup method is not implemented for PromptBenchmarkExperiment.")

    @measure_exec_time
    def run(self, sample: Sample) -> BenchmarkExperimentResult:  # noqa: D102
        _LOGGER.debug("Running prompt experiment")
        _LOGGER.debug("Sample: %s", sample)
        try:
            test_cluster = sample.test_cluster

            pass

            coverage_after = NotImplemented

            return BenchmarkExperimentResult(success=True, score=coverage_after)

        except Exception:
            _LOGGER.exception("Error running prompt experiment")
            return BenchmarkExperimentResult(success=False)

    def __str__(self) -> str:
        raise NotImplementedError("Str method is not implemented for PromptBenchmarkExperiment.")

    def __repr__(self) -> str:
        raise NotImplementedError("Repr method is not implemented for PromptBenchmarkExperiment.")
