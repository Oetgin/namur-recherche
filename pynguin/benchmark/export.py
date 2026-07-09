#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2026 Pynguin Contributors
#
#  SPDX-License-Identifier: MIT
#
"""Exporters for benchmark results."""

import abc
import csv
import logging
from pathlib import Path
from time import localtime, strftime

from benchmark.benchmark import BenchmarkExperiment, BenchmarkExperimentResult, Sample

_LOGGER = logging.getLogger(__name__)


class Exporter(abc.ABC):
    """Abstract class representing an exporter for benchmark results."""

    @classmethod
    @abc.abstractmethod
    def export(
        cls,
        results: dict[Sample, dict[BenchmarkExperiment, list[BenchmarkExperimentResult]]],
        file_path: Path | None = None,
    ):
        """Exports results.

        Args:
            results (dict[Sample, dict[BenchmarkExperiment, list[BenchmarkExperimentResult]]]):
                Results from the BenchmarkSuite.
            file_path (Path | None): Path to the output CSV file.
        """


class CSV(Exporter):
    """Exports benchmark results to CSV."""

    @classmethod
    def export(  # noqa: D102
        cls,
        results: dict[Sample, dict[BenchmarkExperiment, list[BenchmarkExperimentResult]]],
        file_path: Path | None = None,
    ):
        if file_path is None:
            file_path = Path("results.csv")

        _LOGGER.info("Exporting results to %s", file_path)

        with Path.open(file_path, "a", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "sample",
                "experiment",
                "result_duration",
                "result_success",
                "result_score",
                "exported_at",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for sample, experiments in results.items():
                _LOGGER.debug("Exporting results for sample: %s", sample)
                for experiment, experiment_results in experiments.items():
                    _LOGGER.debug("Exporting results for experiment: %s", experiment)
                    for result in experiment_results:
                        writer.writerow({
                            "sample": sample,
                            "experiment": experiment,
                            "result_duration": result.duration,
                            "result_success": result.success,
                            "result_score": result.score,
                            "exported_at": strftime("%d-%m-%Y_%H-%M-%S", localtime()),
                        })
