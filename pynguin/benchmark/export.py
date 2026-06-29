import abc
import csv
from pathlib import Path

from benchmark.benchmark import BenchmarkExperiment, BenchmarkExperimentResult, Sample


class Exporter(abc.ABC):
    """Abstract class representing an exporter for benchmark results."""

    @classmethod
    @abc.abstractmethod
    def export(
        cls, results: dict[Sample, dict[BenchmarkExperiment, list[BenchmarkExperimentResult]]]
    ):
        """Exports results.

        Args:
            results (dict[Sample, dict[BenchmarkExperiment, list[BenchmarkExperimentResult]]]):
                Results from the BenchmarkSuite.
        """


class CSV(Exporter):
    """Exports benchmark results to CSV."""

    @classmethod
    def export(
        cls, results: dict[Sample, dict[BenchmarkExperiment, list[BenchmarkExperimentResult]]]
    ):
        """Exports results to CSV.

        Args:
            results (dict[Sample, dict[BenchmarkExperiment, list[BenchmarkExperimentResult]]]):
                Results from the BenchmarkSuite.
        """
        with Path.open(Path("results.csv"), "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "sample",
                "experiment",
                "result_duration",
                "result_success",
                "result_score",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for sample, experiments in results.items():
                for experiment, experiment_results in experiments.items():
                    for result in experiment_results:
                        writer.writerow(
                            {
                                "sample": sample,
                                "experiment": experiment,
                                "result_duration": result.duration,
                                "result_success": result.success,
                                "result_score": result.score,
                            }
                        )
