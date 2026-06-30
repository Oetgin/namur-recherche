#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2026 Pynguin Contributors
#
#  SPDX-License-Identifier: MIT
#
"""Base for all benchmarks."""

import abc
import datetime
import logging
import sys
import textwrap
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Concatenate, ParamSpec, TypeVar

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress

from pynguin.analyses.module import ModuleTestCluster, generate_test_cluster
from pynguin.configuration import TypeInferenceStrategy

_LOGGER = logging.getLogger(__name__)


class Sample:
    """Code sample to run an experiment on."""

    def __init__(
        self, test_cluster: ModuleTestCluster, module_name: str, module_root: Path, module_code: str
    ) -> None:
        """Initializes the sample.

        Args:
            test_cluster (ModuleTestCluster): The test cluster representing the sample.
            module_name (str): The module name (without .py).
            module_root (Path): The module root path.
            module_code (str): Code contained in the module.
        """
        self.test_cluster = test_cluster
        self.module_name = module_name
        self.module_root = module_root
        self.module_code = module_code

    def __str__(self) -> str:
        return f"Sample from module {self.module_name}"

    def __repr__(self) -> str:
        return textwrap.dedent(f"""Sample(
            test_cluster={self.test_cluster!r},
            module_name="{self.module_name}", 
            module_root={self.module_root!r}, 
            module_code={self.module_code!r})""")  # noqa: W291


class Dataset:
    """Manages a set of code samples to run experiments on."""

    def __init__(self, path: Path) -> None:
        """Creates a dataset.

        Args:
            path (Path): Path of the dataset, including projects and modules.
        """
        self._path = path

    @classmethod
    def _is_project_root(cls, path: Path) -> bool:
        return path.is_dir() and (
            (path / "setup.py").exists() or (path / "pyproject.toml").exists()
        )

    @classmethod
    def _get_projects_from_path(cls, path: Path) -> Iterator[Path]:
        if path.is_file():
            if path.suffix == ".py":
                yield path
            return

        if cls._is_project_root(path):
            yield path
            return

        for child in path.iterdir():
            if child.is_dir():
                yield from cls._get_projects_from_path(child)
            elif child.suffix == ".py":
                yield child

    @staticmethod
    def _module_import_root(project_path: Path, module_path: Path) -> Path:
        src_path = project_path / "src"
        if src_path.is_dir() and module_path.is_relative_to(src_path):
            return src_path
        return project_path

    @staticmethod
    def _module_name(module_root: Path, module_path: Path) -> str:
        relative_path = module_path.relative_to(module_root)
        relative_path = relative_path.with_suffix("")
        module_parts = [part for part in relative_path.parts if part != "src"]
        return ".".join(module_parts) if module_parts else module_path.stem

    @staticmethod
    @contextmanager
    def _prepend_sys_path(path: Path):
        sys.path.insert(0, str(path))
        try:
            yield
        finally:
            del sys.path[0]

    def get_samples(
        self, type_inference_strategy: TypeInferenceStrategy = TypeInferenceStrategy.TYPE_HINTS
    ) -> Iterator[Sample]:
        """Generator for code samples in the dataset.

        Args:
            type_inference_strategy (TypeInferenceStrategy, optional): Strategy for inferring types.
            Defaults to TypeInferenceStrategy.TYPE_HINTS.

        Yields:
            Iterator[Sample]: A sample in the dataset.
        """
        for module_path in self._get_projects_from_path(self._path):
            if module_path.is_file():
                module_root = module_path.parent
                module_name = module_path.stem
            else:
                for candidate in module_path.rglob("*.py"):
                    if not candidate.is_file():
                        continue
                    candidate_root = self._module_import_root(module_path, candidate)
                    candidate_name = self._module_name(module_path, candidate)
                    if any(module_name in candidate_name for module_name in ("setup", "__init__")):
                        _LOGGER.debug("Skipped module %s", candidate_name)
                        continue
                    with self._prepend_sys_path(candidate_root):
                        _LOGGER.debug("Generating test cluster for %s", candidate_name)
                        sample = Sample(
                            generate_test_cluster(candidate_name, type_inference_strategy),
                            candidate_name,
                            candidate_root,
                            candidate.read_text(encoding="utf-8"),
                        )
                        yield sample
                continue

            with self._prepend_sys_path(module_root):
                sample = Sample(
                    generate_test_cluster(module_name, type_inference_strategy),
                    module_name,
                    module_root,
                    module_path.read_text(encoding="utf-8"),
                )
                yield sample

    def __repr__(self) -> str:
        return f"Dataset(path={self._path})"


class BenchmarkExperimentResult:
    """Result of a benchmark experiment.

    Attributes:
        score (float | None):
            Experiment score, to compare the execution with other instances.
        success (bool | None):
            Wether the experiment ended in a success (`True`) or failed (`False`).
        duration (:class:`~datetime.timedelta` | None):
            Experiment duration.
    """

    def __init__(
        self,
        score: float | None = None,
        *,
        success: bool | None = None,
    ) -> None:
        """Initializes the experiment result.

        If used only for measuring execution time, return a BenchmarkExperimentResult initialized
        with empty arguments and use the :decorator:`@BenchmarkExperiment._measure` decorator.

        Args:
            score (float | None, optional):
                Experiment score, to compare the execution with other instances.
                Defaults to None.
            success (bool | None, optional):
                Wether the experiment ended in a success (`True`) or failed (`False`).
                Defaults to None.
        """
        self.score = score
        self.success = success
        self.duration: datetime.timedelta | None = None


class BenchmarkExperiment(abc.ABC):
    """Base class for benchmark experiments, to be ran in BenchMark suites."""

    @abc.abstractmethod
    def setup(self):
        """Execute setup code.

        To be executed before running the experiment.
        """

    @abc.abstractmethod
    def run(self, sample: Sample) -> BenchmarkExperimentResult:
        """Run the experiment once on the provided sample.

        Returns:
            BenchmarkExperimentResult: The experiment results.
        """


T = TypeVar("T", bound=BenchmarkExperiment)


class BenchmarkSuite:
    """Runs benchmark experiments on a dataset and collects results."""

    def __init__(self, experiments: list[T], dataset: Dataset, *, n_runs: int = 1) -> None:
        """Instanciate a Benchmark suite.

        Args:
            experiments (list[T: BenchmarkExperiment]):
                List of experiments to run.
            dataset (Dataset):
                Dataset of code samples to run the experiments on.
            n_runs (int, optional):
                Number of runs.
                Defaults to 1.
        """
        self._experiments = experiments
        self._dataset = dataset
        self._n_runs = n_runs
        self._results: dict[Sample, dict[BenchmarkExperiment, list[BenchmarkExperimentResult]]] = {}

    def run(self) -> None:
        """Run the Benchmark suite and collect results."""
        _LOGGER.info("Setting up benchmark")
        self._setup()
        _LOGGER.info("Running benchmark")

        console = Console()
        status = console.status("Processing samples...")
        progress = Progress(transient=True)
        with Live(Panel(Group(status, progress))):
            for sample in self._dataset.get_samples():
                status.update(f"Benchmarking on module {sample.module_name}")
                sample_task = progress.add_task("Running experiments", total=len(self._experiments))

                for experiment in self._experiments:
                    self._results[sample] = {}
                    if self._n_runs != 1:
                        experiment_task = progress.add_task(
                            "Experiment progress", total=self._n_runs
                        )
                        results: list[BenchmarkExperimentResult] = []
                        for _ in range(self._n_runs):
                            results.append(experiment.run(sample))
                            progress.update(experiment_task, advance=1)
                        progress.remove_task(experiment_task)
                    else:
                        results = [experiment.run(sample)]

                    self._results[sample][experiment] = results
                    progress.update(sample_task, advance=1)

                progress.remove_task(sample_task)
            status.update("[bold green]Benchmark completed")

    def _setup(self):
        """Setup all experiments."""
        for experiment in self._experiments:
            experiment.setup()

    @property
    def results(self) -> dict[Sample, dict[BenchmarkExperiment, list[BenchmarkExperimentResult]]]:
        """Returns results as dicts containing :class:`Sample`, :class:`BenchmarkExperiment` and a
        list of :class:`BenchmarkExperimentResult`.
        """  # noqa: D205
        return self._results

    def __repr__(self) -> str:
        return textwrap.dedent(f"""BenchmarkSuite(
            experiments={self._experiments!r}, 
            dataset={self._dataset!r}, 
            n_runs={self._n_runs})""")  # noqa: W291


P = ParamSpec("P")


def measure_exec_time(
    func: Callable[Concatenate[T, P], BenchmarkExperimentResult],
) -> Callable[Concatenate[T, P], BenchmarkExperimentResult]:
    """Decorator for measuring execution time.

    Overrides the returned :attr:`BenchmarkExperimentResult.duration` attribute.
    The decorated method must inherit :class:`BenchmarkExperiment`,
    and return :class:`BenchmarkExperimentResult`.
    """

    @wraps(func)
    def wrapper(self: T, *args: P.args, **kwargs: P.kwargs) -> BenchmarkExperimentResult:
        start_time = time.perf_counter()
        result: BenchmarkExperimentResult = func(self, *args, **kwargs)
        end_time = time.perf_counter()
        result.duration = datetime.timedelta(seconds=end_time - start_time)
        return result

    return wrapper
