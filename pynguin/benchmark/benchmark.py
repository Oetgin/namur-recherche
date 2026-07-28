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
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from pynguin import configuration as config
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


class Dataset:
    """Manages a set of code samples to run experiments on."""

    def __init__(self, *paths: Path) -> None:
        """Creates a dataset.

        Args:
            paths (Path): Paths of the dataset, including projects and modules.
        """
        self._paths = list(paths)

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
    def _get_module_root_path(module_path: Path) -> Path:
        if module_path == Path("/"):
            raise ValueError("Could not find module root path")

        if module_path.is_file():
            return Dataset._get_module_root_path(module_path.parent)
        if module_path.is_dir() and not (module_path.parent / "__init__.py").exists():
            return module_path
        return Dataset._get_module_root_path(module_path.parent)

    @staticmethod
    @contextmanager
    def _prepend_sys_path(path: Path):
        sys.path.insert(0, str(path))
        try:
            yield
        finally:
            del sys.path[0]

    def _get_samples_from_dir(
        self, dir_path: Path, type_inference_strategy: TypeInferenceStrategy
    ) -> Iterator[Sample]:
        for candidate in dir_path.rglob("*.py"):
            if not candidate.is_file():
                continue
            candidate_root = self._module_import_root(dir_path, candidate)
            candidate_name = self._module_name(candidate_root, candidate)
            if any(module_name in candidate_name for module_name in ("setup", "__init__")):
                _LOGGER.debug("Skipped module %s", candidate_name)
                continue
            _LOGGER.debug(
                "Generating test cluster for %s (%s)",
                candidate_name,
                candidate_root,
            )
            try:
                with self._prepend_sys_path(candidate_root):
                    sample = Sample(
                        generate_test_cluster(candidate_name, type_inference_strategy),
                        candidate_name,
                        candidate_root,
                        candidate.read_text(encoding="utf-8"),
                    )
            except ModuleNotFoundError:
                candidate_root = candidate_root.parent
                candidate_name = Dataset._module_name(candidate_root, candidate)
                _LOGGER.debug(
                    "Retrying generating test cluster for %s (%s)",
                    candidate_name,
                    candidate_root,
                )
                with self._prepend_sys_path(candidate_root):
                    sample = Sample(
                        generate_test_cluster(candidate_name, type_inference_strategy),
                        candidate_name,
                        candidate_root,
                        candidate.read_text(encoding="utf-8"),
                    )
            yield sample

    def _get_samples_from_path(
        self, path: Path, type_inference_strategy: TypeInferenceStrategy
    ) -> Iterator[Sample]:
        for module_path in self._get_projects_from_path(path):
            try:
                if module_path.is_file():
                    module_root = Dataset._get_module_root_path(module_path)
                    module_name = Dataset._module_name(module_root, module_path)
                else:
                    yield from self._get_samples_from_dir(module_path, type_inference_strategy)
                    continue
            except ModuleNotFoundError:
                _LOGGER.exception("Error generating test cluster for %s", module_path)
                continue

            _LOGGER.debug(
                "Generating test cluster for %s (%s)",
                module_name,
                module_root,
            )
            try:
                with self._prepend_sys_path(module_root):
                    sample = Sample(
                        generate_test_cluster(module_name, type_inference_strategy),
                        module_name,
                        module_root,
                        module_path.read_text(encoding="utf-8"),
                    )
            except ModuleNotFoundError:
                module_root = module_root.parent
                module_name = Dataset._module_name(module_root, module_path)
                _LOGGER.debug(
                    "Retrying generating test cluster for %s (%s)",
                    module_name,
                    module_root,
                )
                with self._prepend_sys_path(module_root):
                    sample = Sample(
                        generate_test_cluster(module_name, type_inference_strategy),
                        module_name,
                        module_root,
                        module_path.read_text(encoding="utf-8"),
                    )
            yield sample

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
        for path in self._paths:
            if not path.exists():
                _LOGGER.warning("Path %s does not exist, skipping", path)
                continue
            yield from self._get_samples_from_path(path, type_inference_strategy)

    def __repr__(self) -> str:
        return f"Dataset(path={self._paths})"


class BenchmarkExperimentResult:
    """Result of a benchmark experiment.

    Attributes:
        score (float | None):
            Experiment score, to compare the execution with other instances.
        success (bool | None):
            Whether the experiment ended in a success (`True`) or failed (`False`).
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
                Whether the experiment ended in a success (`True`) or failed (`False`).
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

    def __init__(
        self,
        experiments: list[T],
        dataset: Dataset,
        *,
        n_runs: int = 1,
        max_samples: int | None = None,
    ) -> None:
        """Instanciate a Benchmark suite.

        Args:
            experiments (list[T: BenchmarkExperiment]):
                List of experiments to run.
            dataset (Dataset):
                Dataset of code samples to run the experiments on.
            n_runs (int, optional):
                Number of runs.
                Defaults to 1.
            max_samples (int | None, optional):
                Maximum number of samples to run.
                Defaults to None.
        """
        self._experiments = experiments
        self._dataset = dataset
        self._n_runs = n_runs
        self._max_samples = max_samples
        self._results: dict[Sample, dict[BenchmarkExperiment, list[BenchmarkExperimentResult]]] = {}

    def run(self) -> None:
        """Run the Benchmark suite and collect results."""
        _LOGGER.info("Setting up benchmark")
        self._setup()
        _LOGGER.info("Running benchmark")

        console = Console()
        status = console.status("Processing sample...")
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            SpinnerColumn("simpleDotsScrolling"),
            TextColumn("[progress.percentage]({task.completed}/{task.total})"),
            transient=True,
        )
        with Live(Panel(Group(status, progress)), transient=True):
            for sample in self._dataset.get_samples():
                if self._max_samples is not None and len(self._results) >= self._max_samples:
                    break
                status.update(f"Benchmarking on module {sample.module_name}")
                sample_task = progress.add_task("Running experiments", total=len(self._experiments))
                self._results[sample] = {}

                for experiment in self._experiments:
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
                status.update("Processing sample...")
        _LOGGER.info("Benchmark completed")

    def _setup(self):
        """Setup all experiments."""
        for experiment in self._experiments:
            experiment.setup()
        config.configuration.statistics_output.create_coverage_report = True

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
            n_runs={self._n_runs})""")


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
