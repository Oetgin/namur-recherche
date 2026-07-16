#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2026 Pynguin Contributors
#
#  SPDX-License-Identifier: MIT
#
"""Base for all benchmarks."""

import abc
import importlib.machinery
import importlib.metadata
import importlib.util
import datetime
import logging
import sys
import textwrap
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from types import ModuleType
from typing import Concatenate, ParamSpec, TypeVar

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from pynguin.analyses.module import (
    ModuleTestCluster,
    _ModuleParseResult,
    analyse_module,
    read_module_ast,
)
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

    def __init__(self, path: Path, *args) -> None:
        """Creates a dataset.

        Args:
            path (Path): The path to the dataset, including projects and modules.
                If a path is a directory, all subdirectories and Python files will be included
                recursively.
                If a path is a Python file, it will be included as a sample.
            *args: Additional paths to include in the dataset.
                Non :class:`Path` arguments will be ignored.
        """
        paths: list[Path] = [path]
        paths.extend(arg for arg in args if isinstance(arg, Path))
        self._paths = paths

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
    def _module_root_for_file(module_path: Path) -> Path:
        module_root = module_path.parent
        last_package_root = None
        while (module_root / "__init__.py").exists() and module_root.parent != module_root:
            last_package_root = module_root
            module_root = module_root.parent
        return module_root if last_package_root is not None else module_path.parent

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

    @staticmethod
    def _load_module_from_file(module_name: str, module_path: Path) -> ModuleType:
        package_names: list[str] = []
        package_parts: list[str] = []
        for part in module_name.split(".")[:-1]:
            package_parts.append(part)
            package_names.append(".".join(package_parts))
        for index, package_name in enumerate(package_names):
            package_path = module_path.parents[len(package_names) - index - 1]
            init_path = package_path / "__init__.py"
            package = sys.modules.get(package_name)
            if package is None:
                package = ModuleType(package_name)
                package.__path__ = [str(package_path)]  # type: ignore[attr-defined]
                package.__package__ = package_name
                package.__spec__ = importlib.machinery.ModuleSpec(
                    package_name,
                    loader=None,
                    is_package=True,
                )
                package.__spec__.submodule_search_locations = [str(package_path)]
                if "." not in package_name:
                    try:
                        package.__version__ = importlib.metadata.version(package_name)
                    except importlib.metadata.PackageNotFoundError:
                        pass
                sys.modules[package_name] = package
            if init_path.is_file() and not getattr(package, "__benchmark_initialized__", False):
                spec = importlib.util.spec_from_file_location(
                    package_name,
                    init_path,
                    submodule_search_locations=[str(package_path)],
                )
                if spec is not None and spec.loader is not None:
                    package.__spec__ = spec
                    try:
                        spec.loader.exec_module(package)
                    except Exception as error:  # noqa: BLE001
                        _LOGGER.debug(
                            "Could not fully initialize package %s from %s (%s)",
                            package_name,
                            init_path,
                            error,
                        )
                    finally:
                        package.__benchmark_initialized__ = True  # type: ignore[attr-defined]

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create import specification for {module_name}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    @classmethod
    def _parse_module_from_file(cls, module_name: str, module_path: Path) -> _ModuleParseResult:
        module = cls._load_module_from_file(module_name, module_path)
        syntax_tree: None | object = None
        linenos = -1
        try:
            syntax_tree, source_code = read_module_ast(str(module_path), module_name)
        except (
            TypeError,
            OSError,
            Exception,
        ) as error:
            _LOGGER.debug(
                f"Could not retrieve source code for module {module_name} ({error}). "
                f"Cannot derive syntax tree to allow Pynguin using more precise analysis."
            )
        else:
            linenos = len(source_code.splitlines())

        return _ModuleParseResult(
            linenos=linenos,
            module_name=module_name,
            module=module,
            syntax_tree=syntax_tree,
        )

    def _get_samples_from_module_path(
        self,
        module_path: Path,
        type_inference_strategy: TypeInferenceStrategy,
    ) -> Iterator[Sample]:
        if module_path.is_file():
            module_root = self._module_root_for_file(module_path)
            module_name = self._module_name(module_root, module_path)
            yield Sample(
                analyse_module(
                    self._parse_module_from_file(module_name, module_path),
                    type_inference_strategy,
                ),
                module_name,
                module_root,
                module_path.read_text(encoding="utf-8"),
            )
            return

        for candidate in module_path.rglob("*.py"):
            if not candidate.is_file():
                continue
            candidate_root = self._module_import_root(module_path, candidate)
            candidate_name = self._module_name(module_path, candidate)
            if any(module_name in candidate_name for module_name in ("setup", "__init__")):
                _LOGGER.debug("Skipped module %s", candidate_name)
                continue
            with self._prepend_sys_path(candidate_root):
                _LOGGER.debug(
                    "Generating test cluster for %s (%s)",
                    candidate_name,
                    candidate_root,
                )
                yield Sample(
                    analyse_module(
                        self._parse_module_from_file(candidate_name, candidate),
                        type_inference_strategy,
                    ),
                    candidate_name,
                    candidate_root,
                    candidate.read_text(encoding="utf-8"),
                )

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
            for module_path in self._get_projects_from_path(path):
                try:
                    yield from self._get_samples_from_module_path(
                        module_path, type_inference_strategy
                    )
                    continue
                except ModuleNotFoundError as error:
                    _LOGGER.warning(
                        "Skipping %s because a dependency is missing: %s",
                        module_path,
                        error,
                    )
                    continue
                except ImportError as error:
                    _LOGGER.warning(
                        "Skipping %s because it could not be imported: %s",
                        module_path,
                        error,
                    )
                    continue
                except Exception:
                    _LOGGER.exception("Error generating test cluster for %s", module_path)
                    continue

    def __repr__(self) -> str:
        return f"Dataset(path={self._paths})"


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
