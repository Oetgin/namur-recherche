#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2026 Pynguin Contributors
#
#  SPDX-License-Identifier: MIT
#
from collections.abc import Callable
from datetime import datetime
from typing import Tuple

HEADER = "sample,experiment,result_duration,result_success,result_score,exported_at"


def clean_n(data: str, n: int) -> str:
    """Clean the classic experiment data (keep the n last headers)"""
    return "".join(data.split(HEADER)[-n:])


def clean_default(data: str) -> str:
    """Clean the data (keep the last header)"""
    return "".join(data.split(HEADER)[-1:])


def process_value(value: str, header: str) -> str:
    """Process the value based on the header"""
    if header == "result_duration":
        value_time = datetime.strptime(value, "%H:%M:%S.%f")
        return str(
            value_time.hour * 3600
            + value_time.minute * 60
            + value_time.second
            + value_time.microsecond / 1e6
        )
    if header == "result_success":
        return str(int(value == "True"))
    if header == "result_score" and value == "":
        return "0"
    if header == "sample":
        value = value.split(" ")[-1]
        if value.startswith("torch."):
            return ".".join(value.split(".")[1:])
        return value
    if header == "experiment" and value.startswith("Prompt experiment"):
        return value.split(" ")[-1].strip("()")
    return value


def split_default(data: str) -> Tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Split the data in two"""
    lines = data.splitlines()
    half = len(lines) // 2
    header_split = HEADER.split(",")
    base: list[dict[str, str]] = []
    ours: list[dict[str, str]] = []
    for row, line in enumerate(lines):
        values = line.split(",")
        if len(values) != len(header_split):
            continue
        current = {}
        for col, value in enumerate(values):
            processed_value = process_value(value, header_split[col])
            current[header_split[col]] = processed_value
        if row < half:
            ours.append(current)
        else:
            base.append(current)
    return ours, base


def split_prompt(data: str) -> list[list[dict[str, str]]]:
    """Split the data by prompt"""
    lines = data.splitlines()
    header_split = HEADER.split(",")
    prompt_data: dict[str, list[dict[str, str]]] = {}
    for line in lines:
        values = line.split(",")
        if len(values) != len(header_split):
            continue
        current = {}
        for col, value in enumerate(values):
            processed_value = process_value(value, header_split[col])
            current[header_split[col]] = processed_value
        prompt_name = current["experiment"].split(" ")[-1].strip("()")
        if prompt_name not in prompt_data:
            prompt_data[prompt_name] = []
        prompt_data[prompt_name].append(current)
    split_data = list(prompt_data.values())
    return split_data


def split_model(data: str) -> list[list[dict[str, str]]]:
    """Split the data by models"""
    lines = data.splitlines()
    header_split = HEADER.split(",")
    model_data: dict[str, list[dict[str, str]]] = {}
    for line in lines:
        values = line.split(",")
        if len(values) != len(header_split):
            continue
        current = {}
        for col, value in enumerate(values):
            processed_value = process_value(value, header_split[col])
            current[header_split[col]] = processed_value
        model_name = current["experiment"].split(" ")[-1].strip("()")
        if model_name not in model_data:
            model_data[model_name] = []
        model_data[model_name].append(current)
    split_data = list(model_data.values())
    return split_data


class ExperimentData:
    def __init__(
        self,
        file_name: str,
        data_cleaner: Callable[[str], str] | None = clean_default,
        splitter: Callable[[str], list[list[dict[str, str]]]] | None = None,
    ):
        self.file_name = file_name
        self.data = data_cleaner(self.load_data()) if data_cleaner else self.load_data()
        self.split_data = splitter(self.data) if splitter else []

    def load_data(self):
        # Load the data from the file and return it
        try:
            with open(self.file_name, "r") as f:
                data = f.read()
        except FileNotFoundError:
            with open(f"analysis/{self.file_name}", "r") as f:
                data = f.read()
        return data

    @property
    def samples(self) -> list[str]:
        """Return the list of samples in the experiment"""
        if self.split_data is None:
            return []
        return list(set(row["sample"] for method in self.split_data for row in method))

    @property
    def methods(self) -> list[str]:
        """Return the list of methods in the experiment"""
        if self.split_data is None:
            return []
        return list(
            set(row["experiment"] for method in self.split_data for row in method)
        )


class ExperimentDataWithBase(ExperimentData):
    def __init__(
        self,
        file_name: str,
        splitter: Callable[
            [str], Tuple[list[dict[str, str]], list[dict[str, str]]]
        ] = split_default,
        data_cleaner: Callable[[str], str] | None = clean_default,
    ):
        super().__init__(file_name, data_cleaner)
        self.ours, self.base = splitter(self.data)

    @property
    def samples(self) -> list[str]:
        """Return the list of samples in the experiment"""
        return list(set(row["sample"] for row in self.ours + self.base))


MODEL_EXPERIMENT = ExperimentData(
    "model_results.csv", data_cleaner=lambda x: clean_n(x, 2), splitter=split_model
)
CLASSIC_EXPERIMENT = ExperimentDataWithBase(
    "classic_results.csv", data_cleaner=lambda x: clean_n(x, 2)
)
PROMPT_EXPERIMENT = ExperimentData(
    "prompt_results.csv", data_cleaner=lambda x: clean_n(x, 2), splitter=split_prompt
)


EXPERIMENTS = [
    MODEL_EXPERIMENT,
    CLASSIC_EXPERIMENT,
    PROMPT_EXPERIMENT,
]
