#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2024 Pynguin Contributors
#
#  SPDX-License-Identifier: MIT
#
"""Provides class prompt for generating tests for a module."""

import textwrap

from pynguin.large_language_model.prompts.prompt import Prompt
from pynguin.utils.generic.genericaccessibleobject import (
    GenericCallableAccessibleObject,
    GenericConstructor,
    GenericFunction,
    GenericMethod,
)


class UncoveredTargetsPrompt(Prompt):
    """Implementation prompt for generating tests for a module."""

    def __init__(
        self,
        callables: list[GenericCallableAccessibleObject],
        module_code: str,
        module_path: str,
    ):
        """Initializes the prompt.

        Args:
            callables (list[GenericCallableAccessibleObject]): List of
                uncovered callables.
            module_path (str): Path to the module.
            module_code (str): Source code of the module.
        """
        super().__init__(module_code, module_path)
        self.callables: list[GenericCallableAccessibleObject] = callables
        self.module_path = module_path
        self.module_code = module_code

    def build_callables_prompt_section(self) -> list[str]:
        """Generates a list of function headers and their signatures.

        Returns:
            list[str]: A list of formatted function headers with their signatures.
        """
        callables_list = []

        for gao in self.callables:
            signature = str(gao.inferred_signature)
            if gao.is_method() and isinstance(gao, GenericMethod):
                method_gao: GenericMethod = gao
                callable_list_item = (
                    f"- The method {method_gao.method_name} of class "
                    f"{method_gao.owner.name}{signature}"
                )
            elif gao.is_function() and isinstance(gao, GenericFunction):
                fn_gao: GenericFunction = gao
                callable_list_item = f"- The function {fn_gao.function_name}{signature}"
            elif gao.is_constructor() and isinstance(gao, GenericConstructor):
                constructor_gao: GenericConstructor = gao
                class_name = constructor_gao.owner.name  # type: ignore[union-attr]
                callable_list_item = f"- The constructor of the class {class_name}{signature}"
            else:
                continue  # Skip unknown callable types

            callables_list.append(callable_list_item)

        return callables_list

    def build_prompt(self) -> str:
        """Builds the prompt message."""
        callables_list = self.build_callables_prompt_section()
        callables_section = "\n".join(callables_list)

        return textwrap.dedent(f"""
            You are writing tests to be used as seed for a SBST algorithm. Your goal is to improve as much as possible the coverage of the tests.
            Write unit tests for the following callables that Pynguin failed to cover:
            {callables_section}
            Module path: `{self.module_path}`
            Module source code: `{self.module_code}`
            You answer will be parsed for mutations, so here are the guidelines you need to follow :
            - Answer in a code block only using; one function for each test case, with NO ARGUMENTS, NO HELPER FUNCTIONS AND NO CLASSES.
            - If needed, instantiate vars in the body of the test func or use pytest.parametrize, but DO NOT USE ANY OTHER PYTEST FEATURE (e.g. DO NOT USE FIXTURES), as that will make the parsing fail.
            - Do not rewrite the SUT's code in the tests. If you want for example to call a function or instanciante a class, import it.
            - Don't explain what you do, just answer in simple, concise assertion tests, split in small functions, following the Arrange, Act, Assert pattern.

            Here are some examples; *NEVER  DO* :
            def func_all_tests(param):
              var0 = param
              ...

            *INSTEAD DO* :
            from module_to_test import func
            def test_feat1():
              var0 = func()
              assert ...


            REMEMBER : NO ARGUMENTS IN YOUR TEST FUNCTIONS"
            """)  # noqa: E501
