# -*- coding: utf-8 -*-
# @Date    : 2025-03-31
# @Author  : didi & zhaoyang
# @Desc    : operator demo of aflow

import asyncio
import concurrent.futures
import random
import sys
import traceback
from collections import Counter
from typing import Dict, List, Tuple, Optional

from tenacity import retry, stop_after_attempt, wait_fixed

from scripts.async_llm import AsyncLLM
from scripts.logs import logger
from scripts.formatter import BaseFormatter, FormatError, XmlFormatter, TextFormatter, CodeFormatter
from scripts.operator_an import (
    AnswerGenerateOp,
    CodeGenerateOp,
    FormatOp,
    GenerateOp,
    MdEnsembleOp,
    ReflectionTestOp,
    ReviewOp,
    ReviseOp,
    ScEnsembleOp,
) # All BaseModel

from scripts.prompts.prompt import (
    ANSWER_GENERATION_PROMPT,
    FORMAT_PROMPT,
    MD_ENSEMBLE_PROMPT,
    PYTHON_CODE_VERIFIER_PROMPT,
    REFLECTION_ON_PUBLIC_TEST_PROMPT,
    REVIEW_PROMPT,
    REVISE_PROMPT,
    SC_ENSEMBLE_PROMPT,
)
from scripts.utils.code import (
    extract_test_cases_from_jsonl,
    test_case_2_test_function,
)

class Operator:
    def __init__(self, llm: AsyncLLM, name: str):
        self.name = name
        self.llm = llm

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    async def _fill_node(self, op_class, prompt, mode=None, **extra_kwargs):
        # Create appropriate formatter based on mode
        formatter = self._create_formatter(op_class, mode, **extra_kwargs)
        
        try:
            # Use the formatter with AsyncLLM
            if formatter:
                response = await self.llm.call_with_format(prompt, formatter)
            else:
                # Fallback to direct call if no formatter is needed
                response = await self.llm(prompt)
                
            # Convert to expected format based on the original implementation
            if isinstance(response, dict):
                return response
            else:
                return {"response": response}
        except FormatError as e:
            print(f"Format error in {self.name}: {str(e)}")
            return {"error": str(e)}
    
    def _create_formatter(self, op_class, mode=None, **extra_kwargs) -> Optional[BaseFormatter]:
        """Create appropriate formatter based on operation class and mode"""
        if mode == "xml_fill":
            return XmlFormatter.from_model(op_class)
        elif mode == "code_fill":
            function_name = extra_kwargs.get("function_name")
            return CodeFormatter(function_name=function_name)
        elif mode == "single_fill":
            return TextFormatter()
        else:
            # Return None if no specific formatter is needed
            return None


class Custom(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "Custom"):
        super().__init__(llm, name)

    async def __call__(self, input, instruction):
        prompt = instruction + input
        response = await self._fill_node(GenerateOp, prompt, mode="single_fill")
        # Standardize return format
        if isinstance(response, dict):
            if "error" in response:
                response["success"] = False
            else:
                response["success"] = True
            # Ensure response field exists
            if "response" not in response and "content" in response:
                response["response"] = response["content"]
        return response


class AnswerGenerate(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "AnswerGenerate"):
        super().__init__(llm, name)

    async def __call__(self, input: str) -> Tuple[str, str]:
        prompt = ANSWER_GENERATION_PROMPT.format(input=input)
        response = await self._fill_node(AnswerGenerateOp, prompt, mode="xml_fill")
        # Standardize return format with success flag
        if isinstance(response, dict):
            if "error" in response:
                response["success"] = False
            else:
                response["success"] = True
            # Ensure answer field exists
            if "answer" not in response and "response" in response:
                response["answer"] = response["response"]
        return response


class CustomCodeGenerate(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "CustomCodeGenerate"):
        super().__init__(llm, name)

    async def __call__(self, problem, entry_point, instruction):
        prompt = instruction + problem
        response = await self._fill_node(GenerateOp, prompt, mode="code_fill", function_name=entry_point)
        return response


class ScEnsemble(Operator):
    """
    Paper: Self-Consistency Improves Chain of Thought Reasoning in Language Models
    Link: https://arxiv.org/abs/2203.11171
    Paper: Universal Self-Consistency for Large Language Model Generation
    Link: https://arxiv.org/abs/2311.17311
    """

    def __init__(self, llm: AsyncLLM, name: str = "ScEnsemble"):
        super().__init__(llm, name)

    async def __call__(self, solutions: List[str], problem: str):
        answer_mapping = {}
        solution_text = ""
        for index, solution in enumerate(solutions):
            answer_mapping[chr(65 + index)] = index
            solution_text += f"{chr(65 + index)}: \n{str(solution)}\n\n\n"

        prompt = SC_ENSEMBLE_PROMPT.format(question=problem, solutions=solution_text)
        response = await self._fill_node(ScEnsembleOp, prompt, mode="xml_fill")

        answer = response.get("solution_letter", "")
        answer = answer.strip().upper()

        return {"response": solutions[answer_mapping[answer]]}


def run_code(code):
    try:
        # Create a new global namespace
        global_namespace = {}

        disallowed_imports = [
            "os",
            "sys",
            "subprocess",
            "multiprocessing",
            "matplotlib",
            "seaborn",
            "plotly",
            "bokeh",
            "ggplot",
            "pylab",
            "tkinter",
            "PyQt5",
            "wx",
            "pyglet",
        ]

        # Check for prohibited imports
        for lib in disallowed_imports:
            if f"import {lib}" in code or f"from {lib}" in code:
                logger.info("Detected prohibited import: %s", lib)
                return "Error", f"Prohibited import: {lib} and graphing functionalities"

        # Use exec to execute the code
        exec(code, global_namespace)
        # Assume the code defines a function named 'solve'
        if "solve" in global_namespace and callable(global_namespace["solve"]):
            result = global_namespace["solve"]()
            return "Success", str(result)
        else:
            return "Error", "Function 'solve' not found"
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb_str = traceback.format_exception(exc_type, exc_value, exc_traceback)
        return "Error", f"Execution error: {str(e)}\n{''.join(tb_str)}"


class Programmer(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "Programmer", timeout: int = 60):
        super().__init__(llm, name)
        # Create a class-level process pool, instead of creating a new one for each execution
        self.process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=1)
        # Configurable timeout (default 60s, was hardcoded 30s)
        self.timeout = timeout

    def __del__(self):
        """Ensure the process pool is closed when the object is destroyed"""
        if hasattr(self, 'process_pool'):
            self.process_pool.shutdown(wait=True)

    async def exec_code(self, code, timeout=None):
        """
        Asynchronously execute code and return an error if timeout occurs.
        Use instance timeout if not provided.
        """
        loop = asyncio.get_running_loop()
        # Use provided timeout or fall back to instance timeout
        actual_timeout = timeout if timeout is not None else self.timeout

        try:
            # Use the class-level process pool
            future = loop.run_in_executor(self.process_pool, run_code, code)
            # Wait for the task to complete or timeout
            result = await asyncio.wait_for(future, timeout=actual_timeout)
            return result
        except asyncio.TimeoutError:
            # Only cancel this specific future, not the entire process pool
            future.cancel()
            # Force garbage collection
            import gc
            gc.collect()
            return "Error", "Code execution timed out"
        except concurrent.futures.process.BrokenProcessPool:
            # If the process pool is broken, recreate it
            self.process_pool.shutdown(wait=False)
            self.process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=1)
            return "Error", "Process pool broken, try again"
        except Exception as e:
            return "Error", f"Unknown error: {str(e)}"

    async def code_generate(self, problem, analysis, feedback, mode):
        """
        Asynchronous method to generate code.
        """
        prompt = PYTHON_CODE_VERIFIER_PROMPT.format(
            problem=problem,
            analysis=analysis,
            feedback=feedback
        )
        response = await self._fill_node(CodeGenerateOp, prompt, mode, function_name="solve")
        return response

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    async def __call__(self, problem: str, analysis: str = "None"):
        """
        Call method, generate code and execute, retry up to 3 times.
        Uses configurable timeout for code execution.
        """
        code = None
        output = None
        feedback = ""
        for i in range(3):
            code_response = await self.code_generate(problem, analysis, feedback, mode="code_fill")
            code = code_response.get("code")
            if not code:
                return {
                    "code": code,
                    "output": "No code generated",
                    "success": False,
                    "error": "Code generation failed"
                }
            status, output = await self.exec_code(code, timeout=self.timeout)
            if status == "Success":
                return {
                    "code": code,
                    "output": output,
                    "success": True
                }
            else:
                print(f"Execution error on attempt {i + 1}, error message: {output}")
                feedback = (
                    f"\nThe result of the error from the code you wrote in the previous round:\n"
                    f"Code: {code}\n\nStatus: {status}, {output}"
                )

            # Force garbage collection after each iteration
            import gc
            gc.collect()

        return {
            "code": code,
            "output": output,
            "success": False,
            "error": f"Failed after 3 attempts. Last status: {status}"
        }

class Test(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "Test"):
        super().__init__(llm, name)

    def _extract_test_cases_from_string(self, test_string: str) -> List[str]:
        """
        Extract test cases from test string (e.g., function code with assert statements)

        Args:
            test_string: Complete test function code (e.g., "def check(candidate): assert ...")

        Returns:
            List of test cases (as test function definitions or formatted tests)
        """
        # For now, return the test_string as a single test case item
        # The test_string contains the complete test function that can be executed
        # We return it as a list with one element for compatibility with existing logic
        if test_string and test_string.strip():
            return [test_string]  # Return as single test case
        return None

    def exec_code(self, solution, entry_point, test_string: Optional[str] = None):
        # Two modes:
        # 1. Parameter mode (RL training): Use test_string provided directly (prioritized)
        #    - test_string contains complete test function code
        # 2. File mode (original): Read from JSONL file using entry_point
        #    - test_cases are individual assert statements

        fail_cases = []

        if test_string:
            # Parameter mode: test_string is complete test code ready to execute
            # Combine solution + test_string and execute directly
            test_code = f"{solution}\n\n{test_string}"

            # Count assertions in test_string for granular results
            total_tests = test_string.count('assert')
            if total_tests == 0:
                total_tests = 1  # At least one test if test_string provided

            try:
                exec(test_code, globals())
                # All tests passed
                return {
                    "status": "success",
                    "passed": total_tests,
                    "total": total_tests,
                    "pass_rate": 1.0
                }
            except AssertionError as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                tb_str = traceback.format_exception(exc_type, exc_value, exc_traceback)
                error_information = {
                    "test_fail_case": {
                        "test_case": test_string,
                        "error_type": "AssertionError",
                        "error_message": str(e),
                        "traceback": tb_str,
                    }
                }
                # Conservative: assume first assertion failed
                return {
                    "status": "failed",
                    "passed": 0,
                    "total": total_tests,
                    "pass_rate": 0.0,
                    "error": [error_information]
                }
            except Exception as e:
                with open("tester.txt", "a") as f:
                    f.write(f"Parameter mode error for {entry_point}: {str(e)}\n")
                return {
                    "exec_fail_case": str(e),
                    "status": "error",
                    "passed": 0,
                    "total": total_tests,
                    "pass_rate": 0.0
                }
        else:
            # File mode: Original logic using test_cases from JSONL
            test_cases = extract_test_cases_from_jsonl(entry_point)

            # Handle None case for robustness
            if test_cases is None:
                return {"exec_fail_case": f"No test cases found for entry_point: {entry_point}"}

            for test_case in test_cases:
                test_code = test_case_2_test_function(solution, test_case, entry_point)
                try:
                    exec(test_code, globals())
                except AssertionError as e:
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    tb_str = traceback.format_exception(exc_type, exc_value, exc_traceback)
                    with open("tester.txt", "a") as f:
                        f.write("test_error of " + entry_point + "\n")
                    error_infomation = {
                        "test_fail_case": {
                            "test_case": test_case,
                            "error_type": "AssertionError",
                            "error_message": str(e),
                            "traceback": tb_str,
                        }
                    }
                    fail_cases.append(error_infomation)
                except Exception as e:
                    with open("tester.txt", "a") as f:
                        f.write(entry_point + " " + str(e) + "\n")
                    return {"exec_fail_case": str(e)}
            if fail_cases != []:
                return fail_cases
            else:
                return "no error"

    async def __call__(self, problem, solution, entry_point, test_loop: int = 3, test_string: Optional[str] = None):
        """
        "Test": {
        "description": "Test the solution with test cases, if the solution is correct, return 'no error'; if incorrect, reflect on the solution and the error information",
        "interface": "test(problem: str, solution: str, entry_point: str, test_string: Optional[str] = None) -> str"
        }
        Returns standardized format with success flag and test counts

        Supports two modes:
        1. File mode (original): Reads test cases from JSONL file using entry_point
        2. Parameter mode (new): Uses test_string provided directly for RL training
        """
        for _ in range(test_loop):
            result = self.exec_code(solution, entry_point, test_string)

            # Handle new dict format with test counts
            if isinstance(result, dict) and result.get("status") == "success":
                return {
                    "result": True,
                    "solution": solution,
                    "success": True,
                    "test_passed": True,
                    "passed": result.get("passed", 0),
                    "total": result.get("total", 0),
                    "pass_rate": result.get("pass_rate", 1.0)
                }
            # Backward compatibility: handle old "no error" string
            elif result == "no error":
                return {
                    "result": True,
                    "solution": solution,
                    "success": True,
                    "test_passed": True,
                    "passed": 1,
                    "total": 1,
                    "pass_rate": 1.0
                }
            elif isinstance(result, dict) and "exec_fail_case" in result:
                error_msg = result["exec_fail_case"]
                prompt = REFLECTION_ON_PUBLIC_TEST_PROMPT.format(
                    problem=problem,
                    solution=solution,
                    exec_pass=f"executed unsuccessfully, error: \n {error_msg}",
                    test_fail="executed unsucessfully",
                )
                response = await self._fill_node(ReflectionTestOp, prompt, mode="code_fill")
                solution = response.get("response", solution)
            else:
                prompt = REFLECTION_ON_PUBLIC_TEST_PROMPT.format(
                    problem=problem,
                    solution=solution,
                    exec_pass="executed successfully",
                    test_fail=result,
                )
                response = await self._fill_node(ReflectionTestOp, prompt, mode="code_fill")
                solution = response.get("response", solution)

        # Final attempt after all retries
        result = self.exec_code(solution, entry_point, test_string)

        # Handle new dict format with test counts
        if isinstance(result, dict) and result.get("status") == "success":
            return {
                "result": True,
                "solution": solution,
                "success": True,
                "test_passed": True,
                "passed": result.get("passed", 0),
                "total": result.get("total", 0),
                "pass_rate": result.get("pass_rate", 1.0)
            }
        # Backward compatibility
        elif result == "no error":
            return {
                "result": True,
                "solution": solution,
                "success": True,
                "test_passed": True,
                "passed": 1,
                "total": 1,
                "pass_rate": 1.0
            }
        else:
            # Extract test counts from failed result
            passed = result.get("passed", 0) if isinstance(result, dict) else 0
            total = result.get("total", 1) if isinstance(result, dict) else 1
            pass_rate = result.get("pass_rate", 0.0) if isinstance(result, dict) else 0.0

            return {
                "result": False,
                "solution": solution,
                "success": False,
                "test_passed": False,
                "passed": passed,
                "total": total,
                "pass_rate": pass_rate,
                "error": str(result)
            }


class Format(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "Format"):
        super().__init__(llm, name)

    async def __call__(self, problem, solution, mode: str = None):
        prompt = FORMAT_PROMPT.format(problem_description=problem, solution=solution)
        response = await self._fill_node(FormatOp, prompt, mode)
        return response


class Review(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "Review"):
        super().__init__(llm, name)

    async def __call__(self, problem, solution, mode: str = None):
        prompt = REVIEW_PROMPT.format(problem=problem, solution=solution)
        response = await self._fill_node(ReviewOp, prompt, mode="xml_fill")
        # Standardize return format with success flag
        if isinstance(response, dict):
            if "error" in response:
                response["success"] = False
            else:
                response["success"] = True
            # Ensure feedback field exists
            if "feedback" not in response and "review_result" in response:
                response["feedback"] = response["review_result"]
        return response


class Revise(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "Revise"):
        super().__init__(llm, name)

    async def __call__(self, problem, solution, feedback, mode: str = None):
        prompt = REVISE_PROMPT.format(problem=problem, solution=solution, feedback=feedback)
        response = await self._fill_node(ReviseOp, prompt, mode="xml_fill")
        # Standardize return format with success flag
        if isinstance(response, dict):
            if "error" in response:
                response["success"] = False
                response["solution"] = solution  # Fallback to original solution on error
            else:
                response["success"] = True
                # Ensure solution field exists
                if "solution" not in response and "response" in response:
                    response["solution"] = response["response"]
        return response


class MdEnsemble(Operator):
    """
    Paper: Can Generalist Foundation Models Outcompete Special-Purpose Tuning? Case Study in Medicine
    Link: https://arxiv.org/abs/2311.16452
    """

    def __init__(self, llm: AsyncLLM, name: str = "MdEnsemble", vote_count: int = 5):
        super().__init__(llm, name)
        self.vote_count = vote_count

    @staticmethod
    def shuffle_answers(solutions: List[str]) -> Tuple[List[str], Dict[str, str]]:
        shuffled_solutions = solutions.copy()
        random.shuffle(shuffled_solutions)
        answer_mapping = {chr(65 + i): solutions.index(solution) for i, solution in enumerate(shuffled_solutions)}
        return shuffled_solutions, answer_mapping

    async def __call__(self, solutions: List[str], problem: str, mode: str = None):
        logger.info(f"solution count: {len(solutions)}")
        all_responses = []

        for _ in range(self.vote_count):
            shuffled_solutions, answer_mapping = self.shuffle_answers(solutions)

            solution_text = ""
            for index, solution in enumerate(shuffled_solutions):
                solution_text += f"{chr(65 + index)}: \n{str(solution)}\n\n\n"

            prompt = MD_ENSEMBLE_PROMPT.format(solutions=solution_text, question=problem)
            response = await self._fill_node(MdEnsembleOp, prompt, mode="xml_fill")

            answer = response.get("solution_letter", "A")
            answer = answer.strip().upper()

            if answer in answer_mapping:
                original_index = answer_mapping[answer]
                all_responses.append(original_index)

        most_frequent_index = Counter(all_responses).most_common(1)[0][0]
        final_answer = solutions[most_frequent_index]
        return {"solution": final_answer}
