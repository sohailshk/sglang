"""Unit tests for lang/tracer.py"""

import unittest

import sglang as sgl
from sglang.lang.backend.base_backend import BaseBackend
from sglang.lang.chat_template import get_chat_template
from sglang.lang.ir import SglConstantText, SglGen
from sglang.lang.tracer import extract_prefix_by_tracing
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _ChatMLBackend(BaseBackend):
    def __init__(self):
        super().__init__()
        self.chat_template = get_chat_template("chatml")


@sgl.function
def _with_blocks(s, question):
    with s.system():
        s += "You are a bot."
    with s.user():
        s += question
    with s.assistant():
        s += sgl.gen("answer", max_tokens=8)


@sgl.function
def _with_var_scope(s, question):
    s += "Header: "
    with s.var_scope("scope"):
        s += "body"
    s += question


@sgl.function
def _with_set_var(s, question):
    s["k"] = "v"
    s += "Header: "
    with s.user():
        s += question


@sgl.function
def _role_exprs(s, question):
    s += sgl.system("You are a bot.")
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("answer", max_tokens=8))


def _node_texts(tracer):
    return [
        n.value if isinstance(n, SglConstantText) else type(n).__name__
        for n in tracer.nodes
    ]


class TestTracerWithBlocks(CustomTestCase):
    def setUp(self):
        self.backend = _ChatMLBackend()

    def test_prefix_matches_role_expressions(self):
        """A program written with `with s.user():` blocks yields the same
        cacheable prefix as one written with `sgl.user(...)` expressions."""
        expected = "<|im_start|>system\nYou are a bot.<|im_end|>\n<|im_start|>user\n"
        self.assertEqual(extract_prefix_by_tracing(_role_exprs, self.backend), expected)
        self.assertEqual(
            extract_prefix_by_tracing(_with_blocks, self.backend), expected
        )

    def test_trace_with_role_blocks_matches_role_expressions(self):
        """Tracing with-block roles records the same nodes as the expression form."""
        traced = _with_blocks.trace(backend=self.backend)
        reference = _role_exprs.trace(backend=self.backend)
        self.assertEqual(_node_texts(traced), _node_texts(reference))
        self.assertTrue(any(isinstance(n, SglGen) for n in traced.nodes))

    def test_prefix_survives_state_helpers_before_first_gen(self):
        """Calling other state helpers such as `s["k"] = "v"` while tracing
        must not stop the prefix from being extracted."""
        self.assertEqual(
            extract_prefix_by_tracing(_with_set_var, self.backend),
            "Header: <|im_start|>user\n",
        )

    def test_trace_with_var_scope(self):
        """`s.var_scope` blocks can be traced and record the scoped variable."""
        traced = _with_var_scope.trace(backend=self.backend)
        self.assertEqual(
            extract_prefix_by_tracing(_with_var_scope, self.backend), "Header: body"
        )
        self.assertIn("scope", traced.variables)
        self.assertEqual(_node_texts(traced)[:2], ["Header: ", "body"])


if __name__ == "__main__":
    unittest.main()
