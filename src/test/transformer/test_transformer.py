import unittest
import logging
from src.lib.rltrace.trace import Trace
from src.lib.transformer import Transformer


class DummyCallable:
    @staticmethod
    def no_op_func(s: str) -> str:
        return s

    @staticmethod
    def fixed_transform(_: str = None) -> str:
        return "Always The Same"

    @staticmethod
    def replace_cat(s: str) -> str:
        return s.replace('Cat', 'Hat', 1)

    @staticmethod
    def replace_mat(s: str) -> str:
        return s.replace('Mat', 'Fat', 1)

    def __call__(self, *args, **kwargs):
        return


class TestCapability(unittest.TestCase):
    _run = int(0)

    def __init__(self, *args, **kwargs):
        super(TestCapability, self).__init__(*args, **kwargs)
        self._transformer = None
        return

    @classmethod
    def setUpClass(cls):
        Trace()

    def setUp(self) -> None:
        TestCapability._run += 1
        logging.info("- - - - - - C A S E {} Start - - - - - -".format(TestCapability._run))
        self._transformer = Transformer()
        return

    def tearDown(self) -> None:
        logging.info("- - - - - - C A S E {} Passed - - - - - -".format(TestCapability._run))
        self._transformer = None
        return

    def test_add_fail(self):
        with self.assertRaises(ValueError):
            self._transformer.add_transform(None)
        return

    def test_transform_init_fail(self):
        with self.assertRaises(ValueError):
            _ = Transformer.Transform(None, DummyCallable.no_op_func)

        with self.assertRaises(ValueError):
            _ = Transformer.Transform("^AValidRegExp$", None)

    def test_transform_miss(self):
        string_to_transform = "The Cat Sat On The Mat"
        non_matching_transform = Transformer.Transform(regular_expression="XxXx",
                                                       transform=DummyCallable.fixed_transform)
        self._transformer.add_transform(non_matching_transform)
        actual = self._transformer.transform(string_to_transform=string_to_transform)
        self.assertEqual(string_to_transform, actual)
        return

    def test_transform_hit(self):
        string_to_transform = "The Cat Sat On The Mat"
        non_matching_transform = Transformer.Transform(regular_expression="^T.*Cat.*t$",
                                                       transform=DummyCallable.fixed_transform)
        self._transformer.add_transform(non_matching_transform)
        actual = self._transformer.transform(string_to_transform=string_to_transform)
        self.assertEqual(DummyCallable.fixed_transform(), actual)
        return

    def test_multi_transform_hit(self):
        string_to_transform = "The Cat Sat On The Mat"

        non_matching_transform = Transformer.Transform(regular_expression="^.*Cat.*$",
                                                       transform=DummyCallable.replace_cat)
        self._transformer.add_transform(non_matching_transform)

        non_matching_transform = Transformer.Transform(regular_expression="^.*Mat.*$",
                                                       transform=DummyCallable.replace_mat)
        self._transformer.add_transform(non_matching_transform)

        non_matching_transform = Transformer.Transform(regular_expression="^.*Miss.*$",
                                                       transform=DummyCallable.fixed_transform)
        self._transformer.add_transform(non_matching_transform)

        actual = self._transformer.transform(string_to_transform=string_to_transform)
        self.assertEqual("The Hat Sat On The Fat", actual)
        return


if __name__ == "__main__":
    unittest.main()
