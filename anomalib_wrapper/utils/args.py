import argparse
import copy
import functools
import logging
from argparse import ArgumentParser, Namespace, Action
from typing import Any, Literal, Optional, Sequence, Union

from .logging import add_log_file_handler, setup_logging


@functools.lru_cache()
def _log_level_options():
    # Get all possible log levels.
    # This method is a bit, resource intensive since it checks
    # all possible levels up to 101.
    # But other methods require access to private variables such
    # as :py:attr:`logging._nameToLevel`.
    return {logging.getLevelName(x): x for x in range(1, 101) if not logging.getLevelName(x).startswith("Level")}


class _SetLogLevelAction(argparse.Action):
    """Sets the log level to the numeric value according to the"""

    @staticmethod
    def _try_get_log_level(value: str):
        try:
            return _log_level_options()[value]
        except KeyError:
            raise KeyError(
                f"{value} is not a valid log level. \
                Choose one of {set(_log_level_options().keys())}."
            )

    def __call__(
            self,
            parser: ArgumentParser,
            namespace: Namespace,
            values: Union[str, Sequence[Any], None],
            option_string: Optional[str] = ...,
    ) -> None:
        assert isinstance(values, str), "Log level most be specified as a single value."
        level = self._try_get_log_level(values)
        # TODO: Do not use setup_logging since it overwrites previous changes.
        setup_logging(log_level=level)
        setattr(namespace, self.dest, level)


class _SetLogFileAction(argparse.Action):
    """If called, adds a logging handler which outputs to a local file."""

    def __call__(
            self,
            parser: ArgumentParser,
            namespace: Namespace,
            values: Union[str, Sequence[Any], None],
            option_string: Optional[str] = ...,
    ) -> None:
        assert isinstance(values, str), "Log file most be specified as path."
        add_log_file_handler(values)
        setattr(namespace, self.dest, values)


def add_log_level_arg(parser: ArgumentParser, default: Literal[20] = "INFO"):
    """Adds an option to set the log level of the :py:attr:`logging.root`.
    The log level is automatically set by an `argparse.Action`
    Also sets the default argument immediately.
    Converts the log level to its integer value when adding it to the namespace.

    :param parser: the parser, where the log level argument should be added.
    :param default: the default log level
    """
    # We set the root logger to the default value, since the action is not called,
    # if no value is specified as cmd argument.
    setup_logging(log_level=default)
    parser.add_argument(
        "--log-level",
        default=_log_level_options()[default],
        choices=list(_log_level_options().keys()),
        action=_SetLogLevelAction,
        help="set log level",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        action=_SetLogFileAction,
        help="set log file, which will be used to log stdout and stderr.",
    )


class DictAction(Action):
    """
    Source: https://github.com/open-mmlab/mmengine/blob/main/mmengine/config/config.py#L1729
    argparse action to split an argument into KEY=VALUE form
    on the first = and append to a dictionary. List options can
    be passed as comma separated values, i.e 'KEY=V1,V2,V3', or with explicit
    brackets, i.e. 'KEY=[V1,V2,V3]'. It also support nested brackets to build
    list/tuple values. e.g. 'KEY=[(V1,V2),(V3,V4)]'
    """

    @staticmethod
    def _parse_int_float_bool(val: str) -> Union[int, float, bool, Any]:
        """parse int/float/bool value in the string."""
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        if val.lower() in ['true', 'false']:
            return True if val.lower() == 'true' else False
        if val == 'None':
            return None
        return val

    @staticmethod
    def _parse_iterable(val: str) -> Union[list, tuple, Any]:
        """Parse iterable values in the string.

        All elements inside '()' or '[]' are treated as iterable values.

        Args:
            val (str): Value string.

        Returns:
            list | tuple | Any: The expanded list or tuple from the string,
            or single value if no iterable values are found.

        Examples:
            >>> DictAction._parse_iterable('1,2,3')
            [1, 2, 3]
            >>> DictAction._parse_iterable('[a, b, c]')
            ['a', 'b', 'c']
            >>> DictAction._parse_iterable('[(1, 2, 3), [a, b], c]')
            [(1, 2, 3), ['a', 'b'], 'c']
        """

        def find_next_comma(string):
            """Find the position of next comma in the string.

            If no ',' is found in the string, return the string length. All
            chars inside '()' and '[]' are treated as one element and thus ','
            inside these brackets are ignored.
            """
            assert (string.count('(') == string.count(')')) and (
                    string.count('[') == string.count(']')), \
                f'Imbalanced brackets exist in {string}'
            end = len(string)
            for idx, char in enumerate(string):
                pre = string[:idx]
                # The string before this ',' is balanced
                if ((char == ',') and (pre.count('(') == pre.count(')'))
                        and (pre.count('[') == pre.count(']'))):
                    end = idx
                    break
            return end

        # Strip ' and " characters and replace whitespace.
        val = val.strip('\'\"').replace(' ', '')
        is_tuple = False
        if val.startswith('(') and val.endswith(')'):
            is_tuple = True
            val = val[1:-1]
        elif val.startswith('[') and val.endswith(']'):
            val = val[1:-1]
        elif ',' not in val:
            # val is a single value
            return DictAction._parse_int_float_bool(val)

        values = []
        while len(val) > 0:
            comma_idx = find_next_comma(val)
            element = DictAction._parse_iterable(val[:comma_idx])
            values.append(element)
            val = val[comma_idx + 1:]

        if is_tuple:
            return tuple(values)

        return values

    def __call__(self,
                 parser: ArgumentParser,
                 namespace: Namespace,
                 values: Union[str, Sequence[Any], None],
                 option_string: str = None):
        """Parse Variables in string and add them into argparser.

        Args:
            parser (ArgumentParser): Argument parser.
            namespace (Namespace): Argument namespace.
            values (Union[str, Sequence[Any], None]): Argument string.
            option_string (list[str], optional): Option string.
                Defaults to None.
        """
        # Copied behavior from `argparse._ExtendAction`.
        options = copy.copy(getattr(namespace, self.dest, None) or {})
        if values is not None:
            for kv in values:
                key, val = kv.split('=', maxsplit=1)
                options[key] = self._parse_iterable(val)
        setattr(namespace, self.dest, options)
