# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

import importlib
import inspect
import sys
import warnings
from collections import defaultdict
from typing import Any, Callable, MutableSequence
from unittest.mock import Mock


def get_orig_obj(path: str):
    split_orig_path = path.split(".")
    module_path = ".".join(split_orig_path[:-1])
    modules = module_path.split(".")
    orig_func_name = ".".join(split_orig_path[-1:])

    for i in range(1, len(modules) + 1):
        upper_module = ".".join(modules[: i - 1])
        path = ".".join(modules[:i])
        try:
            importlib.import_module(path)
        except ModuleNotFoundError as err:
            module = getattr(importlib.import_module(upper_module), modules[i - 1])
            if hasattr(module, orig_func_name):
                return module, getattr(module, orig_func_name), orig_func_name
            else:
                raise Exception(f"Get {orig_func_name} origin function object err.") from err

    return sys.modules[module_path], getattr(sys.modules[module_path], orig_func_name), orig_func_name


def _replace_all_obj(orig_obj: Any, wrapper_obj: Any, ignore_keys: tuple[str, ...] = tuple()):
    import gc

    refs = gc.get_referrers(orig_obj)
    obj_id = id(orig_obj)
    for ref in refs:
        if isinstance(ref, MutableSequence):
            for i, v in enumerate(ref):
                if id(v) == obj_id:
                    ref[i] = wrapper_obj
        elif isinstance(ref, dict):
            for k, v in ref.items():
                if id(v) == obj_id and k not in ignore_keys:
                    ref[k] = wrapper_obj
        else:
            pass


def replace_func(
    origin_module_obj: Callable,
    orig_func_obj: Callable,
    wrapper_func_obj: Callable,
    orig_func_name: str,
    ignore_keys: tuple[str] = ("orig_func",),
):
    method_class = False
    method_class = inspect.isclass(origin_module_obj)
    # Assign function
    if not method_class:
        _replace_all_obj(orig_func_obj, wrapper_func_obj, ignore_keys=ignore_keys)
    exec(f"origin_module_obj.{orig_func_name} = wrapper_func_obj")


def _is_mock_obj(obj: Any) -> bool:
    return isinstance(obj, Mock)


class WrapperContext:
    def __init__(self, wrapper_func: Callable, orig_func: Callable, **kwargs):
        self.wrapper_func = wrapper_func
        self.orig_func = orig_func

    def __call__(self, *arg, **kwargs):
        return self.wrapper_func(self, *arg, **kwargs)


class FunctionWrapper:
    def __init__(self):
        self.wrapper_info = dict()
        self.func_contexts = defaultdict(list)

    def _register(self, func_name: str, **kwargs):
        info_dict = kwargs
        if func_name not in self.wrapper_info:
            self.wrapper_info[func_name] = list()
        self.wrapper_info[func_name].append(info_dict)

    def register_wrapper(self, func_name: str, **kwargs):
        def wrapper(object):
            self._register(func_name, wrapper_object=object, **kwargs)
            return object

        return wrapper

    def func_wrapper_apply(self):
        wrapper_funcs = list()
        for orig_func_path, info_dict in self.wrapper_info.items():
            try:
                module_obj, orig_func_obj, orig_func_name = get_orig_obj(orig_func_path)
            except Exception as err:
                raise Exception(f"Get {orig_func_path} object err, the function wrapper can not apply.") from err

            wrapper_func_obj = info_dict[0]["wrapper_object"]
            if _is_mock_obj(module_obj) or _is_mock_obj(orig_func_obj):
                warnings.warn(
                    f"Skip function wrapper for {orig_func_path}: "
                    f"the original object is a mock object. This usually means "
                    f"an optional dependency is not installed in the current environment.",
                    stacklevel=2,
                )
                continue
            wrapper_context = WrapperContext(wrapper_func_obj, orig_func_obj)
            self.func_contexts[orig_func_path].append(wrapper_context)
            wrapper_funcs.append(
                dict(
                    orig_module_obj=module_obj,
                    orig_func_obj=orig_func_obj,
                    wrapper_func_obj=wrapper_func_obj,
                    orig_func_name=orig_func_name,
                    orig_func_path=orig_func_path,
                )
            )
            # replace_func(orig_func_path, wrapper_func_obj)

        # Sovle the problem that wrapper func name is the same,
        # like base class and aderived from this class have the same func name.
        for wrapper_func_dict in wrapper_funcs:
            orig_module_obj = wrapper_func_dict["orig_module_obj"]
            orig_func_obj = wrapper_func_dict["orig_func_obj"]
            wrapper_func_obj = wrapper_func_dict["wrapper_func_obj"]
            orig_func_name = wrapper_func_dict["orig_func_name"]
            orig_func_path = wrapper_func_dict["orig_func_path"]
            try:
                replace_func(orig_module_obj, orig_func_obj, wrapper_func_obj, orig_func_name)
            except Exception as e:
                raise RuntimeError(f"Apply function wrapper for {orig_func_path} failed.") from e

    def get_orig_func(self, orig_func_path: str) -> Callable:
        func_ctxs = self.func_contexts.get(orig_func_path, [])
        if len(func_ctxs) == 0:
            raise Exception(f"Get {orig_func_path} context err, the context is empty.")
        return func_ctxs[0].orig_func


class WrapperManager:
    def __init__(self):
        self.func_wrapper = FunctionWrapper()


WRAPPER_MANAGER = WrapperManager()
FUNCTION_WRAPPER = WRAPPER_MANAGER.func_wrapper
