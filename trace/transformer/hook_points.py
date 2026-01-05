"""
TransformerLens-style hook infrastructure for transformer interpretability.
The code heavily borrows from TransformerLens repository.
Based on TransformerLens hook_points.py but adapted for our custom transformer.
"""
from contextlib import contextmanager

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable, Any, Union, Literal, Tuple
from collections import defaultdict
from functools import partial
import weakref

# Type definitions for hook functions
HookFunction = Callable[[torch.Tensor, str], torch.Tensor]


class LensHandle:
    """Dataclass that holds information about a PyTorch hook."""

    def __init__(self, hook_handle, is_permanent: bool = False, context_level: Optional[int] = None):
        self.hook = hook_handle
        self.is_permanent = is_permanent
        self.context_level = context_level

    def remove(self):
        """Remove the hook."""
        if self.hook is not None:
            self.hook.remove()


class HookPoint(nn.Module):
    """
    A dummy module that acts as an identity function by default -> Can have hooks attached to intercept and modify activations.
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__()
        self.name = name
        self.fwd_hooks: List[HookFunction] = []
        self.bwd_hooks: List[HookFunction] = []

    def add_hook(
            self,
            hook: HookFunction,
            dir: Literal['fwd', 'bwd'] = 'fwd',
            is_permanent: bool = False,
            level: Optional[int] = None,
            prepend: bool = False
    ) -> LensHandle:
        """
        Add a hook to this HookPoint.

        Args:
            hook: Function with signature (activation, hook_name) -> modified_activation
            dir: Direction - 'fwd' for forward hooks, 'bwd' for backward hooks
            is_permanent: Whether this hook persists through reset_hooks()
            level: Context level for nested hook management
            prepend: Whether to add hook at beginning of list

        Returns:
            LensHandle for managing the hook
        """

        def full_hook(module, input, output):
            """Wrapper to convert PyTorch hook to our format."""
            if dir == 'fwd':
                return hook(output, self.name or f"hook_{id(self)}")
            return output

        # Register with PyTorch
        # For info: https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_forward_hook.html
        if dir == 'fwd':
            handle = self.register_forward_hook(full_hook)
        else:
            handle = self.register_full_backward_hook(full_hook)

        return LensHandle(handle, is_permanent, level)

    def add_perma_hook(self, hook: HookFunction, dir: Literal['fwd', 'bwd'] = 'fwd') -> LensHandle:
        """Add a permanent hook that persists through reset_hooks()."""
        return self.add_hook(hook, dir, is_permanent=True)

    def remove_hooks(self, dir: Literal['fwd', 'bwd', 'both'] = 'fwd'):
        """Remove all hooks from this HookPoint."""
        if dir in ['fwd', 'both']:
            # Remove forward hooks
            self._forward_hooks.clear()
        if dir in ['bwd', 'both']:
            # Remove backward hooks
            self._backward_hooks.clear()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - identity function by default."""
        return x


class HookedRootModule(nn.Module):
    """
    Base class for models that support TransformerLens-style hooks.
    Provides hook management, caching, and intervention capabilities.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.hook_dict: Dict[str, HookPoint] = {}
        self.hooks_to_remove: List[LensHandle] = []
        self.mod_dict: Dict[str, nn.Module] = {}
        self.context_level: int = 0

    def setup_hooks(self):
        """
        Set up the hook infrastructure. Call this AFTER defining all layers.
        This builds the hook_dict and mod_dict for the model.
        """
        # Build module name mapping
        for name, module in self.named_modules():
            if name:  # Skip the root module (empty name)
                self.mod_dict[name] = module
                # Add hook points to our hook_dict
                if isinstance(module, HookPoint):
                    self.hook_dict[name] = module
                    if module.name is None:
                        module.name = name

    def add_hook(
            self,
            name: Union[str, Callable[[str], bool]],
            hook: HookFunction,
            dir: Literal['fwd', 'bwd'] = 'fwd',
            is_permanent: bool = False,
            level: Optional[int] = None,
            prepend: bool = False
    ) -> Union[LensHandle, List[LensHandle]]:
        """
        Add hook(s) to the model.

        Args:
            name: Hook point name or function to match names
            hook: Hook function (activation, hook_name) -> modified_activation
            dir: Hook direction
            is_permanent: Whether hook persists through reset
            level: Context level
            prepend: Whether to prepend to hook list

        Returns:
            Handle(s) for the added hook(s)
        """
        if callable(name):
            # Name is a function to match hook points
            handles = []
            for hook_name, hook_point in self.hook_dict.items():
                if name(hook_name):
                    handle = hook_point.add_hook(hook, dir, is_permanent, level, prepend)
                    handles.append(handle)
                    if not is_permanent:
                        self.hooks_to_remove.append(handle)
            return handles
        else:
            # Name is a specific hook point
            if name not in self.hook_dict:
                raise ValueError(f"Hook point '{name}' not found. Available: {list(self.hook_dict.keys())}")

            hook_point = self.hook_dict[name]
            handle = hook_point.add_hook(hook, dir, is_permanent, level, prepend)

            if not is_permanent:
                self.hooks_to_remove.append(handle)

            return handle

    def reset_hooks(self, clear_permanent: bool = False, direction: Literal['fwd', 'bwd', 'both'] = 'fwd'):
        """Remove all temporary hooks, optionally including permanent ones."""
        # Remove temporary hooks
        for handle in self.hooks_to_remove:
            if handle.hook is not None:
                handle.remove()
        self.hooks_to_remove.clear()

        # Remove permanent hooks if requested
        if clear_permanent:
            for hook_point in self.hook_dict.values():
                hook_point.remove_hooks(direction)

    # def hooks(
    #         self,
    #         *model_args,
    #         fwd_hooks: Optional[List[tuple]] = None,
    #         bwd_hooks: Optional[List[tuple]] = None,
    #         reset_hooks_end: bool = True,
    #         clear_internal_hooks: bool = True,
    #         **model_kwargs
    # ) -> Any:
    #     """
    #     Run the model with temporary hooks.
    #
    #     Args:
    #         *model_args: Arguments to pass to model forward()
    #         fwd_hooks: List of (name, hook_fn) tuples for forward hooks
    #         bwd_hooks: List of (name, hook_fn) tuples for backward hooks
    #         reset_hooks_end: Whether to remove hooks after execution
    #         clear_internal_hooks: Whether to clear existing hooks first
    #         **model_kwargs: Keyword arguments to pass to model forward()
    #
    #     Returns:
    #         Model output with hooks applied
    #     """
    #     if clear_internal_hooks:
    #         self.reset_hooks()
    #
    #     # Add temporary hooks
    #     temp_handles = []
    #
    #     if fwd_hooks:
    #         for name, hook_fn in fwd_hooks:
    #             handles = self.add_hook(name, hook_fn, 'fwd', is_permanent=False)
    #             if isinstance(handles, list):
    #                 temp_handles.extend(handles)
    #             else:
    #                 temp_handles.append(handles)
    #
    #     if bwd_hooks:
    #         for name, hook_fn in bwd_hooks:
    #             handles = self.add_hook(name, hook_fn, 'bwd', is_permanent=False)
    #             if isinstance(handles, list):
    #                 temp_handles.extend(handles)
    #             else:
    #                 temp_handles.append(handles)
    #
    #     try:
    #         # Run the model
    #         output = self(*model_args, **model_kwargs)
    #     finally:
    #         # Clean up temporary hooks
    #         if reset_hooks_end:
    #             for handle in temp_handles:
    #                 handle.remove()
    #             # Remove from our tracking list
    #             self.hooks_to_remove = [h for h in self.hooks_to_remove if h not in temp_handles]
    #
    #     return output
    @contextmanager
    def hooks(
            self,
            fwd_hooks: Optional[List[Tuple[Union[str, Callable], Callable]]] = None,
            bwd_hooks: Optional[List[Tuple[Union[str, Callable], Callable]]] = None,
            reset_hooks_end: bool = True,
            clear_contexts: bool = False,
    ):
        """
        Context manager for adding temporary hooks to the model.

        Args:
            fwd_hooks: List of (name, hook_fn) for forward hooks
            bwd_hooks: List of (name, hook_fn) for backward hooks
            reset_hooks_end: Whether to remove hooks when exiting context
            clear_contexts: Whether to clear contexts when exiting
        """
        if fwd_hooks is None:
            fwd_hooks = []
        if bwd_hooks is None:
            bwd_hooks = []

        # Store handles for cleanup
        temp_handles = []

        try:
            # Add all forward hooks
            for name, hook_fn in fwd_hooks:
                try:
                    handles = self.add_hook(name, hook_fn, 'fwd', is_permanent=False)
                    if isinstance(handles, list):
                        temp_handles.extend(handles)
                    else:
                        temp_handles.append(handles)
                except Exception as e:
                    print(f"Warning: Could not add hook {name}: {e}")
                    continue

            # Add all backward hooks
            for name, hook_fn in bwd_hooks:
                try:
                    handles = self.add_hook(name, hook_fn, 'bwd', is_permanent=False)
                    if isinstance(handles, list):
                        temp_handles.extend(handles)
                    else:
                        temp_handles.append(handles)
                except Exception as e:
                    print(f"Warning: Could not add hook {name}: {e}")
                    continue

            # CRITICAL: Just yield control back to caller =  Do NOT call the model here - let the caller do it
            yield

        finally:
            # Clean up hooks
            if reset_hooks_end:
                for handle in temp_handles:
                    try:
                        if handle and hasattr(handle, 'remove'):
                            handle.remove()
                    except:
                        pass  # Ignore cleanup errors

                # Remove from tracking list
                self.hooks_to_remove = [h for h in self.hooks_to_remove if h not in temp_handles]
    def run_with_hooks(
            self,
            *model_args,
            fwd_hooks: Optional[List[tuple]] = None,
            bwd_hooks: Optional[List[tuple]] = None,
            reset_hooks_end: bool = True,
            clear_internal_hooks: bool = True,
            **model_kwargs
    ) -> Any:
        """
        Run the model with temporary hooks.

        Args:
            *model_args: Arguments to pass to model forward()
            fwd_hooks: List of (name, hook_fn) tuples for forward hooks
            bwd_hooks: List of (name, hook_fn) tuples for backward hooks
            reset_hooks_end: Whether to remove hooks after execution
            clear_internal_hooks: Whether to clear existing hooks first
            **model_kwargs: Keyword arguments to pass to model forward()

        Returns:
            Model output with hooks applied
        """
        if clear_internal_hooks:
            self.reset_hooks()

        # Add temporary hooks
        temp_handles = []

        if fwd_hooks:
            for name, hook_fn in fwd_hooks:
                handles = self.add_hook(name, hook_fn, 'fwd', is_permanent=False)
                if isinstance(handles, list):
                    temp_handles.extend(handles)
                else:
                    temp_handles.append(handles)

        if bwd_hooks:
            for name, hook_fn in bwd_hooks:
                handles = self.add_hook(name, hook_fn, 'bwd', is_permanent=False)
                if isinstance(handles, list):
                    temp_handles.extend(handles)
                else:
                    temp_handles.append(handles)

        try:
            # Run the model
            output = self(*model_args, **model_kwargs)
        finally:
            # Clean up temporary hooks
            if reset_hooks_end:
                for handle in temp_handles:
                    handle.remove()
                # Remove from our tracking list
                self.hooks_to_remove = [h for h in self.hooks_to_remove if h not in temp_handles]

        return output




    def run_with_cache(
            self,
            *model_args,
            names_filter: Optional[Union[Callable[[str], bool], List[str]]] = None,
            device: Optional[torch.device] = None,
            remove_batch_dim: bool = False,
            incl_bwd: bool = False,
            reset_hooks_end: bool = True,
            clear_internal_hooks: bool = True,
            **model_kwargs
    ) -> tuple:
        """
        Run model and cache specified activations.

        Args:
            *model_args: Arguments for model forward pass
            names_filter: Function or list to filter which activations to cache
            device: Device to store cached activations
            remove_batch_dim: Whether to remove batch dimension if batch_size=1
            incl_bwd: Whether to include backward pass activations
            reset_hooks_end: Whether to remove hooks after execution
            clear_internal_hooks: Whether to clear existing hooks first
            **model_kwargs: Keyword arguments for model forward pass

        Returns:
            Tuple of (model_output, activation_cache)
        """
        cache = ActivationCache()

        def caching_hook(activation, hook_name):
            """Hook function that stores activations in cache."""
            if device is not None:
                activation = activation.to(device)
            if remove_batch_dim and activation.shape[0] == 1:
                activation = activation.squeeze(0)
            cache[hook_name] = activation.detach()
            return activation

        # Determine which hooks to add
        hook_names_to_cache = []
        if names_filter is None:
            # Cache all hook points
            hook_names_to_cache = list(self.hook_dict.keys())
        elif callable(names_filter):
            # Filter hook names with function
            hook_names_to_cache = [name for name in self.hook_dict.keys() if names_filter(name)]
        else:
            # Use provided list
            hook_names_to_cache = names_filter

        # Add caching hooks
        fwd_hooks = [(name, caching_hook) for name in hook_names_to_cache]
        bwd_hooks = [(name, caching_hook) for name in hook_names_to_cache] if incl_bwd else None

        # Run with hooks
        output = self.run_with_hooks(
            *model_args,
            fwd_hooks=fwd_hooks,
            bwd_hooks=bwd_hooks,
            reset_hooks_end=reset_hooks_end,
            clear_internal_hooks=clear_internal_hooks,
            **model_kwargs
        )

        return output, cache


class ActivationCache:
    """
    Cache for storing model activations with convenient access patterns.
    Borrowed from TransformerLens and CAP
    """

    def __init__(self):
        self.cache: Dict[str, torch.Tensor] = {}

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.cache[key]

    def __setitem__(self, key: str, value: torch.Tensor):
        self.cache[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.cache

    def keys(self):
        return self.cache.keys()

    def values(self):
        return self.cache.values()

    def items(self):
        return self.cache.items()

    def get(self, key: str, default=None):
        return self.cache.get(key, default)

    def to_dict(self) -> Dict[str, torch.Tensor]:
        """Convert to regular dictionary."""
        return dict(self.cache)

    def apply_ln_to_stack(self, *args, **kwargs):
        """Placeholder for LayerNorm - will implement if needed"""
        pass