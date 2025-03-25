from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, Callable, Tuple

import numpy as np
import xarray as xr
import pymc as pm
import pytensor.tensor as pt
from pymc_marketing.prior import handle_dims

__all__ = [
    "HSGPWrapper",
    "FourierWrapper",
    "PriorWrapper",
    "DataWrapper"
]

class GeneralPrior(Protocol):
    def apply(self, data: xr.DataArray | pt.TensorLike) -> pt.TensorLike:
        pass


class Node(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def apply(
        self, data: xr.DataArray | pt.TensorLike, model: pm.Model | None = None
    ) -> pt.TensorLike:
        raise NotImplementedError

    def __add__(self, other: GeneralPrior) -> SumNode:
        return SumNode([self, other], name=f"{self.name}+{other.name}")

    def __mul__(self, other: GeneralPrior) -> ProductNode:
        return ProductNode([self, other], name=f"{self.name}*{other.name}")

    def __call__(self, node: GeneralPrior) -> GeneralPrior:
        return AppliedNode(f"{self.name}({node.name})", self, node)


class CombNode(Node):
    def __init__(self, children: list[GeneralPrior], name: str, agg_fn: Callable):
        super().__init__(name)
        self.children = children
        self.agg_fn = agg_fn
        self._dims = tuple(
            set(
                dim for dims in [child._dims for child in self.children] for dim in dims
            )
        )

    def apply(
        self, 
        data: xr.DataArray | pt.TensorLike, 
        model: pm.Model | None = None
    ) -> pt.TensorLike:
        model = pm.modelcontext(model)
        
        with model:
            self.children[0].apply(data)
            for _child in (self.children[1:]):
                _child.apply(data)
            self._variable = pm.Deterministic(
                self.name,
                self.agg_fn(
                    [
                        handle_dims(child._variable, child._dims, self._dims)
                        for child in self.children
                    ]
                ),
                dims=self._dims,
            )
        return self._variable


class SumNode(CombNode):
    def __init__(self, children: list[GeneralPrior], name: str):
        super().__init__(children, name, sum)


class ProductNode(CombNode):
    def __init__(self, children: list[GeneralPrior], name: str):
        super().__init__(children, name, np.prod)


class AppliedNode(Node):
    def __init__(self, name: str, caller_node: Node, input_node: Node):
        super().__init__(name)
        self._caller_node = caller_node
        self._input_node = input_node
        self._dims = caller_node._dims

    def apply(
        self, data: xr.DataArray | pt.TensorLike, model: pm.Model | None = None
    ) -> pt.TensorLike:
        model = pm.modelcontext(model)

        with model:
            # if isinstance(self._caller_node, CombNode):
            #     data_input = self._input_node.apply(data)
            #     self._variable = self._caller_node.agg_fn(
            #         handle_dims(
            #             child.apply(data_input), child._dims, self._caller_node._dims
            #         )
            #         for child in self._caller_node.children
            #     )
            #     return self._variable

            self._variable = self._caller_node.apply(self._input_node.apply(data), model=model)
        return self._variable


class HSGPWrapper(Node):
    def __init__(self, gp, name: str):
        super().__init__(name)
        self._gp = gp
        self._dims = gp.dims

    def apply(
        self, data: xr.DataArray | pt.TensorLike | None, model: pm.Model | None = None
    ) -> pt.TensorLike:
        model = pm.modelcontext(model)
        with model:
            self._variable = self._gp.register_data(data).create_variable(self.name)
        return self._variable


class FourierWrapper(Node):
    def __init__(self, fourier, dims: Tuple[str], name: str):
        super().__init__(name)
        self._fourier = fourier
        self._dims = dims

    def apply(
        self, data: xr.DataArray | pt.TensorLike | None, model: pm.Model | None = None
    ) -> pt.TensorLike:
        model = pm.modelcontext(model)
        with model:
            self._variable = pm.Deterministic(
                self.name, self._fourier.apply(data), dims=self._dims
            )
        return self._variable


class PriorWrapper(Node):
    def __init__(self, prior, name: str):
        super().__init__(name)
        self._prior = prior
        self._dims = prior.dims

    def apply(
        self,
        data: xr.DataArray | pt.TensorLike | None = None,
        model: pm.Model | None = None,
    ) -> pt.TensorLike:
        model = pm.modelcontext(model)
        with model:
            self._variable = self._prior.create_variable(self.name)
        return self._variable


class DataWrapper(Node):
    def __init__(self, name: str, dims: Tuple[str] | None = None):
        super().__init__(name)
        self._dims = dims

    def apply(
        self, data: xr.DataArray | pt.TensorLike, model: pm.Model | None = None
    ) -> pt.TensorLike:
        model = pm.modelcontext(model)
        with model:
            self._variable = pm.Data(self.name, data, dims=self._dims)
        return self._variable
