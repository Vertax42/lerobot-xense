#!/usr/bin/env python

# Copyright 2025 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from multiprocessing.managers import SharedMemoryManager
from typing import Tuple

import numpy as np
from atomics import UINT, MemoryOrder, atomicview


@dataclass
class ArraySpec:
    name: str
    shape: Tuple[int]
    dtype: np.dtype


class SharedCounter:
    def __init__(self, shm_manager: SharedMemoryManager, size: int = 8):  # 64bit int
        shm = shm_manager.SharedMemory(size=size)
        self.shm = shm
        self.size = size
        self.store(0)  # initialize

    @property
    def buf(self):
        return self.shm.buf[: self.size]

    def load(self) -> int:
        return int(np.frombuffer(self.buf, dtype=np.uint64)[0])

    def store(self, value: int):
        np.frombuffer(self.buf, dtype=np.uint64)[0] = value

    def add(self, value: int):
        val = self.load()
        self.store(val + value)


class SharedAtomicCounter:
    def __init__(self, shm_manager: SharedMemoryManager, size: int = 8):  # 64bit int
        shm = shm_manager.SharedMemory(size=size)
        self.shm = shm
        self.size = size
        self.store(0)  # initialize

    @property
    def buf(self):
        return self.shm.buf[: self.size]

    def load(self) -> int:
        with atomicview(buffer=self.buf, atype=UINT) as a:
            value = a.load(order=MemoryOrder.ACQUIRE)
        return value

    def store(self, value: int):
        with atomicview(buffer=self.buf, atype=UINT) as a:
            a.store(value, order=MemoryOrder.RELEASE)

    def add(self, value: int):
        with atomicview(buffer=self.buf, atype=UINT) as a:
            a.add(value, order=MemoryOrder.ACQ_REL)
