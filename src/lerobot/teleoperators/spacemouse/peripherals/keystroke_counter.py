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

from pynput.keyboard import Key, KeyCode, Listener
from collections import defaultdict
from threading import Lock


class KeystrokeCounter(Listener):
    def __init__(self):
        self.key_count_map = defaultdict(lambda: 0)
        self.key_press_list = list()
        self.lock = Lock()
        super().__init__(on_press=self.on_press, on_release=self.on_release)

    def on_press(self, key):
        with self.lock:
            self.key_count_map[key] += 1
            self.key_press_list.append(key)

    def on_release(self, key):
        pass

    def clear(self):
        with self.lock:
            self.key_count_map = defaultdict(lambda: 0)
            self.key_press_list = list()

    def __getitem__(self, key):
        with self.lock:
            return self.key_count_map[key]

    def get_press_events(self):
        with self.lock:
            events = list(self.key_press_list)
            self.key_press_list = list()
            return events


if __name__ == "__main__":
    import time

    with KeystrokeCounter() as counter:
        try:
            while True:
                print("Space:", counter[Key.space])
                print("q:", counter[KeyCode(char="q")])
                time.sleep(1 / 60)
        except KeyboardInterrupt:
            events = counter.get_press_events()
            print(events)
