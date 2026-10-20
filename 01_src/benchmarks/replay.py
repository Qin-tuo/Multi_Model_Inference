# SPDX-License-Identifier: Apache-2.0
"""Fixed-clock image input; late consumers skip arrivals instead of slowing time."""

import time


class ReplayCapture:
    def __init__(self, images, stop, events, phase, rate=10):
        self.images, self.stop, self.events, self.phase = images, stop, events, phase
        self.rate = rate
        self.origin = time.monotonic()
        self.last = -1
        self.missed = 0
        self.delivered = 0
        self.closed = False

    def read(self):
        now = time.monotonic()
        index = max(self.last + 1, int((now - self.origin) * self.rate))
        due = self.origin + index / self.rate
        if self.stop.wait(max(0, due - now)) or self.closed:
            return False, None
        self.missed += max(0, index - self.last - 1)
        self.last = index
        self.delivered += 1
        name, image = self.images[(index // 20) % len(self.images)]
        self.events.write(
            kind="source_arrival",
            phase=self.phase(),
            source_index=index,
            delivered_index=self.delivered,
            scheduled_monotonic=due,
            delivered_monotonic=time.monotonic(),
            input=name,
            missed_arrivals_total=self.missed,
        )
        return True, image

    def release(self):
        self.closed = True


