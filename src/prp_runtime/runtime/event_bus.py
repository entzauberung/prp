"""Bounded in-memory event sequence hints.

The bus never stores event payloads and is not a source of truth. Consumers use
the received sequence as a prompt to replay facts from the Store. Dropped hints
therefore create a replay gap, not a lost event.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator

__all__ = ["EventBus", "EventSubscription"]


class EventSubscription:
    """One bounded sequence-hint stream for one run."""

    def __init__(self, bus: EventBus, run_id: str, max_buffer: int) -> None:
        self._bus = bus
        self.run_id = run_id
        self._hints: deque[int] = deque(maxlen=max_buffer)
        self._wake = asyncio.Event()
        self._closed = False
        self._dropped_count = 0

    @property
    def closed(self) -> bool:
        """Whether this subscription can receive more hints."""
        return self._closed

    @property
    def buffered_count(self) -> int:
        """Number of currently buffered sequence hints."""
        return len(self._hints)

    @property
    def dropped_count(self) -> int:
        """Number of hints dropped because this consumer was too slow."""
        return self._dropped_count

    @property
    def overflowed(self) -> bool:
        """Whether at least one hint was dropped for this consumer."""
        return self._dropped_count > 0

    def _publish(self, sequence: int) -> None:
        if self._closed:
            return
        if len(self._hints) == self._hints.maxlen:
            self._hints.popleft()
            self._dropped_count += 1
        self._hints.append(sequence)
        self._wake.set()

    async def get(self) -> int | None:
        """Return the next hint, or ``None`` after this stream is closed."""
        while True:
            if self._hints:
                return self._hints.popleft()
            if self._closed:
                return None
            self._wake.clear()
            if self._hints or self._closed:
                continue
            await self._wake.wait()

    async def next(self) -> int | None:
        """Alias for ``get`` useful to callers that treat hints as a cursor."""
        return await self.get()

    def __aiter__(self) -> AsyncIterator[int]:
        return self

    async def __anext__(self) -> int:
        sequence = await self.get()
        if sequence is None:
            raise StopAsyncIteration
        return sequence

    async def close(self) -> None:
        """Close this subscriber and release any waiter."""
        if self._closed:
            return
        self._closed = True
        self._hints.clear()
        self._wake.set()
        self._bus._remove(self)

    async def __aenter__(self) -> EventSubscription:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()


class EventBus:
    """A bounded, per-run hint bus with no event payload storage."""

    def __init__(self, *, max_buffer: int = 64, max_subscriptions: int = 1024) -> None:
        if max_buffer < 1:
            raise ValueError("max_buffer must be at least 1")
        if max_subscriptions < 1:
            raise ValueError("max_subscriptions must be at least 1")
        self._max_buffer = max_buffer
        self._max_subscriptions = max_subscriptions
        self._subscriptions: dict[str, set[EventSubscription]] = {}
        self._closed = False

    @property
    def closed(self) -> bool:
        """Whether the bus has released all subscriptions."""
        return self._closed

    @property
    def subscription_count(self) -> int:
        """Number of live subscriptions across all runs."""
        return sum(len(subscriptions) for subscriptions in self._subscriptions.values())

    async def subscribe(self, run_id: str) -> EventSubscription:
        """Create one bounded subscription for a run."""
        if not run_id.strip():
            raise ValueError("run_id must not be blank")
        if self._closed:
            raise RuntimeError("event bus has been closed")
        if self.subscription_count >= self._max_subscriptions:
            raise RuntimeError("event bus subscription limit reached")
        subscription = EventSubscription(self, run_id, self._max_buffer)
        self._subscriptions.setdefault(run_id, set()).add(subscription)
        return subscription

    async def publish(self, run_id: str, sequence: int) -> None:
        """Publish only a positive sequence hint to subscribers of one run."""
        if not run_id.strip():
            raise ValueError("run_id must not be blank")
        if isinstance(sequence, bool) or sequence < 1:
            raise ValueError("sequence must be a positive integer")
        if self._closed:
            return
        for subscription in tuple(self._subscriptions.get(run_id, ())):
            subscription._publish(sequence)

    def _remove(self, subscription: EventSubscription) -> None:
        subscriptions = self._subscriptions.get(subscription.run_id)
        if subscriptions is None:
            return
        subscriptions.discard(subscription)
        if not subscriptions:
            del self._subscriptions[subscription.run_id]

    async def close(self) -> None:
        """Close every subscription and release every waiter."""
        if self._closed:
            return
        self._closed = True
        subscriptions = tuple(
            subscription
            for run_subscriptions in self._subscriptions.values()
            for subscription in run_subscriptions
        )
        for subscription in subscriptions:
            await subscription.close()
        self._subscriptions.clear()
