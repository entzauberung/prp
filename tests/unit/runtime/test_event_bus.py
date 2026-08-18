"""Targeted tests for bounded event sequence hints."""

import asyncio

import pytest

from prp_runtime.runtime.event_bus import EventBus


async def receive(subscription: object) -> int | None:
    async def get() -> int | None:
        return await subscription.get()  # type: ignore[attr-defined]

    return await asyncio.wait_for(get(), timeout=1.0)


@pytest.mark.asyncio
async def test_hints_are_isolated_by_run_and_carry_no_payload() -> None:
    bus = EventBus(max_buffer=2)
    first = await bus.subscribe("run-a")
    second = await bus.subscribe("run-b")

    await bus.publish("run-a", 3)
    await bus.publish("run-b", 7)

    assert await receive(first) == 3
    assert await receive(second) == 7
    assert first.buffered_count == 0
    await first.close()
    await second.close()
    await bus.close()


@pytest.mark.asyncio
async def test_overflow_drops_hints_but_keeps_latest_replay_cursor() -> None:
    bus = EventBus(max_buffer=2)
    subscription = await bus.subscribe("run-a")

    for sequence in (1, 2, 3):
        await bus.publish("run-a", sequence)

    assert subscription.overflowed is True
    assert subscription.dropped_count == 1
    assert await receive(subscription) == 2
    assert await receive(subscription) == 3
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(subscription.get(), timeout=0.01)
    await bus.close()


@pytest.mark.asyncio
async def test_live_publish_wakes_waiter_and_slow_consumer_does_not_block() -> None:
    bus = EventBus(max_buffer=2)
    subscription = await bus.subscribe("run-a")
    waiting = asyncio.create_task(subscription.get())

    await asyncio.wait_for(bus.publish("run-a", 1), timeout=1.0)
    assert await asyncio.wait_for(waiting, timeout=1.0) == 1

    async def publish_many() -> None:
        for sequence in range(2, 102):
            await bus.publish("run-a", sequence)

    await asyncio.wait_for(publish_many(), timeout=1.0)
    assert subscription.buffered_count == 2
    assert subscription.dropped_count == 98
    await bus.close()


@pytest.mark.asyncio
async def test_subscription_close_releases_a_waiter() -> None:
    bus = EventBus()
    subscription = await bus.subscribe("run-a")
    waiting = asyncio.create_task(subscription.get())
    await asyncio.sleep(0)

    await subscription.close()

    assert await asyncio.wait_for(waiting, timeout=1.0) is None
    assert subscription.closed is True
    assert bus.subscription_count == 0
    await bus.close()


@pytest.mark.asyncio
async def test_bus_close_releases_all_waiters_and_rejects_new_subscribers() -> None:
    bus = EventBus(max_subscriptions=2)
    first = await bus.subscribe("run-a")
    second = await bus.subscribe("run-b")
    first_waiter = asyncio.create_task(first.get())
    second_waiter = asyncio.create_task(second.get())
    await asyncio.sleep(0)

    await bus.close()

    assert await asyncio.wait_for(first_waiter, timeout=1.0) is None
    assert await asyncio.wait_for(second_waiter, timeout=1.0) is None
    assert bus.closed is True
    assert bus.subscription_count == 0
    with pytest.raises(RuntimeError, match="closed"):
        await bus.subscribe("run-c")
