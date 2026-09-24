import time

from timers import TimerService


def test_timer_fires_and_queues_announcement():
    service = TimerService()
    service.start(0.05, "tea")

    time.sleep(0.2)

    assert service.pending_announcements() == ["Your tea timer is done."]
    assert service.active() == []


def test_cancel_by_label_or_id():
    service = TimerService()
    pasta = service.start(60, "pasta")
    eggs = service.start(60, "eggs")

    assert service.cancel("PASTA").id == pasta.id
    assert service.cancel(str(eggs.id)).label == "eggs"
    assert service.cancel("pasta") is None
    assert service.active() == []


def test_cancelled_timer_never_announces():
    service = TimerService()
    service.start(0.05, "tea")
    service.cancel("tea")

    time.sleep(0.2)

    assert service.pending_announcements() == []


def test_active_sorted_by_time_remaining():
    service = TimerService()
    service.start(120, "long")
    service.start(30, "short")

    assert [t.label for t in service.active()] == ["short", "long"]
    service.cancel_all()
