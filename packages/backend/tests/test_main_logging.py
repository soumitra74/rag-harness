import logging

from app.main import _SkipPollLogs


def _record(message: str) -> logging.LogRecord:
    return logging.LogRecord(
        name="uvicorn.access", level=logging.INFO, pathname=__file__, lineno=1,
        msg=message, args=(), exc_info=None,
    )


def test_filters_out_successful_eval_run_poll():
    message = (
        '172.28.0.1:33958 - "GET /api/eval/run/5fca958b-a255-4238-9541-aad4e1a00c2b '
        'HTTP/1.1" 200 OK'
    )
    assert _SkipPollLogs().filter(_record(message)) is False


def test_filters_out_successful_sweep_poll():
    message = '192.168.0.1:53536 - "GET /api/sweep/1 HTTP/1.1" 200 OK'
    assert _SkipPollLogs().filter(_record(message)) is False


def test_keeps_failed_eval_run_poll():
    message = (
        '172.28.0.1:33958 - "GET /api/eval/run/5fca958b-a255-4238-9541-aad4e1a00c2b '
        'HTTP/1.1" 404 Not Found'
    )
    assert _SkipPollLogs().filter(_record(message)) is True


def test_keeps_failed_sweep_poll():
    message = '192.168.0.1:53536 - "GET /api/sweep/1 HTTP/1.1" 500 Internal Server Error'
    assert _SkipPollLogs().filter(_record(message)) is True


def test_keeps_unrelated_routes():
    message = '172.28.0.1:33958 - "GET /api/eval/history HTTP/1.1" 200 OK'
    assert _SkipPollLogs().filter(_record(message)) is True


def test_keeps_the_post_that_starts_a_run():
    message = '172.28.0.1:33958 - "POST /api/eval/run HTTP/1.1" 200 OK'
    assert _SkipPollLogs().filter(_record(message)) is True


def test_keeps_the_post_that_starts_a_sweep():
    message = '192.168.0.1:53536 - "POST /api/sweep/start HTTP/1.1" 200 OK'
    assert _SkipPollLogs().filter(_record(message)) is True
