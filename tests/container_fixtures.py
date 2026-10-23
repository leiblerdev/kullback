"""A scripted World the container tests drive instead of a real container."""

from __future__ import annotations

from kullback.container_world import StepReceipt


def ok_receipt(stdout=b"", stderr=b"", code=0):
    return StepReceipt(stdout, stderr, code, False, False)


class ScriptedWorld:
    """Fake at the World boundary: records step commands, answers by script."""

    def __init__(self, answers=()):
        self._answers = list(answers)
        self.commands = []
        self.reset_calls = 0
        self.close_calls = 0

    def reset(self):
        self.reset_calls += 1
        return "fake-container-id"

    def step(self, command):
        self.commands.append(command)
        item = self._answers.pop(0) if self._answers else ok_receipt()
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self):
        self.close_calls += 1
        return True
