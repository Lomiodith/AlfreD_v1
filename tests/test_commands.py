from commands import INSTANT_COMMAND_DESCRIPTIONS, get_formatted_commands
from intent_detector import INTENT_TRIGGERS

# "affirm" only answers Alfred's own yes/no question; it isn't a command.
NOT_LISTED = {"affirm"}


def test_every_instant_command_is_documented():
    assert set(INTENT_TRIGGERS) - NOT_LISTED == set(INSTANT_COMMAND_DESCRIPTIONS)


def test_reference_lists_every_trigger():
    reference = get_formatted_commands()

    for intent in INSTANT_COMMAND_DESCRIPTIONS:
        for trigger in INTENT_TRIGGERS[intent]:
            assert trigger in reference
