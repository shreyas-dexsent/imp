"""Public API surface tests."""
from algorithms import RobotSystemDescription, WorldDescription


def test_descriptions_public_api():
    assert callable(RobotSystemDescription.from_yaml)
    assert callable(WorldDescription.from_yaml)
