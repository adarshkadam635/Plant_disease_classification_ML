import json

import pytest

import train


def test_atomic_json_retries_windows_file_lock(monkeypatch, tmp_path):
    path = tmp_path / 'status.json'
    path.write_text('{"state":"old"}')
    original_replace = train.os.replace
    attempts = []

    def locked_replace(source, destination):
        attempts.append(1)
        if len(attempts) <= 3:
            assert json.loads(path.read_text()) == {'state': 'old'}
            raise PermissionError('temporary Windows file lock')
        original_replace(source, destination)

    monkeypatch.setattr(train.os, 'replace', locked_replace)
    monkeypatch.setattr(train.time, 'sleep', lambda seconds: None)
    train.atomic_json(path, {'state': 'training'})
    assert len(attempts) == 4
    assert json.loads(path.read_text()) == {'state': 'training'}
    assert not path.with_suffix('.json.tmp').exists()


def test_permanent_lock_preserves_previous_file(monkeypatch, tmp_path):
    path = tmp_path / 'status.json'
    path.write_text('{"state":"old"}')
    def locked_replace(source, destination):
        raise PermissionError('permanent lock')
    monkeypatch.setattr(train.os, 'replace', locked_replace)
    monkeypatch.setattr(train.time, 'sleep', lambda seconds: None)
    with pytest.raises(PermissionError):
        train.atomic_json(path, {'state': 'new'})
    assert json.loads(path.read_text()) == {'state': 'old'}
