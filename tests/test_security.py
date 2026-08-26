
import pytest

from runtime.security.capabilities import CapabilitySet
from runtime.security.policy import SecurityPolicy, reset_policy, set_policy
from runtime.security.sandbox import PathViolation, TriadSandbox


@pytest.fixture(autouse=True)
def restore_policy():
    yield
    reset_policy()


def test_sandbox_rejects_paths_outside_workspace(tmp_path):
    sandbox = TriadSandbox([str(tmp_path)], allow_temp=False)
    with pytest.raises(PathViolation):
        sandbox.resolve('/etc/passwd')


def test_remove_requires_capability(tmp_path):
    from stdlib.universal import fs_remove

    target = tmp_path / 'data.txt'
    target.write_text('data')
    set_policy(SecurityPolicy(
        safe=True,
        sandbox=TriadSandbox([str(tmp_path)], allow_temp=False),
        capabilities=CapabilitySet(),
    ))

    with pytest.raises(PermissionError):
        fs_remove(str(target))
    assert target.exists()


def test_remove_with_capability_stays_inside_workspace(tmp_path):
    from stdlib.universal import fs_remove

    target = tmp_path / 'data.txt'
    target.write_text('data')
    set_policy(SecurityPolicy(
        safe=True,
        sandbox=TriadSandbox([str(tmp_path)], allow_temp=False),
        capabilities=CapabilitySet(fs_remove=True),
    ))

    fs_remove(str(target))
    assert not target.exists()


def test_policy_is_isolated_between_contexts(tmp_path):
    from contextvars import Context

    from runtime.security.policy import get_policy

    first = SecurityPolicy(sandbox=TriadSandbox([str(tmp_path / 'one')]))
    second = SecurityPolicy(sandbox=TriadSandbox([str(tmp_path / 'two')]))
    set_policy(first)

    seen = Context().run(lambda: (set_policy(second), get_policy())[1])

    assert seen is second
    assert get_policy() is first


def test_capability_parser_supports_environment_and_socket():
    from runtime.security.capabilities import parse_capabilities

    caps = parse_capabilities('env.read:true;env.write:yes;net.socket:1')

    assert caps.env_read and caps.env_write and caps.net_socket
