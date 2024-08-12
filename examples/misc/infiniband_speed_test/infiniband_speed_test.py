import subprocess
import sys

from tractorun.backend.generic import GenericBackend
from tractorun.run import prepare_and_get_toolbox


def run_cmd(cmd: list) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout, file=sys.stderr)
    print(result.stderr, file=sys.stderr)
    assert result.returncode == 0, f"command failed with return code {result.returncode}"


toolbox = prepare_and_get_toolbox(backend=GenericBackend())

print("Running ibstatus", file=sys.stderr)
run_cmd(["ibstatus"])

if toolbox.coordinator.is_primary():
    print("Running ib_send_bw as primary", file=sys.stderr)
    run_cmd(["ib_send_bw", "--report_gbits", "-D", "10"])
else:
    print("Running ib_send_bw as secondary", file=sys.stderr)
    primary_address = toolbox.coordinator.get_primary_endpoint()
    print("Primary address", primary_address)
    # primary_host, primary_port = toolbox.coordinator.get_primary_endpoint().split(":")

    # primary_ip = socket.getaddrinfo(primary_host, primary_port)[0][4][0]
    if "e00ypsy4gwrt0b2920" in primary_address:
        primary_ip = "192.168.0.36"
    else:
        assert "e00q0mwgg70qbs7fzd" in primary_address
        primary_ip = "192.168.0.27"

    print("Primary endpoint is ", primary_ip, file=sys.stderr)
    cmd = ["ib_send_bw", "--report_gbits", "-D", "10", primary_ip]
    print(f"run command {cmd}")
    run_cmd(cmd)
