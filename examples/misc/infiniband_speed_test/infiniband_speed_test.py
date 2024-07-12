from tractorun.backend.generic import (
    GenericBackend,
)
from tractorun.run import prepare_and_get_toolbox

import sys
import subprocess


def run_cmd(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout, file=sys.stderr)
    print(result.stderr, file=sys.stderr)
    assert result.returncode == 0, f"command failed with return code {result.returncode}"

toolbox = prepare_and_get_toolbox(backend=GenericBackend())

print("Running ibstatus", file=sys.stderr)
run_cmd(["ibstatus"])

if toolbox.coordinator.is_primary():
    print("Running ib_send_bw as primary", file=sys.stderr)
    run_cmd(["timeout", "10", "ib_send_bw", "--report_gbits", "-D", "10"])
else:
    primary_address = toolbox.coordinator.get_primary_endpoint()
    # TODO: Remove after DNS is set up
    if "a4hfmmhvepq79spp0kce-uxim" in primary_address:
        primary_address = primary_address.replace("a4hfmmhvepq79spp0kce-uxim", "172.20.0.105")
    else:
        assert "a4hfmmhvepq79spp0kce-apub" in primary_address
        primary_address = primary_address.replace("a4hfmmhvepq79spp0kce-apub", "172.20.0.126")

    print("Running ib_send_bw as subordinate", file=sys.stderr)
    print("Primary endpoint is ", primary_address, file=sys.stderr)
    run_cmd(["timeout", "10", "ib_send_bw", primary_address, "--report_gbits", "-D", "10"])
