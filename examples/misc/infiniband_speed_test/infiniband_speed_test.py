import subprocess
import sys

from tractorun.backend.generic import GenericBackend
from tractorun.run import prepare_and_get_toolbox


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
    run_cmd(["ib_send_bw", "--report_gbits", "-D", "10"])
else:
    primary_address = toolbox.coordinator.get_primary_endpoint()
    # TODO: Remove after DNS is set up
    if "a4hfmmhvepq79spp0kce-ohoz" in primary_address:
        primary_address = "172.20.0.54"
    else:
        assert "a4hf4vura28j50o0ju7q-urot" in primary_address
        primary_address = "172.20.0.149"

    print("Running ib_send_bw as subordinate", file=sys.stderr)
    print("Primary endpoint is ", primary_address, file=sys.stderr)
    run_cmd(["ib_send_bw", primary_address, "--report_gbits", "-D", "10"])
