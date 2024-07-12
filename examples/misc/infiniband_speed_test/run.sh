export PYTHONPATH=$PYTHONPATH:../../..

yt create table //tmp/stderr --force

python3 ../../../tractorun/cli/tractorun_runner.py \
    --run-config-path config.yaml
