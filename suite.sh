#!/bin/bash
cd "$(dirname "$0")"

for i in {1..5}
do
    ./undelayed.sh
    (export DELAY=1 && ./delayed.sh)
    (export DELAY=1 && ./delayed_is.sh)
    (export DELAY=5 && ./delayed.sh)
    (export DELAY=5 && ./delayed_is.sh)
    (export DELAY=10 && ./delayed.sh)
    (export DELAY=10 && ./delayed_is.sh)
    (export DELAY=20 && ./delayed.sh)
    (export DELAY=20 && ./delayed_is.sh)
    (export DELAY=100 && ./delayed.sh)
    (export DELAY=100 && ./delayed_is.sh)
done
