#!/bin/bash
set -euo pipefail

## build : build docker file
function task_build {
  echo "Building..."
  docker build -t 3d_deep_teeth_segmentation .
}

## run : builds and runs application as docker container
 function task_run {
   task_build
   echo "Running..."
   docker run --name 3d_deep_teeth_segmentation -d -v "%cd%":/app/workdir/ --gpus all  -p 8811:8888  -e JUPYTER_TOKEN=passwd 3d_deep_teeth_segmentation
 }

## stop: stop and remove docker container
function task_stop {
  docker stop 3d_deep_teeth_segmentation
  docker rm 3d_deep_teeth_segmentation
}


function task_usage {
  echo "Usage: $0"
  sed -n 's/^##//p' <$0 | column -t -s ':' | sed -E $'s/^/\t/' | sort
}


cmd=${1:-}
shift || true
resolved_command=$(echo "task_${cmd}" | sed 's/-/_/g')
if [[ "$(LC_ALL=C type -t "${resolved_command}")" == "function" ]]; then
  ${resolved_command} "$@"
else
  task_usage
  exit 1
fi
