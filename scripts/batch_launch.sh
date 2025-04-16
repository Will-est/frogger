#!/bin/bash

# List of arguments (one per run)
ARGS=(
  "zip_tie_gun_single"
  "black_spray_bottle_single"
  "clamp_single"
  "scissors_single"
  "hot_glue_gun"
  "pliers_single"
  "marker_single"
  "syrup_pourer_single"
  "knife_single"
  "staple_remover_single"
  "zip_tie_tool_single"
)

# Loop over each argument
for ARG in "${ARGS[@]}"; do
  echo "Starting script with argument: $ARG"
  
  # Run the Python script with a 3-hour timeout.
  # If the script finishes early, timeout exits and the next iteration starts immediately.
  # If it runs longer than 3 hours, timeout will kill it.
  timeout 3h python examples/test_batch_width_search.py "$ARG"
  
  echo "Finished run for argument: $ARG"
done