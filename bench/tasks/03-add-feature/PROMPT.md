Add a `--json` flag to the `list` command in todo.py.

When `python3 todo.py list --json` is run, it must print the task list as a JSON
array to stdout and nothing else. Each element must be an object with the keys
`id` (number), `text` (string), and `done` (boolean).

`python3 todo.py list` without the flag must keep its current human-readable
output exactly as it is today. Update README.md to document the new flag.
