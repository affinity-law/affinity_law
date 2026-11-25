"""Run the original benchmark with portable in-memory task paths.

The relocated manifest, original runner, shared scientific code and resume
protocol stay byte-identical to the original experiment. Loaded task paths
resolve to shared repository inputs before the original main() consumes them.
All command-line arguments are handled by the original runner.
"""
import common
import run as runner
from portable_paths import TASKS_PATH, normalize_manifest_paths, normalize_task_paths


def main():
    original_loader = common.load_manifest
    original_path = common.TASKS_PATH
    original_description = runner.__doc__

    def portable_loader():
        manifest, tasks = original_loader()
        return (normalize_manifest_paths(manifest),
                [normalize_task_paths(task) for task in tasks])

    common.load_manifest = portable_loader
    common.TASKS_PATH = TASKS_PATH
    runner.__doc__ = "Run TimesFM 3.0 or Moirai 2.0 on the 63-dataset timeline aligned with L."
    try:
        runner.main()
    finally:
        common.load_manifest = original_loader
        common.TASKS_PATH = original_path
        runner.__doc__ = original_description


if __name__ == "__main__":
    main()
