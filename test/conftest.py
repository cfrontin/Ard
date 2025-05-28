from pathlib import Path


def pytest_sessionfinish(session, exitstatus):
    # cleanup code after tests

    # for each tempdir
    for pytest_out_dir in Path().glob("pytest*_out"):
        for root, dirs, files in pytest_out_dir.walk(
            top_down=False
        ):  # walk the directory
            for name in files:
                (root / name).unlink()  # remove subdirectory files, and
            for name in dirs:
                (root / name).rmdir()  # remove subdirectories
        pytest_out_dir.rmdir()  # then remove that tempdir
