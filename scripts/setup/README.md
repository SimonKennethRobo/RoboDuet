# RoboDuet Setup Helper

`setup.sh` is an interactive bootstrap helper for a new workstation, lab desktop, or GPU node.

It is conservative by default:

- detects existing `conda`, repo checkout, conda env, and IsaacGym before installing;
- shows a built-in terminal module selector before installing anything;
- uses ASCII-only borders and controls for compatibility with plain SSH terminals;
- supports arrow keys, `j`/`k`, Space, Enter, `a` to select all, and `q` to cancel;
- falls back to numbered multi-select when no interactive TTY is available;
- does not delete existing conda envs;
- does not overwrite dirty git worktrees;
- auto-detects IsaacGym Preview 4 archive in common local download locations;
- downloads IsaacGym Preview 4 from NVIDIA's redirecting download page if no local archive is found;
- keeps optional IsaacGym archive path / URL overrides for unusual layouts.
- accepts Anaconda default channel Terms of Service before creating the conda env when the installed conda version requires it.
- adds a user-level writable conda package cache under `$CONDA_DIR/pkgs` or `~/.conda/pkgs` when existing shared cache settings are read-only.

The script uses colored output and emoji by default. Disable them with:

```bash
NO_COLOR=1 ROBODUET_NO_EMOJI=1 bash scripts/setup/setup.sh --verify-only
```

In interactive mode, selecting a module means the script will run that module without asking again before every sub-step. Non-selected modules are skipped and reported in the final summary.

The final summary reports completed, skipped, and failed modules. IsaacGym runs inside the conda env with the env's `lib` directory prepended to `LD_LIBRARY_PATH`, which avoids the common `libpython3.8.so.1.0` import error from IsaacGym Preview 4.

## Recommended Usage

Download and review the script first:

```bash
curl -fsSL https://raw.githubusercontent.com/SimonKennethRobo/RoboDuet/develop/scripts/setup/setup.sh -o setup.sh
bash setup.sh --interactive
```

Quick bootstrap is also supported. The script defaults to interactive mode:

```bash
curl -fsSL https://raw.githubusercontent.com/SimonKennethRobo/RoboDuet/develop/scripts/setup/setup.sh | bash
```

Environment variables can override defaults, which is useful for `curl | bash`:

```bash
BRANCH=develop \
INSTALL_ROOT=~/roboduet \
curl -fsSL https://raw.githubusercontent.com/SimonKennethRobo/RoboDuet/develop/scripts/setup/setup.sh | bash
```

If your system conda config points `pkgs_dirs` at a shared read-only cache, setup will add a writable user-level cache automatically. To do the same manually:

```bash
mkdir -p "$HOME/.conda/pkgs"
conda config --prepend pkgs_dirs "$HOME/.conda/pkgs"
```

For IsaacGym Preview 4, setup first searches for a local archive:

```text
~/Downloads/IsaacGym_Preview_4_Package.tar.gz
```

If the archive is not found, setup downloads it from NVIDIA's public redirecting download page:

```text
https://developer.nvidia.com/isaac-gym-preview-4
```

An explicit archive path or URL can still be supplied:

```bash
bash scripts/setup/setup.sh \
  --interactive \
  --isaacgym-archive ~/Downloads/IsaacGym_Preview_4_Package.tar.gz
```

## Non-Interactive Mode

For controlled CI/self-hosted runner setup:

```bash
bash scripts/setup/setup.sh \
  --yes \
  --repo-dir ~/roboduet/RoboDuet
```

## Verify Only

```bash
bash scripts/setup/setup.sh --verify-only
```

This prints GPU, conda, Python, `torch`, `isaacgym`, `go1_gym`, and repo status checks without installing anything.

## Common Options

```text
--install-root DIR
--repo-dir DIR
--repo-url URL
--branch NAME
--conda-dir DIR
--env-name NAME
--isaacgym-dir DIR
--isaacgym-archive PATH
--isaacgym-url URL
--skip-conda
--skip-repo
--skip-isaacgym
--skip-env
--no-torch
--verify-only
--yes
```

## Notes

The default PyTorch packages are:

```text
torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
```

They are installed from:

```text
https://download.pytorch.org/whl/cu121
```

This is intended for IsaacGym Preview 4 + Python 3.8 based RoboDuet workflows. NVIDIA driver CUDA runtime can be newer than the PyTorch CUDA wheel as long as the driver supports it.

The script follows the RoboDuet `develop` installation path:

```bash
git clone --branch develop https://github.com/SimonKennethRobo/RoboDuet.git
pip install -r requirements.txt
pip install -e .
```

For IsaacGym it follows the package docs at a high level:

```bash
tar -xf IsaacGym_Preview_4_Package.tar.gz
cd isaacgym/python
pip install -e .
```

It intentionally does not use IsaacGym's old conda environment recipe, because RoboDuet currently uses the verified `roboduet` environment with Python 3.8 and PyTorch 2.3.1 CUDA 12.1 wheels.
