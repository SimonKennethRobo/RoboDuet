#!/usr/bin/env bash
set -euo pipefail

# RoboDuet workstation / GPU node setup helper.
#
# Defaults are intentionally conservative:
# - interactive mode is enabled unless --yes is passed;
# - existing directories, conda envs, and repos are reused by default;
# - dirty git worktrees are not overwritten;
# - IsaacGym is auto-discovered locally or downloaded from NVIDIA's redirecting page.

: "${REPO_URL:=https://github.com/SimonKennethRobo/RoboDuet.git}"
: "${BRANCH:=develop}"
: "${INSTALL_ROOT:=$HOME/roboduet}"
: "${REPO_DIR:=}"
: "${CONDA_DIR:=$HOME/miniconda3}"
: "${ENV_NAME:=roboduet}"
: "${PYTHON_VERSION:=3.8}"
: "${ISAACGYM_DIR:=}"
: "${ISAACGYM_ARCHIVE:=}"
: "${ISAACGYM_URL:=https://developer.nvidia.com/isaac-gym-preview-4}"
: "${ISAACGYM_DOWNLOAD_PAGE:=https://developer.nvidia.com/isaac-gym-preview-4}"
: "${TORCH_INDEX_URL:=https://download.pytorch.org/whl/cu121}"
: "${TORCH_PACKAGES:=torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1}"
: "${INTERACTIVE:=1}"
: "${VERIFY_ONLY:=0}"
: "${INSTALL_TORCH:=1}"
: "${SKIP_CONDA:=0}"
: "${SKIP_REPO:=0}"
: "${SKIP_ISAACGYM:=0}"
: "${SKIP_ENV:=0}"
: "${ROBODUET_NO_EMOJI:=0}"

SELECTED_STEPS=()
SUCCESS_STEPS=()
SKIPPED_STEPS=()
FAILED_STEPS=()
STEP_IDS=()
STEP_LABELS=()
STEP_DESCRIPTIONS=()
STEP_DEFAULTS=()

if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
  BOLD=$'\033[1m'
  DIM=$'\033[2m'
  BLUE=$'\033[34m'
  GREEN=$'\033[32m'
  YELLOW=$'\033[33m'
  RED=$'\033[31m'
  RESET=$'\033[0m'
else
  BOLD=""
  DIM=""
  BLUE=""
  GREEN=""
  YELLOW=""
  RED=""
  RESET=""
fi

prefix() {
  local emoji="$1"
  local label="$2"
  local color="$3"
  if [[ "$ROBODUET_NO_EMOJI" -eq 1 ]]; then
    printf '%b[%s]%b' "$color" "$label" "$RESET"
  else
    printf '%b%s %s%b' "$color" "$emoji" "$label" "$RESET"
  fi
}

log() {
  printf '%s %s\n' "$(prefix "🔧" "setup" "$BLUE")" "$*"
}

ok() {
  printf '%s %s\n' "$(prefix "✅" "setup" "$GREEN")" "$*"
}

skip() {
  printf '%s %s\n' "$(prefix "⏭️" "setup skip" "$DIM")" "$*"
}

warn() {
  printf '%s %s\n' "$(prefix "⚠️" "setup warn" "$YELLOW")" "$*" >&2
}

die() {
  printf '%s %s\n' "$(prefix "❌" "setup error" "$RED")" "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  bash scripts/setup/setup.sh [options]

Common modes:
  --interactive              Ask before installing or updating. This is the default.
  --yes                      Non-interactive mode. Install missing pieces with defaults.
  --verify-only              Only print environment checks.

Install locations:
  --install-root DIR         Default: ~/roboduet
  --repo-dir DIR             Default: <install-root>/RoboDuet
  --repo-url URL             Default: SimonKennethRobo/RoboDuet
  --branch NAME              Default: develop
  --conda-dir DIR            Default: ~/miniconda3
  --env-name NAME            Default: roboduet
  --isaacgym-dir DIR         Default: <install-root>/isaacgym

IsaacGym:
  The script auto-detects IsaacGym_Preview_4_Package.tar.gz in common locations:
    ~/Downloads, ./Downloads, <install-root>, and /tmp.
  --isaacgym-archive PATH    Optional explicit local archive path
  --isaacgym-url URL         Download URL. Default: https://developer.nvidia.com/isaac-gym-preview-4

Skips:
  --skip-conda
  --skip-repo
  --skip-isaacgym
  --skip-env
  --no-torch                 Do not install PyTorch packages

Examples:
  bash scripts/setup/setup.sh --interactive
  bash scripts/setup/setup.sh --yes
  curl -fsSL https://raw.githubusercontent.com/SimonKennethRobo/RoboDuet/develop/scripts/setup/setup.sh | bash
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --interactive) INTERACTIVE=1; shift ;;
    --yes|-y|--non-interactive) INTERACTIVE=0; shift ;;
    --verify-only) VERIFY_ONLY=1; shift ;;
    --install-root) INSTALL_ROOT="$2"; shift 2 ;;
    --repo-dir) REPO_DIR="$2"; shift 2 ;;
    --repo-url) REPO_URL="$2"; shift 2 ;;
    --branch) BRANCH="$2"; shift 2 ;;
    --conda-dir) CONDA_DIR="$2"; shift 2 ;;
    --env-name) ENV_NAME="$2"; shift 2 ;;
    --python-version) PYTHON_VERSION="$2"; shift 2 ;;
    --isaacgym-dir) ISAACGYM_DIR="$2"; shift 2 ;;
    --isaacgym-archive) ISAACGYM_ARCHIVE="$2"; shift 2 ;;
    --isaacgym-url) ISAACGYM_URL="$2"; shift 2 ;;
    --isaacgym-download-page) ISAACGYM_DOWNLOAD_PAGE="$2"; shift 2 ;;
    --torch-index-url) TORCH_INDEX_URL="$2"; shift 2 ;;
    --torch-packages) TORCH_PACKAGES="$2"; shift 2 ;;
    --skip-conda) SKIP_CONDA=1; shift ;;
    --skip-repo) SKIP_REPO=1; shift ;;
    --skip-isaacgym) SKIP_ISAACGYM=1; shift ;;
    --skip-env) SKIP_ENV=1; shift ;;
    --no-torch) INSTALL_TORCH=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

REPO_DIR="${REPO_DIR:-$INSTALL_ROOT/RoboDuet}"
ISAACGYM_DIR="${ISAACGYM_DIR:-$INSTALL_ROOT/isaacgym}"

confirm() {
  local prompt="$1"
  if [[ "$INTERACTIVE" -eq 0 ]]; then
    return 0
  fi
  if [[ ! -r /dev/tty ]]; then
    warn "no interactive TTY available; treating prompt as no: $prompt"
    return 1
  fi
  local answer
  printf '%b%s%b %b[y/N]%b ' "$BOLD" "$prompt" "$RESET" "$DIM" "$RESET" >/dev/tty
  read -r answer </dev/tty
  case "$answer" in
    y|Y|yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

prompt_value() {
  local prompt="$1"
  local default_value="$2"
  if [[ "$INTERACTIVE" -eq 0 ]]; then
    printf '%s' "$default_value"
    return
  fi
  if [[ ! -r /dev/tty ]]; then
    warn "no interactive TTY available; using default value for: $prompt"
    printf '%s' "$default_value"
    return
  fi
  local answer
  printf '%b%s%b %b[%s]%b ' "$BOLD" "$prompt" "$RESET" "$DIM" "$default_value" "$RESET" >/dev/tty
  read -r answer </dev/tty
  printf '%s' "${answer:-$default_value}"
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

contains_step() {
  local needle="$1"
  local item
  for item in "${SELECTED_STEPS[@]}"; do
    [[ "$item" == "$needle" ]] && return 0
  done
  return 1
}

record_success() {
  SUCCESS_STEPS+=("$1")
}

record_skip() {
  SKIPPED_STEPS+=("$1")
}

record_failure() {
  FAILED_STEPS+=("$1")
}

conda_exe() {
  if have_cmd conda; then
    command -v conda
  elif [[ -x "$CONDA_DIR/bin/conda" ]]; then
    printf '%s\n' "$CONDA_DIR/bin/conda"
  else
    return 1
  fi
}

is_conda_writable_dir() {
  local dir="$1"
  local probe
  mkdir -p "$dir" 2>/dev/null || return 1
  probe="$dir/.roboduet_write_probe_$$"
  if touch "$probe" 2>/dev/null; then
    rm -f "$probe"
    return 0
  fi
  return 1
}

configure_conda_package_cache() {
  local conda_bin="$1"
  local dir
  local user_cache_dir="$CONDA_DIR/pkgs"
  [[ "$CONDA_DIR" == "$HOME"* ]] || user_cache_dir="$HOME/.conda/pkgs"

  # Ensure a user-writable cache is always prepended so conda can use it,
  # regardless of whether a shared cache passes filesystem permission checks.
  # (bash -w can return true for dirs where conda's own write probe fails.)
  if is_conda_writable_dir "$user_cache_dir"; then
    if ! "$conda_bin" config --show pkgs_dirs 2>/dev/null | grep -qF "$user_cache_dir"; then
      log "adding writable conda package cache: $user_cache_dir"
      "$conda_bin" config --prepend pkgs_dirs "$user_cache_dir" >/dev/null
    else
      log "using conda package cache: $user_cache_dir"
    fi
    return
  fi

  # Fallback: check existing configured dirs with an actual write probe.
  while IFS= read -r dir; do
    [[ -n "$dir" ]] || continue
    if is_conda_writable_dir "$dir"; then
      log "using conda package cache: $dir"
      return
    fi
  done < <("$conda_bin" config --show pkgs_dirs 2>/dev/null | awk '
    /^[[:space:]]*-[[:space:]]/ {
      sub(/^[[:space:]]*-[[:space:]]*/, "")
      print
    }
  ')

  die "no writable conda package cache found; grant write access or add a writable entry with: conda config --prepend pkgs_dirs \$HOME/.conda/pkgs"
}

run_in_env() {
  local conda_bin
  local prefix
  conda_bin="$(conda_exe)" || die "conda is not available"
  if prefix="$(env_prefix 2>/dev/null)"; then
    "$conda_bin" run -n "$ENV_NAME" env LD_LIBRARY_PATH="$prefix/lib:${LD_LIBRARY_PATH:-}" "$@"
  else
    "$conda_bin" run -n "$ENV_NAME" "$@"
  fi
}

env_exists() {
  local conda_bin
  conda_bin="$(conda_exe)" || return 1
  "$conda_bin" env list | awk '{print $1}' | grep -qx "$ENV_NAME"
}

env_prefix() {
  local conda_bin
  conda_bin="$(conda_exe)" || return 1
  "$conda_bin" env list | awk -v env="$ENV_NAME" '
    $1 == env { print $NF; found=1; exit }
    $1 == "*" && prev == env { print $NF; found=1; exit }
    { prev=$1 }
    END { if (!found) exit 1 }
  '
}

configure_env_library_path() {
  local conda_bin="$1"
  local prefix
  prefix="$(env_prefix)" || return
  log "configuring $ENV_NAME LD_LIBRARY_PATH for IsaacGym"
  "$conda_bin" env config vars set -n "$ENV_NAME" LD_LIBRARY_PATH="$prefix/lib" >/dev/null 2>&1 \
    || warn "could not persist LD_LIBRARY_PATH for $ENV_NAME"
}

accept_conda_tos() {
  local conda_bin="$1"
  if ! "$conda_bin" tos --help >/dev/null 2>&1; then
    return
  fi
  log "accepting Anaconda default channel Terms of Service if needed"
  "$conda_bin" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 \
    || warn "could not accept ToS for pkgs/main; conda may ask you to run it manually"
  "$conda_bin" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 \
    || warn "could not accept ToS for pkgs/r; conda may ask you to run it manually"
}

install_miniconda() {
  [[ "$SKIP_CONDA" -eq 1 ]] && return
  if conda_exe >/dev/null 2>&1; then
    log "conda found: $(conda_exe)"
    return
  fi
  confirm "conda not found. Install Miniconda to $CONDA_DIR?" || return
  mkdir -p "$(dirname "$CONDA_DIR")"
  local installer
  installer="$(mktemp /tmp/miniconda.XXXXXX.sh)"
  log "downloading Miniconda installer"
  curl -fsSL "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" -o "$installer"
  bash "$installer" -b -p "$CONDA_DIR"
  rm -f "$installer"
  ok "Miniconda installed: $CONDA_DIR"
}

setup_repo() {
  [[ "$SKIP_REPO" -eq 1 ]] && return
  if [[ -d "$REPO_DIR/.git" ]]; then
    log "repo found: $REPO_DIR"
    git -C "$REPO_DIR" remote -v | sed 's/^/[setup] remote /'
    git -C "$REPO_DIR" status --short --branch | sed 's/^/[setup] /'
    if [[ -n "$(git -C "$REPO_DIR" status --porcelain)" ]]; then
      warn "repo has local changes; skip automatic update"
      return
    fi
    confirm "Update repo $REPO_DIR to $BRANCH?" || return
    git -C "$REPO_DIR" fetch --all --prune
    git -C "$REPO_DIR" switch "$BRANCH"
    git -C "$REPO_DIR" pull --ff-only
    return
  fi

  confirm "Clone RoboDuet into $REPO_DIR?" || return
  mkdir -p "$(dirname "$REPO_DIR")"
  git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
}

ensure_conda_init() {
  local conda_bin="$1"
  local shell_rc="$HOME/.bashrc"
  if grep -q "conda initialize" "$shell_rc" 2>/dev/null; then
    return
  fi
  log "running conda init bash (adds conda to $shell_rc)"
  "$conda_bin" init bash >/dev/null
  ok "conda init bash complete; conda will be available in new shells"
}

setup_conda_env() {
  [[ "$SKIP_ENV" -eq 1 ]] && return
  local conda_bin
  conda_bin="$(conda_exe)" || die "conda is required to create/update $ENV_NAME"
  ensure_conda_init "$conda_bin"
  if env_exists; then
    log "conda env exists: $ENV_NAME"
    configure_env_library_path "$conda_bin"
  else
    confirm "Create conda env $ENV_NAME with python=$PYTHON_VERSION?" || return
    configure_conda_package_cache "$conda_bin"
    accept_conda_tos "$conda_bin"
    "$conda_bin" create -y -n "$ENV_NAME" "python=$PYTHON_VERSION" pip
    configure_env_library_path "$conda_bin"
  fi

  if [[ "$INSTALL_TORCH" -eq 1 ]]; then
    if confirm "Install/update PyTorch in $ENV_NAME from $TORCH_INDEX_URL?"; then
      # shellcheck disable=SC2086
      "$conda_bin" run -n "$ENV_NAME" python -m pip install --index-url "$TORCH_INDEX_URL" $TORCH_PACKAGES
    fi
  fi

  if [[ -f "$REPO_DIR/requirements.txt" ]]; then
    if confirm "Install RoboDuet Python requirements and editable package?"; then
      "$conda_bin" run -n "$ENV_NAME" python -m pip install -r "$REPO_DIR/requirements.txt"
      "$conda_bin" run -n "$ENV_NAME" python -m pip install -e "$REPO_DIR"
    fi
  else
    warn "requirements.txt not found under $REPO_DIR; skip repo package install"
  fi
}

step_conda() {
  if [[ "$SKIP_CONDA" -eq 1 ]]; then
    skip "conda step disabled by --skip-conda"
    return
  fi
  install_miniconda
}

step_repo() {
  if [[ "$SKIP_REPO" -eq 1 ]]; then
    skip "repo step disabled by --skip-repo"
    return
  fi
  setup_repo
}

step_env() {
  if [[ "$SKIP_ENV" -eq 1 ]]; then
    skip "conda env step disabled by --skip-env"
    return
  fi
  setup_conda_env
}

step_isaacgym() {
  if [[ "$SKIP_ISAACGYM" -eq 1 ]]; then
    skip "IsaacGym step disabled by --skip-isaacgym"
    return
  fi
  setup_isaacgym
}

resolve_isaacgym_archive() {
  if [[ -n "$ISAACGYM_ARCHIVE" ]]; then
    printf '%s\n' "$ISAACGYM_ARCHIVE"
    return
  fi

  local candidates=(
    "$HOME/Downloads/IsaacGym_Preview_4_Package.tar.gz"
    "$HOME/Downloads/isaacgym/IsaacGym_Preview_4_Package.tar.gz"
    "$PWD/Downloads/IsaacGym_Preview_4_Package.tar.gz"
    "$INSTALL_ROOT/IsaacGym_Preview_4_Package.tar.gz"
    "$INSTALL_ROOT/downloads/IsaacGym_Preview_4_Package.tar.gz"
    "$INSTALL_ROOT/Downloads/IsaacGym_Preview_4_Package.tar.gz"
    "/tmp/IsaacGym_Preview_4_Package.tar.gz"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  done

  if [[ -n "$ISAACGYM_URL" ]]; then
    local archive
    archive="/tmp/IsaacGym_Preview_4_Package.tar.gz"
    printf '[setup] downloading IsaacGym archive from %s\n' "$ISAACGYM_URL" >&2
    curl -fL --retry 3 --retry-delay 3 -A "Mozilla/5.0" "$ISAACGYM_URL" -o "$archive"
    printf '%s\n' "$archive"
    return
  fi
  warn "IsaacGym Preview 4 archive not found in common locations."
  warn "Download it from $ISAACGYM_DOWNLOAD_PAGE and place IsaacGym_Preview_4_Package.tar.gz in ~/Downloads, or pass --isaacgym-archive / --isaacgym-url."
  return 1
}

validate_isaacgym_archive() {
  local archive="$1"
  tar -tf "$archive" isaacgym/python/setup.py >/dev/null 2>&1 \
    || die "not a valid IsaacGym Preview 4 archive: $archive"
}

setup_isaacgym() {
  [[ "$SKIP_ISAACGYM" -eq 1 ]] && return
  if ! env_exists; then
    warn "conda env $ENV_NAME does not exist; skip IsaacGym install"
    return
  fi
  if run_in_env python -c 'import isaacgym' >/dev/null 2>&1; then
    log "isaacgym import works in env $ENV_NAME"
    return
  fi

  if [[ ! -d "$ISAACGYM_DIR/python" ]]; then
    local archive
    archive="$(resolve_isaacgym_archive)" || {
      warn "skip IsaacGym install"
      return
    }
    [[ -f "$archive" ]] || die "IsaacGym archive not found: $archive"
    validate_isaacgym_archive "$archive"
    log "IsaacGym archive: $archive"
    confirm "Extract IsaacGym archive to $INSTALL_ROOT?" || return
    mkdir -p "$INSTALL_ROOT"
    tar -xf "$archive" -C "$INSTALL_ROOT"
    if [[ ! -d "$ISAACGYM_DIR/python" && -d "$INSTALL_ROOT/isaacgym/python" ]]; then
      ISAACGYM_DIR="$INSTALL_ROOT/isaacgym"
    fi
  fi

  [[ -d "$ISAACGYM_DIR/python" ]] || die "IsaacGym python directory not found: $ISAACGYM_DIR/python"
  confirm "Install IsaacGym editable package from $ISAACGYM_DIR/python?" || return
  run_in_env python -m pip install -e "$ISAACGYM_DIR/python"
}

verify_env() {
  log "system checks"
  if have_cmd nvidia-smi; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
  else
    warn "nvidia-smi not found"
  fi
  if have_cmd git; then
    git --version
  else
    warn "git not found"
  fi
  if conda_exe >/dev/null 2>&1; then
    local conda_bin
    conda_bin="$(conda_exe)"
    log "conda: $conda_bin"
    "$conda_bin" --version
  else
    warn "conda not found"
  fi

  if env_exists; then
    log "python environment checks: $ENV_NAME"
    run_in_env python -c '
import importlib
import sys

print("python", sys.version.split()[0])
for name in ["isaacgym", "torch", "go1_gym"]:
    try:
        module = importlib.import_module(name)
        print(name, "ok", getattr(module, "__version__", ""))
        if name == "torch":
            print("torch_cuda", module.version.cuda, "available", module.cuda.is_available())
    except Exception as exc:
        print(name, "failed", type(exc).__name__, exc)
'
  else
    warn "conda env not found: $ENV_NAME"
  fi

  if [[ -d "$REPO_DIR/.git" ]]; then
    log "repo checks: $REPO_DIR"
    git -C "$REPO_DIR" status --short --branch
    git -C "$REPO_DIR" log -1 --oneline
  else
    warn "repo not found: $REPO_DIR"
  fi
}

print_configuration() {
  log "configuration"
  cat <<EOF
[setup] install_root=$INSTALL_ROOT
[setup] repo_dir=$REPO_DIR
[setup] repo_url=$REPO_URL
[setup] branch=$BRANCH
[setup] conda_dir=$CONDA_DIR
[setup] env_name=$ENV_NAME
[setup] isaacgym_dir=$ISAACGYM_DIR
[setup] interactive=$INTERACTIVE
EOF
}

print_detected_state() {
  log "detected environment"
  if conda_exe >/dev/null 2>&1; then
    printf '[setup] conda: %s\n' "$(conda_exe)"
  else
    printf '[setup] conda: missing\n'
  fi
  if env_exists; then
    printf '[setup] conda env: %s exists\n' "$ENV_NAME"
  else
    printf '[setup] conda env: %s missing\n' "$ENV_NAME"
  fi
  if [[ -d "$REPO_DIR/.git" ]]; then
    printf '[setup] repo: %s exists\n' "$REPO_DIR"
    git -C "$REPO_DIR" status --short --branch | sed 's/^/[setup] repo /'
  else
    printf '[setup] repo: %s missing\n' "$REPO_DIR"
  fi
  if [[ -d "$ISAACGYM_DIR/python" ]]; then
    printf '[setup] IsaacGym dir: %s exists\n' "$ISAACGYM_DIR"
  else
    printf '[setup] IsaacGym dir: %s missing\n' "$ISAACGYM_DIR"
  fi
}

add_step() {
  STEP_IDS+=("$1")
  STEP_LABELS+=("$2")
  STEP_DESCRIPTIONS+=("$3")
  STEP_DEFAULTS+=("$4")
}

build_steps() {
  STEP_IDS=()
  STEP_LABELS=()
  STEP_DESCRIPTIONS=()
  STEP_DEFAULTS=()

  if [[ "$SKIP_CONDA" -eq 0 ]]; then
    if conda_exe >/dev/null 2>&1; then
      add_step "conda" "Conda" "Already found; keep or inspect existing conda" "OFF"
    else
      add_step "conda" "Conda" "Install Miniconda to $CONDA_DIR" "ON"
    fi
  fi

  if [[ "$SKIP_REPO" -eq 0 ]]; then
    if [[ -d "$REPO_DIR/.git" ]]; then
      if [[ -n "$(git -C "$REPO_DIR" status --porcelain)" ]]; then
        add_step "repo" "Repo" "Repo exists with local changes; selected step will only report and skip update" "OFF"
      else
        add_step "repo" "Repo" "Repo exists; update to $BRANCH" "OFF"
      fi
    else
      add_step "repo" "Repo" "Clone $BRANCH into $REPO_DIR" "ON"
    fi
  fi

  if [[ "$SKIP_ENV" -eq 0 ]]; then
    if env_exists; then
      add_step "env" "Python env" "Env exists; optionally update PyTorch and RoboDuet package" "OFF"
    else
      add_step "env" "Python env" "Create $ENV_NAME and install PyTorch/RoboDuet dependencies" "ON"
    fi
  fi

  if [[ "$SKIP_ISAACGYM" -eq 0 ]]; then
    if [[ -d "$ISAACGYM_DIR/python" ]] || (env_exists && run_in_env python -c 'import isaacgym' >/dev/null 2>&1); then
      add_step "isaacgym" "IsaacGym" "IsaacGym appears installed; optionally reinstall editable package" "OFF"
    else
      add_step "isaacgym" "IsaacGym" "Install IsaacGym Preview 4" "ON"
    fi
  fi

  add_step "verify" "Verify" "Run GPU/conda/Python/repo checks" "ON"
}

choose_steps_numbered() {
  if [[ "$INTERACTIVE" -eq 0 || ! -r /dev/tty ]]; then
    SELECTED_STEPS=()
    local i
    for i in "${!STEP_IDS[@]}"; do
      [[ "${STEP_DEFAULTS[$i]}" == "ON" ]] && SELECTED_STEPS+=("${STEP_IDS[$i]}")
    done
    return
  fi

  printf '\n%bRoboDuet setup modules%b\n' "$BOLD" "$RESET" >/dev/tty
  local defaults=()
  local i
  for i in "${!STEP_IDS[@]}"; do
    local marker=" "
    if [[ "${STEP_DEFAULTS[$i]}" == "ON" ]]; then
      marker="*"
      defaults+=("$((i + 1))")
    fi
    printf '  %2d) [%s] %-12s %s\n' "$((i + 1))" "$marker" "${STEP_LABELS[$i]}" "${STEP_DESCRIPTIONS[$i]}" >/dev/tty
  done
  printf '\nSelect modules by number, separated by spaces or commas. Press Enter for defaults: %s\n' "${defaults[*]}" >/dev/tty
  local answer
  read -r answer </dev/tty
  answer="${answer//,/ }"
  if [[ -z "$answer" ]]; then
    answer="${defaults[*]}"
  fi
  SELECTED_STEPS=()
  local token
  for token in $answer; do
    if [[ "$token" =~ ^[0-9]+$ ]] && (( token >= 1 && token <= ${#STEP_IDS[@]} )); then
      SELECTED_STEPS+=("${STEP_IDS[$((token - 1))]}")
    fi
  done
}

draw_steps_tui() {
  local cursor="$1"
  shift
  local selected=("$@")
  local cols
  cols="$(tput cols 2>/dev/null || printf '100')"
  local width=$((cols - 4))
  (( width > 88 )) && width=88
  (( width < 60 )) && width=60
  local line
  local border
  printf -v border '%*s' "$width" ''
  border="${border// /-}"
  printf '\033c' >/dev/tty
  printf '%b%s%b\n' "$BOLD$BLUE" "RoboDuet setup" "$RESET" >/dev/tty
  printf '%b%s%b %s  %b%s%b %s  %b%s%b %s\n\n' \
    "$BOLD" "Move:" "$RESET" "Up/Down or j/k" \
    "$BOLD" "Toggle:" "$RESET" "Space" \
    "$BOLD" "Run:" "$RESET" "Enter  |  a: all  q: cancel" >/dev/tty
  printf '%b+%s+%b\n' "$DIM" "$border" "$RESET" >/dev/tty

  local i
  for i in "${!STEP_IDS[@]}"; do
    local pointer=" "
    local check=" "
    local row_color="$RESET"
    [[ "$i" -eq "$cursor" ]] && pointer=">" && row_color="$BOLD$BLUE"
    [[ "${selected[$i]}" -eq 1 ]] && check="x"
    line=$(printf '%s [%s] %-12s %s' "$pointer" "$check" "${STEP_LABELS[$i]}" "${STEP_DESCRIPTIONS[$i]}")
    printf '%b| %-*.*s |%b\n' "$row_color" "$width" "$width" "$line" "$RESET" >/dev/tty
  done

  printf '%b+%s+%b\n' "$DIM" "$border" "$RESET" >/dev/tty
  printf '%b%s%b\n' "$DIM" "Selected modules will run immediately after Enter. Non-selected modules are skipped." "$RESET" >/dev/tty
}

choose_steps_tui() {
  [[ "$INTERACTIVE" -eq 1 && -r /dev/tty && -t 1 ]] || return 1

  local selected=()
  local i
  for i in "${!STEP_IDS[@]}"; do
    if [[ "${STEP_DEFAULTS[$i]}" == "ON" ]]; then
      selected[$i]=1
    else
      selected[$i]=0
    fi
  done

  local cursor=0
  local key
  local old_stty
  old_stty="$(stty -g </dev/tty)"
  stty -echo -icanon min 1 time 0 </dev/tty
  printf '\033[?25l' >/dev/tty

  while true; do
    draw_steps_tui "$cursor" "${selected[@]}"
    if ! IFS= read -rsn1 key </dev/tty; then
      stty "$old_stty" </dev/tty
      printf '\033[?25h' >/dev/tty
      return 1
    fi
    case "$key" in
      $'\x1b')
        local rest
        IFS= read -rsn2 -t 0.05 rest </dev/tty || rest=""
        case "$rest" in
          "[A") ((cursor > 0)) && ((cursor--)) ;;
          "[B") ((cursor < ${#STEP_IDS[@]} - 1)) && ((cursor++)) ;;
        esac
        ;;
      k|K) ((cursor > 0)) && ((cursor--)) ;;
      j|J) ((cursor < ${#STEP_IDS[@]} - 1)) && ((cursor++)) ;;
      " ") selected[$cursor]=$((1 - selected[$cursor])) ;;
      a|A)
        for i in "${!selected[@]}"; do selected[$i]=1; done
        ;;
      q|Q)
        SELECTED_STEPS=()
        stty "$old_stty" </dev/tty
        printf '\033[?25h' >/dev/tty
        printf '\033c' >/dev/tty
        return 0
        ;;
      ""|$'\r'|$'\n')
        SELECTED_STEPS=()
        for i in "${!STEP_IDS[@]}"; do
          [[ "${selected[$i]}" -eq 1 ]] && SELECTED_STEPS+=("${STEP_IDS[$i]}")
        done
        stty "$old_stty" </dev/tty
        printf '\033[?25h' >/dev/tty
        printf '\033c' >/dev/tty
        return 0
        ;;
    esac
  done
}

choose_steps() {
  build_steps
  if choose_steps_tui; then
    return
  fi
  choose_steps_numbered
}

run_step() {
  local id="$1"
  local label="$2"
  shift 2
  if ! contains_step "$id"; then
    skip "$label not selected"
    record_skip "$label"
    return
  fi

  log "starting $label"
  if "$@"; then
    ok "$label complete"
    record_success "$label"
  else
    warn "$label failed"
    record_failure "$label"
  fi
}

print_next_steps() {
  env_exists || return 0
  printf '\n'
  log "next steps — activate the environment in your shell:"
  printf '    %bsource ~/.bashrc%b\n' "$BOLD" "$RESET"
  printf '    %bconda activate %s%b\n' "$BOLD" "$ENV_NAME" "$RESET"
}

print_summary() {
  printf '\n'
  log "summary"
  if [[ "${#SUCCESS_STEPS[@]}" -gt 0 ]]; then
    ok "completed: ${SUCCESS_STEPS[*]}"
  fi
  if [[ "${#SKIPPED_STEPS[@]}" -gt 0 ]]; then
    skip "skipped: ${SKIPPED_STEPS[*]}"
  fi
  if [[ "${#FAILED_STEPS[@]}" -gt 0 ]]; then
    warn "failed: ${FAILED_STEPS[*]}"
    return 1
  fi
  ok "RoboDuet setup finished"
  print_next_steps
}

print_configuration

if [[ "$VERIFY_ONLY" -eq 1 ]]; then
  verify_env
  exit 0
fi

print_detected_state
choose_steps

if [[ "${#SELECTED_STEPS[@]}" -eq 0 ]]; then
  warn "no setup modules selected"
  exit 0
fi

log "selected modules: ${SELECTED_STEPS[*]}"
INTERACTIVE=0
run_step "conda" "Conda" step_conda
run_step "repo" "Repo" step_repo
run_step "env" "Python env" step_env
run_step "isaacgym" "IsaacGym" step_isaacgym
run_step "verify" "Verify" verify_env
print_summary
