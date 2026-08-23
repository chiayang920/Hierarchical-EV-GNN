#!/bin/bash

transformer_ev_die() {
  echo "ERROR: $*" >&2
  return 2
}

transformer_ev_require_python311() {
  local candidate="${EV_GNN_TRANSFORMER_EV_PYTHON:-}"
  [[ -n "${candidate}" ]] || { transformer_ev_die "EV_GNN_TRANSFORMER_EV_PYTHON is required; no bare-python fallback is permitted"; return $?; }
  [[ -x "${candidate}" ]] || { transformer_ev_die "EV_GNN_TRANSFORMER_EV_PYTHON is not executable: ${candidate}"; return $?; }
  local version
  version="$(${candidate} -I -c 'import platform,sys; print(platform.python_version()); raise SystemExit(0 if sys.version_info[:2] == (3,11) else 11)' 2>/dev/null)" || {
    transformer_ev_die "EV_GNN_TRANSFORMER_EV_PYTHON must be Python 3.11: ${candidate}"
    return $?
  }
  [[ "${version}" == 3.11.* ]] || { transformer_ev_die "Python 3.11 is required; observed ${version:-unknown}"; return $?; }
  PYTHON_BIN="${candidate}"
  PYTHON_VERSION="${version}"
  export PYTHON_BIN PYTHON_VERSION
}

transformer_ev_require_numeric_id() {
  local label="$1" value="$2"
  [[ "${value}" =~ ^[0-9]+$ ]] || { transformer_ev_die "${label} must contain decimal digits only"; return $?; }
}

transformer_ev_require_safe_id() {
  local label="$1" value="$2"
  [[ "${value}" =~ ^[A-Za-z0-9_.-]+$ ]] || { transformer_ev_die "${label} must be a safe identifier"; return $?; }
}

transformer_ev_require_source_identity() {
  local repo_root="$1" expected_source_identity="$2" actual_source_identity=""
  [[ -n "${repo_root}" && -d "${repo_root}" ]] || { transformer_ev_die "source root is missing: ${repo_root}"; return $?; }
  [[ -n "${expected_source_identity}" ]] || { transformer_ev_die "expected source identity is required"; return $?; }
  if git -C "${repo_root}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    actual_source_identity="$(git -C "${repo_root}" rev-parse HEAD)"
    [[ "${actual_source_identity}" == "${expected_source_identity}" ]] || {
      transformer_ev_die "source identity mismatch: expected ${expected_source_identity}, observed ${actual_source_identity}"
      return $?
    }
    git -C "${repo_root}" diff-index --quiet HEAD -- || {
      transformer_ev_die "source identity mismatch: tracked worktree is dirty"
      return $?
    }
    [[ -z "$(git -C "${repo_root}" status --porcelain=v1 --untracked-files=all)" ]] || {
      transformer_ev_die "source identity mismatch: worktree contains untracked files"
      return $?
    }
  else
    local source_file="${repo_root}/SOURCE_COMMIT_SHA.txt"
    [[ -f "${source_file}" ]] || { transformer_ev_die "source identity file is missing: ${source_file}"; return $?; }
    actual_source_identity="$(head -n 1 "${source_file}" | tr -d '[:space:]')"
    [[ "${actual_source_identity}" == "${expected_source_identity}" ]] || {
      transformer_ev_die "source identity mismatch: expected ${expected_source_identity}, observed ${actual_source_identity}"
      return $?
    }
  fi
}

transformer_ev_parse_mapping() {
  local mapping="$1" token key value
  unset task_id scale algorithm seed config training_steps
  for token in ${mapping}; do
    key="${token%%=*}"
    value="${token#*=}"
    case "${key}" in
      task_id|scale|algorithm|seed|config|training_steps) printf -v "${key}" '%s' "${value}" ;;
      *) transformer_ev_die "unsupported task-mapping field: ${key}"; return $? ;;
    esac
  done
  [[ -n "${task_id:-}" && -n "${scale:-}" && -n "${algorithm:-}" && -n "${seed:-}" && -n "${config:-}" && -n "${training_steps:-}" ]] || {
    transformer_ev_die "task mapping is incomplete"; return $?
  }
  [[ "${algorithm}" == "hierarchical_transformer_ev" ]] || {
    transformer_ev_die "unexpected algorithm for transformer-EV workflow: ${algorithm}"; return $?
  }
}

transformer_ev_validate_resource_profile() {
  local workflow_script="$1" profile="$2" scope="$3"
  [[ -n "${profile}" && -f "${profile}" ]] || { transformer_ev_die "reviewed resource profile is required"; return $?; }
  "${PYTHON_BIN}" "${workflow_script}" --validate-resource-profile "${profile}" --resource-profile-scope "${scope}" >/dev/null
  # shellcheck disable=SC1090
  source "${profile}"
}

transformer_ev_require_unique_file() {
  local directory="$1" pattern="$2" label="$3"
  local matches=()
  while IFS= read -r file; do matches+=("${file}"); done < <(find "${directory}" -maxdepth 1 -type f -name "${pattern}" -print | sort)
  [[ "${#matches[@]}" -eq 1 ]] || { transformer_ev_die "${label} requires exactly one file; observed ${#matches[@]}"; return $?; }
  printf '%s\n' "${matches[0]}"
}
