#!/bin/bash

formal75k_die() {
  echo "ERROR: $*" >&2
  return 2
}

formal75k_require_python311() {
  local candidate="${EV_GNN_FORMAL_75K_PYTHON:-}"
  [[ -n "${candidate}" ]] || { formal75k_die "EV_GNN_FORMAL_75K_PYTHON is required; no bare-python fallback is permitted"; return $?; }
  [[ -x "${candidate}" ]] || { formal75k_die "EV_GNN_FORMAL_75K_PYTHON is not executable: ${candidate}"; return $?; }
  local version
  version="$(${candidate} -I -c 'import platform,sys; print(platform.python_version()); raise SystemExit(0 if sys.version_info[:2] == (3,11) else 11)' 2>/dev/null)" || {
    formal75k_die "EV_GNN_FORMAL_75K_PYTHON must be Python 3.11: ${candidate}"
    return $?
  }
  [[ "${version}" == 3.11.* ]] || { formal75k_die "Python 3.11 is required; observed ${version:-unknown}"; return $?; }
  PYTHON_BIN="${candidate}"
  PYTHON_VERSION="${version}"
  export PYTHON_BIN PYTHON_VERSION
}

formal75k_require_numeric_id() {
  local label="$1" value="$2"
  [[ "${value}" =~ ^[0-9]+$ ]] || { formal75k_die "${label} must contain decimal digits only"; return $?; }
}

formal75k_parse_mapping() {
  local mapping="$1" token key value
  unset task_id scale algorithm seed config training_steps
  for token in ${mapping}; do
    key="${token%%=*}"
    value="${token#*=}"
    case "${key}" in
      task_id|scale|algorithm|seed|config|training_steps) printf -v "${key}" '%s' "${value}" ;;
      *) formal75k_die "unsupported task-mapping field: ${key}"; return $? ;;
    esac
  done
  [[ -n "${task_id:-}" && -n "${scale:-}" && -n "${algorithm:-}" && -n "${seed:-}" && -n "${config:-}" && -n "${training_steps:-}" ]] || {
    formal75k_die "task mapping is incomplete"; return $?
  }
}

formal75k_validate_resource_profile() {
  local workflow_script="$1" profile="$2" scope="$3"
  [[ -n "${profile}" && -f "${profile}" ]] || { formal75k_die "reviewed resource profile is required"; return $?; }
  "${PYTHON_BIN}" "${workflow_script}" --validate-resource-profile "${profile}" --resource-profile-scope "${scope}" >/dev/null
  # shellcheck disable=SC1090
  source "${profile}"
}

formal75k_require_unique_file() {
  local directory="$1" pattern="$2" label="$3"
  local matches=()
  while IFS= read -r file; do matches+=("${file}"); done < <(find "${directory}" -maxdepth 1 -type f -name "${pattern}" -print | sort)
  [[ "${#matches[@]}" -eq 1 ]] || { formal75k_die "${label} requires exactly one file; observed ${#matches[@]}"; return $?; }
  printf '%s\n' "${matches[0]}"
}
