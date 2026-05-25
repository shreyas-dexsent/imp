#!/usr/bin/env bash
# Editable install of every Python package under platform/ in dependency order.
# After this runs (in an env with zenoh + protobuf 6.31+ + pin + coal + ompl):
#
#   * `import imp_sdk`             — works without PYTHONPATH gymnastics
#   * `import algorithms`           — motion-core / robot-algorithms
#   * `import imp_module_spatial_tf` and every sibling module — works
#   * entry-point discovery sees every plugin under imp.hal/.modules/.services/.jobs
#
# Used by CI (.github/workflows/ci.yml) and the developer setup docs (P4
# Quality Foundation; closes debt D6 in PLAN.md).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
echo "platform root: ${ROOT}"

# Order matters: SDK + motion-core first so module installs resolve their deps.
PACKAGES=(
  "sdk/py"
  "modules/motion-core/algorithms"

  # Functional modules
  "modules/motion-pinocchio"
  "modules/motion-coal"
  "modules/motion-ompl"
  "modules/motion-cartesian"
  "modules/motion-path-processor"
  "modules/motion-ruckig"
  "modules/motion-grasp-library"
  "modules/spatial-tf"
  "modules/spatial-transform"

  # HAL drivers
  "hal/robot-mujoco-ur5e"
)

PIP="${PIP:-pip}"

for pkg in "${PACKAGES[@]}"; do
  if [[ ! -f "${pkg}/pyproject.toml" ]]; then
    echo "skip ${pkg} (no pyproject.toml)"
    continue
  fi
  echo "==> pip install -e ${pkg}"
  ${PIP} install --no-build-isolation -e "${pkg}"
done

echo
echo "Dev install complete. Sanity:"
python -c "import imp_sdk; print('imp_sdk:', imp_sdk.__file__)"
python -c "from imp_sdk.discover import list_plugins; \
  print('plugins:', [(p.group, p.name) for p in sorted(list_plugins(), key=lambda r: (r.group, r.name))])"
