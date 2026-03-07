#!/usr/bin/env bash

ensure_python_package() {
    local import_name="$1"
    local package_spec="$2"

    python3 - <<PY >/dev/null 2>&1
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec(${import_name@Q}) else 1)
PY
    if [ $? -eq 0 ]; then
        return 0
    fi

    timeout 20s python3 -m pip install --disable-pip-version-check --default-timeout 5 --retries 1 "$package_spec" || true
}

# Ignore failures, see https://github.com/SWE-agent/SWE-agent/issues/1179
ensure_python_package tree_sitter 'tree-sitter==0.21.3'
ensure_python_package tree_sitter_languages 'tree-sitter-languages'
