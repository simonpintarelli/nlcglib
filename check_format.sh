#!/bin/bash

check_diff() {
    local status=0
    for file in "$@"; do
        if ! diff -u "$file" <(clang-format "$file") --label "$file" --label "$file (clang-format)"; then
            status=1
        fi
    done
    return $status
}

export -f check_diff

find . -type f \( -name "*.cpp" -o -name "*.hpp" \) ! -path "./build-env/*" ! -path "./.*" -exec bash -c 'check_diff "$@"' sh {} +
