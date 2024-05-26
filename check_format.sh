#!/usr/bin/bash

check_diff() {
    local status=0
	  for file in "$@"; do
		    if ! diff -q "$file" <(clang-format "$file"); then
            status=1
        fi
	  done
    return $status
}

export -f check_diff

find . -type f \( -name "*.cpp" -o -name "*.hpp" \) ! -path "./build-env/*" ! -path "./.*" -exec sh -c 'check_diff "$@"' sh {} +
