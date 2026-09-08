# Just is a task runner, like Make but without the build system / dependency tracking part.
# docs: https://github.com/casey/just
#
# The `-ci` variants are ran in CI, they do command grouping on GitHub Actions, set consistent env vars etc.,
# but they require bash.
#
# The non`-ci` variants can be run locally without having bash installed.

set dotenv-load

default: list

list:
    just --list

ci: docs msrv miri

nostd:
    rustup target add thumbv8m.main-none-eabihf

    # Run no_std + alloc checks (alloc is required for facet-core)
    cargo check --no-default-features --features alloc -p facet-core --target-dir target/nostd --target thumbv8m.main-none-eabihf
    cargo check --no-default-features --features alloc -p facet --target-dir target/nostd --target thumbv8m.main-none-eabihf
    cargo check --no-default-features --features alloc -p facet-reflect --target-dir target/nostd --target thumbv8m.main-none-eabihf

nostd-ci:
    #!/usr/bin/env -S bash -euo pipefail
    source .envrc

    # Set up target directory for no_std + alloc checks (alloc is required)
    export CARGO_TARGET_DIR=target/nostd

    # Run each check in its own group with the full command as the title
    cmd_group "cargo check --no-default-features --features alloc -p facet-core --target thumbv8m.main-none-eabihf"
    cmd_group "cargo check --no-default-features --features alloc -p facet --target thumbv8m.main-none-eabihf"
    cmd_group "cargo check --no-default-features --features alloc -p facet-reflect --target thumbv8m.main-none-eabihf"

clippy-ci:
    cargo clippy --workspace --all-features --all-targets --keep-going -- -D warnings --allow deprecated

clippy-all:
    cargo clippy --all-targets --all-features -- -D warnings

clippy:
    cargo clippy --all-targets -- -D warnings

clippy-workspace-all:
    cargo clippy --workspace --all-targets --all-features -- -D warnings

clippy-workspace:
    cargo clippy --workspace --all-targets -- -D warnings

test *args:
    cargo nextest run {{ args }} < /dev/null

test-i686:
    rustup target add i686-unknown-linux-gnu
    cargo nextest run --target i686-unknown-linux-gnu --tests --lib < /dev/null

valgrind *args:
    cargo nextest run --profile valgrind {{ args }}

asan *args:
    CARGO_TARGET_DIR=target/asan RUSTFLAGS="-Z sanitizer=address" \
        cargo +nightly nextest run \
        --target "$(rustc -vV | sed -n 's|host: ||p')" \
        -p facet-reflect {{ args }}

# macOS-only: run facet-reflect's tests under `leaks --atExit` to catch
# memory leaks at process exit. Native speed, so it's orders of magnitude
# faster than miri for leak regressions. Exits non-zero if any allocation
# is unreachable at exit.
leaks *args:
    MallocStackLogging=1 CARGO_TARGET_DIR=target/leaks \
        cargo nextest run --profile leaks -p facet-reflect {{ args }}


test-ci *args:
    #!/usr/bin/env -S bash -euo pipefail
    source .envrc
    echo -e "\033[1;33m🏃 Running all but doc-tests with nextest...\033[0m"
    cmd_group "cargo nextest run --features ci {{ args }} < /dev/null"

    echo -e "\033[1;36m📚 Running documentation tests...\033[0m"
    cmd_group "cargo test --features ci --doc {{ args }}"

doc-tests *args:
    cargo test --doc {{ args }}

doc-tests-ci *args:
    #!/usr/bin/env -S bash -euo pipefail
    source .envrc
    echo -e "\033[1;36m📚 Running documentation tests...\033[0m"
    cmd_group "cargo test --doc {{ args }}"

miri *args:
    #!/usr/bin/env -S bash -euo pipefail
    source miri-env.sh
    filter='test(/(arc_vtable|rc_vtable|box_vtable|cow_vtable|test_yoke_|slice_builder|btreeset_vtable|hashset_vtable|list_from_raw_parts|ptr::tagged|partial::array_building::drop_array_partially_initialized|partial::deferred::wip_deferred_drop|partial::fuzz::wip_fuzz_|partial::list_deferred::.*realloc|partial::list_leak::wip_list_leaktest(1|12)$|partial::map::map_partial_initialization_drop|partial::map_deferred_leak|partial::map_leak::wip_map_leaktest(1|8)$|partial::misc::(drop_nested_partially_initialized|drop_partially_initialized_struct|from_raw_drop|from_raw_nested_struct|from_raw_with_vec|gh_354_leak|set_default_drops|set_should_drop)|partial::no_uninit::(array|enum|list|map|smart_pointer|struct)_uninit|partial::option_leak::(fuzz_|wip_option_use_after_free)|partial::pending_smart_pointer_tests::(cow_pending_smart_pointer_drop_uses_actual_staging_shape|external_reinitialization_drains_direct_pending_smart_pointer_staging)|partial::pointer::(cow_|drop_|arc_slice|arc_str|box_str|rc_str)|partial::pointer_complex::arc_slice_(empty|complex)|partial::put_vec_leak|partial::set::set_partial_initialization_drop|partial::struct_leak::wip_struct_testleak(1|14)$|rope_pointer_stability|variance_uaf_regression|opaque_lifetime_laundering)/)'
    cargo miri nextest run --target-dir target/miri -p facet-reflect -p facet-core --features facet-core/bytes,facet-core/yoke -E "$filter" {{ args }}

absolve:
    ./facet-dev/absolve.sh

ship:
    #!/usr/bin/env -S bash -euo pipefail
    # Refuse to run if not on main branch or not up to date with origin/main
    branch="$(git rev-parse --abbrev-ref HEAD)"
    if [[ "$branch" != "main" ]]; then
    echo -e "\033[1;31m❌ Refusing to run: not on 'main' branch (current: $branch)\033[0m"
    exit 1
    fi
    git fetch origin main
    local_rev="$(git rev-parse HEAD)"
    remote_rev="$(git rev-parse origin/main)"
    if [[ "$local_rev" != "$remote_rev" ]]; then
    echo -e "\033[1;31m❌ Refusing to run: local main branch is not up to date with origin/main\033[0m"
    echo -e "Local HEAD:  $local_rev"
    echo -e "Origin HEAD: $remote_rev"
    echo -e "Please pull/rebase to update."
    exit 1
    fi
    release-plz update
    git add .
    git commit -m "Upgrades" || true
    git push
    just publish

publish:
    release-plz release --backend github --git-token $(gh auth token)

docsrs *args:
    #!/usr/bin/env -S bash -eux
    source .envrc
    export RUSTDOCFLAGS="--cfg docsrs"
    cargo +nightly doc {{ args }}

msrv:
    # Check default features compile on MSRV
    cargo hack check --rust-version --workspace --locked --ignore-private --keep-going
    # Check all features compile on MSRV
    cargo hack check --rust-version --workspace --locked --ignore-private --keep-going --all-features

msrv-power:
    cargo hack check --feature-powerset --locked --rust-version --ignore-private --workspace --all-targets --keep-going --exclude-no-default-features -

sync-readme-footer:
    bash scripts/readme-footer.sh sync

check-readme-footer:
    bash scripts/readme-footer.sh check

docs: check-readme-footer
    cargo doc --workspace --all-features --no-deps --document-private-items --keep-going

lockfile:
    cargo update --workspace --locked

docker-build-push-linux-amd64:
    #!/usr/bin/env -S bash -eu
    source .envrc
    echo -e "\033[1;34m🐳 Building and pushing Docker images for CI...\033[0m"

    # Set variables
    IMAGE_NAME="ghcr.io/facet-rs/facet-ci"
    TAG="$(date +%Y%m%d)-$(git rev-parse --short HEAD)"

    # Build tests image using stable Rust
    echo -e "\033[1;36m🔨 Building tests image with stable Rust...\033[0m"
    docker build \
        --push \
        --platform linux/amd64 \
        --build-arg BASE_IMAGE=rust:1.92-slim-trixie \
        --build-arg RUSTUP_TOOLCHAIN=1.92 \
        -t "${IMAGE_NAME}:${TAG}-amd64" \
        -t "${IMAGE_NAME}:latest-amd64" \
        -f Dockerfile \
        .

    # Build miri image using nightly Rust
    echo -e "\033[1;36m🔨 Building miri image with nightly Rust...\033[0m"
    docker build \
    --push \
        --platform linux/amd64 \
        --build-arg BASE_IMAGE=rustlang/rust:nightly-slim \
        --build-arg RUSTUP_TOOLCHAIN=nightly \
        --build-arg ADDITIONAL_RUST_COMPONENTS="miri" \
        -t "${IMAGE_NAME}:${TAG}-miri-amd64" \
        -t "${IMAGE_NAME}:latest-miri-amd64" \
        -f Dockerfile \
        .

docker-build-push-linux-arm64:
    #!/usr/bin/env -S bash -eu
    source .envrc
    echo -e "\033[1;34m🐳 Building and pushing Docker images for CI (arm64)...\033[0m"

    # Set variables
    IMAGE_NAME="ghcr.io/facet-rs/facet-ci"
    TAG="$(date +%Y%m%d)-$(git rev-parse --short HEAD)"

    # Build tests image using stable Rust
    echo -e "\033[1;36m🔨 Building tests image with stable Rust (arm64)...\033[0m"
    docker build \
        --push \
        --platform linux/arm64 \
        --build-arg BASE_IMAGE=rust:1.92-slim-trixie \
        --build-arg RUSTUP_TOOLCHAIN=1.92 \
        -t "${IMAGE_NAME}:${TAG}-arm64" \
        -t "${IMAGE_NAME}:latest-arm64" \
        -f Dockerfile \
        .

    # Build miri image using nightly Rust
    echo -e "\033[1;36m🔨 Building miri image with nightly Rust (arm64)...\033[0m"
    docker build \
        --push \
        --platform linux/arm64 \
        --build-arg BASE_IMAGE=rustlang/rust:nightly-slim \
        --build-arg RUSTUP_TOOLCHAIN=nightly \
        --build-arg ADDITIONAL_RUST_COMPONENTS="miri" \
        -t "${IMAGE_NAME}:${TAG}-miri-arm64" \
        -t "${IMAGE_NAME}:latest-miri-arm64" \
        -f Dockerfile \
        .
