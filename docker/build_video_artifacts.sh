#!/usr/bin/env bash
set -euo pipefail

source_root=${1:-/tmp/video-codec-sources}
artifact_root=${2:-/opt/video_codec_artifacts}
jobs=${CODEC_BUILD_JOBS:-4}
conda_env_root=${CONDA_ENV_ROOT:-/root/miniconda3/envs}

mkdir -p "${artifact_root}"

build_cmake_codec() {
    local name=$1
    local python_path=$2
    local codec_root=$3
    local artifact_subdir=$4
    local build_dir="/tmp/build-${name}"

    rm -rf "${build_dir}"
    PATH="$(dirname "${python_path}"):${PATH}" \
        cmake -S "${codec_root}/src/cpp" -B "${build_dir}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DPYTHON_EXECUTABLE="${python_path}" \
        -DPython_EXECUTABLE="${python_path}"
    PATH="$(dirname "${python_path}"):${PATH}" \
        cmake --build "${build_dir}" --parallel "${jobs}"

    mkdir -p "${artifact_root}/${name}/${artifact_subdir}"
    find "${build_dir}" -type f \( -name 'MLCodec_CXX*.so' -o -name 'MLCodec_rans*.so' \) \
        -exec cp -v {} "${artifact_root}/${name}/${artifact_subdir}/" \;
    local artifact_count
    artifact_count=$(find "${artifact_root}/${name}/${artifact_subdir}" -type f -name '*.so' | wc -l)
    if [[ "${artifact_count}" -lt 2 ]]; then
        echo "Expected two C++ artifacts for ${name}, found ${artifact_count}" >&2
        exit 1
    fi
}

build_cmake_codec \
    dcvc \
    "${conda_env_root}/dcvc/bin/python" \
    "${source_root}/dcvc_family/dcvc" \
    src/entropy_models
build_cmake_codec \
    dcvc_tcm \
    "${conda_env_root}/dcvc_tcm/bin/python" \
    "${source_root}/dcvc_family/dcvc_tcm" \
    src/entropy_models
build_cmake_codec \
    dcvc_hem \
    "${conda_env_root}/dcvc_hem/bin/python" \
    "${source_root}/dcvc_family/dcvc_hem" \
    src/entropy_models
build_cmake_codec \
    dcvc_dc \
    "${conda_env_root}/dcvc_dc/bin/python" \
    "${source_root}/dcvc_family/dcvc_dc" \
    src/models
build_cmake_codec \
    dcvc_fm \
    "${conda_env_root}/dcvc_fm/bin/python" \
    "${source_root}/dcvc_family/dcvc_fm" \
    src/models
build_cmake_codec \
    dcvc_b \
    "${conda_env_root}/dcvc_b/bin/python" \
    "${source_root}/dcvc_b" \
    src/entropy_models
build_cmake_codec \
    dcvc_sdd \
    "${conda_env_root}/dcvc_sdd/bin/python" \
    "${source_root}/dcvc_sdd" \
    src/entropy_models
build_cmake_codec \
    brhvc \
    "${conda_env_root}/brhvc/bin/python" \
    "${source_root}/kwai_nvc/BRHVC" \
    src/models

rt_build=/tmp/build-dcvc-rt
rm -rf "${rt_build}"
mkdir -p "${rt_build}" "${artifact_root}/dcvc_rt"
(
    cd "${source_root}/dcvc_family/dcvc_rt/src/cpp"
    "${conda_env_root}/dcvc_rt/bin/python" setup.py build_ext \
        --build-temp "${rt_build}/temp" \
        --build-lib "${artifact_root}/dcvc_rt"
)

if ! find "${artifact_root}/dcvc_rt" -type f -name 'MLCodec_extensions_cpp*.so' -print -quit | grep -q .; then
    echo "DCVC-RT C++ artifact was not produced" >&2
    exit 1
fi

find "${artifact_root}" -type f -name '*.so' -print
