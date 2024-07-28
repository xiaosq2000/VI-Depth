remove_from_env_var() {
    local env_var_name="$1"
    local path_to_remove="$2"
    local current_value="${(P)env_var_name}"

    current_value="${current_value//:$path_to_remove:/:}"
    current_value="${current_value//:$path_to_remove/}"
    current_value="${current_value//$path_to_remove:/}"

    eval "export ${env_var_name}=\"${current_value}\""
}
# python_version=$(python --version | cut -d' ' -f 2 | cut -d'.' -f1).$(python --version | cut -d' ' -f 2 | cut -d'.' -f2)
# remove_from_env_var LD_LIBRARY_PATH ${CONDA_PREFIX}/lib/python${python_version}/site-packages/torch/lib
unset CUDA_HOME
conda deactivate
