__conda_setup="$("${XDG_PREFIX_HOME}/miniconda3/bin/conda" 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "${XDG_PREFIX_HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
        . "${XDG_PREFIX_HOME}/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="${XDG_PREFIX_HOME}/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

prepend_to_env_var() {
    local env_var_name="$1"
    shift
    local args=("${@}")

    if [[ -z "${(P)env_var_name}" ]]; then
        export ${env_var_name}=""
    fi

    for (( i = $#args; i > 0; i-- )); do
        dir=${args[i]}
        if [[ -d "$dir" && ! :${(P)env_var_name}: =~ :$dir: ]]; then
            if [[ -z "${(P)env_var_name}" ]]; then
                eval "export ${env_var_name}=\"$dir\""
            else
                eval "export ${env_var_name}=\"$dir:\${${env_var_name}}\""
            fi
        fi
    done
}
export conda_env_name=vi-depth
if [[ -n "$(conda env list | grep "${conda_env_name}")" ]]; then 
    conda activate ${conda_env_name};
    # prepend_to_env_var LD_LIBRARY_PATH ${CONDA_PREFIX}/lib
    # Jkpython_version=$(python --version | cut -d' ' -f 2 | cut -d'.' -f1).$(python --version | cut -d' ' -f 2 | cut -d'.' -f2)
    # prepend_to_env_var LD_LIBRARY_PATH ${CONDA_PREFIX}/lib/python${python_version}/site-packages/torch/lib
    export CUDA_HOME=${CONDA_PREFIX}
fi


