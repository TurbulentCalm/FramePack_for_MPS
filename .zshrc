# Function to check if we're in a Cursor terminal
is_cursor_terminal() {
    if [[ -n "$VSCODE_PID" ]] || [[ -n "$CURSOR_PID" ]]; then
        return 0
    else
        return 1
    fi
}

# Conda initialization with Cursor support
if [[ -z "${CONDA_SHLVL}" ]] || is_cursor_terminal; then
    # >>> conda initialize >>>
    # !! Contents within this block are managed by 'conda init' !!
    __conda_setup="$('/Users/paul/miniforge3/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
    if [ $? -eq 0 ]; then
        eval "$__conda_setup"
    else
        if [ -f "/Users/paul/miniforge3/etc/profile.d/conda.sh" ]; then
            . "/Users/paul/miniforge3/etc/profile.d/conda.sh"
        else
            export PATH="/Users/paul/miniforge3/bin:$PATH"
        fi
    fi
    unset __conda_setup
    # <<< conda initialize <<<

    # Activate the project's environment if specified in .envrc
    if [[ -n "$CONDA_DEFAULT_ENV" ]]; then
        conda activate "$CONDA_DEFAULT_ENV"
    fi
fi 