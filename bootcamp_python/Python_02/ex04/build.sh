
# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RESET='\033[0m'

venv_path=".venv"
package_name="my_minipack-0.0.1-py3-none-any.whl"
package_path="./dist"

# Function to display help message
show_help() {
  echo "${CYAN}Usage:${RESET}"
  echo "${GREEN}  source ./my_script.sh <launch|env|django|help>${RESET}"
  echo  "${CYAN}Commands:${RESET}"
#   echo "${YELLOW}  env${RESET}          : Setup Python environment (install virtualenv, dependencies)"
  echo "${YELLOW}  build${RESET}        : Build the package"
  echo "${YELLOW}  install${RESET}      : Install the package"
  echo "${YELLOW}  help${RESET}         : Display this help message"
}

# Function to install the package
install() {
  echo "=============={🧰 ${CYAN}Installing Package${RESET} 🧰}=============="
  pip install --force-reinstall $package_path/$package_name
}

# Function to build the package
build() {
  echo "=============={🧰 ${CYAN}Building Package${RESET} 🧰}=============="
  pip install --upgrade build
  python -m build
}

# Function to setup Python environment
setup_env() {
    if [ -d "$venv_path" ]; then
        echo "${YELLOW}Virtual environment already exists. Activating...${RESET}"
        source $venv_path/bin/activate
        return
    fi
  echo "Setting up Python environment..."
  python -m venv $venv_path
  source $venv_path/bin/activate

  echo "=============={🧰 ${CYAN}Pip Upgrade${RESET} 🧰}=============="
  pip install --upgrade pip

  echo "=============={🧰 ${CYAN}Path Module Installation${RESET} 🧰}=============="
  pip install --upgrade -r requirements.txt
}

# Handle script arguments
case "$1" in
install)
    setup_env
    install
    ;;
env)
  setup_env
  ;;
build)
  setup_env
  build
  ;;
help)
  show_help
  ;;
*)
  echo "${RED}Invalid option! Use 'help' for usage information.${RESET}"
  ;;
esac