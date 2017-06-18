docker run -it --rm -u `id -g` -v $HOME:$HOME -e "HOME=$HOME" --workdir=`pwd` --entrypoint=python jjanzic/docker-python3-opencv $*
