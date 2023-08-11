#!/bin/bash
find "$@" -name state_dict* -printf '%h\n' |xargs -d $'\n' sh -c 'for arg do find "$arg" -name state_dict*.pth |sort |tail -n 1; done'

