#!/usr/bin/env bash
# render.sh NAME NGPU "arg1 arg2 ..."  -> manifest on stdout
NAME=$1; NGPU=$2; shift 2
ARGS=$(for a in $*; do printf "'%s', " "$a"; done | sed 's/, $//')
sed -e "s/NAME/$NAME/g" -e "s/NGPU/$NGPU/" -e "s|ARGS|$ARGS|" "$(dirname "$0")/vllm-dp.tmpl.yaml"
