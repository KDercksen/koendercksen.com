#!/usr/bin/env bash

set -e

cd output

PASS=$(pass show Work/versio_directadmin)
find . -type f -exec curl -u koendfc118:$PASS --ftp-create-dirs -T {} ftp://ftp.koendercksen.com/domains/koendercksen.com/public_html/{} \;
