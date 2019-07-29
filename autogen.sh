#!/bin/sh -exE

mkdir -p config
aclocal -I config
autoreconf --install
automake --foreign --add-missing --copy
autoconf
