# Dialect bytecode parser bootstrap

Simple tool to help in bootstrapping the dialect bytecode parsing definitions.
It is not meant as a full "spec" but rather avoids writing boilerplate.

It is meant to be "redirected" to read/write into other forms and so most
specialization should happen TableGen side. This is not there yet, currently
there are hardcoded behavior C++ side that will be removed, but was able to switch this between different formats in ~a day.