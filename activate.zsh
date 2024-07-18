#!/bin/zsh
export RUSTDOCFLAGS="--html-in-header $(pwd)/docs/html/custom-header.html"
export TOPO_LOG=cg=trace,ls=trace
