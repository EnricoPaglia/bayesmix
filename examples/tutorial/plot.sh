#!/usr/bin/env bash

build/plot_mcmc \
  --grid-file resources/tutorial/grid.csv \
  --dens-file resources/tutorial/out/density.csv \
  --dens-plot resources/tutorial/out/density.png \
  --n-cl-file resources/tutorial/out/numclust.csv \
  --n-cl-trace-plot resources/tutorial/out/traceplot.png \
  --n-cl-bar-plot  resources/tutorial/out/nclus_barplot.png
