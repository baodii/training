SHELL := /bin/zsh

seq := $(shell echo part_{00000..00010}_of_00500)
# files = $(foreach var, $(seq), part_$(var)_of_00500) 
files = $(seq)

tfrecord/%_of_00500 : results/%_of_00500
	@echo 'from $< to $@'
	@touch $@

all: $(addprefix tfrecord/, $(files))

clean:
	rm -rf result/*
