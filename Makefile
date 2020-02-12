RESULT := # result/BaselineModel-kwdlc-4e-large-coref-cz
CONFIG := # config/BaselineModel/kwdlc/4e/large-coref-cz.json
GPUS := -1
# number of train iteration with different random seeds
TRAIN_NUM := 1

# kwdlc or kc
CORPUS := kwdlc
# test or valid
TARGET := test
# pred or noun or all
PAS_TARGET := pred
# which case to calculate confidence interval (ga or wo or ni or ga2 or no or all_case)
AGGR_CASE := all_case

CSV_BASENAME := $(CORPUS)_$(PAS_TARGET).csv

SHELL = /bin/bash -eu
PYTHON := $(shell which python)

ifdef CONFIG
	RESULT := result/$(subst /,-,$(patsubst config/%.json,%,$(CONFIG)))
endif

CHECKPOINTS := $(wildcard $(RESULT)/*/model_best.pth)
NUM_TRAINED := $(words $(CHECKPOINTS))
RESULT_FILES := $(patsubst $(RESULT)/%/model_best.pth,$(RESULT)/%/eval_$(TARGET)/$(CSV_BASENAME),$(CHECKPOINTS))
ifeq ($(AGGR_CASE),all_case)
	AGGR_SCORE_FILE := $(RESULT)/eval_aggr_$(TARGET)/$(CSV_BASENAME)
else
	AGGR_SCORE_FILE := $(RESULT)/eval_aggr_$(TARGET)/$(basename $(CSV_BASENAME))_$(AGGR_CASE).csv
endif
ENS_RESULT_FILE := $(RESULT)/eval_$(TARGET)/$(CSV_BASENAME)

# train and test
.PHONY: all
all: train
	$(MAKE) test TARGET=test

# train (and validation)
N := $(shell expr $(TRAIN_NUM) - $(NUM_TRAINED))
.PHONY: train
train:
	env n=$(N) gpu=$(GPUS) scripts/train.sh $(CONFIG)
	$(MAKE) test TARGET=valid

# test
.PHONY: test
test: $(AGGR_SCORE_FILE)
	$(PYTHON) scripts/confidence_interval.py $<

$(AGGR_SCORE_FILE): $(RESULT_FILES)
	mkdir -p $(dir $@)
	cat <(ls $(RESULT)/*/eval_$(TARGET)/$(CSV_BASENAME) | head -1 | xargs head -1) \
	<(ls $(RESULT)/*/eval_$(TARGET)/$(CSV_BASENAME) | xargs grep -h $(AGGR_CASE),) \
	| tr -d ' ' | sed -r 's/^[^,]+,//' > $@ || rm -f $@

$(RESULT_FILES): %/eval_$(TARGET)/$(CSV_BASENAME): %/model_best.pth
	$(PYTHON) src/test.py -r $< --target $(TARGET) -d $(GPUS)

# ensemble test
.PHONY: test-ens
test-ens: $(ENS_RESULT_FILE)

$(ENS_RESULT_FILE): $(CHECKPOINTS)
	$(PYTHON) src/test.py --ens $(RESULT) -c $(dir $<)config.json --target $(TARGET) -d $(GPUS)

.PHONY: help
help:
	@echo example:
	@echo make train CONFIG=config/BaselineModel/kwdlc/4e/large-coref-cz.json GPUS=0,1 TRAIN_NUM=5
	@echo make test RESULT=result/BaselineModel-kwdlc-4e-large-coref-cz GPU=0
	@echo make all CONFIG=config/BaselineModel/kwdlc/4e/large-coref-cz.json GPUS=0,1 TRAIN_NUM=5
	@echo make test-ens RESULT=result/BaselineModel-kwdlc-4e-large-coref-cz GPU=0
