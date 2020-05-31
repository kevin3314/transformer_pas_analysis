RESULT := # result/BaselineModel-kwdlc-4e-large-coref-cz
CONFIG := # config/BaselineModel-kwdlc-4e-large-coref-cz.json
GPUS := -1
# number of train iteration with different random seeds
TRAIN_NUM := 1

# test or valid
EVAL_SET := test
# which case to calculate confidence interval (ga, wo, ni, ga2, no, or all_case)
CASE := all_case
TARGET := kwdlc_pred

CSV_NAME := $(TARGET).csv
SHELL = /bin/bash -eu
PYTHON := $(shell which python)
AGGR_DIR_NAME := aggregates

ifdef CONFIG
	RESULT := result/$(patsubst config/%.json,%,$(CONFIG))
endif

CHECKPOINTS := $(wildcard $(RESULT)/*/model_best.pth)
NUM_TRAINED := $(words $(CHECKPOINTS))
RESULT_FILES := $(patsubst $(RESULT)/%/model_best.pth,$(RESULT)/%/eval_$(EVAL_SET)/$(CSV_NAME),$(CHECKPOINTS))
ifeq ($(CASE),all_case)
	AGGR_SCORE_FILE := $(RESULT)/$(AGGR_DIR_NAME)/eval_$(EVAL_SET)/$(CSV_NAME)
else
	AGGR_SCORE_FILE := $(RESULT)/$(AGGR_DIR_NAME)/eval_$(EVAL_SET)/$(TARGET)_$(CASE).csv
endif
ENS_RESULT_FILE := $(RESULT)/eval_$(EVAL_SET)/$(CSV_NAME)

# train and test
.PHONY: all
all: train
	$(MAKE) test EVAL_SET=test

# train (and validation)
N := $(shell expr $(TRAIN_NUM) - $(NUM_TRAINED))
.PHONY: train
train:
	for i in $$(seq $(N)); do $(PYTHON) src/train.py -c $(CONFIG) -d $(GPUS) --seed $${RANDOM}; done
	$(MAKE) test EVAL_SET=valid

# test
.PHONY: test
test: $(AGGR_SCORE_FILE)
	$(PYTHON) scripts/confidence_interval.py $<

$(AGGR_SCORE_FILE): $(RESULT_FILES)
	mkdir -p $(dir $@)
	cat <(head -1 $<) <(ls $(RESULT)/*/eval_$(EVAL_SET)/$(CSV_NAME) | xargs grep -h $(CASE),) \
	| tr -d ' ' | sed -r 's/^[^,]+,//' > $@ || rm -f $@

$(RESULT_FILES): %/eval_$(EVAL_SET)/$(CSV_NAME): %/model_best.pth
	$(PYTHON) src/test.py -r $< --target $(EVAL_SET) -d $(GPUS)

# ensemble test
.PHONY: test-ens
test-ens: $(ENS_RESULT_FILE)

$(ENS_RESULT_FILE): $(CHECKPOINTS)
	$(PYTHON) src/test.py --ens $(RESULT) -c $(dir $<)config.json --target $(EVAL_SET) -d $(GPUS)

.PHONY: help
help:
	@echo example:
	@echo make train CONFIG=config/BaselineModel/kwdlc/4e/large-coref-cz.json GPUS=0,1 TRAIN_NUM=5
	@echo make test RESULT=result/BaselineModel-kwdlc-4e-large-coref-cz GPU=0
	@echo make all CONFIG=config/BaselineModel/kwdlc/4e/large-coref-cz.json GPUS=0,1 TRAIN_NUM=5
	@echo make test-ens RESULT=result/BaselineModel-kwdlc-4e-large-coref-cz GPU=0
