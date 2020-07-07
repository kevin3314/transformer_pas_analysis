RESULT := # result/BaselineModel-kwdlc-4e-large-coref-cz
CONFIG := # config/BaselineModel-kwdlc-4e-large-coref-cz.json
GPUS := -1
# number of train iteration with different random seeds
TRAIN_NUM := 1

# test or valid
EVAL_SET := test
# which case to calculate confidence interval (ga, wo, ni, ga2, no, or all_case)
CASE := all_case
TARGET :=

ifndef TARGET
  ifneq ($(shell ls $(RESULT)/eval_$(EVAL_SET)/*pred.csv 2> /dev/null),)
    TARGET=kwdlc_pred
  else
    ifneq ($(shell ls $(RESULT)/eval_$(EVAL_SET)/*noun.csv 2> /dev/null),)
      TARGET=kwdlc_noun
    else
      TARGET=kwdlc
    endif
  endif
endif

CSV_NAME := $(TARGET).csv
SHELL = /bin/bash -eu
PYTHON := $(shell which python)
AGGR_DIR_NAME := aggregates

ifdef CONFIG
  RESULT := result/$(patsubst config/%.json,%,$(CONFIG))
endif

TRAIN_DONES := $(patsubst %,$(RESULT)/.train.done.%,$(shell seq $(TRAIN_NUM)))
CHECKPOINTS := $(wildcard $(RESULT)/*/model_best.pth)
RESULT_FILES := $(patsubst $(RESULT)/%/model_best.pth,$(RESULT)/%/eval_$(EVAL_SET)/$(CSV_NAME),$(CHECKPOINTS))
ifeq ($(CASE),all_case)
  AGGR_SCORE_FILE := $(RESULT)/$(AGGR_DIR_NAME)/eval_$(EVAL_SET)/$(CSV_NAME)
else
  AGGR_SCORE_FILE := $(RESULT)/$(AGGR_DIR_NAME)/eval_$(EVAL_SET)/$(TARGET)_$(CASE).csv
endif
AGGR_TFEVENTS_DONE := $(RESULT)/.aggr_tfevents.done
ENS_RESULT_FILE := $(RESULT)/eval_$(EVAL_SET)/$(CSV_NAME)


# train and test
.PHONY: all
all: train
	$(MAKE) test EVAL_SET=test

# train (and validation)
.PHONY: train
train: $(TRAIN_DONES) $(AGGR_TFEVENTS_DONE)

$(TRAIN_DONES):
	$(PYTHON) src/train.py -c $(CONFIG) -d $(GPUS) --seed $${RANDOM} && touch $@
	$(MAKE) test EVAL_SET=valid

$(AGGR_TFEVENTS_DONE): $(TRAIN_DONES)
	$(PYTHON) scripts/aggregator.py -r $(RESULT) --aggr-name $(AGGR_DIR_NAME) && touch $@

# test
.PHONY: test
test: $(AGGR_SCORE_FILE)
	$(PYTHON) scripts/confidence_interval.py $<

$(AGGR_SCORE_FILE): $(RESULT_FILES)
	mkdir -p $(dir $@)
	cat <(head -1 $<) <(echo $^ | xargs grep -h $(CASE),) | tr -d ' ' | sed -r 's/^[^,]+,//' > $@ || rm -f $@

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
