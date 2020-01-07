RESULT := # result/BaselineModel-kwdlc-4e-large-coref-cz
CONFIG := # config/BaselineModel/kwdlc/4e/large-coref-cz.json
GPUS := # 0,1
TRAIN_NUM := 1  # number of train iteration with different random seed

CORPUS := kwdlc
TARGET := test
CSV_BASENAME := result_$(TARGET)_$(CORPUS).csv

SHELL = /bin/bash -eu
PYTHON := $(shell which python)

ifdef CONFIG
	RESULT := result/$(subst /,-,$(patsubst config/%.json,%,$(CONFIG)))
endif

CHECKPOINTS := $(wildcard $(RESULT)/*/model_best.pth)
NUM_TRAINED := $(words $(CHECKPOINTS))
RESULT_FILES := $(patsubst $(RESULT)/%/model_best.pth,$(RESULT)/%/$(CSV_BASENAME),$(CHECKPOINTS))
SCORE_FILE := $(RESULT)/scores_$(TARGET)_$(CORPUS).csv

.PHONY: all train test help
all: train
	$(MAKE) test TARGET=valid
	$(MAKE) test TARGET=test

N := $(shell expr $(TRAIN_NUM) - $(NUM_TRAINED))
train:
	env n=$(N) gpu=$(GPUS) scripts/train.sh $(CONFIG)

test: $(SCORE_FILE)
	$(PYTHON) scripts/confidence_interval.py $^

$(SCORE_FILE): $(RESULT_FILES)
	cat <(ls $(RESULT)/*/$(CSV_BASENAME) | head -1 | xargs head -1) <(ls $(RESULT)/*/$(CSV_BASENAME) | xargs grep -h all_case) | tr -d ' ' | sed -r 's/[^,]+,//1' > $@ || rm -f $@

$(RESULT_FILES): $(RESULT)/%/$(CSV_BASENAME): $(RESULT)/%/model_best.pth
	$(PYTHON) src/test.py -r $< --target $(TARGET) -d $(GPUS)

help:
	@echo example:
	@echo make train CONFIG=config/BaselineModel/kwdlc/4e/large-coref-cz.json GPUS=0,1 TRAIN_NUM=5
	@echo make test RESULT=result/BaselineModel-kwdlc-4e-large-coref-cz GPU=0
	@echo make all CONFIG=config/BaselineModel/kwdlc/4e/large-coref-cz.json GPUS=0,1 TRAIN_NUM=5
