# Cocotb Makefile for BitbyBit GPU — GPT-2 cosimulation
# Run with: make -C tb/cocotb

TOPLEVEL_LANG = verilog
VERILOG_SOURCES = $(shell pwd)/../../rtl/gpt2/embedding_lookup.v \
                  $(shell pwd)/../../rtl/gpt2/transformer_block.v \
                  $(shell pwd)/../../rtl/gpt2/gpt2_engine.v \
                  $(shell pwd)/../../rtl/transformer/layer_norm.v \
                  $(shell pwd)/../../rtl/transformer/attention_unit.v \
                  $(shell pwd)/../../rtl/transformer/ffn_block.v \
                  $(shell pwd)/../../rtl/transformer/linear_layer.v \
                  $(shell pwd)/../../rtl/compute/gelu_activation.v \
                  $(shell pwd)/../../rtl/compute/softmax_unit.v

TOPLEVEL = gpt2_engine
MODULE = test_gpt2_cosim

SIM = icarus
COCOTB_HDL_TIMEUNIT = 1ns
COCOTB_HDL_TIMEPRECISION = 1ps

# Icarus Verilog location
ICARUS_BIN_DIR = D:/Tools/iverilog/bin
SIM_BUILD = sim_build

include $(shell cocotb-config --makefiles)/Makefile.sim
