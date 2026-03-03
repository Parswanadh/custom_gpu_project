@echo off
REM ============================================================================
REM run_tests.bat - Run all Verilog testbenches (no PowerShell required)
REM Usage: cd custom_gpu_project && scripts\run_tests.bat
REM ============================================================================
setlocal enabledelayedexpansion

set IVERILOG=D:\Tools\iverilog\bin\iverilog.exe
set VVP=D:\Tools\iverilog\bin\vvp.exe
set ROOT=%~dp0..

if not exist "%IVERILOG%" (
    echo [ERROR] iverilog not found at %IVERILOG%
    echo Please update IVERILOG path in this script
    exit /b 1
)

if not exist "%ROOT%\sim" mkdir "%ROOT%\sim"
if not exist "%ROOT%\sim\waveforms" mkdir "%ROOT%\sim\waveforms"

echo.
echo ================================================================
echo        CUSTOM GPU - FULL TEST SUITE
echo        Designed for GPT-2 Inference on FPGA
echo ================================================================
echo.

set TOTAL_PASS=0
set TOTAL_FAIL=0
set TOTAL_MODULES=0

REM -- Phase 1: Core Compute Primitives --
echo --- Phase 1: Core Compute Primitives ---
call :run_test "P1" "zero_detect_mult" "zdm_test" "rtl/primitives/zero_detect_mult.v" "tb/primitives/zero_detect_mult_tb.v"
call :run_test "P1" "variable_precision_alu" "vpa_test" "rtl/primitives/variable_precision_alu.v" "tb/primitives/variable_precision_alu_tb.v"
call :run_test "P1" "sparse_memory_ctrl" "smc_test" "rtl/primitives/sparse_memory_ctrl.v" "tb/primitives/sparse_memory_ctrl_tb.v"
call :run_test "P1" "fused_dequantizer" "fd_test" "rtl/primitives/fused_dequantizer.v" "tb/primitives/fused_dequantizer_tb.v"
call :run_test "P1" "gpu_top" "gt_test" "rtl/primitives/zero_detect_mult.v rtl/primitives/variable_precision_alu.v rtl/primitives/sparse_memory_ctrl.v rtl/primitives/fused_dequantizer.v rtl/primitives/gpu_top.v" "tb/primitives/gpu_top_tb.v"
echo.

REM -- Phase 2: Extended Compute Modules --
echo --- Phase 2: Extended Compute Modules ---
call :run_test "P2" "mac_unit" "mac_test" "rtl/compute/mac_unit.v" "tb/compute/mac_unit_tb.v"
call :run_test "P2" "systolic_array" "sa_test" "rtl/compute/systolic_array.v" "tb/compute/systolic_array_tb.v"
call :run_test "P2" "gelu_activation" "gelu_test" "rtl/compute/gelu_lut_256.v rtl/compute/gelu_activation.v" "tb/compute/gelu_activation_tb.v"
call :run_test "P2" "softmax_unit" "sm_test" "rtl/compute/exp_lut_256.v rtl/compute/softmax_unit.v" "tb/compute/softmax_unit_tb.v"
echo.

REM -- Phase 3: Transformer Building Blocks --
echo --- Phase 3: Transformer Building Blocks ---
call :run_test "P3" "layer_norm" "ln_test" "rtl/compute/inv_sqrt_lut_256.v rtl/transformer/layer_norm.v" "tb/transformer/layer_norm_tb.v"
call :run_test "P3" "linear_layer" "ll_test" "rtl/transformer/linear_layer.v" "tb/transformer/linear_layer_tb.v"
call :run_test "P3" "attention_unit" "au_test" "rtl/compute/exp_lut_256.v rtl/transformer/attention_unit.v" "tb/transformer/attention_unit_tb.v"
call :run_test "P3" "ffn_block" "ffn_test" "rtl/compute/gelu_lut_256.v rtl/transformer/ffn_block.v" "tb/transformer/ffn_block_tb.v"
echo.

REM -- Phase 4: GPT-2 Full Pipeline --
echo --- Phase 4: GPT-2 Full Pipeline ---
call :run_test "P4" "embedding_lookup" "emb_test" "rtl/gpt2/embedding_lookup.v" "tb/gpt2/embedding_lookup_tb.v"
call :run_test "P4" "gpt2_engine_FULL" "gpt2_test" "rtl/gpt2/embedding_lookup.v rtl/gpt2/transformer_block.v rtl/gpt2/gpt2_engine.v rtl/transformer/layer_norm.v rtl/transformer/attention_unit.v rtl/transformer/ffn_block.v rtl/transformer/linear_layer.v rtl/compute/gelu_activation.v rtl/compute/gelu_lut_256.v rtl/compute/softmax_unit.v rtl/compute/exp_lut_256.v rtl/compute/inv_sqrt_lut_256.v" "tb/gpt2/gpt2_engine_tb.v"
call :run_test "P4" "accel_gpt2_engine" "gpt2_acc_test" "rtl/primitives/zero_detect_mult.v rtl/primitives/fused_dequantizer.v rtl/primitives/gpu_core.v rtl/compute/exp_lut_256.v rtl/compute/inv_sqrt_lut_256.v rtl/transformer/layer_norm.v rtl/transformer/accelerated_attention.v rtl/transformer/accelerated_linear_layer.v rtl/transformer/accelerated_transformer_block.v rtl/gpt2/embedding_lookup.v rtl/gpt2/accelerated_gpt2_engine.v" "tb/gpt2/accelerated_gpt2_engine_tb.v"
echo.

REM -- Phase 5: Memory Interface --
echo --- Phase 5: Memory Interface ---
call :run_test "P5" "axi_weight_memory" "axi_test" "rtl/memory/axi_weight_memory.v" "tb/memory/axi_weight_memory_tb.v"
call :run_test "P5" "dma_engine" "dma_test" "rtl/memory/dma_engine.v" "tb/memory/dma_engine_tb.v"
call :run_test "P5" "scratchpad" "sp_test" "rtl/memory/scratchpad.v" "tb/memory/scratchpad_tb.v"
echo.

REM -- Phase 6: Top-Level Control --
echo --- Phase 6: Top-Level Control ---
call :run_test "P6" "command_processor" "cmd_test" "rtl/top/command_processor.v" "tb/top/command_processor_tb.v"
call :run_test "P6" "perf_counters" "perf_test" "rtl/top/perf_counters.v" "tb/top/perf_counters_tb.v"
call :run_test "P6" "gpu_config_regs" "cfg_test" "rtl/top/gpu_config_regs.v" "tb/top/gpu_config_regs_tb.v"
call :run_test "P6" "reset_synchronizer" "rst_test" "rtl/top/reset_synchronizer.v" "tb/top/reset_synchronizer_tb.v"
echo.

REM -- Phase 7: System Integration --
echo --- Phase 7: System Integration ---
call :run_test "P7" "gpu_system_top" "sys_test" "rtl/top/reset_synchronizer.v rtl/top/gpu_config_regs.v rtl/top/command_processor.v rtl/top/perf_counters.v rtl/memory/scratchpad.v rtl/memory/dma_engine.v rtl/top/gpu_system_top.v" "tb/top/gpu_system_top_tb.v"
echo.

REM -- Summary --
echo ================================================================
echo                     TEST RESULTS SUMMARY
echo ================================================================
echo   Modules tested : %TOTAL_MODULES%
echo   Total PASS     : %TOTAL_PASS%
echo   Total FAIL     : %TOTAL_FAIL%
echo ================================================================
if %TOTAL_FAIL% EQU 0 (
    if %TOTAL_PASS% GTR 0 (
        echo   ^>^>^> ALL TESTS PASSED - GPU READY FOR DEPLOYMENT ^<^<^<
    ) else (
        echo   ^>^>^> NO TESTS RAN - CHECK IVERILOG INSTALLATION ^<^<^<
    )
) else (
    echo   ^>^>^> SOME TESTS FAILED - REVIEW REQUIRED ^<^<^<
)
echo.
goto :eof

REM ============================================================================
REM :run_test <Phase> <Name> <OutputBin> <Sources> <Testbench>
REM ============================================================================
:run_test
set PHASE=%~1
set NAME=%~2
set OUTBIN=%~3
set SOURCES=%~4
set TB=%~5

set /a TOTAL_MODULES+=1

REM Build full file paths
set ALL_FILES=
for %%f in (%SOURCES%) do set "ALL_FILES=!ALL_FILES! %ROOT%\%%f"
set "ALL_FILES=!ALL_FILES! %ROOT%\%TB%"
set "OUTPATH=%ROOT%\sim\%OUTBIN%"

REM Compile
echo|set /p="  [%PHASE%] %NAME% ... "
"%IVERILOG%" -o "%OUTPATH%" %ALL_FILES% 2>"%ROOT%\sim\%OUTBIN%_compile.log"
if errorlevel 1 (
    echo COMPILE FAIL
    set /a TOTAL_FAIL+=1
    echo     Compile errors in: sim\%OUTBIN%_compile.log
    goto :eof
)

REM Simulate and capture output
"%VVP%" "%OUTPATH%" > "%ROOT%\sim\%OUTBIN%_output.log" 2>&1

REM Count PASS/FAIL
set PASSES=0
set FAILS=0
for /f %%n in ('findstr /c:"[PASS]" "%ROOT%\sim\%OUTBIN%_output.log" ^| find /c "[PASS]"') do set PASSES=%%n
for /f %%n in ('findstr /c:"[FAIL]" "%ROOT%\sim\%OUTBIN%_output.log" ^| find /c "[FAIL]"') do set FAILS=%%n

if %FAILS% EQU 0 (
    if %PASSES% GTR 0 (
        echo %PASSES% PASS
    ) else (
        echo NO OUTPUT
    )
) else (
    echo %PASSES% PASS, %FAILS% FAIL
)

set /a TOTAL_PASS+=%PASSES%
set /a TOTAL_FAIL+=%FAILS%
goto :eof
