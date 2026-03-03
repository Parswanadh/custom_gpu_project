@echo off
setlocal enabledelayedexpansion
REM Quick test runner - paste into cmd.exe or Anaconda Prompt
REM cd D:\projects\bitbybit\custom_gpu_project && scripts\quick_test.bat

set R=D:\projects\bitbybit\custom_gpu_project
set I=D:\Tools\iverilog\bin\iverilog.exe
set V=D:\Tools\iverilog\bin\vvp.exe
if not exist "%I%" (
    for %%x in (iverilog.exe) do set "I=%%~$PATH:x"
    for %%x in (vvp.exe) do set "V=%%~$PATH:x"
)
if not exist "%I%" (
    echo ERROR: iverilog not found. Set I= path in this script.
    exit /b 1
)
if not exist "%R%\sim" mkdir "%R%\sim"
if not exist "%R%\sim\waveforms" mkdir "%R%\sim\waveforms"

set P=0
set F=0
set M=0

call :T "P1 zero_detect_mult" zdm "rtl\primitives\zero_detect_mult.v" "tb\primitives\zero_detect_mult_tb.v"
call :T "P1 variable_precision_alu" vpa "rtl\primitives\variable_precision_alu.v" "tb\primitives\variable_precision_alu_tb.v"
call :T "P1 sparse_memory_ctrl" smc "rtl\primitives\sparse_memory_ctrl.v" "tb\primitives\sparse_memory_ctrl_tb.v"
call :T "P1 fused_dequantizer" fd "rtl\primitives\fused_dequantizer.v" "tb\primitives\fused_dequantizer_tb.v"
call :T5 "P1 gpu_top" gt "rtl\primitives\zero_detect_mult.v" "rtl\primitives\variable_precision_alu.v" "rtl\primitives\sparse_memory_ctrl.v" "rtl\primitives\fused_dequantizer.v" "rtl\primitives\gpu_top.v" "tb\primitives\gpu_top_tb.v"
call :T "P2 mac_unit" mac "rtl\compute\mac_unit.v" "tb\compute\mac_unit_tb.v"
call :T "P2 systolic_array" sa "rtl\compute\systolic_array.v" "tb\compute\systolic_array_tb.v"
call :T2 "P2 gelu_activation" gelu "rtl\compute\gelu_lut_256.v" "rtl\compute\gelu_activation.v" "tb\compute\gelu_activation_tb.v"
call :T2 "P2 softmax_unit" sm "rtl\compute\exp_lut_256.v" "rtl\compute\softmax_unit.v" "tb\compute\softmax_unit_tb.v"
call :T2 "P3 layer_norm" ln "rtl\compute\inv_sqrt_lut_256.v" "rtl\transformer\layer_norm.v" "tb\transformer\layer_norm_tb.v"
call :T "P3 linear_layer" ll "rtl\transformer\linear_layer.v" "tb\transformer\linear_layer_tb.v"
call :T2 "P3 attention_unit" au "rtl\compute\exp_lut_256.v" "rtl\transformer\attention_unit.v" "tb\transformer\attention_unit_tb.v"
call :T2 "P3 ffn_block" ffn "rtl\compute\gelu_lut_256.v" "rtl\transformer\ffn_block.v" "tb\transformer\ffn_block_tb.v"
call :T "P4 embedding_lookup" emb "rtl\gpt2\embedding_lookup.v" "tb\gpt2\embedding_lookup_tb.v"
call :T "P5 axi_weight_memory" axi "rtl\memory\axi_weight_memory.v" "tb\memory\axi_weight_memory_tb.v"
call :T "P5 dma_engine" dma "rtl\memory\dma_engine.v" "tb\memory\dma_engine_tb.v"
call :T "P5 scratchpad" sp "rtl\memory\scratchpad.v" "tb\memory\scratchpad_tb.v"
call :T "P6 command_processor" cmd "rtl\top\command_processor.v" "tb\top\command_processor_tb.v"
call :T "P6 perf_counters" perf "rtl\top\perf_counters.v" "tb\top\perf_counters_tb.v"
call :T "P6 gpu_config_regs" cfg "rtl\top\gpu_config_regs.v" "tb\top\gpu_config_regs_tb.v"
call :T "P6 reset_synchronizer" rst "rtl\top\reset_synchronizer.v" "tb\top\reset_synchronizer_tb.v"

echo.
echo --- Phase 4 Full Pipeline (may take longer) ---
"%I%" -o "%R%\sim\gpt2" "%R%\rtl\gpt2\embedding_lookup.v" "%R%\rtl\gpt2\transformer_block.v" "%R%\rtl\gpt2\gpt2_engine.v" "%R%\rtl\transformer\layer_norm.v" "%R%\rtl\transformer\attention_unit.v" "%R%\rtl\transformer\ffn_block.v" "%R%\rtl\transformer\linear_layer.v" "%R%\rtl\compute\gelu_activation.v" "%R%\rtl\compute\gelu_lut_256.v" "%R%\rtl\compute\softmax_unit.v" "%R%\rtl\compute\exp_lut_256.v" "%R%\rtl\compute\inv_sqrt_lut_256.v" "%R%\tb\gpt2\gpt2_engine_tb.v" 2>"%R%\sim\gpt2_err.log"
if errorlevel 1 (echo   P4 gpt2_engine: COMPILE FAIL & set /a F+=1) else (echo|set /p="  P4 gpt2_engine: " & "%V%" "%R%\sim\gpt2" > "%R%\sim\gpt2_out.log" 2>&1 & call :count gpt2_out)
set /a M+=1

"%I%" -o "%R%\sim\acc" "%R%\rtl\primitives\zero_detect_mult.v" "%R%\rtl\primitives\fused_dequantizer.v" "%R%\rtl\primitives\gpu_core.v" "%R%\rtl\compute\exp_lut_256.v" "%R%\rtl\compute\inv_sqrt_lut_256.v" "%R%\rtl\transformer\layer_norm.v" "%R%\rtl\transformer\accelerated_attention.v" "%R%\rtl\transformer\accelerated_linear_layer.v" "%R%\rtl\transformer\accelerated_transformer_block.v" "%R%\rtl\gpt2\embedding_lookup.v" "%R%\rtl\gpt2\accelerated_gpt2_engine.v" "%R%\tb\gpt2\accelerated_gpt2_engine_tb.v" 2>"%R%\sim\acc_err.log"
if errorlevel 1 (echo   P4 accel_gpt2: COMPILE FAIL & set /a F+=1) else (echo|set /p="  P4 accel_gpt2: " & "%V%" "%R%\sim\acc" > "%R%\sim\acc_out.log" 2>&1 & call :count acc_out)
set /a M+=1

echo.
echo --- Phase 7 System Integration ---
"%I%" -o "%R%\sim\sys" "%R%\rtl\top\reset_synchronizer.v" "%R%\rtl\top\gpu_config_regs.v" "%R%\rtl\top\command_processor.v" "%R%\rtl\top\perf_counters.v" "%R%\rtl\memory\scratchpad.v" "%R%\rtl\memory\dma_engine.v" "%R%\rtl\top\gpu_system_top.v" "%R%\tb\top\gpu_system_top_tb.v" 2>"%R%\sim\sys_err.log"
if errorlevel 1 (echo   P7 gpu_system_top: COMPILE FAIL & set /a F+=1) else (echo|set /p="  P7 gpu_system_top: " & "%V%" "%R%\sim\sys" > "%R%\sim\sys_out.log" 2>&1 & call :count sys_out)
set /a M+=1

echo.
echo ================================================================
echo   TOTAL: %M% modules, %P% pass, %F% fail
echo ================================================================
if %F% EQU 0 (if %P% GTR 0 echo   ALL TESTS PASSED)
goto :eof

:T
echo|set /p="  %~1: "
set /a M+=1
"%I%" -o "%R%\sim\%~2" "%R%\%~3" "%R%\%~4" 2>"%R%\sim\%~2_err.log"
if errorlevel 1 (echo COMPILE FAIL & set /a F+=1 & goto :eof)
"%V%" "%R%\sim\%~2" > "%R%\sim\%~2_out.log" 2>&1
call :count %~2_out
goto :eof

:T2
echo|set /p="  %~1: "
set /a M+=1
"%I%" -o "%R%\sim\%~2" "%R%\%~3" "%R%\%~4" "%R%\%~5" 2>"%R%\sim\%~2_err.log"
if errorlevel 1 (echo COMPILE FAIL & set /a F+=1 & goto :eof)
"%V%" "%R%\sim\%~2" > "%R%\sim\%~2_out.log" 2>&1
call :count %~2_out
goto :eof

:T5
echo|set /p="  %~1: "
set /a M+=1
"%I%" -o "%R%\sim\%~2" "%R%\%~3" "%R%\%~4" "%R%\%~5" "%R%\%~6" "%R%\%~7" "%R%\%~8" 2>"%R%\sim\%~2_err.log"
if errorlevel 1 (echo COMPILE FAIL & set /a F+=1 & goto :eof)
"%V%" "%R%\sim\%~2" > "%R%\sim\%~2_out.log" 2>&1
call :count %~2_out
goto :eof

:count
set _P=0
set _F=0
for /f %%n in ('findstr /c:"[PASS]" "%R%\sim\%~1.log" 2^>nul ^| find /c /v ""') do set _P=%%n
for /f %%n in ('findstr /c:"[FAIL]" "%R%\sim\%~1.log" 2^>nul ^| find /c /v ""') do set _F=%%n
if !_F! GTR 0 (echo !_P! pass, !_F! FAIL) else if !_P! GTR 0 (echo !_P! pass) else (echo NO OUTPUT)
set /a P+=!_P!
set /a F+=!_F!
goto :eof
