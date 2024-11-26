from REPF_para_opti import REPF_para_opti
import time
plumed_input_file = 'plumed.dat'
plumed_output_file = 'Colvar'
para_opti=REPF_para_opti()
start_time = time.time()
res_parm=para_opti.rest_para_opti(plumed_input_file,plumed_output_file)
end_time = time.time()
execute_time = end_time - start_time
print(f"执行时间：{execute_time:.6f} 秒")
