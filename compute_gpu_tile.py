import math
from collections import defaultdict
import pandas as pd
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import os
from tqdm import tqdm 
import numpy as np
import threading
lock = threading.Lock()
# 这是一个在gpu上计算matmul较优tile的脚本Demo，目前实现较为简单，还有部分细节需要继续优化
# 脚本输入参数为matmul的[M，K，N，lhs_dtype，rhs_dtype，out_dtype]这些参数，
# 输出参数为最佳使用sm数量、sm固定数量下M N在SM上的切分、share memory的切分、stage数目、tensoe core上的M N K的切分等
# 一个block映射为一个SM，sharememory使用stage层流水， tensor core和寄存器使用两级流水



gpu_arch_parma = {
    "num_sm": 132,                   # SM 数量
    "tensor_cores_per_sm": 4,        # 每 SM Tensor Core 数量
    "mma_ops_per_cycle": 4,           # 每 Tensor Core 每周期 MMA 操作数 (8x8x8)
    "tensor_core_mma": (16, 16, 8),    # Tensor Core 的 MMA 操作维度 (M, N, K)
    "clock_rate": 1.7e9,             # 时钟频率 (Hz)
    "global_mem_bandwidth": 3350, # 全局内存带宽 (Bytes/s)
    "shared_mem_per_sm": 256 * 1024,  # 每 SM 共享内存容量 (Bytes)
    "shared_mem_bandwidth": 20000,   # 共享内存带宽 (Bytes/s)
    "kernel_launch_latency": 500      # 内核启动延迟 (ns)
}

class TileConfig:
    def __init__(self):
        self.sm_num = 0
        self.sm_M_num = 0, 
        self.sm_N_num = 0,
        self.M_sm = 0
        self.N_sm = 0
        self.K = 0
        self.stage = 0
        self.share_m_num = 0
        self.mma_m_num = 0
        self.m_mma = 0
        self.lhs_bpe = 0
        self.share_n_num = 0
        self.mma_n_num = 0
        self.n_mma = 0
        self.rhs_bpe = 0
        self.k_mma = 0
        self.k_tile = 0
        self.sm_num = 0
        self.stage = 0
        self.load_time = 0
        self.compute_time = 0
        self.total_time = 0

def save_results_to_csv(results_queue):
        results = []
        while not results_queue.empty():
            result = results_queue.get()
            results.append(result.__dict__)

        df = pd.DataFrame(results)
        df['block_M'] = df['share_m_num'] * df['mma_m_num'] * df['m_mma']
        df['block_N'] = df['share_n_num'] * df['mma_n_num'] * df['n_mma']
        df['block_K'] = df['k_tile']
        df_sorted = df.sort_values(by='total_time')
        time_columns = {'load_time', 'compute_time', 'total_time'}
        for col in df.columns:
            if col not in time_columns:
                df[col] = df[col].astype(int)
        return df_sorted                       

def get_pair(num):
    factor_pairs = []
    for i in range(1, int(num**0.5) + 1):
        if num % i == 0:
            factor_pairs.append([i, num // i])
            factor_pairs.append([num // i, i])
    for i, j in factor_pairs:
        if i == j:
            factor_pairs.remove([i, j])
    return factor_pairs

def get_share_max_mn(shared_memory_size, stage,mma_m_num,  mma_n_num, m_mma, k_mma, n_mma, lhs_bpe, rhs_bpe, out_bpe):
    # share_memory_size =stage*( share_m_num * mma_m_num * m_mma * max_k * lhs_bpe + share_n_num * mma_n_num*n_mma * max_k * rhs_bpe) +  mma_m_num * m_mma* mma_n_num*n_mma*out_bpe
    share_max_m_num =(shared_memory_size/stage - mma_n_num * k_mma * n_mma * lhs_bpe - mma_m_num * m_mma* mma_n_num*n_mma*out_bpe)//(mma_m_num * k_mma * lhs_bpe)
    share_max_n_num =(shared_memory_size/stage - mma_m_num * k_mma * m_mma * rhs_bpe - mma_m_num * m_mma* mma_n_num*n_mma*out_bpe)//(mma_n_num * k_mma * rhs_bpe)
    return share_max_m_num, share_max_n_num

def get_share_max_k(shared_memory_size, stage, share_m_num, share_n_num, mma_m_num,  mma_n_num, m_mma, n_mma, lhs_bpe, rhs_bpe, out_bpe):
    share_max_k = math.floor(shared_memory_size - mma_m_num * m_mma* mma_n_num*n_mma*out_bpe) /stage//(share_m_num * mma_m_num  *m_mma  * lhs_bpe + share_n_num *mma_n_num *n_mma  * rhs_bpe)
    return share_max_k

def verify_share_mamory(share_memory_size, m_mma, k_mma, n_mma, n_num, K, m_num, lhs_dtype, rhs_dtype, out_dtype):
    if share_memory_size - (m_num * (m_mma * K * get_bpe(lhs_dtype)) + n_num * (K * n_mma * get_bpe(rhs_dtype))) < 0:
        return False

def get_bpe(dtype):
    if "32" in dtype:
        return 4
    elif "16" in dtype:
        return 2
    elif "8" in dtype:
        return 1
    else:
        return 1
#
def global2share_time(size, sm_num):
    # 粗略的带宽估计，可以替换为更加准确的模型
    effective_bandwidth = gpu_arch_parma["global_mem_bandwidth"] / sm_num
    return size / effective_bandwidth

def share2register_time(size):
    # 粗略的带宽估计，可以替换为更加准确的模型
    return size / gpu_arch_parma["shared_mem_bandwidth"]

def get_atmo_mma_time():
    cycles_per_mma = 1 
    return cycles_per_mma / gpu_arch_parma["clock_rate"]*1e9

def verify_share_memory(shared_memory_size, stage, m_mma, mma_m_num, k_tile, 
                                    n_mma, mma_n_num, share_m_num, share_n_num, 
                                    lhs_bpe, rhs_bpe, out_bpe):
    lhs_size = stage * share_m_num * mma_m_num * m_mma * k_tile * lhs_bpe
    rhs_size = stage * share_n_num * mma_n_num * n_mma * k_tile * rhs_bpe
    out_size = mma_m_num * m_mma * mma_n_num * n_mma * out_bpe
    return (lhs_size + rhs_size + out_bpe) <= shared_memory_size

def compute_tile_time(M_sm, K, N_sm, sm_num, sm_M_num, sm_N_num, stage, share_m_num, mma_m_num, m_mma, lhs_bpe, share_n_num, mma_n_num, n_mma, rhs_bpe, k_mma,  k_tile, results_queue):
    tile_config = TileConfig()
    lhs_load_bytes = (share_m_num * mma_m_num * m_mma * 
                                                k_tile * lhs_bpe)
    rhs_load_bytes = (k_tile * share_n_num * mma_n_num * 
                    n_mma * rhs_bpe)
    out_store_bytes = (mma_m_num * m_mma * mma_n_num *
                    n_mma * rhs_bpe)
    load_share_lhs_time = global2share_time(lhs_load_bytes, sm_num)
    load_share_rhs_time = global2share_time(rhs_load_bytes, sm_num)
    store_out_time = global2share_time(out_store_bytes, sm_num)
    per_stage_total_load_time = max(load_share_lhs_time , load_share_rhs_time)
    load_reg_lhs_time = share2register_time(mma_m_num*m_mma*k_mma)
    load_reg_rhs_time = share2register_time(mma_n_num * n_mma*k_mma)
    store_res_out_time = share2register_time(mma_m_num * m_mma * mma_n_num * n_mma)
    load_per_mma_time = max(load_reg_lhs_time , load_reg_rhs_time)
    mma_exec_time = get_atmo_mma_time()
    k_mma_loops = math.ceil(k_tile // k_mma)

    
    core_time = max(
        load_per_mma_time,
        mma_exec_time,
        store_res_out_time
    ) * k_mma_loops
                            
    m_mma_loops = math.ceil(share_m_num / (mma_m_num * m_mma))
    n_mma_loops = math.ceil(share_n_num / (mma_n_num * n_mma))
    sm_total_loop = m_mma_loops * n_mma_loops
    per_stage_total_compute_time = sm_total_loop * core_time
    
    
    K_iterations = math.ceil(K / k_tile)
    M_iterations = math.ceil(M_sm / (share_m_num * mma_m_num * m_mma))
    N_iterations = math.ceil(N_sm / (share_n_num * mma_n_num * n_mma))

    sm_compute_time = per_stage_total_compute_time * K_iterations * M_iterations * N_iterations
    sm_load_store_time = per_stage_total_load_time * K_iterations * M_iterations * N_iterations
    if (per_stage_total_load_time < store_out_time) :
        sm_load_store_time = per_stage_total_load_time * (K_iterations - 1)* M_iterations * N_iterations   +  M_iterations * N_iterations * store_out_time
    last_compute_store_time = mma_exec_time + store_res_out_time + store_out_time
    if sm_compute_time > sm_load_store_time:
        total_time = gpu_arch_parma['kernel_launch_latency'] + per_stage_total_load_time + sm_compute_time + last_compute_store_time
    else:
        total_time = gpu_arch_parma['kernel_launch_latency'] + sm_load_store_time + last_compute_store_time

    tile_config.sm_num = int(sm_num)
    tile_config.sm_M_num = int(sm_M_num)
    tile_config.sm_N_num = int(sm_N_num)
    tile_config.M_sm = int(M_sm)
    tile_config.N_sm = int(N_sm)
    tile_config.K = int(K)
    tile_config.stage = int(stage)
    tile_config.share_m_num = int(share_m_num)
    tile_config.mma_m_num = int(mma_m_num)
    tile_config.m_mma = int(m_mma)
    tile_config.lhs_bpe = int(lhs_bpe)
    tile_config.share_n_num = int(share_n_num)
    tile_config.mma_n_num = int(mma_n_num)
    tile_config.n_mma = int(n_mma)
    tile_config.rhs_bpe = int(rhs_bpe)
    tile_config.k_mma = int(k_mma)
    tile_config.k_tile = int(k_tile)
    tile_config.sm_num = int(sm_num)
    tile_config.load_time = sm_load_store_time
    tile_config.compute_time = sm_compute_time
    tile_config.total_time = total_time

    with lock:
        results_queue.put(tile_config)
    return 


    

def suggest_tile_configuration(M, K, N, lhs_dtype, rhs_dtype, out_dtype):
    tile_param = defaultdict(list)
    lhs_bpe = get_bpe(lhs_dtype)
    rhs_bpe = get_bpe(rhs_dtype)
    out_bpe = get_bpe(out_dtype)
    for sm_num in range(gpu_arch_parma["num_sm"]//4, gpu_arch_parma["num_sm"] + 1):
        sm_pair = get_pair(sm_num)
        for sm_M_num, sm_N_num in sm_pair:
            M_sm = M//sm_M_num
            N_sm = N//sm_N_num
            M_sm_remain = M % sm_M_num
            N_sm_remain = N % sm_N_num
            
            while M_sm_remain != 0:
                if M_sm_remain  < sm_M_num:
                    M_sm += 1
                    break
                M_sm += M_sm_remain//sm_M_num
                M_sm_remain = M_sm_remain % sm_M_num

            while N_sm_remain!= 0:
                if N_sm_remain  < sm_N_num:
                    N_sm += 1
                    break
                N_sm += N_sm_remain//sm_N_num
                N_sm_remain = N_sm_remain % sm_N_num
            for mma_per_warp in range(1, gpu_arch_parma["tensor_cores_per_sm"]):
                m_mma = gpu_arch_parma["tensor_core_mma"][0]
                k_mma = gpu_arch_parma["tensor_core_mma"][1]
                n_mma = gpu_arch_parma["tensor_core_mma"][2]
                mma_m_num = mma_per_warp
                shared_memory_size = gpu_arch_parma["shared_mem_per_sm"]
                mma_n_num = gpu_arch_parma["tensor_cores_per_sm"]  - mma_per_warp
                for stage in range(2, 5):
                    share_max_m_num, share_max_n_num = get_share_max_mn(shared_memory_size, stage,mma_m_num,  mma_n_num, m_mma, k_mma, n_mma, lhs_bpe, rhs_bpe, out_bpe)
                    for share_m_num in range(1, min(share_max_m_num, M_sm//(mma_m_num*m_mma)) + 1):
                        for share_n_num in range(1, min(share_max_n_num, N_sm//(mma_n_num*n_mma)) + 1):
                            share_max_k_val = get_share_max_k(
                                shared_memory_size, stage, share_m_num, share_n_num, mma_m_num,  mma_n_num, m_mma, n_mma, lhs_bpe, rhs_bpe, out_bpe
                            )
                            
                            for k_tile in range(k_mma, min(K, int(share_max_k_val)) + 1, k_mma):
                                if not verify_share_memory(
                                    shared_memory_size, stage, m_mma, mma_m_num, k_tile, 
                                    n_mma, mma_n_num, share_m_num, share_n_num, 
                                    lhs_bpe, rhs_bpe, out_bpe
                                ):
                                    continue
                                tile_param['sm_M_num'].append(sm_M_num)
                                tile_param['sm_N_num'].append(sm_N_num)
                                tile_param['M_sm'].append(M_sm)
                                tile_param['N_sm'].append(N_sm)
                                tile_param['K'].append(K)
                                tile_param['stage'].append(stage)
                                tile_param['share_m_num'].append(share_m_num)
                                tile_param['mma_m_num'].append(mma_m_num)
                                tile_param['m_mma'].append(m_mma)
                                tile_param['lhs_bpe'].append(lhs_bpe)
                                tile_param['share_n_num'].append(share_n_num)
                                tile_param['mma_n_num'].append(mma_n_num)
                                tile_param['n_mma'].append(n_mma)
                                tile_param['rhs_bpe'].append(rhs_bpe)
                                tile_param['k_mma'].append(k_mma)
                                tile_param['k_tile'].append(k_tile)
                                tile_param['sm_num'].append(sm_num)

    tile_space = pd.DataFrame(tile_param)
    
    max_workers = os.cpu_count()
    # max_workers = 1
    with tqdm(total=len(tile_space), desc="Processing tasks", unit="task") as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results_queue = queue.Queue()
            futures = [
                executor.submit(
                    compute_tile_time,
                    M_sm=row['M_sm'],
                    K=row['K'],
                    N_sm=row['N_sm'],
                    sm_num=row['sm_num'],
                    sm_M_num=row['sm_M_num'], 
                    sm_N_num=row['sm_N_num'],
                    stage = row['stage'],
                    share_m_num=row['share_m_num'],
                    mma_m_num=row['mma_m_num'],
                    m_mma=row['m_mma'],
                    lhs_bpe=lhs_bpe,
                    share_n_num=row['share_n_num'],
                    mma_n_num=row['mma_n_num'],
                    n_mma=row['n_mma'],
                    rhs_bpe=rhs_bpe,
                    k_mma=row['k_mma'],
                    k_tile=row['k_tile'],
                    results_queue = results_queue
                ) for index, row in tile_space.iterrows()
            ]
            for future in as_completed(futures):
                pbar.update(1)
    result_df = save_results_to_csv(results_queue)
    return result_df
   

def print_first_row_formatted(df):
    if isinstance(df, pd.Series):
        df = df.to_frame().T
    
    if df.empty:
        print("DataFrame is empty!")
        return
    
    first_row = df.iloc[0]
    
    for col_name, value in first_row.items():
        if 'time' in col_name.lower():
            print(f"{col_name}: {value}")
            continue
        
        if pd.isna(value):
            formatted_value = 0  
        elif isinstance(value, (int, np.integer)):
            formatted_value = value  
        elif isinstance(value, (float, np.floating)):
            formatted_value = int(value)  
        else:
            formatted_value = value  
        print(f"{col_name}: {formatted_value}")
results_queue = queue.Queue()
# compute_tile_time(M_sm = 342, K = 4096, N_sm=372, sm_num=132, stage=2, share_m_num=11, mma_m_num=1, m_mma=16, lhs_bpe=2, share_n_num=8, mma_n_num=3, n_mma=8, rhs_bpe=2, k_mma=16,  k_tile=64, results_queue=results_queue)
result_df = suggest_tile_configuration(1024, 4096, 4096, "float16", "float16", "float16")
print("\n=== 建议的配置参数 ===")
print_first_row_formatted(result_df)

# 打印结果如下：
# sm_num: 64
# sm_M_num: 4
# sm_N_num: 16
# M_sm: 256
# N_sm: 256
# K: 4096
# stage: 4
# share_m_num: 8
# mma_m_num: 2
# m_mma: 16
# lhs_bpe: 2
# share_n_num: 16
# mma_n_num: 2
# n_mma: 8
# rhs_bpe: 2
# k_mma: 16
# k_tile: 16
# load_time: 40064.99343283582
# compute_time: 150.58823529411762
# total_time: 40585.17025320457
# block_M: 256
# block_N: 256
# block_K: 16
