# =============================================================================
# 1. 甲醇+DAC
# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from gurobipy import Model, GRB

# =============================================================================
# 1. 数据加载
# =============================================================================
wind_solar_data = pd.read_csv('wind&solar_NM.csv')
pv_output   = wind_solar_data.iloc[:, 0].values  # 假设第一列为 'pv'
wind_output = wind_solar_data.iloc[:, 1].values  # 假设第二列为 'wind'

# =============================================================================
# 2. 参数定义
# =============================================================================
# 2.1 可再生能源与电池参数
wind_cost = 800            # 风电安装成本 ($/kW)
pv_cost   = 300            # 光伏安装成本 ($/kW)
battery_cost = 300         # 储能安装成本 ($/kW)
battery_charge_penalty = 0.001  # ($/kWh)
battery_discharge_penalty = 0.000 # ($/kWh)
battery_om = battery_cost * 0.03
battery_lifetime = 15
battery_efficiency = 0.98   
# 2.2 电解槽与甲醇合成参数
electrolyzer_cost = 300      # ($/kW)
electrolyzer_eff = 0.7   
methanol_synthesis_cost = 5000  # ($/kg·h)
electrolyzer_om = electrolyzer_cost * 0.03
methanol_synthesis_om = methanol_synthesis_cost * 0.03
electrolyzer_lifetime = 15
methanol_synthesis_lifetime = 30
Flex_up = 0.05  # 用于逐时爬坡率约束
Flex_mid = 0.1  
Flex_down = 0.2  
surplus_penalty = 0.001  
# 2.3 氢气存储参数
hydrogen_storage_cost = 600 # ($/kg)
hydrogen_storage_om   = hydrogen_storage_cost * 0.03
hydrogen_storage_efficiency = 1
hydrogen_storage_elec =  1  # kWh/kg H2（充氢所需电耗）
hydrogen_storage_lifetime = 25
# 2.4 热泵与热储参数
heat_pump_cost = 800       # ($/kW_th)
heat_pump_om = heat_pump_cost * 0.03
heat_pump_cop = 3         # COP
heat_pump_lifetime = 20
thermal_storage_cost = 30     # ($/kWh_th)
thermal_storage_om = thermal_storage_cost * 0.03
thermal_storage_efficiency = 0.98
thermal_storage_lifetime = 20
# 2.5 DAC参数
dac_cost = 3000         # ($/(kgCO2/h))
dac_om   = dac_cost * 0.03
dac_lifetime = 30
dac_elec_per_kg = 0.3   # kWh/kg CO2
dac_heat_per_kg = 1.0   # kWh/kg CO2
# 2.6 CO2压缩与储存参数
co2_compressor_cost = 350      # ($/(kgCO2/h))
co2_compressor_om = co2_compressor_cost * 0.03
co2_compressor_lifetime = 20
co2_compression_energy = 0.1  # kWh/kg CO2
co2_storage_cost = 0.05       # ($/kg CO2储存容量)
co2_storage_om = co2_storage_cost * 0.03
co2_storage_lifetime = 30
# 2.7 甲醇储存参数
methanol_storage_cost_param = 0.050  # ($/kg)
methanol_storage_om = methanol_storage_cost_param * 0.03
methanol_storage_lifetime = 20
# 2.8 目标与其它参数
target_methanol_production = 10 * 1000  # kg/年，即10吨/年
methanol_hydrogen_ratio = 0.19          # kg H2/kg CH3OH
methanol_co2_ratio = 1.4                # kg CO2/kg CH3OH
methanol_electricity_ratio = 0.5        # kWh/kg CH3OH
target_hydrogen_for_methanol = target_methanol_production * methanol_hydrogen_ratio
# 折现率与设备寿命
discount_rate = 0.07
wind_lifetime = 20
pv_lifetime = 25
battery_lifetime = 10
electrolyzer_lifetime = 15
methanol_synthesis_lifetime = 30
# 其它设备运维成本
wind_om = wind_cost * 0.03
pv_om = pv_cost * 0.03
battery_om = battery_cost * 0.03
electrolyzer_om = electrolyzer_cost * 0.03
methanol_synthesis_om = methanol_synthesis_cost * 0.03
# 每小时甲醇需求（均匀分布假设）
hourly_methanol_demand = target_methanol_production / 8760
# 2.9 年化成本计算
wind_annual_cost = wind_cost * discount_rate / (1 - (1 + discount_rate) ** -wind_lifetime) + wind_om
pv_annual_cost = pv_cost * discount_rate / (1 - (1 + discount_rate) ** -pv_lifetime) + pv_om
battery_annual_cost = battery_cost * discount_rate / (1 - (1 + discount_rate) ** -battery_lifetime) + battery_om
electrolyzer_annual_cost = electrolyzer_cost * discount_rate / (1 - (1 + discount_rate) ** -electrolyzer_lifetime) + electrolyzer_om
methanol_synthesis_annual_cost = methanol_synthesis_cost * discount_rate / (1 - (1 + discount_rate) ** -methanol_synthesis_lifetime) + methanol_synthesis_om
hydrogen_storage_annual_cost = hydrogen_storage_cost * discount_rate / (1 - (1 + discount_rate) ** -battery_lifetime) + hydrogen_storage_om
dac_annual_cost = dac_cost * discount_rate / (1 - (1 + discount_rate) ** (-dac_lifetime)) + dac_om
co2_compressor_annual_cost = co2_compressor_cost * discount_rate / (1 - (1 + discount_rate) ** (-co2_compressor_lifetime)) + co2_compressor_om
co2_storage_annual_cost = co2_storage_cost * discount_rate / (1 - (1 + discount_rate) ** (-co2_storage_lifetime)) + co2_storage_om
heat_pump_annual_cost = heat_pump_cost * discount_rate / (1 - (1 + discount_rate) ** (-heat_pump_lifetime)) + heat_pump_om
thermal_storage_annual_cost = thermal_storage_cost * discount_rate / (1 - (1 + discount_rate) ** (-thermal_storage_lifetime)) + thermal_storage_om
methanol_storage_annual_cost = methanol_storage_cost_param * discount_rate / (1 - (1 + discount_rate)**(-methanol_storage_lifetime)) + methanol_storage_om

# ----- 新增：定义供需平衡聚合尺度的开关 -----
# 可选值："hourly"（小时级）、"3hourly"、"daily"、"weekly"、"monthly"、"annual"
aggregation_mode = "daily"   # 修改此处选择不同的时间聚合尺度

# =============================================================================
# 3. 模型创建与变量定义
# =============================================================================
model = Model("HydrogenMethanolOptimization")
model.setParam('TimeLimit', 1000)

# 3.1 装机设备决策变量
wind_capacity = model.addVar(lb=0, name="wind_capacity")
pv_capacity   = model.addVar(lb=0, name="pv_capacity")
battery_capacity = model.addVar(lb=0, name="battery_capacity")
electrolyzer_capacity = model.addVar(lb=0, name="electrolyzer_capacity")
methanol_synthesis_capacity = model.addVar(lb=0, name="methanol_capacity")
heat_pump_capacity = model.addVar(lb=0, name="heat_pump_capacity")
thermal_storage_capacity = model.addVar(lb=0, name="thermal_storage_capacity")
methanol_storage_capacity = model.addVar(lb=0, name="methanol_storage_capacity")
dac_capacity = model.addVar(lb=0, name="dac_capacity")
co2_compressor_capacity = model.addVar(lb=0, name="co2_compressor_capacity")
co2_storage_capacity = model.addVar(lb=0, name="co2_storage_capacity")
hydrogen_storage_capacity = model.addVar(lb=0, name="hydrogen_storage_capacity")

# 3.2 时间步相关变量（共8760小时）
time_steps = 8760

# 电池及氢气变量
energy_balance = model.addVars(time_steps, lb=0, name="energy_balance")
battery_charge = model.addVars(time_steps, lb=0, name="battery_charge")
battery_discharge = model.addVars(time_steps, lb=0, name="battery_discharge")
electrolyzer_power = model.addVars(time_steps, lb=0, name="electrolyzer_power")
methanol_synthesis_power = model.addVars(time_steps, lb=0, name="methanol_synthesis_power")
surplus = model.addVars(time_steps, lb=0, name="surplus")
hydrogen_storage_balance = model.addVars(time_steps, lb=0, name="hydrogen_storage_balance")
hydrogen_charge = model.addVars(time_steps, lb=0, name="hydrogen_charge")
hydrogen_discharge = model.addVars(time_steps, lb=0, name="hydrogen_discharge")

# 甲醇储存变量
methanol_storage_balance = model.addVars(time_steps, lb=0, name="methanol_storage_balance")
methanol_storage_charge = model.addVars(time_steps, lb=0, name="methanol_storage_charge")
methanol_storage_discharge = model.addVars(time_steps, lb=0, name="methanol_storage_discharge")

# DAC相关变量
dac_CO2 = model.addVars(time_steps, lb=0, name="dac_CO2")
dac_electricity = model.addVars(time_steps, lb=0, name="dac_electricity")
dac_heat_input = model.addVars(time_steps, lb=0, name="dac_heat_input")

# 热泵与热储系统变量
heat_pump_output = model.addVars(time_steps, lb=0, name="heat_pump_output")
thermal_charge = model.addVars(time_steps, lb=0, name="thermal_charge")
thermal_discharge = model.addVars(time_steps, lb=0, name="thermal_discharge")
thermal_storage_level = model.addVars(time_steps, lb=0, name="thermal_storage_level")

# CO2压缩与储存变量
co2_comp_elec = model.addVars(time_steps, lb=0, name="co2_comp_elec")
co2_storage_level = model.addVars(time_steps, lb=0, name="co2_storage_level")
co2_charge = model.addVars(time_steps, lb=0, name="co2_charge")
co2_discharge = model.addVars(time_steps, lb=0, name="co2_discharge")

# CO2分流变量：直接供给与进入储存
direct_co2 = model.addVars(time_steps, lb=0, name="direct_co2")
stored_co2 = model.addVars(time_steps, lb=0, name="stored_co2")

# =============================================================================
# 4. 目标函数
# =============================================================================
model.setObjective(
    (wind_annual_cost * wind_capacity +
     pv_annual_cost * pv_capacity +
     battery_annual_cost * battery_capacity +
     electrolyzer_annual_cost * electrolyzer_capacity +
     methanol_synthesis_annual_cost * methanol_synthesis_capacity +
     heat_pump_annual_cost * heat_pump_capacity +
     thermal_storage_annual_cost * thermal_storage_capacity +
     hydrogen_storage_annual_cost * hydrogen_storage_capacity +
     methanol_storage_annual_cost * methanol_storage_capacity +
     dac_annual_cost * dac_capacity +
     co2_compressor_annual_cost * co2_compressor_capacity +
     co2_storage_annual_cost * co2_storage_capacity +
     surplus_penalty * sum(surplus[t] for t in range(time_steps)) +
     battery_charge_penalty * sum(battery_charge[t] for t in range(time_steps)) +
     battery_discharge_penalty * sum(battery_discharge[t] for t in range(time_steps))
    ),
    GRB.MINIMIZE
)

# =============================================================================
# 5. 约束条件
# =============================================================================
# 5.1 逐时设备灵活性约束（以装机容量为基准）
for t in range(1, time_steps):
    model.addConstr(electrolyzer_power[t] - electrolyzer_power[t-1] <= Flex_down * electrolyzer_capacity,
                    name=f"electrolyzer_power_increase_limit_{t}")
    model.addConstr(electrolyzer_power[t] - electrolyzer_power[t-1] >= -Flex_down * electrolyzer_capacity,
                    name=f"electrolyzer_power_decrease_limit_{t}")
    model.addConstr(methanol_synthesis_power[t] - methanol_synthesis_power[t-1] <= Flex_up * methanol_synthesis_capacity,
                    name=f"methanol_power_increase_limit_{t}")
    model.addConstr(methanol_synthesis_power[t] - methanol_synthesis_power[t-1] >= -Flex_down * methanol_synthesis_capacity,
                    name=f"methanol_power_decrease_limit_{t}")
    model.addConstr(dac_CO2[t] - dac_CO2[t-1] <= Flex_up * dac_capacity,
                    name=f"dac_increase_limit_{t}")
    model.addConstr(dac_CO2[t] - dac_CO2[t-1] >= -Flex_down * dac_capacity,
                    name=f"dac_decrease_limit_{t}")

# 5.2 甲醇供需平衡约束（包含甲醇储存充放），采用不同时间聚合尺度
start_index = 168  # 起始时间，避开初始不稳定阶段
if aggregation_mode == "hourly":
    for t in range(start_index, time_steps):
        model.addConstr(
            methanol_synthesis_power[t] + methanol_storage_discharge[t] ==
            hourly_methanol_demand + methanol_storage_charge[t],
            name=f"methanol_balance_{t}"
        )
elif aggregation_mode == "3hourly":
    interval = 3
    n_intervals = (time_steps - start_index) // interval
    for i in range(n_intervals):
        t_start = start_index + i * interval
        t_end = t_start + interval
        model.addConstr(
            gp.quicksum(methanol_synthesis_power[t] for t in range(t_start, t_end)) +
            gp.quicksum(methanol_storage_discharge[t] for t in range(t_start, t_end))
            ==
            hourly_methanol_demand * interval +
            gp.quicksum(methanol_storage_charge[t] for t in range(t_start, t_end)),
            name=f"methanol_balance_3hourly_{i}"
        )
elif aggregation_mode == "daily":
    interval = 24
    n_intervals = (time_steps - start_index) // interval
    for i in range(n_intervals):
        t_start = start_index + i * interval
        t_end = t_start + interval
        model.addConstr(
            gp.quicksum(methanol_synthesis_power[t] for t in range(t_start, t_end)) +
            gp.quicksum(methanol_storage_discharge[t] for t in range(t_start, t_end))
            >=
            0.8 * hourly_methanol_demand * interval +
            gp.quicksum(methanol_storage_charge[t] for t in range(t_start, t_end)),
            name=f"methanol_balance_daily_{i}"
        )
elif aggregation_mode == "weekly":
    interval = 168  # 1周 = 168小时
    n_intervals = (time_steps - start_index) // interval
    for i in range(n_intervals):
        t_start = start_index + i * interval
        t_end = t_start + interval
        model.addConstr(
            gp.quicksum(methanol_synthesis_power[t] for t in range(t_start, t_end)) +
            gp.quicksum(methanol_storage_discharge[t] for t in range(t_start, t_end))
            ==
            hourly_methanol_demand * interval +
            gp.quicksum(methanol_storage_charge[t] for t in range(t_start, t_end)),
            name=f"methanol_balance_weekly_{i}"
        )
elif aggregation_mode == "monthly":
    # 近似按每月730小时（实际可根据具体月份调整）
    interval = 730
    n_intervals = (time_steps - start_index) // interval
    for i in range(n_intervals):
        t_start = start_index + i * interval
        t_end = t_start + interval
        model.addConstr(
            gp.quicksum(methanol_synthesis_power[t] for t in range(t_start, t_end)) +
            gp.quicksum(methanol_storage_discharge[t] for t in range(t_start, t_end))
            ==
            hourly_methanol_demand * interval +
            gp.quicksum(methanol_storage_charge[t] for t in range(t_start, t_end)),
            name=f"methanol_balance_monthly_{i}"
        )
elif aggregation_mode == "annual":
    model.addConstr(
        gp.quicksum(methanol_synthesis_power[t] for t in range(start_index, time_steps)) +
        gp.quicksum(methanol_storage_discharge[t] for t in range(start_index, time_steps))
        ==
        hourly_methanol_demand * (time_steps - start_index) +
        gp.quicksum(methanol_storage_charge[t] for t in range(start_index, time_steps)),
        name="methanol_balance_annual"
    )
else:
    raise ValueError("Unsupported aggregation_mode value!")

# 附加：年生产总量约束
model.addConstr(
    gp.quicksum(methanol_synthesis_power[t] for t in range(time_steps)) >= hourly_methanol_demand * 8760,
    name="annual_methanol_production_constraint"
)

# 5.3 甲醇储存动态约束及容量限制
model.addConstr(methanol_storage_balance[0] ==
                methanol_storage_charge[0] - methanol_storage_discharge[0],
                name="methanol_storage_balance_0")
for t in range(1, time_steps):
    model.addConstr(
        methanol_storage_balance[t] == methanol_storage_balance[t-1] + methanol_storage_charge[t] - methanol_storage_discharge[t],
        name=f"methanol_storage_balance_{t}"
    )
    model.addConstr(methanol_storage_balance[t] <= methanol_storage_capacity,
                    name=f"methanol_storage_level_limit_{t}")
    
# 5.4 其它设备及能量平衡约束
for t in range(time_steps):

    wind_power = wind_output[t] * wind_capacity
    pv_power = pv_output[t] * pv_capacity
    
    # 甲醇储存约束
    model.addConstr(methanol_synthesis_power[t] + methanol_storage_discharge[t] >=
             methanol_storage_charge[t],
            name=f"methanol_min_balance_{t}")
                    
    # 电池储能约束
    model.addConstr(energy_balance[t] <= battery_capacity,
                    name=f"energy_balance_limit_{t}")
    model.addConstr(battery_charge[t] + battery_discharge[t] <= battery_capacity,
                    name=f"battery_charge_discharge_limit_{t}")
    # 氢气存储上限
    model.addConstr(hydrogen_storage_balance[t] <= hydrogen_storage_capacity,
                    name=f"hydrogen_storage_capacity_limit_{t}")
    model.addConstr(hydrogen_charge[t] + hydrogen_discharge[t] <= hydrogen_storage_capacity,
            name=f"hydrogen_charge_limit_{t}")
    
    # 电力平衡约束（含热泵及CO2压缩电耗）
    model.addConstr(
        wind_power + pv_power + battery_discharge[t] ==
        electrolyzer_power[t] + methanol_synthesis_power[t] * methanol_electricity_ratio +
        battery_charge[t] + hydrogen_charge[t] * hydrogen_storage_elec +
        (heat_pump_output[t] / heat_pump_cop) + dac_electricity[t] + surplus[t] + co2_comp_elec[t],
        name=f"power_balance_{t}"
    )
    # 电池动态平衡
    if t == 0:
        model.addConstr(
            energy_balance[t] == battery_charge[t] * battery_efficiency - battery_discharge[t] / battery_efficiency,
            name=f"energy_balance_{t}"
        )
    else:
        model.addConstr(
            energy_balance[t] == energy_balance[t-1] + battery_charge[t] * battery_efficiency - battery_discharge[t] / battery_efficiency,
            name=f"energy_balance_{t}"
        )
    # 氢气存储动态平衡
    if t == 0:
        model.addConstr(
            hydrogen_storage_balance[t] == hydrogen_charge[t] * hydrogen_storage_efficiency - hydrogen_discharge[t],
            name=f"hydrogen_balance_{t}"
        )
    else:
        model.addConstr(
            hydrogen_storage_balance[t] == hydrogen_storage_balance[t-1] + hydrogen_charge[t] * hydrogen_storage_efficiency - hydrogen_discharge[t],
            name=f"hydrogen_balance_{t}"
        )
    # 甲醇合成出力上下限
    model.addConstr(methanol_synthesis_power[t] <= methanol_synthesis_capacity,
                    name=f"methanol_synthesis_max_output_{t}")
    model.addConstr(methanol_synthesis_power[t] >= 0.2 * methanol_synthesis_capacity,
                    name=f"methanol_synthesis_min_output_{t}")
    # 氢气供需平衡约束
    model.addConstr(
        electrolyzer_power[t] * electrolyzer_eff * 3.6 / 120 + hydrogen_discharge[t] >=
        methanol_synthesis_power[t] * methanol_hydrogen_ratio + hydrogen_charge[t],
        name=f"hydrogen_PD_balance_{t}"
    )
    # DAC部分：热、电耗与捕集CO2量关系
    model.addConstr(dac_heat_input[t] == dac_CO2[t] * dac_heat_per_kg,
                    name=f"dac_heat_consumption_{t}")
    model.addConstr(dac_electricity[t] == dac_CO2[t] * dac_elec_per_kg,
                    name=f"dac_electricity_consumption_{t}")
    model.addConstr(dac_CO2[t] <= dac_capacity,
                    name=f"dac_capacity_limit_{t}")
    # CO2分流：直接供给与进入储存
    model.addConstr(direct_co2[t] + stored_co2[t] == dac_CO2[t],
                    name=f"co2_split_{t}")
    model.addConstr(stored_co2[t] <= co2_compressor_capacity,
                    name=f"co2_compressor_capacity_limit_{t}")
    model.addConstr(co2_comp_elec[t] == stored_co2[t] * co2_compression_energy,
                    name=f"co2_compression_energy_{t}")
    model.addConstr(co2_charge[t] == stored_co2[t],
                    name=f"co2_charge_relation_{t}")
    # 热泵与热储供热平衡
    model.addConstr(
        heat_pump_output[t] - thermal_charge[t] + thermal_discharge[t] >= dac_heat_input[t],
        name=f"heat_supply_balance_{t}"
    )
    # CO2储存动态及容量约束
    if t == 0:
        model.addConstr(
            co2_storage_level[t] == co2_charge[t] - co2_discharge[t],
            name="co2_storage_balance_0"
        )
    else:
        model.addConstr(
            co2_storage_level[t] == co2_storage_level[t-1] + co2_charge[t] - co2_discharge[t],
            name=f"co2_storage_balance_{t}"
        )
    model.addConstr(co2_storage_level[t] <= co2_storage_capacity,
                    name=f"co2_storage_level_limit_{t}")
    
# 5.5 储热系统动态及容量约束
for t in range(time_steps):
    if t == 0:
        model.addConstr(
            thermal_storage_level[t] == thermal_charge[t] * thermal_storage_efficiency - thermal_discharge[t] / thermal_storage_efficiency,
            name="thermal_storage_balance_0"
        )
    else:
        model.addConstr(
            thermal_storage_level[t] == thermal_storage_level[t-1] + thermal_charge[t] * thermal_storage_efficiency - thermal_discharge[t] / thermal_storage_efficiency,
            name=f"thermal_storage_balance_{t}"
        )
    model.addConstr(thermal_storage_level[t] <= thermal_storage_capacity,
                    name=f"thermal_storage_level_limit_{t}")
    
    # CO2储存动态及容量约束（已在上层循环中处理）
    # 此处不再重复

# 5.6 热泵与电解槽装机容量约束
model.addConstrs((heat_pump_capacity >= heat_pump_output[t] for t in range(time_steps)),
                 name="heat_pump_capacity_constraint")
model.addConstrs((electrolyzer_capacity >= electrolyzer_power[t] for t in range(time_steps)),
                 name="electrolyzer_capacity_max_constraint")

# 5.7 每小时CO2供需平衡约束：直接供给加储存放出需满足当期甲醇CO2需求
for t in range(time_steps):
    model.addConstr(
        direct_co2[t] + co2_discharge[t] >= methanol_synthesis_power[t] * methanol_co2_ratio,
        name=f"co2_hourly_balance_{t}"
    )

# 5.8 年度CO2供给约束
model.addConstr(sum(direct_co2[t] + co2_discharge[t] for t in range(time_steps)) >= target_methanol_production * methanol_co2_ratio,
                name="annual_co2_supply_constraint")

# =============================================================================
# 6. 求解模型
# =============================================================================
model.optimize()

# =============================================================================
# 7. 输出结果
# =============================================================================
if model.status == GRB.OPTIMAL:
    total_methanol_production = sum(methanol_synthesis_power[t].x for t in range(time_steps))
    hydrogen_production_kwh = sum(electrolyzer_power[t].x for t in range(time_steps)) * electrolyzer_eff 
    electrolyzer_utilization_hours = hydrogen_production_kwh / (electrolyzer_capacity.x * electrolyzer_eff * 8760) * 100
    methanol_utilization_hours = target_methanol_production / (methanol_synthesis_capacity.x * 8760) * 100
    total_CO2_captured = sum(dac_CO2[t].x for t in range(time_steps))
    
    total_Elec_generation = (sum(wind_output[t] for t in range(time_steps)) * wind_capacity.x +
                             sum(pv_output[t] for t in range(time_steps)) * pv_capacity.x -
                             sum(surplus[t].x for t in range(time_steps)))
    
    total_Heat_generation = sum(heat_pump_output[t].x for t in range(time_steps))
    
    dac_utilization_hours = total_CO2_captured / (dac_capacity.x * 8760) * 100
    levelized_dac_cost = (dac_annual_cost * dac_capacity.x +
                          co2_compressor_annual_cost * co2_compressor_capacity.x +
                          co2_storage_annual_cost * co2_storage_capacity.x) / (total_CO2_captured / 1000)
         
    levelized_Elec_cost = (wind_annual_cost * wind_capacity.x +
                          pv_annual_cost * pv_capacity.x +
                          battery_annual_cost * battery_capacity.x) / (total_Elec_generation)

    levelized_Heat_cost = (heat_pump_annual_cost * heat_pump_capacity.x +
                           thermal_storage_annual_cost * thermal_storage_capacity.x +
                           levelized_Elec_cost * total_Heat_generation / heat_pump_cop) / (total_Heat_generation)
    
    levelized_H2_cost = (electrolyzer_annual_cost * electrolyzer_capacity.x +
                           levelized_Elec_cost * hydrogen_production_kwh / electrolyzer_eff +
                           hydrogen_storage_annual_cost * hydrogen_storage_capacity.x) / (hydrogen_production_kwh / 33.3)
    
    print("Optimal system configuration:")
    print(f"Wind capacity: {wind_capacity.x:.2f} kW")
    print(f"PV capacity: {pv_capacity.x:.2f} kW")
    print(f"Battery capacity: {battery_capacity.x:.2f} kW")
    print(f"Electrolyzer capacity: {electrolyzer_capacity.x:.2f} kW")
    print(f"Heat pump capacity: {heat_pump_capacity.x:.2f} kW_th")
    print(f"Thermal storage capacity: {thermal_storage_capacity.x:.2f} kWh_th")
    print(f"Methanol synthesis capacity: {methanol_synthesis_capacity.x * 8760 / 1000:.2f} t/y")
    print(f"Methanol storage capacity: {methanol_storage_capacity.x:.2f} kgMeOH/h")
    print(f"Hydrogen storage capacity: {hydrogen_storage_capacity.x:.2f} kgH2/h")
    print(f"DAC capacity: {dac_capacity.x:.2f} kgCO2/h")
    print(f"CO₂ compressor capacity: {co2_compressor_capacity.x:.2f} kgCO2/h")
    print(f"CO₂ storage capacity: {co2_storage_capacity.x:.2f} kg")
    print(f"Total CO₂ captured: {total_CO2_captured:.2f} kg")
    print(f"Electrolyzer utilization hours: {electrolyzer_utilization_hours:.2f} %")
    print(f"Methanol utilization hours: {methanol_utilization_hours:.2f} %")
    print(f"DAC utilization hours: {dac_utilization_hours:.2f} %")
    print(f"Levelized DAC cost: ${levelized_dac_cost:.2f} per ton CO2")
    print(f"Levelized ELec cost: ${levelized_Elec_cost:.2f} per kWh")
    print(f"Levelized Heat cost: ${levelized_Heat_cost:.2f} per kWh_th")
    print(f"Levelized H2 cost: ${levelized_H2_cost:.2f} per kgH2")
    print(f"Levelized methanol production cost: ${model.objVal / total_methanol_production * 1000:.2f} per ton")
else:
    print("No feasible solution found.")

# =============================================================================
# 8. 折线图
# =============================================================================
if model.status == GRB.OPTIMAL:
    # 计算各时段输出与状态曲线
    wind_power_profile = wind_output * wind_capacity.x
    pv_power_profile = pv_output * pv_capacity.x
    # 电解槽实际出力（考虑效率），单位为 kW
    electrolyzer_demand = [electrolyzer_power[t].x * electrolyzer_eff for t in range(time_steps)]
    # 甲醇生产曲线，单位为 kg/h（满足小时供需平衡约束）
    methanol_synthesis_profile = [methanol_synthesis_power[t].x for t in range(time_steps)]
    # 甲醇储存状态曲线，单位为 kg（这里不做缩放，直接展示）
    methanol_storage_profile = [methanol_storage_balance[t].x/100 for t in range(time_steps)]
    
    # 电池充放电及电池储能状态
    battery_charge_profile = [battery_charge[t].x for t in range(time_steps)]
    battery_discharge_profile = [battery_discharge[t].x for t in range(time_steps)]
    energy_balance_profile = [energy_balance[t].x for t in range(time_steps)]
    
    # 提取各时刻电力平衡约束的对偶价格（电价）
    electricity_prices = [model.getConstrByName(f"power_balance_{t}").pi for t in range(time_steps)]
    # 对于甲醇供需平衡约束（在小时模式下，从 t=168 开始定义），提取其对偶价格作为甲醇价格
    methanol_prices = [0] * time_steps
    for t in range(168, time_steps):
        constr = model.getConstrByName(f"methanol_balance_{t}")
        if constr is not None:
            methanol_prices[t] = constr.pi
        else:
            methanol_prices[t] = 0

    # 绘制样本周（168小时）时序图，示例取第11周（从第 168*10 小时开始）
    week_start = 168 * 10
    week_end = week_start + 168
    
    battery_charge_week = battery_charge_profile[week_start:week_end]
    battery_discharge_week = battery_discharge_profile[week_start:week_end]
    energy_balance_week = energy_balance_profile[week_start:week_end]
    wind_power_week = wind_power_profile[week_start:week_end]
    pv_power_week = pv_power_profile[week_start:week_end]
    electrolyzer_demand_week = electrolyzer_demand[week_start:week_end]
    methanol_synthesis_week = methanol_synthesis_profile[week_start:week_end]
    methanol_storage_week = methanol_storage_profile[week_start:week_end]
    electricity_prices_week = electricity_prices[week_start:week_end]
    methanol_prices_week = methanol_prices[week_start:week_end]
    
    plt.figure(figsize=(15, 8))
    plt.plot(wind_power_week, label='Wind Output (kW)', linestyle='-', color='blue')
    plt.plot(pv_power_week, label='PV Output (kW)', linestyle='-', color='orange')
    plt.plot(battery_charge_week, label='Battery Charge (kW)', linestyle='--', color='green')
    plt.plot(battery_discharge_week, label='Battery Discharge (kW)', linestyle='--', color='red')
    plt.plot(electrolyzer_demand_week, label='Electrolyzer Demand (kW)', linestyle='-', color='purple')
    plt.plot(methanol_synthesis_week, label='Methanol Synthesis Production (kg/h)', linestyle='-', color='magenta')
    plt.plot(methanol_storage_week, label='Methanol Storage Balance (100kg)', linestyle='-', color='cyan')
    plt.plot(electricity_prices_week, label='Electricity Price ($/kWh)', linestyle='-', color='brown')
    plt.plot(methanol_prices_week, label='Methanol Price ($/kg)', linestyle='-', color='black')
    
    plt.xlabel('Hour')
    plt.ylabel('Value')
    plt.title('Power, Production and Storage Profiles (Sample Week)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 按月绘图（以每月小时数为分段，近似划分）
    hours_in_month = [0, 744, 1416, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016, 8760]
    month_names = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']

    for i in range(12):
        start = hours_in_month[i]
        end = hours_in_month[i + 1]
        
        battery_charge_month = battery_charge_profile[start:end]
        battery_discharge_month = battery_discharge_profile[start:end]
        energy_balance_month = energy_balance_profile[start:end]
        wind_power_month = wind_power_profile[start:end]
        pv_power_month = pv_power_profile[start:end]
        electrolyzer_demand_month = electrolyzer_demand[start:end]
        methanol_synthesis_month = methanol_synthesis_profile[start:end]
        methanol_storage_month = methanol_storage_profile[start:end]
        electricity_prices_month = electricity_prices[start:end]
        methanol_prices_month = methanol_prices[start:end]
    
        plt.figure(figsize=(15, 8))
        plt.plot(wind_power_month, label='Wind Output (kW)', linestyle='-', color='blue')
        plt.plot(pv_power_month, label='PV Output (kW)', linestyle='-', color='orange')
        plt.plot(battery_charge_month, label='Battery Charge (kW)', linestyle='--', color='green')
        plt.plot(battery_discharge_month, label='Battery Discharge (kW)', linestyle='--', color='red')
        plt.plot(electrolyzer_demand_month, label='Electrolyzer Demand (kW)', linestyle='-', color='purple')
        plt.plot(methanol_synthesis_month, label='Methanol Synthesis Production (kg/h)', linestyle='-', color='magenta')
        plt.plot(methanol_storage_month, label='Methanol Storage Balance (kg)', linestyle='-', color='cyan')
        plt.plot(electricity_prices_month, label='Electricity Price ($/kWh)', linestyle='-', color='brown')
        plt.plot(methanol_prices_month, label='Methanol Price ($/kg)', linestyle='-', color='black')
    
        plt.xlabel('Hour')
        plt.ylabel('Value')
        plt.title(f'Power, Production and Storage Profiles ({month_names[i]})')
        plt.legend()
        plt.grid(True)
        plt.show()
else:
    print("No feasible solution found.")

# =============================================================================
# 9. 堆积图
# =============================================================================
if model.status == GRB.OPTIMAL:
    # ---------------------------
    # 样本周（168小时）堆积柱状图
    # ---------------------------
    week_start = 168 * 10   # 例如取第11周（从第168*10小时开始）
    week_end = week_start + 168
    x = np.arange(week_end - week_start)
    
    # 供给侧（正值）
    supply_wind = np.array([wind_output[t] * wind_capacity.x for t in range(week_start, week_end)])
    supply_pv = np.array([pv_output[t] * pv_capacity.x for t in range(week_start, week_end)])
    supply_battery = np.array([battery_discharge[t].x for t in range(week_start, week_end)])
    
    # 需求侧（负值）
    demand_electrolyzer = np.array([electrolyzer_power[t].x for t in range(week_start, week_end)])
    demand_methanol = np.array([methanol_synthesis_power[t].x * methanol_electricity_ratio for t in range(week_start, week_end)])
    demand_battery = np.array([battery_charge[t].x for t in range(week_start, week_end)])
    demand_hydrogen = np.array([hydrogen_charge[t].x * hydrogen_storage_elec for t in range(week_start, week_end)])
    demand_heatpump = np.array([heat_pump_output[t].x / heat_pump_cop for t in range(week_start, week_end)])
    demand_dac = np.array([dac_electricity[t].x for t in range(week_start, week_end)])
    demand_surplus = np.array([surplus[t].x for t in range(week_start, week_end)])
    demand_co2 = np.array([co2_comp_elec[t].x for t in range(week_start, week_end)])
    
    plt.figure(figsize=(15, 8))
    # 绘制供给堆积柱状图（正值）
    bar1 = plt.bar(x, supply_wind, label='Wind Supply', color='blue')
    bar2 = plt.bar(x, supply_pv, bottom=supply_wind, label='PV Supply', color='orange')
    bar3 = plt.bar(x, supply_battery, bottom=supply_wind + supply_pv, label='Battery Discharge Supply', color='cyan')
    
    # 绘制需求堆积柱状图（负值）
    # 依次累计：先绘制电解槽，再叠加甲醇合成、再叠加电池充电、氢气充电、热泵、DAC、过剩和CO2压缩
    bar4 = plt.bar(x, -demand_electrolyzer, label='Electrolyzer Demand', color='purple')
    bar5 = plt.bar(x, -demand_methanol, bottom=-demand_electrolyzer, label='Methanol Synthesis Demand', color='magenta')
    bar6 = plt.bar(x, -demand_battery, bottom=-(demand_electrolyzer + demand_methanol), label='Battery Charge Demand', color='red')
    bar7 = plt.bar(x, -demand_hydrogen, bottom=-(demand_electrolyzer + demand_methanol + demand_battery), label='Hydrogen Charge Demand', color='green')
    bar8 = plt.bar(x, -demand_heatpump, bottom=-(demand_electrolyzer + demand_methanol + demand_battery + demand_hydrogen), label='Heat Pump Demand', color='brown')
    bar9 = plt.bar(x, -demand_dac, bottom=-(demand_electrolyzer + demand_methanol + demand_battery + demand_hydrogen + demand_heatpump), label='DAC Electricity Demand', color='gray')
    bar10 = plt.bar(x, -demand_surplus, bottom=-(demand_electrolyzer + demand_methanol + demand_battery + demand_hydrogen + demand_heatpump + demand_dac), label='Surplus Demand', color='black')
    bar11 = plt.bar(x, -demand_co2, bottom=-(demand_electrolyzer + demand_methanol + demand_battery + demand_hydrogen + demand_heatpump + demand_dac + demand_surplus), label='CO₂ Compression Demand', color='darkred')
    
    plt.axhline(0, color='black', linewidth=0.8)
    plt.xlabel('Hour')
    plt.ylabel('Power (kW)')
    plt.title('Stacked Power Supply (Positive) & Demand (Negative) - Sample Week (Methanol+DAC)')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()
    
    # ---------------------------
    # 按月堆积柱状图
    # ---------------------------
    # 这里采用近似的月度划分（小时数），实际可根据具体月份调整
    hours_in_month = [0, 744, 1416, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016, 8760]
    month_names = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']
    
    for i in range(12):
        start = hours_in_month[i]
        end = hours_in_month[i + 1]
        x_month = np.arange(end - start)
        
        supply_wind_month = np.array([wind_output[t] * wind_capacity.x for t in range(start, end)])
        supply_pv_month = np.array([pv_output[t] * pv_capacity.x for t in range(start, end)])
        supply_battery_month = np.array([battery_discharge[t].x for t in range(start, end)])
        
        demand_electrolyzer_month = np.array([electrolyzer_power[t].x for t in range(start, end)])
        demand_methanol_month = np.array([methanol_synthesis_power[t].x * methanol_electricity_ratio for t in range(start, end)])
        demand_battery_month = np.array([battery_charge[t].x for t in range(start, end)])
        demand_hydrogen_month = np.array([hydrogen_charge[t].x * hydrogen_storage_elec for t in range(start, end)])
        demand_heatpump_month = np.array([heat_pump_output[t].x / heat_pump_cop for t in range(start, end)])
        demand_dac_month = np.array([dac_electricity[t].x for t in range(start, end)])
        demand_surplus_month = np.array([surplus[t].x for t in range(start, end)])
        demand_co2_month = np.array([co2_comp_elec[t].x for t in range(start, end)])
        
        plt.figure(figsize=(15, 8))
        plt.bar(x_month, supply_wind_month, label='Wind Supply', color='blue')
        plt.bar(x_month, supply_pv_month, bottom=supply_wind_month, label='PV Supply', color='orange')
        plt.bar(x_month, supply_battery_month, bottom=supply_wind_month + supply_pv_month, label='Battery Discharge Supply', color='cyan')
        
        plt.bar(x_month, -demand_electrolyzer_month, label='Electrolyzer Demand', color='purple')
        plt.bar(x_month, -demand_methanol_month, bottom=-demand_electrolyzer_month, label='Methanol Synthesis Demand', color='magenta')
        plt.bar(x_month, -demand_battery_month, bottom=-(demand_electrolyzer_month + demand_methanol_month), label='Battery Charge Demand', color='red')
        plt.bar(x_month, -demand_hydrogen_month, bottom=-(demand_electrolyzer_month + demand_methanol_month + demand_battery_month), label='Hydrogen Charge Demand', color='green')
        plt.bar(x_month, -demand_heatpump_month, bottom=-(demand_electrolyzer_month + demand_methanol_month + demand_battery_month + demand_hydrogen_month), label='Heat Pump Demand', color='brown')
        plt.bar(x_month, -demand_dac_month, bottom=-(demand_electrolyzer_month + demand_methanol_month + demand_battery_month + demand_hydrogen_month + demand_heatpump_month), label='DAC Electricity Demand', color='gray')
        plt.bar(x_month, -demand_surplus_month, bottom=-(demand_electrolyzer_month + demand_methanol_month + demand_battery_month + demand_hydrogen_month + demand_heatpump_month + demand_dac_month), label='Surplus Demand', color='black')
        plt.bar(x_month, -demand_co2_month, bottom=-(demand_electrolyzer_month + demand_methanol_month + demand_battery_month + demand_hydrogen_month + demand_heatpump_month + demand_dac_month + demand_surplus_month), label='CO₂ Compression Demand', color='darkred')
        
        plt.axhline(0, color='black', linewidth=0.8)
        plt.xlabel('Hour')
        plt.ylabel('Power (kW)')
        plt.title(f'Stacked Power Supply & Demand ({month_names[i]}) - Methanol+DAC')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()
else:
    print("No feasible solution found.")
