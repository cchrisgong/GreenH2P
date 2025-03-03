# =============================================================================
# 2. 合成氨
# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import gurobipy as gp
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
wind_lifetime = 20
pv_lifetime = 25
battery_efficiency = 0.98    

# 2.2 电解槽与合成氨参数
electrolyzer_cost = 300      # ($/kW)
electrolyzer_eff = 0.7     
# 假设合成氨过程成本参数（示例值）
ammonia_synthesis_cost = 4500  # ($/kg·h)
electrolyzer_om = electrolyzer_cost * 0.03
ammonia_synthesis_om = ammonia_synthesis_cost * 0.03
electrolyzer_lifetime = 15
ammonia_synthesis_lifetime = 30

# 逐时灵活性参数（用于限制逐时爬坡率）
Flex_up = 0.05  
Flex_mid = 0.1  
Flex_down = 0.2  
surplus_penalty = 0.001  

# 2.3 氢气存储参数
hydrogen_storage_cost = 600 # ($/kg)
hydrogen_storage_om   = hydrogen_storage_cost * 0.03
hydrogen_storage_efficiency = 1
hydrogen_storage_elec =  1  # kWh/kg H2（充氢所需电耗）
hydrogen_storage_lifetime = 25

# 2.4 合成氨参数
# 反应：N2 + 3H2 -> 2NH3
# 氢气消耗比 (kg H2 per kg NH3) — 示例值取 0.18
ammonia_hydrogen_ratio = 0.18  
target_ammonia_production = 10 * 1000  # kg/年，即10吨/年
hourly_ammonia_demand = target_ammonia_production / 8760
ammonia_electricity_ratio = 0.303  # Electricity requirement for 1 ton of ammonia (kWh/kg NH3)

# 2.5 氨储存参数（类似甲醇储存）
ammonia_storage_cost_param = 0.59   # ($/kg) 示例值
ammonia_storage_om = ammonia_storage_cost_param * 0.03
ammonia_storage_lifetime = 30

# 折现率
discount_rate = 0.07

# 2.6 年化成本计算
wind_annual_cost = wind_cost * discount_rate / (1 - (1 + discount_rate) ** -wind_lifetime) + wind_cost * 0.03
pv_annual_cost = pv_cost * discount_rate / (1 - (1 + discount_rate) ** -pv_lifetime) + pv_cost * 0.03
battery_annual_cost = battery_cost * discount_rate / (1 - (1 + discount_rate) ** -battery_lifetime) + battery_om
electrolyzer_annual_cost = electrolyzer_cost * discount_rate / (1 - (1 + discount_rate) ** -electrolyzer_lifetime) + electrolyzer_om
ammonia_synthesis_annual_cost = ammonia_synthesis_cost * discount_rate / (1 - (1 + discount_rate) ** -ammonia_synthesis_lifetime) + ammonia_synthesis_om
hydrogen_storage_annual_cost = hydrogen_storage_cost * discount_rate / (1 - (1 + discount_rate) ** -hydrogen_storage_lifetime) + hydrogen_storage_om
ammonia_storage_annual_cost = ammonia_storage_cost_param * discount_rate / (1 - (1 + discount_rate) ** -ammonia_storage_lifetime) + ammonia_storage_om

# ----- 新增：定义供需平衡聚合尺度的开关 -----
# 可选值："hourly"（小时级）、"3hourly"、"daily"、"weekly"、"monthly"、"annual"
aggregation_mode = "daily"  # <-- 修改此处选择不同的时间尺度

# =============================================================================
# 3. 模型创建与变量定义
# =============================================================================
model = Model("HydrogenAmmoniaOptimization")
model.setParam('TimeLimit', 1000)

# 3.1 装机设备决策变量
wind_capacity = model.addVar(lb=0, name="wind_capacity")
pv_capacity   = model.addVar(lb=0, name="pv_capacity")
battery_capacity = model.addVar(lb=0, name="battery_capacity")
electrolyzer_capacity = model.addVar(lb=0, name="electrolyzer_capacity")
ammonia_synthesis_capacity = model.addVar(lb=0, name="ammonia_synthesis_capacity")
hydrogen_storage_capacity = model.addVar(lb=0, name="hydrogen_storage_capacity")
ammonia_storage_capacity = model.addVar(lb=0, name="ammonia_storage_capacity")

# 3.2 时间步相关变量（共8760小时）
time_steps = 8760

# 电池变量
energy_balance = model.addVars(time_steps, lb=0, name="energy_balance")
battery_charge = model.addVars(time_steps, lb=0, name="battery_charge")
battery_discharge = model.addVars(time_steps, lb=0, name="battery_discharge")

# 电解槽与合成氨变量
electrolyzer_power = model.addVars(time_steps, lb=0, name="electrolyzer_power")
ammonia_synthesis_power = model.addVars(time_steps, lb=0, name="ammonia_synthesis_power")

# 过剩电量变量（未利用电量）
surplus = model.addVars(time_steps, lb=0, name="surplus")

# 氢气存储变量
hydrogen_storage_balance = model.addVars(time_steps, lb=0, name="hydrogen_storage_balance")
hydrogen_charge = model.addVars(time_steps, lb=0, name="hydrogen_charge")
hydrogen_discharge = model.addVars(time_steps, lb=0, name="hydrogen_discharge")

# 氨储存变量（类似甲醇储存）
ammonia_storage_balance = model.addVars(time_steps, lb=0, name="ammonia_storage_balance")
ammonia_storage_charge = model.addVars(time_steps, lb=0, name="ammonia_storage_charge")
ammonia_storage_discharge = model.addVars(time_steps, lb=0, name="ammonia_storage_discharge")

# =============================================================================
# 4. 目标函数
# =============================================================================
model.setObjective(
    (wind_annual_cost * wind_capacity +
     pv_annual_cost * pv_capacity +
     battery_annual_cost * battery_capacity +
     electrolyzer_annual_cost * electrolyzer_capacity +
     ammonia_synthesis_annual_cost * ammonia_synthesis_capacity +
     hydrogen_storage_annual_cost * hydrogen_storage_capacity +
     ammonia_storage_annual_cost * ammonia_storage_capacity +
     surplus_penalty * sum(surplus[t] for t in range(time_steps)) +
     battery_charge_penalty * sum(battery_charge[t] for t in range(time_steps)) +
     battery_discharge_penalty * sum(battery_discharge[t] for t in range(time_steps))
    ),
    GRB.MINIMIZE
)

# =============================================================================
# 5. 约束条件
# =============================================================================
# 5.1 逐时设备灵活性约束（限制电解槽与合成氨的逐时变化）
for t in range(1, time_steps):
    model.addConstr(electrolyzer_power[t] - electrolyzer_power[t-1] <= Flex_down * electrolyzer_capacity,
                    name=f"electrolyzer_power_increase_limit_{t}")
    model.addConstr(electrolyzer_power[t] - electrolyzer_power[t-1] >= -Flex_down * electrolyzer_capacity,
                    name=f"electrolyzer_power_decrease_limit_{t}")
    model.addConstr(ammonia_synthesis_power[t] - ammonia_synthesis_power[t-1] <= Flex_up * ammonia_synthesis_capacity,
                    name=f"ammonia_power_increase_limit_{t}")
    model.addConstr(ammonia_synthesis_power[t] - ammonia_synthesis_power[t-1] >= -Flex_down * ammonia_synthesis_capacity,
                    name=f"ammonia_power_decrease_limit_{t}")

# 5.2 合成氨供需平衡（包含氨储存充放），采用不同时间聚合尺度
start_index = 168  # 起始时间（避开初始不稳定阶段）
if aggregation_mode == "hourly":
    for t in range(start_index, time_steps):
        model.addConstr(
            ammonia_synthesis_power[t] + ammonia_storage_discharge[t] ==
            hourly_ammonia_demand + ammonia_storage_charge[t],
            name=f"ammonia_balance_{t}"
        )
elif aggregation_mode == "3hourly":
    interval = 3
    n_intervals = (time_steps - start_index) // interval
    for i in range(n_intervals):
        t_start = start_index + i * interval
        t_end = t_start + interval
        model.addConstr(
            gp.quicksum(ammonia_synthesis_power[t] for t in range(t_start, t_end)) +
            gp.quicksum(ammonia_storage_discharge[t] for t in range(t_start, t_end))
            ==
            hourly_ammonia_demand * interval +
            gp.quicksum(ammonia_storage_charge[t] for t in range(t_start, t_end)),
            name=f"ammonia_balance_3hourly_{i}"
        )
elif aggregation_mode == "daily":
    interval = 24
    n_intervals = (time_steps - start_index) // interval
    for i in range(n_intervals):
        t_start = start_index + i * interval
        t_end = t_start + interval
        model.addConstr(
            gp.quicksum(ammonia_synthesis_power[t] for t in range(t_start, t_end)) +
            gp.quicksum(ammonia_storage_discharge[t] for t in range(t_start, t_end))
            >=
            0.8 * hourly_ammonia_demand * interval +
            gp.quicksum(ammonia_storage_charge[t] for t in range(t_start, t_end)),
            name=f"ammonia_balance_daily_{i}"
        )
elif aggregation_mode == "weekly":
    interval = 168  # 1周 = 168小时
    n_intervals = (time_steps - start_index) // interval
    for i in range(n_intervals):
        t_start = start_index + i * interval
        t_end = t_start + interval
        model.addConstr(
            gp.quicksum(ammonia_synthesis_power[t] for t in range(t_start, t_end)) +
            gp.quicksum(ammonia_storage_discharge[t] for t in range(t_start, t_end))
            ==
            hourly_ammonia_demand * interval +
            gp.quicksum(ammonia_storage_charge[t] for t in range(t_start, t_end)),
            name=f"ammonia_balance_weekly_{i}"
        )
elif aggregation_mode == "monthly":
    # 近似按每月730小时（实际月份可调整）
    interval = 730
    n_intervals = (time_steps - start_index) // interval
    for i in range(n_intervals):
        t_start = start_index + i * interval
        t_end = t_start + interval
        model.addConstr(
            gp.quicksum(ammonia_synthesis_power[t] for t in range(t_start, t_end)) +
            gp.quicksum(ammonia_storage_discharge[t] for t in range(t_start, t_end))
            ==
            hourly_ammonia_demand * interval +
            gp.quicksum(ammonia_storage_charge[t] for t in range(t_start, t_end)),
            name=f"ammonia_balance_monthly_{i}"
        )
elif aggregation_mode == "annual":
    model.addConstr(
        gp.quicksum(ammonia_synthesis_power[t] for t in range(start_index, time_steps)) +
        gp.quicksum(ammonia_storage_discharge[t] for t in range(start_index, time_steps))
        ==
        hourly_ammonia_demand * (time_steps - start_index) +
        gp.quicksum(ammonia_storage_charge[t] for t in range(start_index, time_steps)),
        name="ammonia_balance_annual"
    )
else:
    raise ValueError("Unsupported aggregation_mode value!")

# 附加：年生产总量约束
model.addConstr(
    gp.quicksum(ammonia_synthesis_power[t] for t in range(time_steps)) >= hourly_ammonia_demand * 8760,
    name="annual_ammonia_production_constraint"
)

# 5.3 氢气存储及电力平衡约束
for t in range(time_steps):
    wind_power = wind_output[t] * wind_capacity
    pv_power = pv_output[t] * pv_capacity

    # 合成氨储存约束
    model.addConstr(ammonia_synthesis_power[t] + ammonia_storage_discharge[t] >=
            ammonia_storage_charge[t],
            name=f"ammonia_min_balance_{t}")
    
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
    
    # 电力平衡约束（风电、光伏与电池放电需满足电解槽运行、充电及氢气存储充氢的电耗）
    model.addConstr(
        wind_power + pv_power + battery_discharge[t] ==
        electrolyzer_power[t] + battery_charge[t] + hydrogen_charge[t] * hydrogen_storage_elec + ammonia_synthesis_power[t] * ammonia_electricity_ratio + surplus[t],
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
    
    # 合成氨出力受限于其装机容量（上下限约束）
    model.addConstr(ammonia_synthesis_power[t] <= ammonia_synthesis_capacity,
                    name=f"ammonia_synthesis_max_output_{t}")
    model.addConstr(ammonia_synthesis_power[t] >= 0.2 * ammonia_synthesis_capacity,
                    name=f"ammonia_synthesis_min_output_{t}")
    
    # 氢气供需平衡约束：
    # 电解槽生产的氢气（按系数 3.6/120 转换）加上氢气存储释放量，
    # 应满足合成氨生产所需的氢气（按氢消耗比）加上充氢量
    model.addConstr(
        electrolyzer_power[t] * electrolyzer_eff * 3.6 / 120 + hydrogen_discharge[t] >=
        ammonia_synthesis_power[t] * ammonia_hydrogen_ratio + hydrogen_charge[t],
        name=f"hydrogen_PD_balance_{t}"
    )

# 5.4 氨储存动态约束及容量限制（类似甲醇储存）
model.addConstr(ammonia_storage_balance[0] ==
                ammonia_storage_charge[0] - ammonia_storage_discharge[0],
                name="ammonia_storage_balance_0")
for t in range(1, time_steps):
    model.addConstr(
        ammonia_storage_balance[t] == ammonia_storage_balance[t-1] + ammonia_storage_charge[t] - ammonia_storage_discharge[t],
        name=f"ammonia_storage_balance_{t}"
    )
    model.addConstr(ammonia_storage_balance[t] <= ammonia_storage_capacity,
                    name=f"ammonia_storage_level_limit_{t}")

# 5.5 装机容量约束
model.addConstrs((ammonia_synthesis_capacity >= ammonia_synthesis_power[t] for t in range(time_steps)),
                 name="ammonia_synthesis_capacity_constraint")
model.addConstrs((electrolyzer_capacity >= electrolyzer_power[t] for t in range(time_steps)),
                 name="electrolyzer_capacity_max_constraint")

# =============================================================================
# 6. 求解模型
# =============================================================================
model.optimize()

# =============================================================================
# 7. 输出结果
# =============================================================================
if model.status == GRB.OPTIMAL:
    total_ammonia_production = sum(ammonia_synthesis_power[t].x for t in range(time_steps))
    hydrogen_production_kwh = sum(electrolyzer_power[t].x for t in range(time_steps)) * electrolyzer_eff 
    electrolyzer_utilization_hours = hydrogen_production_kwh / (electrolyzer_capacity.x * electrolyzer_eff * 8760) * 100
    ammonia_utilization_hours = target_ammonia_production / (ammonia_synthesis_capacity.x * 8760) * 100
    total_Elec_generation = sum(wind_output[t] for t in range(time_steps)) * wind_capacity.x + \
                            sum(pv_output[t] for t in range(time_steps)) * pv_capacity.x - \
                            sum(surplus[t].x for t in range(time_steps))
    
    levelized_Elec_cost = (wind_annual_cost * wind_capacity.x +
                           pv_annual_cost * pv_capacity.x +
                           battery_annual_cost * battery_capacity.x) / total_Elec_generation
    
    levelized_H2_cost = (electrolyzer_annual_cost * electrolyzer_capacity.x +
                           levelized_Elec_cost * hydrogen_production_kwh/electrolyzer_eff +
                        hydrogen_storage_annual_cost * hydrogen_storage_capacity.x) / (hydrogen_production_kwh/33.3)
    
    print("Optimal system configuration:")
    print(f"Wind capacity: {wind_capacity.x:.2f} kW")
    print(f"PV capacity: {pv_capacity.x:.2f} kW")
    print(f"Battery capacity: {battery_capacity.x:.2f} kW")
    print(f"Electrolyzer capacity: {electrolyzer_capacity.x:.2f} kW")
    print(f"Ammonia synthesis capacity: {ammonia_synthesis_capacity.x * 8760 / 1000:.2f} t/y")
    print(f"Hydrogen storage capacity: {hydrogen_storage_capacity.x:.2f} kgH2/h")
    print(f"Ammonia storage capacity: {ammonia_storage_capacity.x:.2f} kgNH3/h")
    print(f"Total ammonia production: {total_ammonia_production:.2f} kg")
    print(f"Electrolyzer utilization hours: {electrolyzer_utilization_hours:.2f} %")
    print(f"Ammonia synthesis utilization hours: {ammonia_utilization_hours:.2f} %")
    print(f"Levelized ELec cost: ${levelized_Elec_cost:.2f} per kWh")
    print(f"Levelized H2 cost: ${levelized_H2_cost:.2f} per kgH2")
    print(f"Levelized ammonia production cost: ${model.objVal / total_ammonia_production * 1000:.2f} per ton")
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
    # 合成氨生产曲线，单位为 kg/h（满足小时供需平衡约束）
    ammonia_synthesis_profile = [ammonia_synthesis_power[t].x for t in range(time_steps)]
    # 氨储存状态曲线，单位为 100kg NH3
    ammonia_storage_profile = [ammonia_storage_balance[t].x/100 for t in range(time_steps)]
    
    # 电池充放电及电池储能状态
    battery_charge_profile = [battery_charge[t].x for t in range(time_steps)]
    battery_discharge_profile = [battery_discharge[t].x for t in range(time_steps)]
    energy_balance_profile = [energy_balance[t].x for t in range(time_steps)]
    
    # 提取各时刻电力平衡约束的对偶价格（电价）
    electricity_prices = [model.getConstrByName(f"power_balance_{t}").pi for t in range(time_steps)]
    # 对于合成氨供需平衡约束（在小时模式下，从 t=168 开始定义），提取其对偶价格作为氨价
    ammonia_prices = [0]*time_steps
    for t in range(168, time_steps):
        constr = model.getConstrByName(f"ammonia_balance_{t}")
        if constr is not None:
            ammonia_prices[t] = constr.pi
        else:
            ammonia_prices[t] = 0

    # 绘制样本周（168小时）时序图，示例取第11周（从第 168*10 小时开始）
    week_start = 168 * 10
    week_end = week_start + 168
    
    battery_charge_week = battery_charge_profile[week_start:week_end]
    battery_discharge_week = battery_discharge_profile[week_start:week_end]
    energy_balance_week = energy_balance_profile[week_start:week_end]
    wind_power_week = wind_power_profile[week_start:week_end]
    pv_power_week = pv_power_profile[week_start:week_end]
    electrolyzer_demand_week = electrolyzer_demand[week_start:week_end]
    ammonia_synthesis_week = ammonia_synthesis_profile[week_start:week_end]
    ammonia_storage_week = ammonia_storage_profile[week_start:week_end]
    electricity_prices_week = electricity_prices[week_start:week_end]
    ammonia_prices_week = ammonia_prices[week_start:week_end]
    
    plt.figure(figsize=(15, 8))
    plt.plot(wind_power_week, label='Wind Output (kW)', linestyle='-', color='blue')
    plt.plot(pv_power_week, label='PV Output (kW)', linestyle='-', color='orange')
    plt.plot(battery_charge_week, label='Battery Charge (kW)', linestyle='--', color='green')
    plt.plot(battery_discharge_week, label='Battery Discharge (kW)', linestyle='--', color='red')
    plt.plot(electrolyzer_demand_week, label='Electrolyzer Demand (kW)', linestyle='-', color='purple')
    plt.plot(ammonia_synthesis_week, label='Ammonia Synthesis Production (kg/h)', linestyle='-', color='magenta')
    plt.plot(ammonia_storage_week, label='Ammonia Storage Balance (kg)', linestyle='-', color='cyan')
    plt.plot(electricity_prices_week, label='Electricity Price ($/kWh)', linestyle='-', color='brown')
    plt.plot(ammonia_prices_week, label='Ammonia Price ($/kg)', linestyle='-', color='black')
    
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
        ammonia_synthesis_month = ammonia_synthesis_profile[start:end]
        ammonia_storage_month = ammonia_storage_profile[start:end]
        electricity_prices_month = electricity_prices[start:end]
        ammonia_prices_month = ammonia_prices[start:end]
    
        plt.figure(figsize=(15, 8))
        plt.plot(wind_power_month, label='Wind Output (kW)', linestyle='-', color='blue')
        plt.plot(pv_power_month, label='PV Output (kW)', linestyle='-', color='orange')
        plt.plot(battery_charge_month, label='Battery Charge (kW)', linestyle='--', color='green')
        plt.plot(battery_discharge_month, label='Battery Discharge (kW)', linestyle='--', color='red')
        plt.plot(electrolyzer_demand_month, label='Electrolyzer Demand (kW)', linestyle='-', color='purple')
        plt.plot(ammonia_synthesis_month, label='Ammonia Synthesis Production (kg/h)', linestyle='-', color='magenta')
        plt.plot(ammonia_storage_month, label='Ammonia Storage Balance (100kg)', linestyle='-', color='cyan')
        plt.plot(electricity_prices_month, label='Electricity Price ($/kWh)', linestyle='-', color='brown')
        plt.plot(ammonia_prices_month, label='Ammonia Price ($/kg)', linestyle='-', color='black')
    
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
    week_start = 168 * 10   # 例如取第11周
    week_end = week_start + 168
    x = np.arange(week_end - week_start)

    # 供给侧：风电、光伏、以及电池放电（均为正值）
    supply_wind = np.array([wind_output[t] * wind_capacity.x for t in range(week_start, week_end)])
    supply_pv = np.array([pv_output[t] * pv_capacity.x for t in range(week_start, week_end)])
    supply_battery = np.array([battery_discharge[t].x for t in range(week_start, week_end)])
    
    # 需求侧：电解槽、电池充电，以及氢气充电（乘以充氢所需电耗），显示为负值
    demand_electrolyzer = np.array([electrolyzer_power[t].x for t in range(week_start, week_end)])
    demand_battery = np.array([battery_charge[t].x for t in range(week_start, week_end)])
    demand_hydrogen = np.array([hydrogen_charge[t].x * hydrogen_storage_elec for t in range(week_start, week_end)])
    demand_ammonia = np.array([ammonia_synthesis_power[t].x * ammonia_electricity_ratio for t in range(week_start, week_end)])
    demand_surplus = np.array([surplus[t].x  for t in range(week_start, week_end)])
    
    plt.figure(figsize=(15, 8))
    # 绘制供给堆积柱状图（正值）
    bar1 = plt.bar(x, supply_wind, label='Wind Supply', color='blue')
    bar2 = plt.bar(x, supply_pv, bottom=supply_wind, label='PV Supply', color='orange')
    bar3 = plt.bar(x, supply_battery, bottom=supply_wind + supply_pv, label='Battery Discharge Supply', color='cyan')
    
    # 绘制需求堆积柱状图（负值）
    bar4 = plt.bar(x, -demand_electrolyzer, label='Electrolyzer Demand', color='purple')
    bar5 = plt.bar(x, -demand_battery, bottom=-demand_electrolyzer, label='Battery Charge Demand', color='red')
    bar6 = plt.bar(x, -demand_hydrogen, bottom=-(demand_electrolyzer + demand_battery), label='Hydrogen Charge Demand', color='green')
    bar7 = plt.bar(x, -demand_ammonia, bottom=-(demand_electrolyzer + demand_battery + demand_hydrogen), label='Ammonia Synthesis Demand', color='mediumvioletred')
    bar7 = plt.bar(x, -demand_surplus, bottom=-(demand_electrolyzer + demand_battery + demand_hydrogen + demand_ammonia), label='Surplus', color='brown')
    
    plt.axhline(0, color='black', linewidth=0.8)
    plt.xlabel('Hour')
    plt.ylabel('Power (kW)')
    plt.title('Stacked Power Supply (Positive) & Demand (Negative) - Sample Week')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

    # ---------------------------
    # 按月堆积柱状图
    # ---------------------------
    # 近似每月小时数划分（可根据实际情况调整）
    hours_in_month = [0, 744, 1416, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016, 8760]
    month_names = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']

    for i in range(12):
        start = hours_in_month[i]
        end = hours_in_month[i + 1]
        x_month = np.arange(end - start)
        
        # 供给侧数据
        supply_wind_month = np.array([wind_output[t] * wind_capacity.x for t in range(start, end)])
        supply_pv_month = np.array([pv_output[t] * pv_capacity.x for t in range(start, end)])
        supply_battery_month = np.array([battery_discharge[t].x for t in range(start, end)])
        
        # 需求侧数据
        demand_electrolyzer_month = np.array([electrolyzer_power[t].x for t in range(start, end)])
        demand_battery_month = np.array([battery_charge[t].x for t in range(start, end)])
        demand_hydrogen_month = np.array([hydrogen_charge[t].x * hydrogen_storage_elec for t in range(start, end)])
        demand_ammonia_month = np.array([ammonia_synthesis_power[t].x * ammonia_electricity_ratio for t in range(start, end)])
        demand_surplus_month = np.array([surplus[t].x  for t in range(start, end)])
        
        plt.figure(figsize=(15, 8))
        plt.bar(x_month, supply_wind_month, label='Wind Supply', color='blue')
        plt.bar(x_month, supply_pv_month, bottom=supply_wind_month, label='PV Supply', color='orange')
        plt.bar(x_month, supply_battery_month, bottom=supply_wind_month + supply_pv_month, label='Battery Discharge Supply', color='cyan')
        
        plt.bar(x_month, -demand_electrolyzer_month, label='Electrolyzer Demand', color='purple')
        plt.bar(x_month, -demand_battery_month, bottom=-demand_electrolyzer_month, label='Battery Charge Demand', color='red')
        plt.bar(x_month, -demand_hydrogen_month, bottom=-(demand_electrolyzer_month + demand_battery_month), label='Hydrogen Charge Demand', color='green')
        plt.bar(x_month, -demand_ammonia_month, bottom=-(demand_electrolyzer_month + demand_battery_month + demand_hydrogen_month), label='Ammonia Synthesis Demand', color='mediumvioletred')
        plt.bar(x_month, -demand_surplus_month, bottom=-(demand_electrolyzer_month + demand_battery_month + demand_hydrogen_month + demand_ammonia_month), label='Surplus', color='brown')
        
        plt.axhline(0, color='black', linewidth=0.8)
        plt.xlabel('Hour')
        plt.ylabel('Power (kW)')
        plt.title(f'Stacked Power Supply & Demand ({month_names[i]})')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()
else:
    print("No feasible solution found.")
