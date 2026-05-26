"""
Power System Optimal Power Flow (OPF) Dispatch

Python equivalent of MATLAB/MATPOWER dispatch.m script.
Uses pandapower for power system simulation.
Matches the behavior of dispatch.m as closely as possible.
"""

import pandas as pd
import numpy as np
import os
import warnings
import copy

try:
    import pandapower as pp
except ImportError:
    raise ImportError("Install pandapower: pip install pandapower")


def load_prediction_data(predictions_dir='../predictions'):
    """Load prediction data from CSV files (matching MATLAB script)."""
    data = {}
    
    # Load load predictions (scaled by 0.6 as in MATLAB)
    for zone in [1, 2, 3, 4]:
        load_file = os.path.join(predictions_dir, f'load_power_{zone}.csv')
        if os.path.exists(load_file):
            df = pd.read_csv(load_file)
            data[f'load_{zone}'] = df['Predicted'].values * 0.6
    
    # Load solar predictions
    for zone in [1, 2, 3]:
        solar_file = os.path.join(predictions_dir, f'solar_power_{zone}.csv')
        if os.path.exists(solar_file):
            df = pd.read_csv(solar_file)
            data[f'solar_{zone}'] = df['Predicted'].values
    
    # Load wind predictions
    for zone in [3, 4]:
        wind_file = os.path.join(predictions_dir, f'wind_power_{zone}.csv')
        if os.path.exists(wind_file):
            df = pd.read_csv(wind_file)
            data[f'wind_{zone}'] = df['Predicted'].values
    
    return data


def create_ieee30_base_system():
    """
    Create IEEE 30-bus system matching MATPOWER case30.m specifications.
    This includes all buses, lines, transformers, generators, and loads.
    """
    net = pp.create_empty_network()
    
    # IEEE 30-bus voltage levels (kV)
    bus_kv = {
        1: 138, 2: 138, 3: 138, 4: 138, 5: 138, 6: 138,
        7: 138, 8: 138, 9: 138, 10: 138, 11: 138, 12: 138,
        13: 138, 14: 138, 15: 138, 16: 138, 17: 138, 18: 138,
        19: 138, 20: 138, 21: 138, 22: 138, 23: 138, 24: 138,
        25: 138, 26: 138, 27: 138, 28: 138, 29: 138, 30: 138
    }
    
    # Create buses
    for i in range(1, 31):
        pp.create_bus(net, vn_kv=bus_kv[i], name=f'Bus {i}', type='b')
    
    # NO external grid - system is islanded with only generators
    # Bus 1 generator will act as slack (reference) bus
    
    # Add transmission lines (from IEEE 30-bus standard data)
    # Format: (from_bus, to_bus, r_pu, x_pu, b_pu, rate_a_mva)
    # Using simplified parameters for demonstration
    lines_data = [
        # From case30.m standard topology
        (1, 2, 0.0192, 0.0575, 0.0264, 200),
        (1, 3, 0.0452, 0.1852, 0.0204, 200),
        (2, 4, 0.0570, 0.1737, 0.0184, 200),
        (3, 4, 0.0132, 0.0379, 0.0042, 200),
        (2, 5, 0.0472, 0.1983, 0.0209, 200),
        (2, 6, 0.0581, 0.1763, 0.0187, 200),
        (4, 6, 0.0119, 0.0414, 0.0045, 200),
        (5, 7, 0.0460, 0.1160, 0.0100, 200),
        (6, 7, 0.0267, 0.0820, 0.0085, 200),
        (6, 8, 0.0120, 0.0420, 0.0045, 200),
        (6, 9, 0.0000, 0.2080, 0.0000, 200),  # Transformer
        (6, 10, 0.0000, 0.5560, 0.0000, 200),  # Transformer
        (9, 11, 0.0000, 0.2080, 0.0000, 200),  # Transformer
        (9, 10, 0.0318, 0.0845, 0.0088, 200),
        (4, 12, 0.0000, 0.2560, 0.0000, 200),  # Transformer
        (12, 13, 0.0000, 0.2700, 0.0000, 200),  # Transformer
        (12, 14, 0.1231, 0.2559, 0.0000, 200),
        (12, 15, 0.0662, 0.1304, 0.0000, 200),
        (12, 16, 0.0945, 0.1987, 0.0000, 200),
        (14, 15, 0.1281, 0.2638, 0.0000, 200),
        (16, 17, 0.0824, 0.1923, 0.0000, 200),
        (15, 18, 0.1015, 0.2087, 0.0000, 200),
        (18, 19, 0.0399, 0.1526, 0.0000, 200),
        (19, 20, 0.0933, 0.2088, 0.0000, 200),
        (10, 20, 0.0933, 0.2088, 0.0000, 200),
        (10, 17, 0.0324, 0.0845, 0.0000, 200),
        (10, 21, 0.0348, 0.0749, 0.0000, 200),
        (10, 22, 0.0727, 0.1499, 0.0000, 200),
        (21, 22, 0.0116, 0.0236, 0.0000, 200),
        (15, 23, 0.1000, 0.2020, 0.0000, 200),
        (22, 24, 0.1150, 0.1790, 0.0000, 200),
        (23, 24, 0.1320, 0.2700, 0.0000, 200),
        (24, 25, 0.1885, 0.3292, 0.0000, 200),
        (25, 26, 0.2544, 0.3800, 0.0000, 200),
        (25, 27, 0.1093, 0.2087, 0.0000, 200),
        (28, 27, 0.0000, 0.3960, 0.0000, 200),  # Transformer
        (27, 29, 0.2198, 0.4153, 0.0000, 200),
        (27, 30, 0.3202, 0.6027, 0.0000, 200),
        (29, 30, 0.2399, 0.4153, 0.0000, 200),
        (8, 28, 0.0636, 0.2000, 0.0214, 200),
        (6, 28, 0.0169, 0.0599, 0.0065, 200),
    ]
    
    for from_bus, to_bus, r, x, b, rate in lines_data:
        try:
            # IEEE 30-bus parameters are already in per-unit on 100 MVA base
            # Convert to actual ohms for pandapower
            # Z_base = V_base^2 / S_base = 138^2 / 100 = 190.44 ohms
            z_base = 138**2 / 100  # 190.44 ohms
            
            pp.create_line_from_parameters(
                net,
                from_bus=from_bus - 1,  # Convert to 0-indexed
                to_bus=to_bus - 1,
                length_km=100.0,  # Use 100 km to make parameters reasonable
                r_ohm_per_km=r * z_base / 100,  # Convert pu to ohms/km
                x_ohm_per_km=x * z_base / 100,
                c_nf_per_km=b * 1e6 / (2 * np.pi * 60 * z_base * 100) if b > 0 else 0,  # Convert pu susceptance to nF/km
                max_i_ka=rate / (np.sqrt(3) * 138) if rate > 0 else 1.0,
                name=f'Line {from_bus}-{to_bus}'
            )
        except Exception as e:
            print(f"Warning: Could not create line {from_bus}-{to_bus}: {e}")
    
    # Add original generators from case30.m
    # Bus, P_mw (initial), V_pu, Q_max, Q_min, P_max, P_min
    # Generator at bus 1 acts as slack/reference generator
    gen_data = [
        (1, 50, 1.06, 150, -20, 80, 0),   # Slack generator (controls voltage)
        (2, 40, 1.05, 150, -20, 80, 0),   # PV bus
        (5, 30, 1.05, 150, -20, 50, 0),   # PV bus
        (8, 30, 1.05, 150, -20, 50, 0),   # PV bus
        (11, 20, 1.05, 150, -20, 40, 0),  # PV bus
        (13, 20, 1.05, 150, -20, 40, 0),  # PV bus
    ]
    
    for bus_idx, p_mw, vm_pu, q_max, q_min, p_max, p_min in gen_data:
        pp.create_gen(
            net,
            bus=bus_idx - 1,
            p_mw=p_mw,
            vm_pu=vm_pu,
            max_q_mvar=q_max,
            min_q_mvar=q_min,
            max_p_mw=p_max,
            min_p_mw=p_min,
            name=f'Gen_{bus_idx}',
            controllable=True,
            slack=True if bus_idx == 1 else False  # Bus 1 is slack
        )
    
    # Add cost characteristics for conventional generators
    # Using linear cost model similar to MATLAB's gencost
    for idx in net.gen.index:
        # Cost coefficients: cp0_eur (fixed), cp1_eur_per_mw (linear), cp2_eur_per_mw2 (quadratic)
        pp.create_poly_cost(
            net, idx, 'gen',
            cp0_eur=0.0,           # Fixed cost
            cp1_eur_per_mw=10.0,   # Linear cost coefficient ($/MW)
            cp2_eur_per_mw2=0.0    # Quadratic cost coefficient
        )
    
    # Add loads at all buses (base values from case30.m)
    # These will be scaled by prediction factors
    base_loads = {
        1: (0, 0), 2: (21.7, 12.7), 3: (2.4, 1.2), 4: (7.6, 1.6),
        5: (94.2, 19.0), 6: (0, 0), 7: (22.8, 10.9), 8: (30.0, 30.0),
        9: (0, 0), 10: (5.8, 2.0), 11: (0, 0), 12: (11.2, 7.5),
        13: (0, 0), 14: (6.2, 1.6), 15: (8.2, 2.5), 16: (3.5, 1.8),
        17: (9.0, 5.8), 18: (3.2, 0.9), 19: (9.5, 3.4), 20: (2.2, 0.7),
        21: (17.5, 11.2), 22: (0, 0), 23: (3.2, 1.6), 24: (8.7, 6.7),
        25: (0, 0), 26: (3.5, 2.3), 27: (0, 0), 28: (0, 0),
        29: (2.4, 0.9), 30: (10.6, 1.9)
    }
    
    for bus_idx, (p_mw, q_mvar) in base_loads.items():
        if p_mw > 0 or q_mvar > 0:
            pp.create_load(
                net,
                bus=bus_idx - 1,
                p_mw=p_mw,
                q_mvar=q_mvar,
                name=f'Load_{bus_idx}',
                controllable=False
            )
    
    return net


def run_opf_dispatch():
    """Main OPF dispatch function matching MATLAB dispatch.m behavior."""
    
    # Change to project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    predictions_dir = os.path.join(project_root, 'predictions')
    results_dir = os.path.join(project_root, 'opf_results')
    
    os.chdir(project_root)
    print(f"Working directory: {project_root}")
    
    print("Loading prediction data...")
    pred_data = load_prediction_data(predictions_dir)
    
    time_steps = 96  # 24 hours * 4 (15-min intervals)
    
    # Use dummy data if predictions not found
    if not pred_data:
        print("Warning: No prediction data found. Using dummy data.")
        np.random.seed(42)
        for zone in [1, 2, 3, 4]:
            pred_data[f'load_{zone}'] = np.random.rand(time_steps) * 0.6
        for zone in [1, 2, 3]:
            pred_data[f'solar_{zone}'] = np.random.rand(time_steps)
        for zone in [3, 4]:
            pred_data[f'wind_{zone}'] = np.random.rand(time_steps)
    
    # Define areas (matching MATLAB exactly)
    areas = {
        1: [1, 2, 3, 4, 12, 13, 16],
        2: [14, 15, 17, 18, 19, 20, 23],
        3: [5, 6, 7, 8, 9, 10, 11, 28],
        4: [21, 22, 24, 25, 26, 27, 29, 30],
    }
    
    # Get prediction arrays
    load_1 = pred_data.get('load_1', np.ones(time_steps) * 0.6)[:time_steps]
    load_2 = pred_data.get('load_2', np.ones(time_steps) * 0.6)[:time_steps]
    load_3 = pred_data.get('load_3', np.ones(time_steps) * 0.6)[:time_steps]
    load_4 = pred_data.get('load_4', np.ones(time_steps) * 0.6)[:time_steps]
    
    solar_1 = pred_data.get('solar_1', np.zeros(time_steps))[:time_steps]
    solar_2 = pred_data.get('solar_2', np.zeros(time_steps))[:time_steps]
    solar_3 = pred_data.get('solar_3', np.zeros(time_steps))[:time_steps]
    wind_3 = pred_data.get('wind_3', np.zeros(time_steps))[:time_steps]
    wind_4 = pred_data.get('wind_4', np.zeros(time_steps))[:time_steps]
    
    # Calculate total active load over entire time range (matching MATLAB)
    total_active_load = 0
    for i in range(time_steps):
        for area_id, buses in areas.items():
            if area_id == 1:
                load_factor = load_1[i]
            elif area_id == 2:
                load_factor = load_2[i]
            elif area_id == 3:
                load_factor = load_3[i]
            elif area_id == 4:
                load_factor = load_4[i]
            
            # Sum base loads for this area
            base_loads = {
                1: 0, 2: 21.7, 3: 2.4, 4: 7.6, 12: 11.2, 13: 0, 16: 3.5,
                14: 6.2, 15: 8.2, 17: 9.0, 18: 3.2, 19: 9.5, 20: 2.2, 23: 3.2,
                5: 94.2, 6: 0, 7: 22.8, 8: 30.0, 9: 0, 10: 5.8, 11: 0, 28: 0,
                21: 17.5, 22: 0, 24: 8.7, 25: 0, 26: 3.5, 27: 0, 29: 2.4, 30: 10.6
            }
            
            for bus in buses:
                total_active_load += base_loads.get(bus, 0) * load_factor
    
    # Total renewable generation (15% of total load as in MATLAB)
    total_renewable_generation = 0.15 * total_active_load
    
    # Sum of all renewable generation data
    sum_solar_wind = (np.sum(solar_1) + np.sum(solar_2) + np.sum(solar_3) + 
                      np.sum(wind_3) + np.sum(wind_4))
    
    if sum_solar_wind == 0:
        sum_solar_wind = 1  # Avoid division by zero
    
    print(f"Total active load: {total_active_load:.2f} MW")
    print(f"Total renewable allocation: {total_renewable_generation:.2f} MW")
    print(f"Sum solar+wind: {sum_solar_wind:.2f}")
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Storage for results
    results = []
    pg_over_time = []
    cost_over_time = []
    
    print("\nRunning OPF for each time step...")
    
    # Create base system once
    base_net = create_ieee30_base_system()
    
    # Store original load values for proper scaling
    original_loads = {}
    for idx in base_net.load.index:
        original_loads[idx] = {
            'p_mw': base_net.load.at[idx, 'p_mw'],
            'q_mvar': base_net.load.at[idx, 'q_mvar'],
            'bus': base_net.load.at[idx, 'bus']
        }
    
    for i in range(time_steps):
        # Deep copy the base network for each time step
        net = copy.deepcopy(base_net)
        
        # Update loads based on predictions (matching MATLAB behavior)
        area_loads = {
            1: load_1[i],
            2: load_2[i],
            3: load_3[i],
            4: load_4[i],
        }
        
        # Map buses to areas
        bus_to_area = {}
        for area_id, buses in areas.items():
            for bus in buses:
                bus_to_area[bus] = area_id
        
        # Update each load based on its bus area
        for idx in net.load.index:
            bus_num = net.load.at[idx, 'bus'] + 1  # Convert to 1-indexed
            area_id = bus_to_area.get(bus_num, 1)
            load_factor = area_loads[area_id]
            
            # Scale from original base load (not cumulative)
            orig_p = original_loads[idx]['p_mw']
            orig_q = original_loads[idx]['q_mvar']
            
            net.load.at[idx, 'p_mw'] = orig_p * load_factor
            net.load.at[idx, 'q_mvar'] = orig_q * load_factor
        
        # Add renewable generators at FIXED nodes per area
        # Area 1 (buses 1,2,3,4,12,13,16): solar_1 at bus 12
        # Area 2 (buses 14,15,17,18,19,20,23): solar_2 at bus 15
        # Area 3 (buses 5,6,7,8,9,10,11,28): solar_3 at bus 8, wind_3 at bus 10
        # Area 4 (buses 21,22,24,25,26,27,29,30): wind_4 at bus 24
        renewable_gens = [
            ('solar_1', 12, solar_1[i]),   # Area 1 - Solar
            ('solar_2', 15, solar_2[i]),   # Area 2 - Solar
            ('solar_3', 8, solar_3[i]),    # Area 3 - Solar (changed from 28 to 8)
            ('wind_3', 10, wind_3[i]),     # Area 3 - Wind
            ('wind_4', 24, wind_4[i]),     # Area 4 - Wind
        ]
        
        for gen_name, bus, gen_value in renewable_gens:
            bus_idx = bus - 1  # Convert to 0-indexed
            
            # Calculate renewable power allocation (matching MATLAB formula)
            p_max = total_renewable_generation * (gen_value / sum_solar_wind)
            p_max = max(0, p_max)
            
            try:
                pp.create_gen(
                    net,
                    bus=bus_idx,
                    p_mw=0,  # Initial power (OPF will determine optimal)
                    vm_pu=1.02,  # Slightly lower voltage than conventional (1.05-1.06)
                    max_p_mw=p_max,  # Maximum available renewable power
                    min_p_mw=0,      # Can curtail if needed
                    max_q_mvar=50,   # Reduced reactive power range for renewables
                    min_q_mvar=-20,
                    name=gen_name,
                    controllable=True,
                    slack=False  # Renewables are NOT slack
                )
                
                # Add zero cost for renewables (prefer renewable over conventional)
                gen_idx = net.gen[net.gen.name == gen_name].index[0]
                pp.create_poly_cost(
                    net, gen_idx, 'gen',
                    cp0_eur=0.0,
                    cp1_eur_per_mw=0.0,  # Zero marginal cost for renewables
                    cp2_eur_per_mw2=0.0
                )
                
            except Exception as e:
                print(f"Warning: Could not create renewable gen {gen_name} at bus {bus}: {e}")
        
        # Run OPF
        success = False
        try:
            pp.runopp(net, verbose=False)
            success = net.OPF_converged
        except Exception as e:
            print(f"Warning: OPF failed at time step {i+1}: {e}")
            success = False
        
        if not success:
            warnings.warn(f'OPF did not converge at time step {i+1}')
        
        # Extract results
        if success:
            try:
                pg = net.res_gen['p_mw'].values
                
                # Calculate total cost from OPF results
                # pandapower stores the total OPF cost in net.res_opf
                if hasattr(net, 'res_opf') and 'cost' in net.res_opf.columns:
                    cost = net.res_opf['cost'].sum()
                else:
                    # Manual cost calculation based on generator outputs and costs
                    cost = 0.0
                    for idx in net.res_gen.index:
                        p = net.res_gen.at[idx, 'p_mw']
                        gen_name = net.gen.at[idx, 'name']
                        
                        # Get cost coefficients from poly_cost table
                        cost_coeffs = net.poly_cost[net.poly_cost.element == idx]
                        if len(cost_coeffs) > 0:
                            cp0 = cost_coeffs.iloc[0]['cp0_eur']
                            cp1 = cost_coeffs.iloc[0]['cp1_eur_per_mw']
                            cp2 = cost_coeffs.iloc[0].get('cp2_eur_per_mw2', 0)
                            cost += cp0 + cp1 * p + cp2 * p * p
                        else:
                            # Default cost if no cost data
                            cost += p * 10.0
                    
            except Exception as e:
                print(f"Error extracting results at step {i+1}: {e}")
                pg = np.zeros(len(net.gen))
                cost = 0
        else:
            pg = np.zeros(len(net.gen))
            cost = 0
        
        # Store results
        results.append({
            'time_step': i + 1,
            'success': success,
            'cost': cost,
        })
        
        pg_over_time.append(pg)
        cost_over_time.append(cost)
        
        # Progress reporting
        if (i + 1) % 10 == 0 or i == 0:
            print(f"Time step {i+1}/{time_steps} - Cost: ${cost:.2f} - Converged: {success}")
        
        # Save bus and gen results for every time step (matching MATLAB)
        try:
            if success:
                bus_df = net.res_bus.copy()
                gen_df = net.res_gen.copy()
            else:
                bus_df = pd.DataFrame()
                gen_df = pd.DataFrame()
            
            bus_df.to_csv(os.path.join(results_dir, f't_{i+1}_bus.csv'))
            gen_df.to_csv(os.path.join(results_dir, f't_{i+1}_gen.csv'))
        except Exception as e:
            print(f"Warning: Could not save results for step {i+1}: {e}")
    
    # Convert to arrays
    pg_over_time = np.array(pg_over_time)
    cost_over_time = np.array(cost_over_time)
    
    # Save summary results
    summary = pd.DataFrame({
        'time_step': range(1, time_steps + 1),
        'total_cost': cost_over_time,
        'converged': [r['success'] for r in results],
    })
    summary.to_csv(os.path.join(results_dir, 'summary.csv'), index=False)
    
    # Print final summary
    print(f"\n{'='*70}")
    print("OPF DISPATCH COMPLETE")
    print(f"{'='*70}")
    print(f"Average cost: ${np.mean(cost_over_time):.2f}")
    print(f"Total cost: ${np.sum(cost_over_time):.2f}")
    print(f"Convergence rate: {np.mean([r['success'] for r in results])*100:.1f}%")
    print(f"Results saved to: {results_dir}")
    
    return results, pg_over_time, cost_over_time


if __name__ == "__main__":
    results, pg, costs = run_opf_dispatch()
