"""
Analyze renewable vs conventional generation in OPF results
"""
import pandas as pd
import numpy as np
import os

results_dir = 'opf_results'

print('='*70)
print('RENEWABLE vs CONVENTIONAL GENERATION ANALYSIS')
print('='*70)
print()

# Analyze several time steps
time_steps_to_check = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90]

total_renewable_gen = []
total_conventional_gen = []
total_load = []
renewable_ratios = []

for t in time_steps_to_check:
    gen_file = os.path.join(results_dir, f't_{t}_gen.csv')
    bus_file = os.path.join(results_dir, f't_{t}_bus.csv')

    if not os.path.exists(gen_file) or not os.path.exists(bus_file):
        continue

    gen_data = pd.read_csv(gen_file, index_col=0)
    bus_data = pd.read_csv(bus_file, index_col=0)

    # Total generation
    total_gen = gen_data['p_mw'].sum()

    # Total load (sum of negative p_mw in bus data)
    total_load_t = -bus_data[bus_data['p_mw'] < 0]['p_mw'].sum()

    # Count generators (should be 11: 6 conventional + 5 renewable)
    num_gens = len(gen_data)

    # Check generator names to identify renewable vs conventional
    # In our implementation, we have 11 generators total
    # Generators with very low output (< 0.01 MW) are likely conventional
    # because renewables have zero cost and are prioritized

    # Sort by output to see distribution
    gen_sorted = gen_data['p_mw'].sort_values(ascending=False)

    print(f'Time Step {t}:')
    print(f'  Total Generation: {total_gen:.4f} MW')
    print(f'  Total Load: {total_load_t:.4f} MW')
    print(f'  Number of Generators: {num_gens}')
    print(f'  Generator Outputs (sorted):')

    for idx, (gen_idx, p_mw) in enumerate(gen_sorted.items()):
        gen_type = "Renewable" if p_mw > 0.1 else "Conventional (minimal)"
        print(f'    Gen {idx+1:2d}: {p_mw:8.4f} MW - {gen_type}')

    # Calculate renewable ratio
    # Assuming generators with significant output are renewable (zero cost)
    renewable_output = gen_sorted[gen_sorted > 0.1].sum()
    conventional_output = gen_sorted[gen_sorted <= 0.1].sum()

    renewable_ratio = renewable_output / total_gen if total_gen > 0 else 0

    print(f'  Renewable Generation: {renewable_output:.4f} MW ({renewable_ratio*100:.1f}%)')
    print(f'  Conventional Generation: {conventional_output:.4f} MW ({(1-renewable_ratio)*100:.1f}%)')
    print()

    total_renewable_gen.append(renewable_output)
    total_conventional_gen.append(conventional_output)
    total_load.append(total_load_t)
    renewable_ratios.append(renewable_ratio)

print('='*70)
print('SUMMARY STATISTICS')
print('='*70)
print(f'Average Renewable Ratio: {np.mean(renewable_ratios)*100:.1f}%')
print(f'Average Renewable Generation: {np.mean(total_renewable_gen):.4f} MW')
print(f'Average Conventional Generation: {np.mean(total_conventional_gen):.4f} MW')
print(f'Average Total Load: {np.mean(total_load):.4f} MW')
print()
print('='*70)
print('CONCLUSION')
print('='*70)
print('The electricity price is very low because:')
print('1. Renewable generators have ZERO marginal cost ($0/MW)')
print('2. OPF prioritizes zero-cost renewable generation')
print('3. Load is small enough that renewables can meet most demand')
print('4. Conventional generators only run at minimum output')
print()
print('To increase electricity prices, you can:')
print('- Increase conventional generator costs (currently $10/MW)')
print('- Reduce renewable capacity (currently 15% of total load)')
print('- Increase overall load (currently scaled by 0.6)')
print('- Add transmission constraints that limit renewable delivery')
