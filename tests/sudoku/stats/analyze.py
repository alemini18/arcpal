import pandas as pd
import os
import matplotlib.pyplot as plt
import glob

def categorize_operation(name):
    name_lower = name.lower()
    
    # Memoria
    if 'malloc' in name_lower or 'cudafree' in name_lower or  'cudalaunch' in name_lower:
        return 'Malloc'
    elif 'memcpy' in name_lower or 'memset' in name_lower:
        return 'Memoria'
        
    # CPU (CUDA API or NVTX)
    if 'pushpop' in name_lower or 'build_reverse_tables' in name_lower or \
       'cudadevice' in name_lower or \
       'cumodule' in name_lower or 'cukernel' in name_lower or 'culibrary' in name_lower or \
       'cuevent' in name_lower or 'cudastream' in name_lower:
        return 'CPU'
        
    # Kernel
    return 'Kernel'

def parse_csv_custom(file_path):
    data = []
    with open(file_path, 'r') as f:
        header = f.readline().strip().split(',')
        for line in f:
            line = line.strip()
            if not line: continue
            
            parts = []
            in_quotes = False
            current_part = []
            for char in line:
                if char == '"':
                    in_quotes = not in_quotes
                elif char == ',' and not in_quotes:
                    parts.append(''.join(current_part))
                    current_part = []
                else:
                    current_part.append(char)
            parts.append(''.join(current_part))
            
            if len(parts) > len(header):
                kernel_name = ','.join(parts[len(header)-1:])
                parts = parts[:len(header)-1] + [kernel_name]
            
            while len(parts) < len(header):
                parts.append('')
                
            data.append(parts)
            
    df = pd.DataFrame(data, columns=header)
    df['Total Time (ns)'] = pd.to_numeric(df['Total Time (ns)'], errors='coerce')
    return df

def analyze_stats(directory):
    csv_files = glob.glob(os.path.join(directory, '*_summary.csv'))
    results = []
    
    for file in csv_files:
        config_name = os.path.basename(file).replace('_nsys_summary.csv', '')
        try:
            df = parse_csv_custom(file)
            
            # Categorize each operation
            df['Category'] = df['Kernel Name'].apply(categorize_operation)
            
            # Group by Test File and Category
            grouped = df.groupby(['Test File', 'Category'])['Total Time (ns)'].sum().reset_index()
            
            # Calculate average for each category across tests
            avg_per_category = grouped.groupby('Category')['Total Time (ns)'].mean()
            
            res_dict = {'Configuration': config_name}
            res_dict['Kernel Time (ns)'] = avg_per_category.get('Kernel', 0.0)
            res_dict['Memoria Time (ns)'] = avg_per_category.get('Memoria', 0.0)
            res_dict['Malloc Time (ns)'] = avg_per_category.get('Malloc', 0.0)
            res_dict['CPU Time (ns)'] = avg_per_category.get('CPU', 0.0)
            
            res_dict['Total Time (ns)'] = res_dict['Kernel Time (ns)'] + res_dict['Memoria Time (ns)'] + res_dict['Malloc Time (ns)'] + res_dict['CPU Time (ns)']
            res_dict['Num Tests'] = df['Test File'].nunique()
            
            results.append(res_dict)
            
        except Exception as e:
            print(f"Error processing {file}: {e}")
            
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values('Total Time (ns)')
    return results_df

if __name__ == '__main__':
    directory = '.'
    df = analyze_stats(directory)
    
    with open('performance_report_full.md', 'w') as f:
        f.write('# Sudoku Performance Report (Full Profiling)\n\n')
        f.write('This report compares the performance of different Sudoku solver configurations. ')
        f.write('The times reported represent the average total time across all 9x9 test instances, categorized by component.\n\n')
        f.write('## Summary Table\n\n')
        f.write(df.to_markdown(index=False))
        f.write('\n\n')
        f.write('## Performance Graph\n\n')
        f.write('![Performance Comparison](/home/ale/arcpal/tests/sudoku/stats/performance_comparison_full.png)\n')
        
    # Convert ns to ms for the graph
    df_ms = df.copy()
    for col in ['Kernel Time (ns)', 'Memoria Time (ns)', 'Malloc Time (ns)', 'CPU Time (ns)', 'Total Time (ns)']:
        df_ms[col.replace('(ns)', '(ms)')] = df[col] / 1000000.0
    
    # Sort for plotting based on total time
    df_ms = df_ms.sort_values('Total Time (ms)', ascending=True)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Stacked bar chart (horizontal)
    categories = ['Kernel Time (ms)', 'Memoria Time (ms)', 'Malloc Time (ms)', 'CPU Time (ms)']
    colors = ['#ff9999', '#66b3ff', '#c2c2f0', '#99ff99']
    
    left = [0] * len(df_ms)
    for cat, color in zip(categories, colors):
        ax.barh(df_ms['Configuration'], df_ms[cat], left=left, label=cat.replace(' Time (ms)', ''), color=color)
        left = [l + v for l, v in zip(left, df_ms[cat])]
        
    ax.set_xlabel('Average Time (ms)')
    ax.set_ylabel('Configuration')
    ax.set_title('Detailed Performance Breakdown: Kernel vs Memoria vs CPU')
    ax.legend()
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add a logarithmic scale plot because Memoria/CPU usually dwarves Kernel
    # but let's stick to linear first as requested, and add log scale on x if needed.
    # We will just do linear scale. If the user wants log, they can ask.
    
    plt.tight_layout()
    plt.savefig('performance_comparison_full.png')
    
    # Create an alternative graph in microseconds for just the kernel
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    
    # Stacked column chart (Vertical) as requested by user ("un grafico a colonne in cui ogni colonna è suddivisa in 3 tipi")
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    
    df_ms = df_ms.sort_values('Total Time (ms)', ascending=False)
    
    bottom = [0] * len(df_ms)
    for cat, color in zip(categories, colors):
        ax3.bar(df_ms['Configuration'], df_ms[cat], bottom=bottom, label=cat.replace(' Time (ms)', ''), color=color)
        bottom = [b + v for b, v in zip(bottom, df_ms[cat])]
        
    ax3.set_ylabel('Average Time (ms)')
    ax3.set_xlabel('Configuration')
    ax3.set_title('Detailed Performance Breakdown (Stacked Columns)')
    ax3.legend()
    ax3.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('performance_comparison_columns.png')
    
    # Third graph: Stacked column chart WITHOUT Malloc
    fig4, ax4 = plt.subplots(figsize=(12, 8))
    
    # Sort based on Total Time excluding Malloc
    df_ms['Total Time No Malloc (ms)'] = df_ms['Kernel Time (ms)'] + df_ms['Memoria Time (ms)'] + df_ms['CPU Time (ms)']
    df_no_malloc_sorted = df_ms.sort_values('Total Time No Malloc (ms)', ascending=False)
    
    categories_no_malloc = ['Kernel Time (ms)', 'Memoria Time (ms)', 'CPU Time (ms)']
    colors_no_malloc = ['#ff9999', '#66b3ff', '#99ff99']
    
    bottom_nm = [0] * len(df_no_malloc_sorted)
    for cat, color in zip(categories_no_malloc, colors_no_malloc):
        ax4.bar(df_no_malloc_sorted['Configuration'], df_no_malloc_sorted[cat], bottom=bottom_nm, label=cat.replace(' Time (ms)', ''), color=color)
        bottom_nm = [b + v for b, v in zip(bottom_nm, df_no_malloc_sorted[cat])]
        
    ax4.set_ylabel('Average Time (ms)')
    ax4.set_xlabel('Configuration')
    ax4.set_title('Detailed Performance Breakdown (No Malloc)')
    ax4.legend()
    ax4.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('performance_comparison_no_malloc.png')
    
    print("Analysis complete.")
