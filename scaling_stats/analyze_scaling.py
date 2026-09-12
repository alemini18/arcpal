import pandas as pd
import os
import matplotlib.pyplot as plt
import glob

SERIAL_CONFIG = 'serial_naive'

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
                name = ','.join(parts[len(header)-1:])
                parts = parts[:len(header)-1] + [name]
            
            while len(parts) < len(header):
                parts.append('')
                
            data.append(parts)
            
    df = pd.DataFrame(data, columns=header)
    df['Total Time (ns)'] = pd.to_numeric(df['Total Time (ns)'], errors='coerce')
    df['Rules'] = pd.to_numeric(df['Rules'], errors='coerce')
    df['Lits'] = pd.to_numeric(df['Lits'], errors='coerce')
    return df

def fixpoint_times(df):
    # Un punto per istanza: somma delle regioni NVTX 'fixpoint' di ogni esecuzione e mediana
    # sulle ripetizioni. La baseline seriale non ha regioni NVTX, per lei vale il tempo di parete.
    rows = df[df['Report'].str.startswith('nvtx')]
    rows = rows[rows['Name'].str.contains('fixpoint', case=False, na=False)]
    if rows.empty:
        rows = df[df['Report'] == 'wall']
    per_run = rows.groupby(['Test File', 'Rules', 'Lits', 'Run'])['Total Time (ns)'].sum()
    return per_run.groupby(['Test File', 'Rules', 'Lits']).median().reset_index()

def analyze_scaling(directory):
    csv_files = glob.glob(os.path.join(directory, '*_scaling.csv'))
    results = []
    
    for file in csv_files:
        config_name = os.path.basename(file).replace('_scaling.csv', '')
        try:
            points = fixpoint_times(parse_csv_custom(file))
            if points.empty:
                continue
            points['Configuration'] = config_name
            # La famiglia e' il nome dell'istanza senza la taglia: sudoku, synth_L008, synth_L200
            points['Family'] = points['Test File'].str.replace('.in', '', regex=False).str.rsplit('_', n=1).str[0]
            points['Time (ms)'] = points['Total Time (ns)'] / 1000000.0
            results.append(points)
            
        except Exception as e:
            print(f"Error processing {file}: {e}")
            
    if not results:
        return pd.DataFrame()
    
    return pd.concat(results).sort_values(['Family', 'Rules', 'Configuration'])

if __name__ == '__main__':
    directory = '.'
    df = analyze_scaling(directory)
    
    families = sorted(df['Family'].unique())
    
    with open('scaling_report.md', 'w') as f:
        f.write('# Scaling Report\n\n')
        f.write('Propagation time against the number of rules, for each variant. ')
        f.write('Each point is the sum of the NVTX `fixpoint` ranges of one execution, ')
        f.write(f'taken as the median over the repetitions; the `{SERIAL_CONFIG}` baseline has no NVTX range, ')
        f.write('so its wall time is used instead.\n\n')
        f.write('The `synth_L008` and `synth_L200` families have the same shape and the same number of ')
        f.write('fixpoint iterations, and differ only in the number of literals per rule: 8 literals fit in ')
        f.write('one tile, 200 force a tile to scan the rule in several passes.\n\n')
        
        for family in families:
            sub = df[df['Family'] == family]
            f.write(f'## {family}\n\n')
            table = sub.pivot_table(index='Rules', columns='Configuration', values='Time (ms)')
            f.write(table.to_markdown())
            f.write('\n\n')
        
        f.write('## Scaling Graph\n\n')
        f.write('![Scaling Comparison](scaling_comparison.png)\n')
    
    fig, axes = plt.subplots(1, len(families), figsize=(7 * len(families), 6), squeeze=False)
    
    for ax, family in zip(axes[0], families):
        sub = df[df['Family'] == family]
        
        # Nei Sudoku i letterali per regola crescono con la taglia, nelle sintetiche sono fissi
        lits = (sub['Lits'] / sub['Rules']).round().astype(int)
        label = f'{lits.min()} literals per rule' if lits.min() == lits.max() else f'{lits.min()} to {lits.max()} literals per rule'
        
        for config in sorted(sub['Configuration'].unique()):
            line = sub[sub['Configuration'] == config].sort_values('Rules')
            ax.plot(line['Rules'], line['Time (ms)'], marker='o', label=config)
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks(sorted(sub['Rules'].unique()))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(plt.NullFormatter())
        ax.set_xlabel('Number of rules')
        ax.set_ylabel('Fixpoint Time (ms)')
        ax.set_title(f'{family} ({label})')
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.legend(fontsize='small')
    
    plt.tight_layout()
    plt.savefig('scaling_comparison.png')
