import pandas as pd
import matplotlib.pyplot as plt

def main():
    csv_file = 'gpu_utilization_log.csv'
    df = pd.read_csv(csv_file)
    # Clean up whitespace in memory_used
    # Robustly convert memory_used to int, handling both string and integer types
    df['memory_used'] = df['memory_used'].apply(lambda x: int(str(x).strip()))

    # Plot with dual y-axes
    fig, ax1 = plt.subplots(figsize=(10, 5))
    color1 = 'tab:blue'
    ax1.set_xlabel('Timestamp (s)')
    ax1.set_ylabel('GPU Utilization (%)', color=color1)
    l1, = ax1.plot(df['timestamp'], df['gpu_utilization'], color=color1, marker='o', label='GPU Utilization (%)')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True)

    ax2 = ax1.twinx()
    color2 = 'tab:orange'
    ax2.set_ylabel('Memory Used (MiB)', color=color2)
    l2, = ax2.plot(df['timestamp'], df['memory_used'], color=color2, marker='x', label='Memory Used (MiB)')
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title('GPU Utilization and Memory Usage Over Time')
    # Combine legends from both axes
    lines = [l1, l2]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper left')
    plt.tight_layout()
    plt.savefig('gpu_utilization_plot.png')
    plt.show()

if __name__ == '__main__':
    main()