#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import JointState
import numpy as np
from scipy import stats
from collections import defaultdict
import time
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime

class JointNoiseAnalyzer:
    def __init__(self, duration=30, target_rate=500):
        rospy.init_node('joint_noise_analyzer')

        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = f'joint_noise_analysis_{self.timestamp}'
        os.makedirs(self.output_dir, exist_ok=True)

        self.joint_data = defaultdict(lambda: {
            'position': [],
            'velocity': [],
            'effort': [],
            'time': []
        })

        self.start_time = time.time()
        self.last_sample_time = self.start_time
        self.duration = duration
        self.target_rate = target_rate
        self.sample_interval = 1.0 / target_rate
        self.data_collection_complete = False
        self.stability_threshold = 0.05
        self.min_samples = 100

        print(f"\nCollecting data for {duration} seconds at {target_rate}Hz...")
        print("Please ensure the robot is stationary during data collection.")

        # 使用定时器来控制采样率
        rospy.Timer(rospy.Duration(self.sample_interval), self.timer_callback)
        self.latest_msg = None
        self.sub = rospy.Subscriber('/joint_states', JointState, self.msg_callback)

    def msg_callback(self, msg):
        """只存储最新的消息"""
        self.latest_msg = msg

    def timer_callback(self, event):
        """按固定频率处理数据"""
        if self.data_collection_complete or self.latest_msg is None:
            return

        current_time = time.time()
        elapsed_time = current_time - self.start_time

        if elapsed_time > self.duration:
            self.data_collection_complete = True
            print("\nData collection completed. Processing results...")
            self.process_and_save_data()
            rospy.signal_shutdown("Data collection completed")
            return

        msg = self.latest_msg
        for i, joint_name in enumerate(msg.name):
            if i < len(msg.position) and i < len(msg.velocity) and i < len(msg.effort):
                self.joint_data[joint_name]['position'].append(msg.position[i])
                self.joint_data[joint_name]['velocity'].append(msg.velocity[i])
                self.joint_data[joint_name]['effort'].append(msg.effort[i])
                self.joint_data[joint_name]['time'].append(elapsed_time)

        actual_rate = 1.0 / (current_time - self.last_sample_time) if current_time > self.last_sample_time else 0
        self.last_sample_time = current_time
        print(f"Sampling rate: {actual_rate:.2f} Hz | Time: {elapsed_time:.1f}/{self.duration}s", end='\r')

    def check_stability(self, data):
        if len(data) < self.min_samples:
            return False, "Insufficient samples"

        # Calculate trend
        time_indices = np.arange(len(data))
        slope, _, _, _, _ = stats.linregress(time_indices, data)

        # Calculate range
        data_range = np.max(data) - np.min(data)
        mean_abs = np.mean(np.abs(data))
        relative_range = data_range / mean_abs if mean_abs > 1e-10 else data_range

        if relative_range > self.stability_threshold:
            return False, f"Range too large: {relative_range:.6f}"
        if abs(slope) > self.stability_threshold/len(data):
            return False, f"Trend detected: {slope:.6f}"

        return True, "Data stable"

    def test_distribution(self, data):
        if len(data) < self.min_samples:
            return None, None, None, None, None

        mean_val = np.mean(data)
        std_val = np.std(data)
        min_val = np.min(data)
        max_val = np.max(data)

        # Calculate relative range
        range_val = max_val - min_val
        relative_range = range_val / (abs(mean_val) if abs(mean_val) > 1e-10 else 1)

        # Normalize data for distribution tests
        normalized_data = (data - mean_val) / (std_val if std_val > 1e-10 else 1)

        # Very small variation - probably constant
        if relative_range < 1e-6:
            return "Constant", 0, 1.0, mean_val, std_val

        # Test for normality
        shapiro_stat, normal_p = stats.shapiro(normalized_data)

        # Test for uniformity
        ks_stat, uniform_p = stats.kstest(normalized_data, 'uniform',
                                          args=(np.min(normalized_data), np.max(normalized_data)))

        if normal_p > uniform_p and normal_p > 0.05:
            return "Gaussian", std_val/abs(mean_val)*100 if abs(mean_val) > 1e-10 else std_val, normal_p, mean_val, std_val
        else:
            return "Uniform", relative_range*100, uniform_p, min_val, max_val

    def save_joint_analysis(self, joint_name, data):
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'Joint Analysis: {joint_name} (Sample Rate: {self.target_rate}Hz)', fontsize=16)

        data_types = ['position', 'velocity', 'effort']
        for idx, data_type in enumerate(data_types):
            times = np.array(data['time'])
            values = np.array(data[data_type])

            # Time series plot
            axes[idx, 0].plot(times, values, '-b', label='Measured', linewidth=1)
            axes[idx, 0].set_title(f'{data_type.capitalize()} Time Series')
            axes[idx, 0].set_xlabel('Time (s)')
            axes[idx, 0].set_ylabel(data_type.capitalize())
            axes[idx, 0].grid(True)

            # Distribution plot
            axes[idx, 1].hist(values, bins=50, density=True, alpha=0.7, color='g')
            dist_type, std_pct, p_value, param1, param2 = self.test_distribution(values)

            if dist_type == "Gaussian":
                title = f'Distribution: {dist_type}\nMean: {param1:.6f}\nStd: {param2:.6f}\np-value: {p_value:.6f}'
            elif dist_type == "Constant":
                title = f'Distribution: {dist_type}\nValue: {param1:.6f}'
            else:
                title = f'Distribution: {dist_type}\nMin: {param1:.6f}\nMax: {param2:.6f}\np-value: {p_value:.6f}'

            axes[idx, 1].set_title(title)
            axes[idx, 1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{joint_name}_analysis.png'), dpi=300)
        plt.close()

        # Save data
        df = pd.DataFrame({
            'time': data['time'],
            'position': data['position'],
            'velocity': data['velocity'],
            'effort': data['effort']
        })
        df.to_csv(os.path.join(self.output_dir, f'{joint_name}_data.csv'), index=False)

    def process_and_save_data(self):
        results = []

        for joint_name, data in self.joint_data.items():
            # Save combined plot and data
            self.save_joint_analysis(joint_name, data)

            actual_rate = len(data['time']) / (data['time'][-1] - data['time'][0])

            for data_type in ['position', 'velocity', 'effort']:
                values = np.array(data[data_type])
                is_stable, stability_msg = self.check_stability(values)
                dist_type, std_pct, p_value, param1, param2 = self.test_distribution(values)

                result_dict = {
                    'joint_name': joint_name,
                    'data_type': data_type,
                    'distribution': dist_type,
                    'stability': "stable" if is_stable else "unstable",
                    'stability_message': stability_msg,
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'range': np.max(values) - np.min(values),
                    'sample_count': len(values),
                    'actual_sample_rate': actual_rate,
                    'p_value': p_value
                }

                results.append(result_dict)

        # Save analysis results
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(self.output_dir, 'noise_analysis_results.csv'), index=False)

        # Print analysis results
        print("\nNoise Analysis Results:")
        print("="*80)
        print(f"Target sampling rate: {self.target_rate} Hz")

        for joint_name in self.joint_data.keys():
            print(f"\nJoint: {joint_name}")
            joint_results = results_df[results_df['joint_name'] == joint_name]
            for _, row in joint_results.iterrows():
                print(f"\n{row['data_type'].capitalize()}:")
                print(f"  Distribution: {row['distribution']}")
                print(f"  Stability: {row['stability']}")
                print(f"  Sample Count: {row['sample_count']}")
                print(f"  Actual Sample Rate: {row['actual_sample_rate']:.2f} Hz")

                if row['distribution'] == "Constant":
                    print(f"  Value: {row['mean']:.8f}")
                elif row['distribution'] == "Gaussian":
                    print(f"  Mean: {row['mean']:.8f}")
                    print(f"  Std: {row['std']:.8f}")
                    print(f"  Confidence (p-value): {row['p_value']:.6f}")
                else:  # Uniform
                    print(f"  Min: {row['min']:.8f}")
                    print(f"  Max: {row['max']:.8f}")
                    print(f"  Range: {row['range']:.8f}")

        print(f"\nResults saved to: {self.output_dir}")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        analyzer = JointNoiseAnalyzer(duration=30, target_rate=500)  # 30秒数据，500Hz采样率
        analyzer.run()
    except rospy.ROSInterruptException:
        pass