import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

class FlightScenarioAnalyzer:
    def __init__(self):
        self.sample_rate = 5.0  # writes/sec
        self.scenarios = ['cruising', 'climbing', 'rolling', 'descending']
        
    def read_xplane_data(self, filepath):
        """Read X-Plane data from txt file"""
        data = []
        with open(filepath, 'r') as file:
            lines = file.readlines()
            data_lines = [line.strip() for line in lines if line.strip() and '|' in line and 'pitch' not in line]
            
            for line in data_lines:
                values = [float(x.strip()) for x in line.split('|') if x.strip()]
                data.append(values)
        
        columns = ['pitch', 'roll', 'heading_true', 'heading_mag', 
                  'mag_var', 'mag_comp', 'P', 'Q', 'R']
        return pd.DataFrame(data, columns=columns)

    def analyze_scenario(self, data, scenario_name):
        """Analyze flight data for a specific scenario"""
        # Basic analysis
        euler_data = np.array([
            [row.pitch, row.roll, row.heading_true] 
            for _, row in data.iterrows()
        ])
        
        angular_rates = np.array([
            [row.P, row.Q, row.R]
            for _, row in data.iterrows()
        ])
        
        # Convert to quaternions
        quaternions = np.array([
            R.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_quat()
            for pitch, roll, yaw in euler_data
        ])
        
        # Calculate statistics
        stats = {
            'euler': {
                'pitch_mean': np.mean(euler_data[:, 0]),
                'pitch_std': np.std(euler_data[:, 0]),
                'pitch_range': np.ptp(euler_data[:, 0]),
                'roll_mean': np.mean(euler_data[:, 1]),
                'roll_std': np.std(euler_data[:, 1]),
                'roll_range': np.ptp(euler_data[:, 1]),
                'heading_mean': np.mean(euler_data[:, 2]),
                'heading_std': np.std(euler_data[:, 2]),
                'heading_range': np.ptp(euler_data[:, 2])
            },
            'rates': {
                'P_max': np.max(np.abs(angular_rates[:, 0])),
                'Q_max': np.max(np.abs(angular_rates[:, 1])),
                'R_max': np.max(np.abs(angular_rates[:, 2])),
                'P_mean': np.mean(angular_rates[:, 0]),
                'Q_mean': np.mean(angular_rates[:, 1]),
                'R_mean': np.mean(angular_rates[:, 2])
            },
            'quaternion': {
                'w_std': np.std(quaternions[:, 0]),
                'x_std': np.std(quaternions[:, 1]),
                'y_std': np.std(quaternions[:, 2]),
                'z_std': np.std(quaternions[:, 3])
            }
        }
        
        return {
            'time': np.arange(len(data)) / self.sample_rate,
            'euler_data': euler_data,
            'quaternions': quaternions,
            'angular_rates': angular_rates,
            'stats': stats,
            'scenario': scenario_name
        }

    def plot_scenario(self, results):
        """Create plots for a specific scenario"""
        fig = plt.figure(figsize=(15, 12))
        fig.suptitle(f'Flight Analysis: {results["scenario"].title()}', size=14)
        
        # Plot 1: Euler Angles
        ax1 = plt.subplot(311)
        ax1.plot(results['time'], results['euler_data'][:, 0], 
                label='Pitch', color='blue')
        ax1.plot(results['time'], results['euler_data'][:, 1], 
                label='Roll', color='red')
        ax1.plot(results['time'], results['euler_data'][:, 2], 
                label='Heading', color='green')
        ax1.set_title('Euler Angles')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Degrees')
        ax1.grid(True)
        ax1.legend()
        
        # Plot 2: Quaternion Components
        ax2 = plt.subplot(312)
        ax2.plot(results['time'], results['quaternions'][:, 0], 
                label='w', color='purple')
        ax2.plot(results['time'], results['quaternions'][:, 1], 
                label='x', color='orange')
        ax2.plot(results['time'], results['quaternions'][:, 2], 
                label='y', color='brown')
        ax2.plot(results['time'], results['quaternions'][:, 3], 
                label='z', color='pink')
        ax2.set_title('Quaternion Components')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Component Value')
        ax2.grid(True)
        ax2.legend()
        
        # Plot 3: Angular Rates
        ax3 = plt.subplot(313)
        ax3.plot(results['time'], results['angular_rates'][:, 0], 
                label='P (Roll Rate)', color='red')
        ax3.plot(results['time'], results['angular_rates'][:, 1], 
                label='Q (Pitch Rate)', color='blue')
        ax3.plot(results['time'], results['angular_rates'][:, 2], 
                label='R (Yaw Rate)', color='green')
        ax3.set_title('Angular Rates')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Degrees/s')
        ax3.grid(True)
        ax3.legend()
        
        plt.tight_layout()
        return fig

def main():
    analyzer = FlightScenarioAnalyzer()
    scenarios = {
        'cruising': 'cruising_data.txt',
        'climbing': 'climbing_data.txt',
        'rolling': 'rolling_data.txt',
        'descending': 'descending_data.txt'
    }
    
    all_results = {}
    
    # Analyze each scenario
    for scenario, filepath in scenarios.items():
        try:
            print(f"\nAnalyzing {scenario} scenario...")
            data = analyzer.read_xplane_data(filepath)
            results = analyzer.analyze_scenario(data, scenario)
            
            # Print statistics
            print(f"\n{scenario.upper()} Statistics:")
            print("\nEuler Angles:")
            for key, value in results['stats']['euler'].items():
                print(f"{key}: {value:.6f}")
            
            print("\nAngular Rates (deg/s):")
            for key, value in results['stats']['rates'].items():
                print(f"{key}: {value:.6f}")
            
            print("\nQuaternion Stability:")
            for key, value in results['stats']['quaternion'].items():
                print(f"{key}: {value:.6f}")
            
            # Create and save plot
            fig = analyzer.plot_scenario(results)
            plt.savefig(f'{scenario}_analysis.png')
            plt.close()
            
            all_results[scenario] = results
            
        except FileNotFoundError:
            print(f"Warning: Data file for {scenario} not found")
            continue
        except Exception as e:
            print(f"Error analyzing {scenario}: {str(e)}")
            continue

if __name__ == "__main__":
    main()