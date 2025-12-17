#!/usr/bin/env python3
"""
Performance analysis and comparison script for TurboMind vs DriftEngine benchmarks.
This script analyzes benchmark results and generates comprehensive performance reports.
"""

import json
import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import argparse
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")


class BenchmarkAnalyzer:
    """Analyze and compare benchmark results."""
    
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self.results = {}
        self.analysis_results = {}
        
    def load_results(self) -> Dict[str, Any]:
        """Load all benchmark results from directory."""
        print(f"Loading results from: {self.results_dir}")
        
        for config_dir in glob.glob(os.path.join(self.results_dir, "*")):
            config_name = os.path.basename(config_dir)
            if not os.path.isdir(config_dir):
                continue
                
            self.results[config_name] = {}
            
            for scenario_dir in glob.glob(os.path.join(config_dir, "*")):
                scenario_name = os.path.basename(scenario_dir)
                if not os.path.isdir(scenario_dir):
                    continue
                    
                self.results[config_name][scenario_name] = self._load_scenario_results(scenario_dir)
        
        print(f"Loaded results for {len(self.results)} configurations")
        return self.results
    
    def _load_scenario_results(self, scenario_dir: str) -> Dict[str, Any]:
        """Load results for a specific scenario."""
        scenario_results = {
            'csv_files': {},
            'system_metrics': {},
            'gpu_monitoring': {},
            'logs': []
        }
        
        # Load CSV files
        for csv_file in glob.glob(os.path.join(scenario_dir, "*.csv")):
            try:
                df = pd.read_csv(csv_file)
                scenario_results['csv_files'][os.path.basename(csv_file)] = df
            except Exception as e:
                print(f"Warning: Failed to load {csv_file}: {e}")
        
        # Load system metrics
        system_metrics_file = os.path.join(scenario_dir, "system_metrics.json")
        if os.path.exists(system_metrics_file):
            try:
                with open(system_metrics_file, 'r') as f:
                    scenario_results['system_metrics'] = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load system metrics: {e}")
        
        # Load GPU monitoring
        gpu_monitor_file = os.path.join(scenario_dir, "gpu_monitoring.csv")
        if os.path.exists(gpu_monitor_file):
            try:
                df = pd.read_csv(gpu_monitor_file, header=None, 
                               names=['timestamp', 'gpu_utilization', 'memory_used_gb', 'memory_total_gb', 'temperature'])
                scenario_results['gpu_monitoring'] = df
            except Exception as e:
                print(f"Warning: Failed to load GPU monitoring: {e}")
        
        return scenario_results
    
    def extract_key_metrics(self) -> Dict[str, Any]:
        """Extract key performance metrics from loaded results."""
        print("Extracting key performance metrics...")
        
        key_metrics = {
            'throughput': {},
            'latency': {},
            'memory': {},
            'efficiency': {}
        }
        
        for config_name, config_data in self.results.items():
            key_metrics['throughput'][config_name] = {}
            key_metrics['latency'][config_name] = {}
            key_metrics['memory'][config_name] = {}
            key_metrics['efficiency'][config_name] = {}
            
            for scenario_name, scenario_data in config_data.items():
                # Extract throughput metrics
                throughput_metrics = self._extract_throughput_metrics(scenario_data)
                key_metrics['throughput'][config_name][scenario_name] = throughput_metrics
                
                # Extract latency metrics
                latency_metrics = self._extract_latency_metrics(scenario_data)
                key_metrics['latency'][config_name][scenario_name] = latency_metrics
                
                # Extract memory metrics
                memory_metrics = self._extract_memory_metrics(scenario_data)
                key_metrics['memory'][config_name][scenario_name] = memory_metrics
                
                # Extract efficiency metrics
                efficiency_metrics = self._extract_efficiency_metrics(scenario_data)
                key_metrics['efficiency'][config_name][scenario_name] = efficiency_metrics
        
        self.analysis_results['key_metrics'] = key_metrics
        return key_metrics
    
    def _extract_throughput_metrics(self, scenario_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract throughput metrics from scenario data."""
        metrics = {}
        
        for csv_name, df in scenario_data['csv_files'].items():
            if df.empty:
                continue
                
            # Look for throughput-related columns
            throughput_cols = [col for col in df.columns if 'throughput' in col.lower() or 
                            'token' in col.lower() and 'sec' in col.lower() or
                            'request' in col.lower() and 'sec' in col.lower()]
            
            for col in throughput_cols:
                metrics[f'{csv_name}_{col}'] = df[col].mean() if not df[col].empty else 0
        
        return metrics
    
    def _extract_latency_metrics(self, scenario_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract latency metrics from scenario data."""
        metrics = {}
        
        for csv_name, df in scenario_data['csv_files'].items():
            if df.empty:
                continue
                
            # Look for latency-related columns
            latency_cols = [col for col in df.columns if 'latency' in col.lower() or
                          'p50' in col.lower() or 'p95' in col.lower() or 'p99' in col.lower() or
                          'time' in col.lower()]
            
            for col in latency_cols:
                metrics[f'{csv_name}_{col}'] = df[col].mean() if not df[col].empty else 0
        
        return metrics
    
    def _extract_memory_metrics(self, scenario_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract memory metrics from scenario data."""
        metrics = {}
        
        # Extract memory from GPU monitoring
        if 'gpu_monitoring' in scenario_data and not scenario_data['gpu_monitoring'].empty:
            gpu_df = scenario_data['gpu_monitoring']
            if 'memory_used_gb' in gpu_df.columns:
                metrics['peak_memory_gb'] = gpu_df['memory_used_gb'].max()
                metrics['avg_memory_gb'] = gpu_df['memory_used_gb'].mean()
        
        # Extract memory from system metrics
        system_metrics = scenario_data.get('system_metrics', {})
        if 'system_info' in system_metrics:
            gpu_memory_gb = system_metrics['system_info'].get('gpu_memory_gb', 0)
            if gpu_memory_gb > 0:
                metrics['gpu_memory_total_gb'] = gpu_memory_gb
        
        return metrics
    
    def _extract_efficiency_metrics(self, scenario_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract efficiency metrics from scenario data."""
        metrics = {}
        
        # Extract GPU utilization from monitoring
        if 'gpu_monitoring' in scenario_data and not scenario_data['gpu_monitoring'].empty:
            gpu_df = scenario_data['gpu_monitoring']
            if 'gpu_utilization' in gpu_df.columns:
                metrics['avg_gpu_utilization'] = gpu_df['gpu_utilization'].mean()
                metrics['peak_gpu_utilization'] = gpu_df['gpu_utilization'].max()
        
        return metrics
    
    def compare_engines(self) -> Dict[str, Any]:
        """Compare performance between engines."""
        print("Comparing engines...")
        
        if 'key_metrics' not in self.analysis_results:
            self.extract_key_metrics()
        
        key_metrics = self.analysis_results['key_metrics']
        comparisons = {}
        
        # Get engine configurations
        configs = list(key_metrics['throughput'].keys())
        
        if 'turbomind_baseline' in configs and 'drift_baseline' in configs:
            comparisons['baseline_comparison'] = self._compare_configs(
                key_metrics, 'turbomind_baseline', 'drift_baseline'
            )
        
        if 'turbomind_baseline' in configs and 'drift_optimized' in configs:
            comparisons['optimized_comparison'] = self._compare_configs(
                key_metrics, 'turbomind_baseline', 'drift_optimized'
            )
        
        if 'drift_baseline' in configs and 'drift_optimized' in configs:
            comparisons['drift_optimization_impact'] = self._compare_configs(
                key_metrics, 'drift_baseline', 'drift_optimized'
            )
        
        self.analysis_results['comparisons'] = comparisons
        return comparisons
    
    def _compare_configs(self, key_metrics: Dict[str, Any], config1: str, config2: str) -> Dict[str, Any]:
        """Compare two configurations."""
        comparison = {
            'config1': config1,
            'config2': config2,
            'throughput_improvement': {},
            'latency_improvement': {},
            'memory_comparison': {},
            'efficiency_comparison': {}
        }
        
        # Compare throughput
        for scenario in key_metrics['throughput'][config1]:
            if scenario in key_metrics['throughput'][config2]:
                throughput1 = sum(key_metrics['throughput'][config1][scenario].values())
                throughput2 = sum(key_metrics['throughput'][config2][scenario].values())
                
                if throughput1 > 0:
                    improvement = ((throughput2 - throughput1) / throughput1) * 100
                    comparison['throughput_improvement'][scenario] = improvement
        
        # Compare latency
        for scenario in key_metrics['latency'][config1]:
            if scenario in key_metrics['latency'][config2]:
                latency1 = sum(key_metrics['latency'][config1][scenario].values())
                latency2 = sum(key_metrics['latency'][config2][scenario].values())
                
                if latency1 > 0:
                    improvement = ((latency1 - latency2) / latency1) * 100
                    comparison['latency_improvement'][scenario] = improvement
        
        # Compare memory
        for scenario in key_metrics['memory'][config1]:
            if scenario in key_metrics['memory'][config2]:
                memory1 = sum(key_metrics['memory'][config1][scenario].values())
                memory2 = sum(key_metrics['memory'][config2][scenario].values())
                
                if memory1 > 0:
                    change = ((memory2 - memory1) / memory1) * 100
                    comparison['memory_comparison'][scenario] = change
        
        return comparison
    
    def generate_visualizations(self, output_dir: str) -> List[str]:
        """Generate performance comparison charts."""
        print("Generating visualizations...")
        
        os.makedirs(output_dir, exist_ok=True)
        chart_files = []
        
        if 'key_metrics' not in self.analysis_results:
            self.extract_key_metrics()
        
        key_metrics = self.analysis_results['key_metrics']
        
        # 1. Throughput comparison chart
        throughput_chart = self._create_throughput_chart(key_metrics, output_dir)
        if throughput_chart:
            chart_files.append(throughput_chart)
        
        # 2. Latency comparison chart
        latency_chart = self._create_latency_chart(key_metrics, output_dir)
        if latency_chart:
            chart_files.append(latency_chart)
        
        # 3. Memory usage chart
        memory_chart = self._create_memory_chart(key_metrics, output_dir)
        if memory_chart:
            chart_files.append(memory_chart)
        
        # 4. Overall performance radar chart
        radar_chart = self._create_radar_chart(key_metrics, output_dir)
        if radar_chart:
            chart_files.append(radar_chart)
        
        return chart_files
    
    def _create_throughput_chart(self, key_metrics: Dict[str, Any], output_dir: str) -> Optional[str]:
        """Create throughput comparison chart."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Throughput Comparison: TurboMind vs DriftEngine', fontsize=16)
            
            scenarios = ['baseline', 'throughput', 'latency', 'memory']
            metrics_data = key_metrics['throughput']
            
            for i, scenario in enumerate(scenarios):
                ax = axes[i//2, i%2]
                
                data_for_scenario = {}
                for config, scenario_data in metrics_data.items():
                    if scenario in scenario_data:
                        # Sum all throughput metrics for this scenario
                        total_throughput = sum(v for v in scenario_data[scenario].values() if v > 0)
                        data_for_scenario[config] = total_throughput
                
                if data_for_scenario:
                    configs = list(data_for_scenario.keys())
                    values = list(data_for_scenario.values())
                    
                    bars = ax.bar(configs, values)
                    ax.set_title(f'{scenario.capitalize()} Scenario')
                    ax.set_ylabel('Tokens/sec')
                    ax.tick_params(axis='x', rotation=45)
                    
                    # Add value labels on bars
                    for bar, value in zip(bars, values):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{value:.1f}', ha='center', va='bottom')
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            
            plt.tight_layout()
            chart_file = os.path.join(output_dir, 'throughput_comparison.png')
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            return chart_file
            
        except Exception as e:
            print(f"Warning: Failed to create throughput chart: {e}")
            return None
    
    def _create_latency_chart(self, key_metrics: Dict[str, Any], output_dir: str) -> Optional[str]:
        """Create latency comparison chart."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Latency Comparison: TurboMind vs DriftEngine', fontsize=16)
            
            scenarios = ['baseline', 'throughput', 'latency', 'memory']
            metrics_data = key_metrics['latency']
            
            for i, scenario in enumerate(scenarios):
                ax = axes[i//2, i%2]
                
                data_for_scenario = {}
                for config, scenario_data in metrics_data.items():
                    if scenario in scenario_data:
                        # Average all latency metrics for this scenario
                        latency_values = [v for v in scenario_data[scenario].values() if v > 0]
                        avg_latency = np.mean(latency_values) if latency_values else 0
                        data_for_scenario[config] = avg_latency
                
                if data_for_scenario:
                    configs = list(data_for_scenario.keys())
                    values = list(data_for_scenario.values())
                    
                    bars = ax.bar(configs, values)
                    ax.set_title(f'{scenario.capitalize()} Scenario')
                    ax.set_ylabel('Latency (ms)')
                    ax.tick_params(axis='x', rotation=45)
                    
                    # Add value labels on bars
                    for bar, value in zip(bars, values):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{value:.1f}', ha='center', va='bottom')
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            
            plt.tight_layout()
            chart_file = os.path.join(output_dir, 'latency_comparison.png')
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            return chart_file
            
        except Exception as e:
            print(f"Warning: Failed to create latency chart: {e}")
            return None
    
    def _create_memory_chart(self, key_metrics: Dict[str, Any], output_dir: str) -> Optional[str]:
        """Create memory usage comparison chart."""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('Memory Usage Comparison: TurboMind vs DriftEngine', fontsize=16)
            
            metrics_data = key_metrics['memory']
            
            # Peak memory chart
            ax1 = axes[0]
            peak_memory = {}
            for config, scenario_data in metrics_data.items():
                peak_values = []
                for scenario, metrics in scenario_data.items():
                    if 'peak_memory_gb' in metrics:
                        peak_values.append(metrics['peak_memory_gb'])
                if peak_values:
                    peak_memory[config] = max(peak_values)
            
            if peak_memory:
                configs = list(peak_memory.keys())
                values = list(peak_memory.values())
                bars = ax1.bar(configs, values)
                ax1.set_title('Peak Memory Usage')
                ax1.set_ylabel('Memory (GB)')
                ax1.tick_params(axis='x', rotation=45)
                
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.1f}', ha='center', va='bottom')
            
            # Average memory chart
            ax2 = axes[1]
            avg_memory = {}
            for config, scenario_data in metrics_data.items():
                avg_values = []
                for scenario, metrics in scenario_data.items():
                    if 'avg_memory_gb' in metrics:
                        avg_values.append(metrics['avg_memory_gb'])
                if avg_values:
                    avg_memory[config] = np.mean(avg_values)
            
            if avg_memory:
                configs = list(avg_memory.keys())
                values = list(avg_memory.values())
                bars = ax2.bar(configs, values)
                ax2.set_title('Average Memory Usage')
                ax2.set_ylabel('Memory (GB)')
                ax2.tick_params(axis='x', rotation=45)
                
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.1f}', ha='center', va='bottom')
            
            plt.tight_layout()
            chart_file = os.path.join(output_dir, 'memory_comparison.png')
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            return chart_file
            
        except Exception as e:
            print(f"Warning: Failed to create memory chart: {e}")
            return None
    
    def _create_radar_chart(self, key_metrics: Dict[str, Any], output_dir: str) -> Optional[str]:
        """Create overall performance radar chart."""
        try:
            from math import pi
            
            # Prepare data for radar chart
            categories = ['Throughput', 'Latency', 'Memory Efficiency', 'GPU Utilization']
            
            # Normalize metrics to 0-100 scale for radar chart
            configs = []
            values = []
            
            for config in key_metrics['throughput'].keys():
                config_scores = []
                
                # Throughput score (normalized)
                throughput_values = []
                for scenario_data in key_metrics['throughput'][config].values():
                    throughput_values.extend([v for v in scenario_data.values() if v > 0])
                throughput_score = min(100, np.mean(throughput_values) / 10) if throughput_values else 0
                config_scores.append(throughput_score)
                
                # Latency score (inverse, normalized)
                latency_values = []
                for scenario_data in key_metrics['latency'][config].values():
                    latency_values.extend([v for v in scenario_data.values() if v > 0])
                latency_score = max(0, 100 - np.mean(latency_values) / 10) if latency_values else 0
                config_scores.append(latency_score)
                
                # Memory efficiency score
                memory_values = []
                for scenario_data in key_metrics['memory'][config].values():
                    memory_values.extend([v for v in scenario_data.values() if v > 0])
                memory_score = max(0, 100 - np.mean(memory_values)) if memory_values else 0
                config_scores.append(memory_score)
                
                # GPU utilization score
                eff_values = []
                for scenario_data in key_metrics['efficiency'][config].values():
                    eff_values.extend([v for v in scenario_data.values() if v > 0])
                gpu_score = np.mean(eff_values) if eff_values else 0
                config_scores.append(gpu_score)
                
                configs.append(config)
                values.append(config_scores)
            
            if not values:
                return None
            
            # Create radar chart
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
            angles += angles[:1]  # Complete the circle
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(configs)))
            
            for i, (config, config_values) in enumerate(zip(configs, values)):
                config_values += config_values[:1]  # Complete the circle
                ax.plot(angles, config_values, 'o-', linewidth=2, label=config, color=colors[i])
                ax.fill(angles, config_values, alpha=0.25, color=colors[i])
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_ylim(0, 100)
            ax.set_title('Overall Performance Comparison', size=16, pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
            
            plt.tight_layout()
            chart_file = os.path.join(output_dir, 'performance_radar.png')
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            return chart_file
            
        except Exception as e:
            print(f"Warning: Failed to create radar chart: {e}")
            return None
    
    def generate_report(self, output_dir: str) -> str:
        """Generate comprehensive analysis report."""
        print("Generating comprehensive report...")
        
        if 'comparisons' not in self.analysis_results:
            self.compare_engines()
        
        os.makedirs(output_dir, exist_ok=True)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': self._generate_summary(),
            'detailed_metrics': self.analysis_results.get('key_metrics', {}),
            'comparisons': self.analysis_results.get('comparisons', {}),
            'recommendations': self._generate_recommendations()
        }
        
        report_file = os.path.join(output_dir, 'performance_analysis_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate HTML report
        html_file = os.path.join(output_dir, 'performance_analysis_report.html')
        self._generate_html_report(report, html_file)
        
        return report_file
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate executive summary."""
        summary = {
            'engines_tested': list(self.results.keys()),
            'scenarios_tested': set(),
            'key_findings': []
        }
        
        # Collect all scenarios
        for config_data in self.results.values():
            summary['scenarios_tested'].update(config_data.keys())
        
        summary['scenarios_tested'] = list(summary['scenarios_tested'])
        
        # Generate key findings
        comparisons = self.analysis_results.get('comparisons', {})
        
        if 'baseline_comparison' in comparisons:
            baseline_comp = comparisons['baseline_comparison']
            throughput_improvements = list(baseline_comp['throughput_improvement'].values())
            if throughput_improvements:
                avg_improvement = np.mean(throughput_improvements)
                if avg_improvement > 5:
                    summary['key_findings'].append(f"DriftEngine shows {avg_improvement:.1f}% average throughput improvement over TurboMind baseline")
                elif avg_improvement < -5:
                    summary['key_findings'].append(f"DriftEngine shows {abs(avg_improvement):.1f}% average throughput degradation vs TurboMind baseline")
                else:
                    summary['key_findings'].append("DriftEngine performance is comparable to TurboMind baseline")
        
        return summary
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        comparisons = self.analysis_results.get('comparisons', {})
        
        if 'drift_optimization_impact' in comparisons:
            opt_impact = comparisons['drift_optimization_impact']
            throughput_improvements = list(opt_impact['throughput_improvement'].values())
            
            if throughput_improvements:
                avg_improvement = np.mean(throughput_improvements)
                if avg_improvement > 10:
                    recommendations.append("DriftEngine optimizations provide significant performance benefits and should be enabled in production")
                elif avg_improvement > 5:
                    recommendations.append("DriftEngine optimizations provide moderate benefits and should be considered for performance-critical applications")
                else:
                    recommendations.append("DriftEngine optimizations show minimal impact and may not be worth the complexity")
        
        recommendations.append("Monitor GPU utilization and memory usage to identify bottlenecks")
        recommendations.append("Consider workload-specific tuning for optimal performance")
        recommendations.append("Implement continuous performance monitoring to detect regressions")
        
        return recommendations
    
    def _generate_html_report(self, report: Dict[str, Any], html_file: str):
        """Generate HTML report."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>TurboMind vs DriftEngine Performance Analysis</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; border-bottom: 2px solid #007acc; }}
        h2 {{ color: #555; }}
        .metric {{ margin: 10px 0; padding: 10px; background: #f5f5f5; border-radius: 5px; }}
        .recommendation {{ margin: 10px 0; padding: 10px; background: #e8f5e8; border-left: 4px solid #28a745; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>TurboMind vs DriftEngine Performance Analysis</h1>
    <p><strong>Generated:</strong> {report['timestamp']}</p>
    
    <h2>Executive Summary</h2>
    <div class="metric"><strong>Engines Tested:</strong> {', '.join(report['summary']['engines_tested'])}</div>
    <div class="metric"><strong>Scenarios Tested:</strong> {', '.join(report['summary']['scenarios_tested'])}</div>
    
    <h2>Key Findings</h2>
    {"".join([f"<div class='metric'>• {finding}</div>" for finding in report['summary']['key_findings']])}
    
    <h2>Recommendations</h2>
    {"".join([f"<div class='recommendation'>{rec}</div>" for rec in report['recommendations']])}
    
    <h2>Detailed Metrics</h2>
    <p>Detailed performance metrics and comparisons are available in the JSON report file.</p>
    
</body>
</html>
"""
        
        with open(html_file, 'w') as f:
            f.write(html_content)


def main():
    parser = argparse.ArgumentParser(description='Analyze TurboMind vs DriftEngine benchmark results')
    parser.add_argument('--results-dir', type=str, required=True,
                      help='Directory containing benchmark results')
    parser.add_argument('--output-dir', type=str, default='./analysis_output',
                      help='Directory to save analysis results')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = BenchmarkAnalyzer(args.results_dir)
    
    # Load and analyze results
    analyzer.load_results()
    analyzer.extract_key_metrics()
    analyzer.compare_engines()
    
    # Generate visualizations
    chart_files = analyzer.generate_visualizations(args.output_dir)
    print(f"Generated {len(chart_files)} charts in {args.output_dir}")
    
    # Generate comprehensive report
    report_file = analyzer.generate_report(args.output_dir)
    print(f"Generated comprehensive report: {report_file}")
    
    print(f"\nAnalysis complete! Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()