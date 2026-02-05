"""
Analysis and visualization script for judge evaluation results.

This script provides utilities to:
- Load and analyze evaluation results
- Generate detailed breakdowns by failure type
- Compare resolved vs unresolved instances
- Export analysis reports
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter, defaultdict
import argparse


class ResultsAnalyzer:
    """Analyzer for judge evaluation results"""
    
    def __init__(self, results_path: str):
        self.results_path = Path(results_path)
        self.results = self._load_results()
        self.df = self._to_dataframe()
    
    def _load_results(self) -> List[Dict]:
        """Load results from JSONL file"""
        results = []
        with open(self.results_path, 'r') as f:
            for line in f:
                results.append(json.loads(line))
        print(f"Loaded {len(results)} evaluation results")
        return results
    
    def _to_dataframe(self) -> pd.DataFrame:
        """Convert results to a flat DataFrame for analysis"""
        rows = []
        
        for result in self.results:
            instance_id = result.get('instance_id')
            resolved = result.get('resolved')
            
            if 'error' in result:
                rows.append({
                    'instance_id': instance_id,
                    'resolved': resolved,
                    'has_error': True,
                    'error_type': result.get('error')
                })
                continue
            
            judge_eval = result.get('judge_evaluation', {})
            
            # Extract key fields
            outcome = judge_eval.get('outcome', {})
            patch_alignment = judge_eval.get('patch_alignment', {})
            primary_failure = judge_eval.get('primary_failure', {})
            quality_scores = judge_eval.get('quality_scores', {})
            
            row = {
                'instance_id': instance_id,
                'resolved': resolved,
                'has_error': False,
                
                # Outcome
                'outcome_status': outcome.get('status'),
                'final_test_state': outcome.get('final_test_state'),
                
                # Alignment
                'alignment_score': patch_alignment.get('alignment_score'),
                'accidental_pass_risk': patch_alignment.get('accidental_pass_risk'),
                'num_missing_requirements': len(patch_alignment.get('missing_requirements', [])),
                'num_extra_behaviors': len(patch_alignment.get('extra_behavior_changes', [])),
                
                # Primary failure
                'failure_class': primary_failure.get('class'),
                'failure_inferability': primary_failure.get('inferability'),
                'failure_reason_code': primary_failure.get('reason_code'),
                'failure_stage': primary_failure.get('stage'),
                
                # Quality scores
                'spec_alignment': quality_scores.get('spec_alignment'),
                'repo_exploration': quality_scores.get('repo_exploration'),
                'root_cause_quality': quality_scores.get('root_cause_quality'),
                'patch_correctness': quality_scores.get('patch_correctness'),
                'validation_rigor': quality_scores.get('validation_rigor'),
                'iteration_efficiency': quality_scores.get('iteration_efficiency'),
                
                # Secondary failures
                'num_secondary_failures': len(judge_eval.get('secondary_failures', [])),
            }
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def print_overview(self):
        """Print high-level overview statistics"""
        print("\n" + "="*80)
        print("EVALUATION RESULTS OVERVIEW")
        print("="*80)
        
        total = len(self.df)
        errors = self.df['has_error'].sum()
        valid = total - errors
        
        print(f"\nTotal instances: {total}")
        print(f"  Valid evaluations: {valid}")
        print(f"  Errors: {errors}")
        
        if errors > 0:
            print("\nError breakdown:")
            error_counts = self.df[self.df['has_error']]['error_type'].value_counts()
            for error_type, count in error_counts.items():
                print(f"  {error_type}: {count}")
        
        # Valid results only
        valid_df = self.df[~self.df['has_error']].copy()
        valid_df = valid_df.reset_index(drop=True)
        
        if len(valid_df) == 0:
            print("\nNo valid results to analyze.")
            return
        
        # Resolution status
        print(f"\nResolution status:")
        resolved_count = (valid_df['resolved'] == True).sum()
        unresolved_count = (valid_df['resolved'] == False).sum()
        print(f"  Resolved: {resolved_count}")
        print(f"  Unresolved: {unresolved_count}")
        
        # Outcome distribution
        print(f"\nOutcome distribution:")
        outcome_counts = valid_df['outcome_status'].value_counts()
        for outcome, count in outcome_counts.items():
            pct = 100 * count / len(valid_df)
            print(f"  {outcome}: {count} ({pct:.1f}%)")
        
        # Alignment scores
        print(f"\nAlignment scores (0-4):")
        alignment_stats = valid_df['alignment_score'].describe()
        print(f"  Mean: {alignment_stats['mean']:.2f}")
        print(f"  Median: {alignment_stats['50%']:.2f}")
        print(f"  Std: {alignment_stats['std']:.2f}")
        
        # Quality scores
        print(f"\nAverage quality scores (0-4):")
        for col in ['spec_alignment', 'repo_exploration', 'root_cause_quality',
                    'patch_correctness', 'validation_rigor', 'iteration_efficiency']:
            if col in valid_df.columns:
                mean_score = valid_df[col].mean()
                print(f"  {col}: {mean_score:.2f}")
    
    def analyze_failure_patterns(self):
        """Analyze failure patterns in detail"""
        valid_df = self.df[~self.df['has_error']].copy()
        valid_df = valid_df.reset_index(drop=True)
        
        if len(valid_df) == 0:
            return
        
        print("\n" + "="*80)
        print("FAILURE PATTERN ANALYSIS")
        print("="*80)
        
        # Failure class distribution
        print("\nPrimary failure class:")
        failure_class_counts = valid_df['failure_class'].value_counts()
        for fc, count in failure_class_counts.items():
            pct = 100 * count / len(valid_df)
            print(f"  {fc}: {count} ({pct:.1f}%)")
        
        # Failure stage distribution
        print("\nFailure stage:")
        stage_counts = valid_df['failure_stage'].value_counts()
        for stage, count in stage_counts.items():
            pct = 100 * count / len(valid_df)
            print(f"  {stage}: {count} ({pct:.1f}%)")
        
        # Top reason codes
        print("\nTop 10 failure reason codes:")
        reason_counts = valid_df['failure_reason_code'].value_counts().head(10)
        for reason, count in reason_counts.items():
            pct = 100 * count / len(valid_df)
            print(f"  {reason}: {count} ({pct:.1f}%)")
        
        # Inferability breakdown
        print("\nInferability distribution:")
        infer_counts = valid_df['failure_inferability'].value_counts()
        for infer, count in infer_counts.items():
            pct = 100 * count / len(valid_df)
            print(f"  {infer}: {count} ({pct:.1f}%)")
    
    def compare_resolved_vs_unresolved(self):
        """Compare outcomes between resolved and unresolved instances"""
        valid_df = self.df[~self.df['has_error']].copy()
        
        if len(valid_df) == 0:
            return
        
        print("\n" + "="*80)
        print("RESOLVED vs UNRESOLVED COMPARISON")
        print("="*80)
        
        # Check if 'resolved' column exists and has valid data
        if 'resolved' not in valid_df.columns:
            print("\nError: 'resolved' column not found in results")
            print(f"Available columns: {list(valid_df.columns)}")
            return
        
        # Reset index to avoid index mismatch issues
        valid_df = valid_df.reset_index(drop=True)
        
        # Filter by resolved status
        resolved_df = valid_df[valid_df['resolved'] == True].copy()
        unresolved_df = valid_df[valid_df['resolved'] == False].copy()
        
        print(f"\nResolved instances: {len(resolved_df)}")
        print(f"Unresolved instances: {len(unresolved_df)}")
        
        if len(resolved_df) == 0 or len(unresolved_df) == 0:
            print("\nNot enough data for comparison")
            return
        
        # Outcome distribution comparison
        print("\nOutcome distribution:")
        print("\n  Resolved:")
        for outcome, count in resolved_df['outcome_status'].value_counts().items():
            pct = 100 * count / len(resolved_df)
            print(f"    {outcome}: {count} ({pct:.1f}%)")
        
        print("\n  Unresolved:")
        for outcome, count in unresolved_df['outcome_status'].value_counts().items():
            pct = 100 * count / len(unresolved_df)
            print(f"    {outcome}: {count} ({pct:.1f}%)")
        
        # Alignment score comparison
        print("\nAlignment scores:")
        print(f"  Resolved - Mean: {resolved_df['alignment_score'].mean():.2f}, Median: {resolved_df['alignment_score'].median():.2f}")
        print(f"  Unresolved - Mean: {unresolved_df['alignment_score'].mean():.2f}, Median: {unresolved_df['alignment_score'].median():.2f}")
        
        # Quality score comparison
        print("\nQuality scores (Resolved vs Unresolved):")
        for col in ['spec_alignment', 'repo_exploration', 'root_cause_quality',
                    'patch_correctness', 'validation_rigor', 'iteration_efficiency']:
            if col in valid_df.columns:
                resolved_mean = resolved_df[col].mean()
                unresolved_mean = unresolved_df[col].mean()
                print(f"  {col}: {resolved_mean:.2f} vs {unresolved_mean:.2f}")
        
        # Failure class comparison
        print("\nFailure class (Resolved vs Unresolved):")
        resolved_fc = resolved_df['failure_class'].value_counts()
        unresolved_fc = unresolved_df['failure_class'].value_counts()
        
        all_classes = set(resolved_fc.index) | set(unresolved_fc.index)
        for fc in sorted(all_classes):
            r_count = resolved_fc.get(fc, 0)
            u_count = unresolved_fc.get(fc, 0)
            r_pct = 100 * r_count / len(resolved_df) if len(resolved_df) > 0 else 0
            u_pct = 100 * u_count / len(unresolved_df) if len(unresolved_df) > 0 else 0
            print(f"  {fc}: {r_count} ({r_pct:.1f}%) vs {u_count} ({u_pct:.1f}%)")
    
    def analyze_accidental_passes(self):
        """Analyze instances with potential accidental passes"""
        valid_df = self.df[~self.df['has_error']].copy()
        valid_df = valid_df.reset_index(drop=True)
        
        if len(valid_df) == 0:
            return
        
        print("\n" + "="*80)
        print("ACCIDENTAL PASS ANALYSIS")
        print("="*80)
        
        # Accidental pass risk distribution
        print("\nAccidental pass risk distribution:")
        risk_counts = valid_df['accidental_pass_risk'].value_counts()
        for risk, count in risk_counts.items():
            pct = 100 * count / len(valid_df)
            print(f"  {risk}: {count} ({pct:.1f}%)")
        
        # High risk instances
        high_risk = valid_df[valid_df['accidental_pass_risk'] == 'high'].copy()
        print(f"\nHigh-risk instances: {len(high_risk)}")
        
        if len(high_risk) > 0:
            print("\nHigh-risk outcome distribution:")
            for outcome, count in high_risk['outcome_status'].value_counts().items():
                pct = 100 * count / len(high_risk)
                print(f"  {outcome}: {count} ({pct:.1f}%)")
            
            print("\nHigh-risk instances by ID:")
            for instance_id in high_risk['instance_id'].head(20):
                print(f"  {instance_id}")
    
    def export_detailed_report(self, output_path: str):
        """Export detailed analysis to CSV"""
        self.df.to_csv(output_path, index=False)
        print(f"\nExported detailed results to {output_path}")
    
    def get_instance_details(self, instance_id: str) -> Dict:
        """Get full details for a specific instance"""
        for result in self.results:
            if result.get('instance_id') == instance_id:
                return result
        return None
    
    def print_instance_report(self, instance_id: str):
        """Print a detailed report for a specific instance"""
        result = self.get_instance_details(instance_id)
        
        if result is None:
            print(f"Instance {instance_id} not found")
            return
        
        print("\n" + "="*80)
        print(f"INSTANCE REPORT: {instance_id}")
        print("="*80)
        
        if 'error' in result:
            print(f"\nERROR: {result['error']}")
            return
        
        judge_eval = result.get('judge_evaluation', {})
        
        # Print narrative
        narrative = judge_eval.get('narrative', {})
        print("\n### Diagnosis")
        print(narrative.get('one_paragraph_diagnosis', 'N/A'))
        
        print("\n### Counterfactual Fix")
        print(narrative.get('counterfactual_fix', 'N/A'))
        
        # Print outcome
        outcome = judge_eval.get('outcome', {})
        print(f"\n### Outcome")
        print(f"Status: {outcome.get('status')}")
        print(f"Test State: {outcome.get('final_test_state')}")
        
        # Print intent
        intent = judge_eval.get('intent', {})
        print("\n### Inferred Intent")
        print("Requirements:")
        for req in intent.get('requirements', []):
            print(f"  - {req}")
        print("Edge Cases:")
        for ec in intent.get('edge_cases', []):
            print(f"  - {ec}")
        
        # Print alignment
        alignment = judge_eval.get('patch_alignment', {})
        print(f"\n### Patch Alignment")
        print(f"Score: {alignment.get('alignment_score')}/4")
        print(f"Accidental Pass Risk: {alignment.get('accidental_pass_risk')}")
        print(f"Notes: {alignment.get('notes')}")
        
        # Print primary failure
        primary = judge_eval.get('primary_failure', {})
        print(f"\n### Primary Failure")
        print(f"Class: {primary.get('class')}")
        print(f"Reason: {primary.get('reason_code')}")
        print(f"Stage: {primary.get('stage')}")
        print(f"Inferability: {primary.get('inferability')}")
        
        # Print quality scores
        scores = judge_eval.get('quality_scores', {})
        print(f"\n### Quality Scores (0-4)")
        for key, value in scores.items():
            print(f"  {key}: {value}")


def main():
    parser = argparse.ArgumentParser(description='Analyze judge evaluation results')
    parser.add_argument(
        '--results',
        required=True,
        help='Path to evaluation_results.jsonl file'
    )
    parser.add_argument(
        '--export-csv',
        help='Export detailed analysis to CSV file'
    )
    parser.add_argument(
        '--instance',
        help='Print detailed report for specific instance ID'
    )
    
    args = parser.parse_args()
    
    # Load and analyze
    analyzer = ResultsAnalyzer(args.results)
    
    if args.instance:
        analyzer.print_instance_report(args.instance)
    else:
        analyzer.print_overview()
        analyzer.analyze_failure_patterns()
        analyzer.compare_resolved_vs_unresolved()
        analyzer.analyze_accidental_passes()
    
    if args.export_csv:
        analyzer.export_detailed_report(args.export_csv)


if __name__ == '__main__':
    main()
