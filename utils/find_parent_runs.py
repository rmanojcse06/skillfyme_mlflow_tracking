"""
Utility to Find Parent Runs with Model Artifacts (UPDATED)
Now handles custom artifact paths like 'best_model'

Usage:
    # Check both 'model' and 'best_model' paths (default)
    python find_parent_runs.py
    
    # Check only 'best_model' path
    python find_parent_runs.py --artifact-path best_model
    
    # Check multiple custom paths
    python find_parent_runs.py --artifact-path best_model --artifact-path models
"""

import mlflow
from mlflow.tracking import MlflowClient
import json
from pathlib import Path
import argparse


def get_parent_run_id(run_id):
    """Get the parent run ID for a child run"""
    client = MlflowClient()
    run = client.get_run(run_id)
    return run.data.tags.get("mlflow.parentRunId")


def has_model_artifact(run_id, artifact_paths=None):
    """
    Check if a run has a model artifact in any of the specified paths
    
    Args:
        run_id (str): Run ID to check
        artifact_paths (list): List of paths to check (default: ['model', 'best_model'])
    
    Returns:
        tuple: (has_artifact, found_path) - True/False and path where found
    """
    if artifact_paths is None:
        artifact_paths = ['model', 'best_model']
    
    client = MlflowClient()
    
    for artifact_path in artifact_paths:
        try:
            artifacts = client.list_artifacts(run_id, artifact_path)
            if len(artifacts) > 0:
                return True, artifact_path
        except Exception:
            continue
    
    return False, None


def find_registrable_run(run_id, artifact_paths=None):
    """
    Find the run that actually has the model artifact
    
    Args:
        run_id (str): Starting run ID
        artifact_paths (list): Paths to check
    
    Returns:
        tuple: (registrable_run_id, artifact_path) or (None, None)
    """
    client = MlflowClient()
    
    # Check if current run has model
    has_artifact, found_path = has_model_artifact(run_id, artifact_paths)
    if has_artifact:
        print(f"✅ Run {run_id[:16]}... has model artifact at: {found_path}/")
        return run_id, found_path
    
    # Check if it's a child run
    parent_id = get_parent_run_id(run_id)
    
    if parent_id:
        print(f"⚠️  Run {run_id[:16]}... is a child run (no artifact)")
        print(f"   Checking parent: {parent_id[:16]}...")
        
        has_artifact, found_path = has_model_artifact(parent_id, artifact_paths)
        if has_artifact:
            print(f"   ✅ Parent run has model artifact at: {found_path}/")
            return parent_id, found_path
        else:
            print(f"   ❌ Parent run also has no model artifact")
            print(f"   Checked paths: {artifact_paths}")
            return None, None
    else:
        print(f"❌ Run {run_id[:16]}... has no model artifact and no parent")
        print(f"   Checked paths: {artifact_paths}")
        return None, None


def generate_corrected_best_runs_json(
    input_json="session5_best_runs.json",
    output_json="session5_best_runs_corrected.json",
    artifact_paths=None
):
    """
    Generate corrected JSON file with parent run IDs
    
    Args:
        input_json (str): Original JSON file
        output_json (str): Corrected JSON file to create
        artifact_paths (list): Paths to check for model artifacts
    """
    
    if artifact_paths is None:
        artifact_paths = ['model', 'best_model']
    
    print("=" * 70)
    print("🔧 CORRECTING SESSION 5 BEST RUNS FOR MODEL REGISTRATION")
    print("=" * 70)
    print(f"\n📁 Will check for models in paths: {artifact_paths}")
    
    # Load original JSON
    input_path = Path(input_json)
    if not input_path.exists():
        print(f"\n❌ Error: {input_json} not found!")
        print(f"   Please generate it from Session 5 first.")
        return False
    
    print(f"\n📂 Loading: {input_json}")
    with open(input_path) as f:
        best_runs = json.load(f)
    
    best_runs = best_runs['top_runs']
    
    print(f"   Found {len(best_runs)} runs")
    
    # Process each run
    corrected_runs = []
    
    for i, run_info in enumerate(best_runs, 1):
        print(f"\n{'='*70}")
        print(f"Processing Rank {run_info.get('rank', i)}:")
        print(f"{'='*70}")
        
        original_run_id = run_info.get('run_id')
        print(f"Original run_id: {original_run_id[:16]}...")
        
        # Find registrable run (with model artifact)
        registrable_run_id, artifact_path = find_registrable_run(original_run_id, artifact_paths)
        
        if registrable_run_id:
            # Update with corrected run ID
            corrected_info = run_info.copy()
            corrected_info['run_id'] = registrable_run_id
            corrected_info['artifact_path'] = artifact_path  # Save where we found it!
            
            # Add metadata about correction
            if registrable_run_id != original_run_id:
                corrected_info['original_run_id'] = original_run_id
                corrected_info['corrected'] = True
                corrected_info['correction_reason'] = 'Parent run contains model artifact'
                print(f"✅ Corrected to parent: {registrable_run_id[:16]}...")
                print(f"   Found model at: {artifact_path}/")
            else:
                corrected_info['corrected'] = False
                print(f"✅ Run already had model artifact at: {artifact_path}/")
            
            corrected_runs.append(corrected_info)
        else:
            print(f"⚠️  Could not find model artifact for Rank {i}")
            print(f"   Checked paths: {artifact_paths}")
            print(f"   💡 Check your Session 5 code:")
            print(f"      mlflow.sklearn.log_model(model, 'YOUR_PATH')")
            print(f"   Skipping this run...")
    
    # Save corrected JSON
    if corrected_runs:
        print(f"\n{'='*70}")
        print(f"💾 Saving corrected runs to: {output_json}")
        print(f"{'='*70}")
        
        with open(output_json, 'w') as f:
            json.dump(corrected_runs, f, indent=2)
        
        print(f"\n✅ Saved {len(corrected_runs)} corrected runs")
        
        # Summary
        corrected_count = sum(1 for r in corrected_runs if r.get('corrected', False))
        print(f"\n📊 Summary:")
        print(f"   Total runs: {len(corrected_runs)}")
        print(f"   Corrected to parent: {corrected_count}")
        print(f"   Already had artifacts: {len(corrected_runs) - corrected_count}")
        
        # Show artifact paths found
        artifact_paths_found = {}
        for r in corrected_runs:
            path = r.get('artifact_path', 'unknown')
            artifact_paths_found[path] = artifact_paths_found.get(path, 0) + 1
        
        print(f"\n📁 Artifact paths found:")
        for path, count in artifact_paths_found.items():
            print(f"   {path}/: {count} model(s)")
        
        return True
    else:
        print(f"\n❌ No valid runs found with model artifacts!")
        print(f"   Checked paths: {artifact_paths}")
        print(f"\n💡 Troubleshooting:")
        print(f"   1. Check your Session 5 code for where models were logged")
        print(f"   2. Use --artifact-path to specify custom path:")
        print(f"      python find_parent_runs.py --artifact-path YOUR_PATH")
        print(f"   3. Verify runs in MLflow UI")
        return False


def verify_run_structure(run_id, artifact_paths=None):
    """
    Verify and display run structure for debugging
    
    Args:
        run_id (str): Run ID to verify
        artifact_paths (list): Paths to check
    """
    if artifact_paths is None:
        artifact_paths = ['model', 'best_model']
    
    client = MlflowClient()
    
    print("\n" + "=" * 70)
    print("🔍 RUN STRUCTURE VERIFICATION")
    print("=" * 70)
    
    run = client.get_run(run_id)
    
    print(f"\nRun ID: {run_id}")
    print(f"Run Name: {run.data.tags.get('mlflow.runName', 'N/A')}")
    
    # Check if child run
    parent_id = run.data.tags.get("mlflow.parentRunId")
    if parent_id:
        print(f"\n⚠️  This is a CHILD run")
        print(f"   Parent ID: {parent_id}")
    else:
        print(f"\n✅ This is a PARENT run (or standalone)")
    
    # List ALL artifacts
    print(f"\n📦 All Artifacts:")
    try:
        all_artifacts = client.list_artifacts(run_id)
        if all_artifacts:
            for artifact in all_artifacts:
                print(f"   - {artifact.path}")
        else:
            print(f"   ❌ No artifacts found at root level")
    except Exception as e:
        print(f"   ❌ Error listing artifacts: {e}")
    
    # Check specific paths
    print(f"\n🔍 Checking specific model paths:")
    for path in artifact_paths:
        try:
            artifacts = client.list_artifacts(run_id, path)
            if artifacts:
                print(f"   ✅ {path}/ EXISTS ({len(artifacts)} items)")
                for artifact in artifacts[:3]:  # Show first 3
                    print(f"      - {artifact.path}")
            else:
                print(f"   ❌ {path}/ NOT FOUND")
        except Exception:
            print(f"   ❌ {path}/ NOT FOUND")
    
    # Metrics and params
    print(f"\n📊 Metrics:")
    for key, value in list(run.data.metrics.items())[:5]:
        print(f"   {key}: {value:.4f}")
    
    print(f"\n⚙️  Parameters:")
    for key, value in list(run.data.params.items())[:5]:
        print(f"   {key}: {value}")
    
    print("\n" + "=" * 70)
    
    # Recommendation
    has_artifact, found_path = has_model_artifact(run_id, artifact_paths)
    if has_artifact:
        print(f"\n✅ This run can be registered!")
        print(f"   Model found at: {found_path}/")
    else:
        print(f"\n⚠️  This run cannot be registered directly")
        if parent_id:
            print(f"   Try checking parent run: {parent_id}")
        else:
            print(f"   No model artifact found in checked paths")
            print(f"   Checked: {artifact_paths}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fix Session 5 best runs for model registration (supports custom artifact paths)"
    )
    
    parser.add_argument(
        "--input-json",
        type=str,
        default="session5_best_runs.json",
        help="Original best runs JSON"
    )
    
    parser.add_argument(
        "--output-json",
        type=str,
        default="session5_best_runs_corrected.json",
        help="Corrected JSON output file"
    )
    
    parser.add_argument(
        "--artifact-path",
        type=str,
        action='append',
        help="Artifact path(s) to check (can specify multiple times). Default: model, best_model"
    )
    
    parser.add_argument(
        "--verify-run",
        type=str,
        help="Verify specific run structure (for debugging)"
    )
    
    args = parser.parse_args()
    
    # Handle artifact paths
    artifact_paths = args.artifact_path if args.artifact_path else ['model', 'best_model']
    
    if args.verify_run:
        # Verify specific run
        verify_run_structure(args.verify_run, artifact_paths)
    else:
        # Generate corrected JSON
        success = generate_corrected_best_runs_json(
            input_json=args.input_json,
            output_json=args.output_json,
            artifact_paths=artifact_paths
        )
        
        if success:
            print("\n🎉 Correction complete! Ready for model registration!")
        else:
            print("\n❌ Correction failed. Please check your Session 5 data.")
            print("\n💡 Try:")
            print(f"   python find_parent_runs.py --verify-run YOUR_RUN_ID")
