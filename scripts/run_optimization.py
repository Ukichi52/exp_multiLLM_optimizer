"""Main optimization script"""
import sys
sys.path.append('.')

import yaml
from src.models.target_model import TargetModel
from src.models.judge_model import JudgeModel
from src.models.mutator_model import MutatorModel
from src.models.analyzer_model import AnalyzerModel
from src.retrieval.clip_retriever import CLIPRetriever
from src.storage.index_manager import IndexManager
from src.storage.mutation_logger import MutationLogger
from src.storage.success_db import SuccessDB
from src.optimization.optimizer import ItemOptimizer
from src.optimization.convergence import ConvergenceDetector
from src.utils.metrics import calculate_asr

def main():
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize models
    target_model = TargetModel(config['models']['target'])
    judge_model = JudgeModel(config['models']['judge'])
    mutator_model = MutatorModel(config['models']['mutator'])
    analyzer_model = AnalyzerModel(config['models']['analyzer'])
    
    # Initialize CLIP
    retriever = CLIPRetriever(
        model_path=config['retrieval']['clip_model_path'],
        gpu_ids=config['retrieval']['gpu_ids']
    )
    
    # Initialize storage
    index_mgr = IndexManager(config['paths']['index'])
    mutation_logger = MutationLogger(config['paths']['mutations'])
    success_db = SuccessDB(config['paths']['success_db'])
    
    # Initialize optimizer
    optimizer = ItemOptimizer(
        target_model=target_model,
        judge_model=judge_model,
        mutator_model=mutator_model,
        analyzer_model=analyzer_model,
        retriever=retriever,
        success_db=success_db,
        threshold=config['optimization']['jailbreak_threshold']
    )
    
    # Initialize convergence detector
    convergence = ConvergenceDetector(
        window=config['optimization']['convergence_window'],
        epsilon=config['optimization']['convergence_epsilon']
    )
    
    # Main loop
    max_iters = config['optimization']['max_global_iterations']
    
    for global_iter in range(max_iters):
        print(f"\n{'='*60}")
        print(f"Global Iteration {global_iter + 1}/{max_iters}")
        print(f"{'='*60}")
        
        # Get pending items
        pending_items = index_mgr.get_pending_items()
        
        if not pending_items:
            print("All items successful, terminating early")
            break
        
        print(f"Optimizing {len(pending_items)} pending items...")
        
        # Optimize each item
        for idx, item in enumerate(pending_items):
            print(f"  [{idx+1}/{len(pending_items)}] {item['image_id']}")
            
            updated_item, mutation_entry = optimizer.optimize_step(item)
            
            if mutation_entry:
                # Update index
                index_mgr.update_entry(
                    image_id=item['image_id'],
                    updates=updated_item
                )
                
                # Log mutation
                mutation_logger.log_mutation(
                    image_id=item['image_id'],
                    **mutation_entry
                )
                
                print(f"    Iter {mutation_entry['iteration']}: "
                      f"Score {mutation_entry['score']:.3f}, "
                      f"AS={updated_item['AS']}")
        
        # Calculate ASR
        all_items = index_mgr.load_all()
        current_asr = calculate_asr(all_items)
        print(f"\nCurrent ASR: {current_asr:.2%}")
        
        # Check convergence
        if convergence.check(current_asr):
            print(f"\nASR converged at {current_asr:.2%}, terminating")
            break
    
    # Final report
    print(f"\n{'='*60}")
    print("Optimization Complete")
    print(f"{'='*60}")
    final_asr = calculate_asr(index_mgr.load_all())
    print(f"Final ASR: {final_asr:.2%}")
    print(f"Total iterations: {global_iter + 1}")

if __name__ == '__main__':
    main()