"""Single item optimizer"""
from typing import Dict, Tuple, Optional
from ..models.target_model import TargetModel
from ..models.judge_model import JudgeModel
from ..models.mutator_model import MutatorModel
from ..models.analyzer_model import AnalyzerModel
from ..retrieval.clip_retriever import CLIPRetriever
from ..storage.success_db import SuccessDB

class ItemOptimizer:
    """Optimizer for single item with score-guided optimization"""
    
    def __init__(
        self,
        target_model: TargetModel,
        judge_model: JudgeModel,
        mutator_model: MutatorModel,
        analyzer_model: AnalyzerModel,
        retriever: CLIPRetriever,
        success_db: SuccessDB,
        threshold: float = 0.7
    ):
        self.target = target_model
        self.judge = judge_model
        self.mutator = mutator_model
        self.analyzer = analyzer_model
        self.retriever = retriever
        self.success_db = success_db
        self.threshold = threshold
    
    def optimize_step(self, item: Dict) -> Tuple[Dict, Optional[Dict]]:
        """
        Execute one optimization step with score guidance
        
        Args:
            item: Current item state
            
        Returns:
            (updated_item, mutation_entry)
        """
        if item.get('AS') == 1:
            return item, None
        
        # Get current best score
        current_score = item.get('best_score', 0.0)
        
        # Find similar successful case
        similar_case = self.retriever.find_most_similar(
            query_image_path=item['image_path'],
            candidate_cases=self.success_db.load_all()
        )
        
        failure_analysis = None
        
        # Generate new query
        if similar_case:
            # Reference-based rewriting with score guidance
            new_query = self.mutator.rewrite_with_reference(
                current_query=item['final_query'],
                reference_query=similar_case['successful_query'],
                image_path=item['image_path'],
                current_score=current_score,
                target_score=self.threshold
            )
            strategy = f"reference_{similar_case['image_id']}"
        else:
            # Failure analysis path with score guidance
            last_response = item.get('last_response', '')
            
            # Analyze failure with score metrics
            failure_analysis = self.analyzer.analyze_failure(
                query=item['final_query'],
                response=last_response,
                image_path=item['image_path'],
                current_score=current_score,
                best_score=item.get('best_score', 0.0),
                threshold=self.threshold
            )
            
            # Rewrite with score target
            new_query = self.mutator.rewrite_with_analysis(
                current_query=item['final_query'],
                failure_reason=failure_analysis,
                image_path=item['image_path'],
                current_score=current_score,
                target_score=self.threshold
            )
            strategy = "failure_analysis"
        
        # Test new query
        response = self.target.call(
            image_path=item['image_path'],
            query=new_query
        )
        
        score = self.judge.score(response)
        
        # Calculate score improvement
        score_improvement = score - current_score
        previous_score = current_score
        
        # Update state
        item['total_iterations'] += 1
        item['last_response'] = response
        
        if score > item['best_score']:
            item['best_score'] = score
            item['final_query'] = new_query
        
        if score > self.threshold:
            item['AS'] = 1
            
            # Add to success database
            image_features = self.retriever.encode_image(item['image_path'])
            self.success_db.add_case(
                image_id=item['image_id'],
                image_path=item['image_path'],
                query=new_query,
                score=score,
                image_features=image_features.tolist()
            )
        
        mutation_entry = {
            'iteration': item['total_iterations'],
            'query': new_query,
            'response': response,
            'score': score,
            'previous_score': previous_score,
            'score_improvement': score_improvement,
            'gap_to_threshold': self.threshold - score,
            'strategy': strategy,
            'failure_analysis': failure_analysis
        }
        
        return item, mutation_entry
