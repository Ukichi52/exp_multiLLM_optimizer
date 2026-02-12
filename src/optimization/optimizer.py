"""Single item optimizer"""
from typing import Dict, Tuple
from ..models.target_model import TargetModel
from ..models.judge_model import JudgeModel
from ..models.mutator_model import MutatorModel
from ..models.analyzer_model import AnalyzerModel
from ..retrieval.clip_retriever import CLIPRetriever
from ..storage.success_db import SuccessDB

class ItemOptimizer:
    """Optimizer for single item"""
    
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
    
    def optimize_step(self, item: Dict) -> Tuple[Dict, Dict]:
        """
        Execute one optimization step
        
        Args:
            item: Current item state
            
        Returns:
            (updated_item, mutation_entry)
        """
        if item.get('AS') == 1:
            return item, None
        
        # Find similar successful case
        similar_case = self.retriever.find_most_similar(
            query_image_path=item['image_path'],
            candidate_cases=self.success_db.load_all()
        )
        
        # Generate new query
        if similar_case:
            new_query = self.mutator.rewrite_with_reference(
                current_query=item['final_query'],
                reference_query=similar_case['successful_query'],
                image_path=item['image_path']
            )
            strategy = f"reference_{similar_case['image_id']}"
        else:
            last_response = item.get('last_response', '')
            failure_reason = self.analyzer.analyze_failure(
                query=item['final_query'],
                response=last_response,
                image_path=item['image_path']
            )
            
            new_query = self.mutator.rewrite_with_analysis(
                current_query=item['final_query'],
                failure_reason=failure_reason,
                image_path=item['image_path']
            )
            strategy = "failure_analysis"
        
        # Test new query
        response = self.target.call(
            image_path=item['image_path'],
            query=new_query
        )
        
        score = self.judge.score(response)
        
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
            'strategy': strategy
        }
        
        return item, mutation_entry