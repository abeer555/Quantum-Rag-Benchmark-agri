"""
Human Evaluation Framework

This module provides tools for conducting human evaluation of RAG systems,
including relevance scoring, factual accuracy assessment, and comparative testing.
"""

import json
import random
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import csv


@dataclass
class EvaluationQuestion:
    """Container for evaluation questions."""
    query: str
    context: str
    answer: str
    system_name: str
    question_id: str


@dataclass
class RelevanceRating:
    """Container for relevance ratings."""
    question_id: str
    rating: int  # 1-5 scale
    explanation: str
    evaluator_id: str
    timestamp: str


@dataclass
class FactualAccuracyRating:
    """Container for factual accuracy ratings."""
    question_id: str
    is_accurate: bool
    error_type: Optional[str]  # 'factual_error', 'hallucination', 'misleading', etc.
    explanation: str
    evaluator_id: str
    timestamp: str


@dataclass
class PreferenceRating:
    """Container for comparative preference ratings."""
    question_id: str
    preferred_system: str
    confidence: int  # 1-5 scale
    reason: str
    evaluator_id: str
    timestamp: str


class RelevanceScorer:
    """
    Tool for scoring answer relevance on a 1-5 scale.
    """
    
    def __init__(self):
        self.rating_scale = {
            1: "Not relevant - Answer does not address the query at all",
            2: "Slightly relevant - Answer partially addresses the query but misses key points",
            3: "Moderately relevant - Answer addresses the query but lacks detail or has minor issues",
            4: "Highly relevant - Answer addresses the query well with good detail",
            5: "Perfectly relevant - Answer completely and accurately addresses all aspects of the query"
        }
    
    def create_evaluation_questions(
        self,
        queries: List[str],
        contexts: List[List[str]],
        answers: List[str],
        system_name: str
    ) -> List[EvaluationQuestion]:
        """
        Create evaluation questions for relevance scoring.
        
        Args:
            queries: List of queries
            contexts: List of context lists for each query
            answers: List of generated answers
            system_name: Name of the system that generated answers
            
        Returns:
            List of evaluation questions
        """
        questions = []
        
        for i, (query, context_list, answer) in enumerate(zip(queries, contexts, answers)):
            context_text = "\n".join(context_list)
            
            question = EvaluationQuestion(
                query=query,
                context=context_text,
                answer=answer,
                system_name=system_name,
                question_id=f"{system_name}_{i}"
            )
            questions.append(question)
        
        return questions
    
    def generate_evaluation_form(
        self,
        questions: List[EvaluationQuestion],
        output_file: str
    ):
        """
        Generate an HTML form for relevance evaluation.
        
        Args:
            questions: List of evaluation questions
            output_file: Path to save the HTML form
        """
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>RAG System Relevance Evaluation</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .question {{ border: 1px solid #ccc; padding: 15px; margin: 20px 0; }}
        .query {{ font-weight: bold; color: #333; }}
        .context {{ background-color: #f5f5f5; padding: 10px; margin: 10px 0; }}
        .answer {{ background-color: #e8f4fd; padding: 10px; margin: 10px 0; }}
        .rating {{ margin: 10px 0; }}
        .explanation {{ width: 100%; height: 60px; }}
        .scale {{ font-size: 12px; color: #666; }}
    </style>
</head>
<body>
    <h1>RAG System Relevance Evaluation</h1>
    <p>Please evaluate the relevance of each answer to its corresponding query using the 1-5 scale below:</p>
    
    <div class="scale">
        <strong>Rating Scale:</strong><br>
        1 = Not relevant - Answer does not address the query at all<br>
        2 = Slightly relevant - Answer partially addresses the query but misses key points<br>
        3 = Moderately relevant - Answer addresses the query but lacks detail or has minor issues<br>
        4 = Highly relevant - Answer addresses the query well with good detail<br>
        5 = Perfectly relevant - Answer completely and accurately addresses all aspects of the query
    </div>
    
    <form id="evaluationForm">
"""
        
        for i, question in enumerate(questions):
            html_content += f"""
        <div class="question">
            <h3>Question {i+1}</h3>
            <div class="query">
                <strong>Query:</strong> {question.query}
            </div>
            <div class="context">
                <strong>Context:</strong><br>
                {question.context[:500]}{"..." if len(question.context) > 500 else ""}
            </div>
            <div class="answer">
                <strong>Answer ({question.system_name}):</strong><br>
                {question.answer}
            </div>
            <div class="rating">
                <strong>Relevance Rating:</strong><br>
                <input type="radio" name="rating_{question.question_id}" value="1"> 1 (Not relevant)
                <input type="radio" name="rating_{question.question_id}" value="2"> 2 (Slightly relevant)
                <input type="radio" name="rating_{question.question_id}" value="3"> 3 (Moderately relevant)
                <input type="radio" name="rating_{question.question_id}" value="4"> 4 (Highly relevant)
                <input type="radio" name="rating_{question.question_id}" value="5"> 5 (Perfectly relevant)
            </div>
            <div>
                <strong>Explanation (optional):</strong><br>
                <textarea name="explanation_{question.question_id}" class="explanation" 
                          placeholder="Please explain your rating..."></textarea>
            </div>
        </div>
"""
        
        html_content += """
        <button type="button" onclick="exportResults()">Export Results</button>
    </form>
    
    <script>
        function exportResults() {
            const form = document.getElementById('evaluationForm');
            const formData = new FormData(form);
            const results = {};
            
            for (let [key, value] of formData.entries()) {
                results[key] = value;
            }
            
            const json = JSON.stringify(results, null, 2);
            const blob = new Blob([json], {type: 'application/json'});
            const url = URL.createObjectURL(blob);
            
            const a = document.createElement('a');
            a.href = url;
            a.download = 'evaluation_results.json';
            a.click();
        }
    </script>
</body>
</html>
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Evaluation form saved to {output_file}")


class FactualAccuracyEvaluator:
    """
    Tool for evaluating factual accuracy of answers.
    """
    
    def __init__(self):
        self.error_types = {
            'factual_error': 'Contains incorrect factual information',
            'hallucination': 'Contains information not present in the context',
            'misleading': 'Technically correct but misleading interpretation',
            'incomplete': 'Missing important factual details',
            'outdated': 'Contains outdated information',
            'none': 'No accuracy issues identified'
        }
    
    def create_accuracy_questions(
        self,
        queries: List[str],
        contexts: List[List[str]],
        answers: List[str],
        system_name: str
    ) -> List[EvaluationQuestion]:
        """
        Create evaluation questions for factual accuracy assessment.
        """
        return RelevanceScorer().create_evaluation_questions(
            queries, contexts, answers, system_name
        )
    
    def generate_accuracy_form(
        self,
        questions: List[EvaluationQuestion],
        output_file: str
    ):
        """
        Generate an HTML form for factual accuracy evaluation.
        """
        error_options = "".join([
            f'<option value="{key}">{value}</option>' 
            for key, value in self.error_types.items()
        ])
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>RAG System Factual Accuracy Evaluation</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .question {{ border: 1px solid #ccc; padding: 15px; margin: 20px 0; }}
        .query {{ font-weight: bold; color: #333; }}
        .context {{ background-color: #f5f5f5; padding: 10px; margin: 10px 0; }}
        .answer {{ background-color: #e8f4fd; padding: 10px; margin: 10px 0; }}
        .evaluation {{ margin: 10px 0; }}
        .explanation {{ width: 100%; height: 60px; }}
    </style>
</head>
<body>
    <h1>RAG System Factual Accuracy Evaluation</h1>
    <p>Please evaluate the factual accuracy of each answer based on the provided context.</p>
    
    <form id="accuracyForm">
"""
        
        for i, question in enumerate(questions):
            html_content += f"""
        <div class="question">
            <h3>Question {i+1}</h3>
            <div class="query">
                <strong>Query:</strong> {question.query}
            </div>
            <div class="context">
                <strong>Context:</strong><br>
                {question.context}
            </div>
            <div class="answer">
                <strong>Answer ({question.system_name}):</strong><br>
                {question.answer}
            </div>
            <div class="evaluation">
                <strong>Is the answer factually accurate?</strong><br>
                <input type="radio" name="accurate_{question.question_id}" value="true"> Yes
                <input type="radio" name="accurate_{question.question_id}" value="false"> No
            </div>
            <div class="evaluation">
                <strong>If not accurate, what type of error?</strong><br>
                <select name="error_type_{question.question_id}">
                    {error_options}
                </select>
            </div>
            <div>
                <strong>Explanation:</strong><br>
                <textarea name="explanation_{question.question_id}" class="explanation" 
                          placeholder="Please explain your assessment..."></textarea>
            </div>
        </div>
"""
        
        html_content += """
        <button type="button" onclick="exportResults()">Export Results</button>
    </form>
    
    <script>
        function exportResults() {
            const form = document.getElementById('accuracyForm');
            const formData = new FormData(form);
            const results = {};
            
            for (let [key, value] of formData.entries()) {
                results[key] = value;
            }
            
            const json = JSON.stringify(results, null, 2);
            const blob = new Blob([json], {type: 'application/json'});
            const url = URL.createObjectURL(blob);
            
            const a = document.createElement('a');
            a.href = url;
            a.download = 'accuracy_results.json';
            a.click();
        }
    </script>
</body>
</html>
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Accuracy evaluation form saved to {output_file}")


class ComparativePreferenceTest:
    """
    Tool for comparative evaluation between different RAG systems.
    """
    
    def __init__(self):
        self.confidence_scale = {
            1: "Very low confidence",
            2: "Low confidence", 
            3: "Moderate confidence",
            4: "High confidence",
            5: "Very high confidence"
        }
    
    def create_comparative_questions(
        self,
        queries: List[str],
        contexts: List[List[str]],
        answers_a: List[str],
        answers_b: List[str],
        system_a_name: str,
        system_b_name: str,
        randomize: bool = True
    ) -> List[Tuple[EvaluationQuestion, EvaluationQuestion]]:
        """
        Create paired questions for comparative evaluation.
        
        Args:
            queries: List of queries
            contexts: List of context lists
            answers_a: Answers from system A
            answers_b: Answers from system B
            system_a_name: Name of system A
            system_b_name: Name of system B
            randomize: Whether to randomize answer order
            
        Returns:
            List of question pairs
        """
        question_pairs = []
        
        for i, (query, context_list, ans_a, ans_b) in enumerate(
            zip(queries, contexts, answers_a, answers_b)
        ):
            context_text = "\n".join(context_list)
            
            # Create questions for both systems
            question_a = EvaluationQuestion(
                query=query,
                context=context_text,
                answer=ans_a,
                system_name=system_a_name,
                question_id=f"comp_{i}_a"
            )
            
            question_b = EvaluationQuestion(
                query=query,
                context=context_text,
                answer=ans_b,
                system_name=system_b_name,
                question_id=f"comp_{i}_b"
            )
            
            # Randomize order if requested
            if randomize and random.random() < 0.5:
                question_pairs.append((question_b, question_a))
            else:
                question_pairs.append((question_a, question_b))
        
        return question_pairs
    
    def generate_comparative_form(
        self,
        question_pairs: List[Tuple[EvaluationQuestion, EvaluationQuestion]],
        output_file: str
    ):
        """
        Generate an HTML form for comparative evaluation.
        """
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>RAG System Comparative Evaluation</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .comparison { border: 1px solid #ccc; padding: 15px; margin: 20px 0; }
        .query { font-weight: bold; color: #333; margin-bottom: 15px; }
        .context { background-color: #f5f5f5; padding: 10px; margin: 10px 0; }
        .answers { display: flex; gap: 20px; }
        .answer { flex: 1; background-color: #e8f4fd; padding: 10px; }
        .preference { margin: 15px 0; text-align: center; }
        .explanation { width: 100%; height: 60px; }
    </style>
</head>
<body>
    <h1>RAG System Comparative Evaluation</h1>
    <p>For each query, compare the two answers and indicate which you prefer and why.</p>
    
    <form id="comparativeForm">
"""
        
        for i, (question_a, question_b) in enumerate(question_pairs):
            html_content += f"""
        <div class="comparison">
            <h3>Comparison {i+1}</h3>
            <div class="query">
                <strong>Query:</strong> {question_a.query}
            </div>
            <div class="context">
                <strong>Context:</strong><br>
                {question_a.context[:500]}{"..." if len(question_a.context) > 500 else ""}
            </div>
            <div class="answers">
                <div class="answer">
                    <strong>Answer A ({question_a.system_name}):</strong><br>
                    {question_a.answer}
                </div>
                <div class="answer">
                    <strong>Answer B ({question_b.system_name}):</strong><br>
                    {question_b.answer}
                </div>
            </div>
            <div class="preference">
                <strong>Which answer do you prefer?</strong><br>
                <input type="radio" name="preference_{i}" value="A"> Answer A ({question_a.system_name})
                <input type="radio" name="preference_{i}" value="B"> Answer B ({question_b.system_name})
                <input type="radio" name="preference_{i}" value="tie"> Both equally good
            </div>
            <div class="preference">
                <strong>Confidence in your preference:</strong><br>
                <input type="radio" name="confidence_{i}" value="1"> 1 (Very low)
                <input type="radio" name="confidence_{i}" value="2"> 2 (Low)
                <input type="radio" name="confidence_{i}" value="3"> 3 (Moderate)
                <input type="radio" name="confidence_{i}" value="4"> 4 (High)
                <input type="radio" name="confidence_{i}" value="5"> 5 (Very high)
            </div>
            <div>
                <strong>Reason for preference:</strong><br>
                <textarea name="reason_{i}" class="explanation" 
                          placeholder="Please explain your preference..."></textarea>
            </div>
        </div>
"""
        
        html_content += """
        <button type="button" onclick="exportResults()">Export Results</button>
    </form>
    
    <script>
        function exportResults() {
            const form = document.getElementById('comparativeForm');
            const formData = new FormData(form);
            const results = {};
            
            for (let [key, value] of formData.entries()) {
                results[key] = value;
            }
            
            const json = JSON.stringify(results, null, 2);
            const blob = new Blob([json], {type: 'application/json'});
            const url = URL.createObjectURL(blob);
            
            const a = document.createElement('a');
            a.href = url;
            a.download = 'comparative_results.json';
            a.click();
        }
    </script>
</body>
</html>
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Comparative evaluation form saved to {output_file}")


class HumanEvaluationFramework:
    """
    Main framework for conducting human evaluation of RAG systems.
    """
    
    def __init__(self):
        self.relevance_scorer = RelevanceScorer()
        self.accuracy_evaluator = FactualAccuracyEvaluator()
        self.preference_tester = ComparativePreferenceTest()
    
    def conduct_full_evaluation(
        self,
        queries: List[str],
        contexts: List[List[str]],
        system_answers: Dict[str, List[str]],
        output_dir: str = "human_evaluation"
    ):
        """
        Conduct a comprehensive human evaluation.
        
        Args:
            queries: List of queries
            contexts: List of context lists
            system_answers: Dictionary mapping system names to their answers
            output_dir: Directory to save evaluation forms
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        system_names = list(system_answers.keys())
        
        # Generate relevance evaluation forms for each system
        for system_name in system_names:
            print(f"Generating relevance evaluation for {system_name}...")
            questions = self.relevance_scorer.create_evaluation_questions(
                queries, contexts, system_answers[system_name], system_name
            )
            self.relevance_scorer.generate_evaluation_form(
                questions, 
                os.path.join(output_dir, f"relevance_{system_name}.html")
            )
            
            # Generate accuracy evaluation
            print(f"Generating accuracy evaluation for {system_name}...")
            self.accuracy_evaluator.generate_accuracy_form(
                questions,
                os.path.join(output_dir, f"accuracy_{system_name}.html")
            )
        
        # Generate comparative evaluations between systems
        if len(system_names) >= 2:
            for i in range(len(system_names)):
                for j in range(i + 1, len(system_names)):
                    system_a, system_b = system_names[i], system_names[j]
                    print(f"Generating comparative evaluation: {system_a} vs {system_b}...")
                    
                    question_pairs = self.preference_tester.create_comparative_questions(
                        queries, contexts,
                        system_answers[system_a], system_answers[system_b],
                        system_a, system_b
                    )
                    
                    self.preference_tester.generate_comparative_form(
                        question_pairs,
                        os.path.join(output_dir, f"comparative_{system_a}_vs_{system_b}.html")
                    )
        
        print(f"Human evaluation forms generated in {output_dir}")
        print("Instructions:")
        print("1. Open the HTML files in a web browser")
        print("2. Complete the evaluations")
        print("3. Export results using the 'Export Results' button")
        print("4. Use the analyze_human_evaluation_results() function to process results")
    
    def analyze_human_evaluation_results(
        self,
        results_files: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze results from human evaluation.
        
        Args:
            results_files: List of paths to result JSON files
            
        Returns:
            Analysis summary
        """
        analysis = {
            "relevance_scores": {},
            "accuracy_scores": {},
            "preference_scores": {},
            "summary": {}
        }
        
        for results_file in results_files:
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            # Analyze relevance scores
            relevance_ratings = []
            for key, value in results.items():
                if key.startswith('rating_'):
                    relevance_ratings.append(int(value))
            
            if relevance_ratings:
                system_name = results_file.split('_')[1].split('.')[0] if '_' in results_file else 'unknown'
                analysis["relevance_scores"][system_name] = {
                    "mean_score": sum(relevance_ratings) / len(relevance_ratings),
                    "scores": relevance_ratings,
                    "count": len(relevance_ratings)
                }
            
            # Analyze accuracy scores
            accuracy_ratings = []
            for key, value in results.items():
                if key.startswith('accurate_'):
                    accuracy_ratings.append(value == 'true')
            
            if accuracy_ratings:
                system_name = results_file.split('_')[1].split('.')[0] if '_' in results_file else 'unknown'
                analysis["accuracy_scores"][system_name] = {
                    "accuracy_rate": sum(accuracy_ratings) / len(accuracy_ratings),
                    "correct_count": sum(accuracy_ratings),
                    "total_count": len(accuracy_ratings)
                }
            
            # Analyze preference scores
            preferences = []
            for key, value in results.items():
                if key.startswith('preference_'):
                    preferences.append(value)
            
            if preferences:
                comparison_name = results_file.split('.')[0] if '.' in results_file else 'unknown'
                preference_counts = {pref: preferences.count(pref) for pref in set(preferences)}
                analysis["preference_scores"][comparison_name] = preference_counts
        
        return analysis