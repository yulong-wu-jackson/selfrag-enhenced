"""
Main module to demonstrate the Self-RAG system.
"""
import os
import logging
import shutil
from typing import Dict, List, Any, Tuple
import time
import json
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain.vectorstores.base import VectorStore
from langchain.schema import BaseRetriever

from src.config.config import get_config
from src.embeddings.embedding import get_embeddings
from src.agents.self_rag import SelfRAG
from src.agents.rag import TraditionalRAG
from src.utils.langsmith import setup_langsmith
from src.models.llm import get_llm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_vectorstore() -> VectorStore:
    """
    Create a sample vector store with some documents using Chroma DB.
    In a real application, this would load and index your actual documents.
    """
    # Set a directory for persistence
    persist_directory = "./chroma_db"
    
    # Check if vector store already exists and clear it if it does
    if os.path.exists(persist_directory):
        logger.info(f"Existing vector store found at {persist_directory}. Clearing it...")
        try:
            shutil.rmtree(persist_directory)
            logger.info(f"Successfully cleared existing vector store at {persist_directory}")
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
    
    # Sample documents for basic testing
    basic_documents = [
        Document(
            page_content="Jackson is a 3rd year uoft student majoring in Computer Science. He has a GPA of 3.8.",
            metadata={"source": "personal_data.txt", "topic": "profile"}
        ),
        Document(
            page_content="Jackson is originally from Vancouver and moved to Toronto for university.",
            metadata={"source": "background.txt", "topic": "profile"}
        ),
        Document(
            page_content="The University of Toronto (U of T or UToronto) is a public research university in Toronto, Ontario, Canada, located on the grounds that surround Queen's Park.",
            metadata={"source": "universities.txt", "topic": "education"}
        ),
        Document(
            page_content="Computer Science at UofT focuses on the fundamentals of computer programming and design, theories of computation, AI, systems programming, and software engineering.",
            metadata={"source": "programs.txt", "topic": "education"}
        ),
        Document(
            page_content="Jackson works part-time as a teaching assistant for first-year programming courses.",
            metadata={"source": "work_experience.txt", "topic": "employment"}
        ),
        Document(
            page_content="Jackson is interested in artificial intelligence and machine learning. He has completed several projects in these areas.",
            metadata={"source": "interests.txt", "topic": "personal"}
        ),
    ]
    
    # Complex documents about a fictional company for testing advanced RAG capabilities
    complex_documents = [
        Document(
            page_content="TechNova Corporation reported Q1 2023 revenues of $245.8 million, a 12.3% increase from Q1 2022. Operating expenses were $198.2 million, resulting in an operating profit of $47.6 million. The cloud services division generated $125.3 million in revenue, while the AI solutions division contributed $85.1 million. The remaining $35.4 million came from professional services and other sources.",
            metadata={"source": "financial_report_q1_2023.pdf", "topic": "finance", "date": "2023-04-15"}
        ),
        Document(
            page_content="TechNova Corporation's Q2 2023 financial results showed revenues of $267.5 million, a 15.8% year-over-year increase. Operating expenses increased to $210.4 million due to expanded R&D investments in the AI division. Operating profit reached $57.1 million, representing a 20% increase from the previous quarter.",
            metadata={"source": "financial_report_q2_2023.pdf", "topic": "finance", "date": "2023-07-20"}
        ),
        Document(
            page_content="In Q3 2023, TechNova Corporation reported a decline in revenue to $251.2 million due to market challenges and delayed contract renewals. Operating expenses remained relatively stable at $208.9 million, resulting in an operating profit of $42.3 million, a 25.9% decrease from Q2 2023.",
            metadata={"source": "financial_report_q3_2023.pdf", "topic": "finance", "date": "2023-10-18"}
        ),
        Document(
            page_content="TechNova's Board of Directors approved a new Research & Development investment plan allocating $75 million for AI innovation in 2023, a 25% increase from 2022. The plan specifically earmarks $45 million for machine learning model development and $30 million for cloud infrastructure optimization.",
            metadata={"source": "board_minutes_jan2023.pdf", "topic": "strategy", "date": "2023-01-12"}
        ),
        Document(
            page_content="TechNova's Remote Work Policy (Policy ID: HR-RWP-2023) states that employees may work remotely up to 3 days per week, subject to manager approval. Exceptions for full remote work require VP-level approval and are granted based on role requirements and performance history. All remote employees must maintain core hours of 10:00 AM to 3:00 PM in their local time zone for collaboration purposes.",
            metadata={"source": "hr_policies.pdf", "topic": "human resources", "date": "2023-02-01"}
        ),
        Document(
            page_content="According to TechNova's Expense Reimbursement Policy (Policy ID: FIN-ERP-2023), employees must submit all expense reports within 30 days of incurring the expense. Expenses submitted after this period require director-level approval. Technology purchases over $2,000 require pre-approval from the IT department to ensure compatibility with company systems.",
            metadata={"source": "finance_policies.pdf", "topic": "finance", "date": "2023-03-15"}
        ),
        Document(
            page_content="TechNova's AI Ethics Committee meeting on April 5, 2023, established guidelines for responsible AI development. The committee, chaired by Dr. Emily Wong, mandated bias testing for all AI models before deployment and required quarterly reviews of model performance to identify potential unintended consequences. The committee also approved a $1.5 million budget for developing explainable AI tools.",
            metadata={"source": "ai_ethics_minutes.pdf", "topic": "governance", "date": "2023-04-05"}
        ),
        Document(
            page_content="TechNova's Project Aurora, aimed at enhancing the company's cloud infrastructure, faced delays due to supply chain issues as reported in the July 2023 project status update. The project, initially scheduled for completion in Q3 2023, has been postponed to Q1 2024. The budget remains unchanged at $28.5 million, but resource allocation has been adjusted to focus on software development until hardware availability improves.",
            metadata={"source": "project_status_july2023.pdf", "topic": "projects", "date": "2023-07-10"}
        ),
        Document(
            page_content="TechNova's CEO Sarah Johnson announced in the September 2023 Town Hall that the company plans to acquire DataSphere Inc., a data analytics startup, for $135 million. The acquisition, expected to close in Q1 2024, aims to strengthen TechNova's data processing capabilities and expand its market share in the financial services sector.",
            metadata={"source": "townhall_transcript.pdf", "topic": "corporate", "date": "2023-09-15"}
        ),
        Document(
            page_content="TechNova's Annual Holiday Policy 2023 (Policy ID: HR-AHP-2023) provides all full-time employees with 12 paid holidays and a flexible personal holiday that can be taken anytime during the calendar year with manager approval. Additionally, the office will be closed from December 24, 2023, to January 1, 2024, with employees receiving these days as paid time off without deducting from their personal vacation allowance.",
            metadata={"source": "hr_holiday_policy.pdf", "topic": "human resources", "date": "2023-10-30"}
        ),
        Document(
            page_content="DataSphere Inc., founded in 2018, specializes in real-time data analytics for financial institutions. Their flagship product, FinanceIQ, processes transaction data to identify patterns and anomalies for fraud detection. The company reported $28.7 million in revenue for 2022 and employs 85 people, primarily data scientists and software engineers.",
            metadata={"source": "datasphere_profile.pdf", "topic": "acquisition", "date": "2023-08-25"}
        ),
        Document(
            page_content="TechNova's September 2023 Market Analysis Report indicates that enterprise demand for AI solutions is expected to grow by 35% in 2024, with financial services and healthcare showing the strongest interest. The report notes that competitors XYZ Technologies and AI Innovations have recently lowered their pricing, which may impact TechNova's premium pricing strategy in the short term.",
            metadata={"source": "market_analysis.pdf", "topic": "strategy", "date": "2023-09-30"}
        ),
    ]
    
    # Combine both document sets
    documents = basic_documents + complex_documents
    
    # Create vector store with Chroma
    embeddings = get_embeddings()
    
    # Create or load the Chroma vector store
    # Note: Chroma automatically persists docs since v0.4.x
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="self-rag-collection",
    )
    
    logger.info(f"Vector store created with {len(documents)} documents and persisted to {persist_directory}")
    
    return vectorstore

# Define ground truth answers for evaluation
def get_ground_truth_answers() -> Dict[str, str]:
    """
    Define the correct/ground truth answers for each query for evaluation purposes.
    
    Returns:
        Dictionary mapping queries to their ground truth answers
    """
    return {
        # Basic queries
        "Tell me a joke": "No specific ground truth as this is subjective, but should be a joke.",
        "Who is Jackson?": "Jackson is a 3rd year University of Toronto student majoring in Computer Science with a GPA of 3.8. He is originally from Vancouver and moved to Toronto for university. He works part-time as a teaching assistant for first-year programming courses and is interested in artificial intelligence and machine learning, having completed several projects in these areas.",
        "What is Jackson studying and where?": "Jackson is studying Computer Science at the University of Toronto (UofT).",
        
        # Complex queries
        "How did TechNova's financial performance in Q3 2023 compare to their R&D investment plans, and what might explain any discrepancies?": 
            "TechNova's Q3 2023 financial performance showed a decline in revenue to $251.2 million and operating profit of $42.3 million, which was a 25.9% decrease from Q2 2023. This contrasts with their R&D investment plans from January 2023, which allocated $75 million for AI innovation ($45 million for machine learning model development and $30 million for cloud infrastructure optimization). The discrepancy might be explained by market challenges, delayed contract renewals mentioned in the Q3 report, and the delay in Project Aurora due to supply chain issues, which was postponed from Q3 2023 to Q1 2024. Despite financial challenges, their operating expenses remained stable at $208.9 million, suggesting they maintained their R&D commitments.",
        
        "Has TechNova's AI development strategy changed since the formation of their ethics committee, and how might this relate to their acquisition plans?": 
            "TechNova's AI development strategy has evolved since forming their AI Ethics Committee on April 5, 2023. The committee established responsible AI development guidelines, mandated bias testing for all AI models before deployment, required quarterly reviews of model performance, and approved a $1.5 million budget for developing explainable AI tools. This focus on ethical AI development likely influenced their September 2023 decision to acquire DataSphere Inc., a data analytics startup specializing in real-time data analytics for financial institutions. The acquisition aims to strengthen TechNova's data processing capabilities and expand their market share in the financial services sector, which aligns with their ethical AI approach since DataSphere's FinanceIQ product focuses on pattern identification and anomaly detection for fraud prevention—areas where ethical considerations are paramount.",
        
        "Based on TechNova's policies, if an employee working remotely needs to purchase a $2,500 laptop but submits the expense 45 days later, what approvals would they need?": 
            "The employee would need three approvals: 1) IT department pre-approval for the technology purchase over $2,000 (as per the Expense Reimbursement Policy), 2) director-level approval for submitting the expense after the 30-day deadline (as per the Expense Reimbursement Policy), and 3) manager approval for the remote work arrangement itself (as per the Remote Work Policy). The expense report falls outside the 30-day submission window and involves a technology purchase above the $2,000 threshold that requires special approval.",
        
        "What impact might the delay in Project Aurora have on TechNova's revenue projections for Q1 2024, especially considering their acquisition plans?": 
            "The delay in Project Aurora from Q3 2023 to Q1 2024 will likely have a negative impact on TechNova's Q1 2024 revenue projections for several reasons: 1) The cloud infrastructure enhancement that Project Aurora represents won't be contributing revenue during that period as originally planned, 2) The simultaneous closing of the DataSphere acquisition in Q1 2024 will create integration challenges and potential resource conflicts, 3) The $28.5 million budget for Project Aurora, combined with the $135 million acquisition cost for DataSphere, represents significant Q1 2024 expenditure without immediate revenue returns, and 4) The company already experienced revenue decline in Q3 2023 due to market challenges and delayed contract renewals, which may continue to affect projections. However, DataSphere's analytics capabilities might eventually complement the cloud infrastructure improvements once both projects are completed, potentially leading to stronger revenue growth later in 2024.",
        
        "How does TechNova's holiday policy compare to their remote work policy in terms of flexibility, and when would an employee need VP-level approval?": 
            "TechNova's holiday policy provides 12 paid holidays plus a flexible personal holiday that can be taken anytime with manager approval, and guaranteed paid time off from December 24, 2023, to January 1, 2024, without deducting from vacation allowance. The remote work policy allows employees to work remotely up to 3 days per week with manager approval. In terms of flexibility, the holiday policy is more structured with specific dates but includes one flexible day, while the remote work policy offers regular weekly flexibility. An employee would need VP-level approval only for exceptions to the remote work policy, specifically for full remote work arrangements. This approval is based on role requirements and performance history. There are no mentioned scenarios in the holiday policy that would require VP-level approval.",
        
        "Given the market analysis and financial reports, should TechNova reconsider their premium pricing strategy while still meeting their R&D investment goals?": 
            "Based on the market analysis and financial reports, TechNova should likely reconsider their premium pricing strategy while maintaining their R&D investment goals. The market analysis from September 2023 indicates that competitors XYZ Technologies and AI Innovations have recently lowered their pricing, which could directly impact TechNova's premium strategy. Additionally, TechNova experienced a revenue decline in Q3 2023 (to $251.2 million) due to market challenges and delayed contract renewals, suggesting pricing sensitivity in the market. However, they need to maintain their R&D investments ($75 million allocated for AI innovation in 2023) to remain competitive, especially since the market analysis shows enterprise demand for AI solutions is expected to grow by 35% in 2024. A potential approach would be to implement a more flexible pricing tier system that preserves premium options while offering more competitive alternatives, particularly in the financial services sector where they're expanding through the DataSphere acquisition.",
        
        "What financial trends can be observed across TechNova's quarterly reports for 2023, and how do these align with their strategic investments?": 
            "Financial trends across TechNova's 2023 quarterly reports show: 1) Revenue growth from Q1 to Q2 (from $245.8M to $267.5M, representing a 15.8% year-over-year increase) followed by a decline in Q3 (to $251.2M); 2) Operating expenses increased from Q1 to Q2 (from $198.2M to $210.4M) then stabilized in Q3 (at $208.9M); 3) Operating profit grew from Q1 to Q2 (from $47.6M to $57.1M, a 20% increase) then declined significantly in Q3 (to $42.3M, a 25.9% decrease). These trends partially align with their strategic investments: the increased R&D spending in the AI division explains the higher operating expenses, while the $75M R&D investment plan from January 2023 is consistent with maintaining high operating expenses even during revenue decline. However, the Q3 revenue decline due to market challenges and delayed contract renewals suggests potential misalignment between their premium pricing strategy and market conditions, as highlighted in their September market analysis. The delay of Project Aurora and the planned DataSphere acquisition represent strategic pivots that acknowledge the need to strengthen capabilities in response to financial performance challenges.",
        
        "If DataSphere's FinanceIQ product is integrated into TechNova's AI solutions division, how might this affect their revenue breakdown in future quarters?": 
            "If DataSphere's FinanceIQ product is integrated into TechNova's AI solutions division, it would likely impact TechNova's revenue breakdown in future quarters by: 1) Increasing the AI solutions division's contribution beyond the $85.1 million reported in Q1 2023, potentially making it a larger percentage of total revenue; 2) Specifically boosting revenue from the financial services sector, which was identified as showing strong interest in AI solutions in the market analysis report; 3) Creating new cross-selling opportunities between TechNova's existing cloud services (which generated $125.3 million in Q1 2023) and FinanceIQ's data analytics capabilities, potentially increasing cloud services revenue; 4) Adding a new revenue stream from FinanceIQ's fraud detection capabilities, possibly under a specialized security services category; and 5) Potentially increasing recurring revenue through subscription-based services, as FinanceIQ's real-time processing of transaction data lends itself to a subscription model. Given DataSphere's 2022 revenue of $28.7 million, the integration could initially add approximately 10-15% to TechNova's total quarterly revenue once fully integrated."
    }

def judge_answer(query: str, generated_answer: str, ground_truth: str) -> Tuple[float, str]:
    """
    Use an LLM to judge the quality of a generated answer compared to ground truth.
    
    Args:
        query: The original query
        generated_answer: The answer generated by the RAG system
        ground_truth: The correct answer
        
    Returns:
        Tuple containing score (0-1) and reasoning
    """
    llm = get_llm()
    
    prompt = f"""You are an expert judge evaluating the quality of answers to complex questions.
    
    Question: {query}
    
    Ground Truth Answer: {ground_truth}
    
    Generated Answer: {generated_answer}
    
    Your task is to evaluate the generated answer against the ground truth answer.
    
    Evaluation criteria:
    1. Factual correctness: Does the answer contain factual errors or contradict the ground truth?
    2. Completeness: Does the answer address all aspects of the question?
    3. Relevance: Is the information in the answer relevant to the question?
    4. Coherence: Is the answer well-structured and logical?
    
    First, provide your reasoning about the quality of the answer based on these criteria.
    Then, give a score between 0 and 1, where:
    - 0.0-0.2: Poor (significant factual errors or missing critical information)
    - 0.3-0.5: Fair (some factual errors or incomplete, but partially correct)
    - 0.6-0.7: Good (mostly correct with minor omissions or errors)
    - 0.8-0.9: Very good (correct with very minor issues)
    - 1.0: Excellent (completely correct and comprehensive)
    
    Format your response exactly as follows:
    
    REASONING: <your detailed reasoning>
    SCORE: <numerical score between 0 and 1>
    """
    
    response = llm.invoke(prompt).strip()
    
    # Extract score from response
    reasoning = ""
    score = 0.0
    
    for line in response.split("\n"):
        if line.startswith("REASONING:"):
            reasoning = line[len("REASONING:"):].strip()
        elif line.startswith("SCORE:"):
            try:
                score = float(line[len("SCORE:"):].strip())
            except ValueError:
                score = 0.0
    
    return score, reasoning

def compare_rag_systems(query: str, self_rag: SelfRAG, traditional_rag: TraditionalRAG, ground_truth_answers: Dict[str, str], evaluation_results: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Compare the Self-RAG and Traditional RAG systems on the same query.
    
    Args:
        query: The query to process
        self_rag: The Self-RAG system
        traditional_rag: The Traditional RAG system
        ground_truth_answers: Dictionary of ground truth answers
        evaluation_results: Dictionary to store evaluation results
        
    Returns:
        Dictionary with comparison results
    """
    logger.info(f"\n\n========== COMPARING RAG SYSTEMS ON QUERY: {query} ==========")
    
    # Process with Self-RAG
    start_time_self = time.time()
    self_rag_result = self_rag.invoke(query)
    self_rag_time = time.time() - start_time_self
    
    # Process with Traditional RAG
    start_time_trad = time.time()
    trad_rag_result = traditional_rag.invoke(query)
    trad_rag_time = time.time() - start_time_trad
    
    # Get ground truth answer
    ground_truth = ground_truth_answers.get(query, "No ground truth available")
    
    # Compile results
    comparison = {
        "query": query,
        "self_rag": {
            "response": self_rag_result.get("response", ""),
            "time": self_rag_time,
            "retrieve_decision": self_rag_result.get("retrieve_decision", ""),
            "documents_retrieved": len(self_rag_result.get("documents", [])),
            "relevant_docs": self_rag_result.get("relevant_docs_indices", []),
            "iterations": self_rag_result.get("loop_count", 1)
        },
        "traditional_rag": {
            "response": trad_rag_result.get("response", ""),
            "time": trad_rag_time,
            "documents_retrieved": len(trad_rag_result.get("documents", []))
        }
    }
    
    # Print comparison
    print("\n" + "="*80)
    print(f"QUERY: {query}")
    print("="*80)
    
    print("\n----- SELF-RAG RESPONSE -----")
    print(f"Time: {self_rag_time:.2f}s")
    print(f"Retrieve Decision: {self_rag_result.get('retrieve_decision', '')}")
    print(f"Documents Retrieved: {len(self_rag_result.get('documents', []))}")
    print(f"Relevant Documents: {self_rag_result.get('relevant_docs_indices', [])}")
    print(f"Iterations: {self_rag_result.get('loop_count', 1)}")
    print(f"\nResponse: {self_rag_result.get('response', '')}")
    
    print("\n----- TRADITIONAL RAG RESPONSE -----")
    print(f"Time: {trad_rag_time:.2f}s")
    print(f"Documents Retrieved: {len(trad_rag_result.get('documents', []))}")
    print(f"\nResponse: {trad_rag_result.get('response', '')}")
    print("="*80 + "\n")
    
    # Store evaluation data
    if query in evaluation_results:
        evaluation_data = evaluation_results[query]
    else:
        evaluation_data = {
            "query": query,
            "ground_truth": ground_truth,
            "self_rag_response": self_rag_result.get("response", ""),
            "traditional_rag_response": trad_rag_result.get("response", ""),
            "self_rag_score": None,
            "traditional_rag_score": None,
            "self_rag_reasoning": None,
            "traditional_rag_reasoning": None
        }
        evaluation_results[query] = evaluation_data
    
    return comparison

def evaluate_answers(evaluation_results: Dict[str, Dict]) -> Dict[str, float]:
    """
    Evaluate the quality of answers from both RAG systems.
    
    Args:
        evaluation_results: Dictionary containing queries, responses, and ground truth
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("\n\n========== EVALUATING ANSWER QUALITY ==========")
    
    # Track scores
    self_rag_scores = []
    traditional_rag_scores = []
    
    # Process each query
    for query, data in evaluation_results.items():
        print(f"\nEvaluating answers for query: {query}")
        
        # Skip if no ground truth (e.g., for jokes)
        if data["ground_truth"] == "No specific ground truth as this is subjective, but should be a joke.":
            print("Skipping evaluation as this query has subjective answers")
            continue
        
        # Judge Self-RAG answer
        self_score, self_reasoning = judge_answer(
            query, 
            data["self_rag_response"], 
            data["ground_truth"]
        )
        data["self_rag_score"] = self_score
        data["self_rag_reasoning"] = self_reasoning
        self_rag_scores.append(self_score)
        
        # Judge Traditional RAG answer
        trad_score, trad_reasoning = judge_answer(
            query, 
            data["traditional_rag_response"], 
            data["ground_truth"]
        )
        data["traditional_rag_score"] = trad_score
        data["traditional_rag_reasoning"] = trad_reasoning
        traditional_rag_scores.append(trad_score)
        
        # Print scores
        print(f"Self-RAG Score: {self_score:.2f}")
        print(f"Traditional RAG Score: {trad_score:.2f}")
    
    # Calculate summary metrics
    avg_self_score = sum(self_rag_scores) / len(self_rag_scores) if self_rag_scores else 0
    avg_trad_score = sum(traditional_rag_scores) / len(traditional_rag_scores) if traditional_rag_scores else 0
    
    self_rag_win_rate = sum(1 for s, t in zip(self_rag_scores, traditional_rag_scores) if s > t) / len(self_rag_scores) if self_rag_scores else 0
    
    metrics = {
        "avg_self_rag_score": avg_self_score,
        "avg_traditional_rag_score": avg_trad_score,
        "self_rag_win_rate": self_rag_win_rate
    }
    
    return metrics

def main():
    """Main function to run the Self-RAG demo."""
    config = get_config()
    
    # Setup LangSmith
    langsmith_client = setup_langsmith()
    if langsmith_client:
        logger.info("LangSmith tracing enabled")
    
    # Create a sample vector store and retriever
    vectorstore = create_sample_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # Create the RAG agents
    self_rag = SelfRAG(retriever)
    traditional_rag = TraditionalRAG(retriever)
    
    # Get ground truth answers
    ground_truth_answers = get_ground_truth_answers()
    
    # Dictionary to store evaluation results
    evaluation_results = {}
    
    # Basic queries for comparison
    basic_queries = [
        "Tell me a joke",
        "Who is Jackson?",
        "What is Jackson studying and where?",
    ]
    
    # Complex multi-hop and ambiguous queries that test advanced RAG capabilities
    complex_queries = [
        "How did TechNova's financial performance in Q3 2023 compare to their R&D investment plans, and what might explain any discrepancies?",
        "Has TechNova's AI development strategy changed since the formation of their ethics committee, and how might this relate to their acquisition plans?",
        "Based on TechNova's policies, if an employee working remotely needs to purchase a $2,500 laptop but submits the expense 45 days later, what approvals would they need?",
        "What impact might the delay in Project Aurora have on TechNova's revenue projections for Q1 2024, especially considering their acquisition plans?",
        "How does TechNova's holiday policy compare to their remote work policy in terms of flexibility, and when would an employee need VP-level approval?",
        "Given the market analysis and financial reports, should TechNova reconsider their premium pricing strategy while still meeting their R&D investment goals?",
        "What financial trends can be observed across TechNova's quarterly reports for 2023, and how do these align with their strategic investments?",
        "If DataSphere's FinanceIQ product is integrated into TechNova's AI solutions division, how might this affect their revenue breakdown in future quarters?",
    ]
    
    # Compare systems on basic queries
    print("\n\n========== TESTING BASIC QUERIES ==========")
    for query in basic_queries:
        compare_rag_systems(query, self_rag, traditional_rag, ground_truth_answers, evaluation_results)
    
    # Compare systems on complex queries
    print("\n\n========== TESTING COMPLEX MULTI-HOP QUERIES ==========")
    for query in complex_queries:
        compare_rag_systems(query, self_rag, traditional_rag, ground_truth_answers, evaluation_results)
    
    # Evaluate answer quality
    metrics = evaluate_answers(evaluation_results)
    
    # Print summary
    print("\n\n========== EVALUATION SUMMARY ==========")
    print(f"Average Self-RAG Score: {metrics['avg_self_rag_score']:.2f}")
    print(f"Average Traditional RAG Score: {metrics['avg_traditional_rag_score']:.2f}")
    print(f"Self-RAG Win Rate: {metrics['self_rag_win_rate']:.2%}")
    
    # Save detailed results to file
    with open("evaluation_results.json", "w") as f:
        json.dump(evaluation_results, f, indent=2)
    print(f"Detailed evaluation results saved to evaluation_results.json")

if __name__ == "__main__":
    main() 