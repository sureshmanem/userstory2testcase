"""
ui_runner.py
Implements the UI backend logic for Streamlit.
"""

import os
import configparser
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any
from us_to_mtc_file.ModelManualTestLLM import LLM
from us_to_mtc_file.ChromaDBConnector import ChromaDBConnector
from us_to_mtc_file.GenerateManualTestResults import ManualTestGenerator


@dataclass
class UIRunOptions:
    """Configuration options for UI-based manual test generation."""
    test_type: str = "Both"
    use_context: bool = False
    num_context_retrieve: int = 3
    enable_additional_intelligence: bool = True
    generate_additional_acceptance_criteria: bool = False


def generate_manual_tests_dataframe(
    user_stories_df: pd.DataFrame,
    additional_context_df: Optional[pd.DataFrame] = None,
    llm_family: str = "GPT",
    persist_directory: str = "Data/SavedContexts/EmbedDataMTC_UI",
    options: Optional[UIRunOptions] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> pd.DataFrame:
    """
    Generate manual tests and return results as a DataFrame.

    Args:
        user_stories_df: DataFrame containing user stories.
        additional_context_df: Optional DataFrame with additional context.
        llm_family: LLM family to use.
        persist_directory: Directory for storing vector database.
        options: UI run options.
        progress_callback: Callback function(current, total, message).

    Returns:
        DataFrame with columns: UserStoryID, TestType, UsedContext, UserStory,
        Context, PromptSentToLLM, ManualTest, plus metrics columns.

    Raises:
        RuntimeError: If Azure OpenAI connection fails.
    """
    if options is None:
        options = UIRunOptions()

    # Load configuration
    config = configparser.ConfigParser()
    config.read("Config/Config.properties")

    # Preflight check - validate Azure OpenAI connection
    if progress_callback:
        progress_callback(0, len(user_stories_df), "Validating Azure OpenAI connection...")

    try:
        llm = LLM()
        client_info = llm.get_azure_client_info()
        llm.ping()
        
        if progress_callback:
            progress_callback(
                0, 
                len(user_stories_df), 
                f"✓ Connected to Azure OpenAI ({client_info['endpoint']})"
            )
    except Exception as e:
        raise RuntimeError(
            f"Failed to connect to Azure OpenAI: {str(e)}\n\n"
            f"Please check your .env file and ensure all required environment variables are set:\n"
            f"- AZURE_OPENAI_API_KEY\n"
            f"- AZURE_OPENAI_ENDPOINT\n"
            f"- AZURE_OPENAI_API_VERSION\n"
            f"- AZURE_OPENAI_DEPLOYMENT_NAME"
        )

    # Initialize generator
    generator = ManualTestGenerator(llm_family=llm_family)

    # Setup context store if needed
    context_store = None
    threshold = float(config.get("AdvancedConfigurations", "default_model_threshold"))

    if options.use_context and additional_context_df is not None and len(additional_context_df) > 0:
        if progress_callback:
            progress_callback(0, len(user_stories_df), "Building vector store from additional context...")

        # Clean up and recreate persist directory
        os.makedirs(persist_directory, exist_ok=True)

        # Save uploaded context
        context_csv_path = os.path.join(persist_directory, "uploaded_context.csv")
        additional_context_df.to_csv(context_csv_path, index=False, encoding="utf-8-sig")

        # Build vector store
        context_store = ChromaDBConnector(persist_directory=persist_directory)
        context_store.vector_store(context_csv_path)

        # Update threshold from store metadata
        import json
        meta_file = os.path.join(persist_directory, "store_meta.json")
        if os.path.exists(meta_file):
            with open(meta_file, "r") as f:
                meta = json.load(f)
                threshold = meta.get("threshold", threshold)

        if progress_callback:
            progress_callback(0, len(user_stories_df), "✓ Vector store ready")

    # Process each user story
    results = []
    total_stories = len(user_stories_df)

    for idx, row in user_stories_df.iterrows():
        current_num = idx + 1

        if progress_callback:
            progress_callback(
                current_num - 1,
                total_stories,
                f"Processing story {current_num}/{total_stories}..."
            )

        # Extract user story ID
        user_story_id = _extract_user_story_id(row, idx)

        # Build user story text
        description = str(row.get("Description", "")) if pd.notna(row.get("Description")) else ""
        acceptance_criteria = str(row.get("AcceptanceCriteria", "")) if pd.notna(row.get("AcceptanceCriteria")) else ""
        user_story_text = f"User Story Description: {description}\nAcceptance Criteria: \n{acceptance_criteria}"

        # Track metrics and prompt
        story_metrics = {
            "LLMCalls": 0,
            "InputTokens": 0,
            "OutputTokens": 0,
            "TotalTokens": 0,
            "LatencySeconds": 0.0,
            "ResponseModel": "",
            "OutputWords": 0,
            "OutputChars": 0
        }
        captured_prompt = ""

        def prompt_callback(prompt: str):
            nonlocal captured_prompt
            captured_prompt = prompt

        def metrics_callback(metrics: Dict[str, Any]):
            nonlocal story_metrics
            story_metrics["LLMCalls"] += 1
            story_metrics["InputTokens"] += metrics.get("input_tokens", 0)
            story_metrics["OutputTokens"] += metrics.get("output_tokens", 0)
            story_metrics["TotalTokens"] += metrics.get("total_tokens", 0)
            story_metrics["LatencySeconds"] += metrics.get("latency_seconds", 0.0)
            story_metrics["ResponseModel"] = metrics.get("response_model", "")
            story_metrics["OutputWords"] += metrics.get("output_words", 0)
            story_metrics["OutputChars"] += metrics.get("output_chars", 0)

        # Determine whether to use context
        used_context = "No"
        final_context = ""
        manual_test = ""

        if context_store:
            # Retrieve context
            combined_context, docs_with_scores, _ = context_store.retrieval_context(
                query=user_story_text,
                k=options.num_context_retrieve
            )

            # Filter by threshold
            filtered_docs = {score: doc for score, doc in docs_with_scores.items() if score < threshold}

            if filtered_docs:
                # Use context
                used_context = "Yes"
                final_context = "\n\n".join(filtered_docs.values())

                result = generator.get_manual_test(
                    test_type=options.test_type,
                    input_length=total_stories,
                    query=user_story_text,
                    context=final_context,
                    enable_additional_intelligence=options.enable_additional_intelligence,
                    generate_additional_acceptance_criteria=options.generate_additional_acceptance_criteria,
                    auto_send_to_llm=True,
                    prompt_callback=prompt_callback,
                    metrics_callback=metrics_callback
                )
                manual_test = result[0]
            else:
                # No relevant context - fall back to no-context
                result = generator.get_manual_test_no_context(
                    test_type=options.test_type,
                    input_length=total_stories,
                    query=user_story_text,
                    enable_additional_intelligence=options.enable_additional_intelligence,
                    generate_additional_acceptance_criteria=options.generate_additional_acceptance_criteria,
                    auto_send_to_llm=True,
                    prompt_callback=prompt_callback,
                    metrics_callback=metrics_callback
                )
                manual_test = result[0]
        else:
            # No context store - generate without context
            result = generator.get_manual_test_no_context(
                test_type=options.test_type,
                input_length=total_stories,
                query=user_story_text,
                enable_additional_intelligence=options.enable_additional_intelligence,
                generate_additional_acceptance_criteria=options.generate_additional_acceptance_criteria,
                auto_send_to_llm=True,
                prompt_callback=prompt_callback,
                metrics_callback=metrics_callback
            )
            manual_test = result[0]

        # Calculate derived metrics
        tokens_per_second = (
            story_metrics["TotalTokens"] / story_metrics["LatencySeconds"]
            if story_metrics["LatencySeconds"] > 0 else 0
        )
        estimated_effort_saved_minutes = story_metrics["OutputWords"] / 40.0  # 40 words per minute

        # Build result row
        result_row = {
            "UserStoryID": user_story_id,
            "TestType": options.test_type,
            "UsedContext": used_context,
            "UserStory": user_story_text,
            "Context": final_context,
            "PromptSentToLLM": captured_prompt,
            "ManualTest": manual_test,
            "LLMCalls": story_metrics["LLMCalls"],
            "InputTokens": story_metrics["InputTokens"],
            "OutputTokens": story_metrics["OutputTokens"],
            "TotalTokens": story_metrics["TotalTokens"],
            "LatencySeconds": round(story_metrics["LatencySeconds"], 2),
            "TokensPerSecond": round(tokens_per_second, 2),
            "ResponseModel": story_metrics["ResponseModel"],
            "OutputWords": story_metrics["OutputWords"],
            "OutputChars": story_metrics["OutputChars"],
            "EstimatedEffortSavedMinutes": round(estimated_effort_saved_minutes, 2)
        }

        results.append(result_row)

        if progress_callback:
            progress_callback(
                current_num,
                total_stories,
                f"✓ Completed story {current_num}/{total_stories}"
            )

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    if progress_callback:
        progress_callback(total_stories, total_stories, "✓ All stories processed!")

    return results_df


def _extract_user_story_id(row: pd.Series, index: int) -> str:
    """
    Extract user story ID from row, trying multiple columns.

    Args:
        row: DataFrame row.
        index: Row index as fallback.

    Returns:
        User story ID as string.
    """
    # Try ID, FormattedID, UserStoryID columns
    for col in ["ID", "FormattedID", "UserStoryID"]:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            return str(row[col]).strip()
    
    # Fallback to index
    return str(index)
