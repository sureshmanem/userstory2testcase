"""
GenerateManualTest.py
Implements the ManualTestProcessor for CLI-based batch processing.
"""

import os
import configparser
import pandas as pd
from datetime import datetime
from typing import Optional
from us_to_mtc_file.ChromaDBConnector import ChromaDBConnector
from us_to_mtc_file.GenerateManualTestResults import ManualTestGenerator


class ManualTestProcessor:
    """
    CLI-based processor for generating manual tests from user stories.
    Reads configuration from properties files and processes CSV inputs.
    """

    def __init__(self):
        """Initialize the processor with configuration."""
        # Load configuration
        self.config = configparser.ConfigParser()
        self.config.read("Config/Config.properties")
        self.config_io = configparser.ConfigParser()
        self.config_io.read("Config/ConfigIO.properties")

        # Read settings
        self.llm_family = self.config.get("LLM", "LLM_Family")
        self.num_context_retrieve = int(self.config_io.get("Output", "num_context_retrieve"))
        self.retrieval_context_dir = self.config_io.get("Output", "retrieval_context")

        # Read thresholds from Config.properties
        self.external_model_threshold = float(self.config.get("AdvancedConfigurations", "external_model_threshold"))
        self.default_model_threshold = float(self.config.get("AdvancedConfigurations", "default_model_threshold"))

        # Initialize generator
        self.generator = ManualTestGenerator(llm_family=self.llm_family)

    def _extract_user_story_id(self, row: pd.Series, index: int) -> str:
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

    def generate_manual_test(
        self,
        input_us_df: pd.DataFrame,
        test_type: str,
        output_dir: str
    ):
        """
        Generate manual tests without context.

        Args:
            input_us_df: DataFrame containing user stories.
            test_type: "Both", "Positive", or "Negative".
            output_dir: Directory to save output CSV files.
        """
        os.makedirs(output_dir, exist_ok=True)
        total_stories = len(input_us_df)

        print(f"\nProcessing {total_stories} user stories without context...")
        print("=" * 80)

        for idx, row in input_us_df.iterrows():
            print(f"\nProcessing story {idx + 1}/{total_stories}")

            # Extract user story ID
            user_story_id = self._extract_user_story_id(row, idx)

            # Build user story text
            description = str(row.get("Description", "")) if pd.notna(row.get("Description")) else ""
            acceptance_criteria = str(row.get("AcceptanceCriteria", "")) if pd.notna(row.get("AcceptanceCriteria")) else ""
            
            user_story_text = f"User Story Description: {description}\nAcceptance Criteria: \n{acceptance_criteria}"

            # Generate manual test
            print(f"Generating manual test for User Story ID: {user_story_id}")
            result = self.generator.get_manual_test_no_context(
                test_type=test_type,
                input_length=total_stories,
                query=user_story_text
            )

            # Create output DataFrame
            output_df = pd.DataFrame([{
                "UserStoryID": user_story_id,
                "TestType": test_type,
                "UsedContext": "No",
                "UserStory": user_story_text,
                "Context": "",
                "ManualTest": result[0]
            }])

            # Save to CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_dir, f"manual_test_{idx}_{timestamp}.csv")
            output_df.to_csv(output_file, index=False, encoding="utf-8-sig")
            print(f"Saved: {output_file}")

        print("\n" + "=" * 80)
        print(f"Completed processing {total_stories} user stories!")

    def gen_manual_test_context(
        self,
        input_us_df: pd.DataFrame,
        test_type: str,
        input_context_df: pd.DataFrame,
        output_dir: str
    ):
        """
        Generate manual tests with context retrieval.

        Args:
            input_us_df: DataFrame containing user stories.
            test_type: "Both", "Positive", or "Negative".
            input_context_df: DataFrame containing additional context.
            output_dir: Directory to save output CSV files.
        """
        os.makedirs(output_dir, exist_ok=True)
        total_stories = len(input_us_df)

        # Prepare context store
        persist_directory = "Data/SavedContexts/EmbedDataMTC"
        os.makedirs(persist_directory, exist_ok=True)

        # Save context CSV
        context_csv_path = "Data/SavedContexts/Contexts.csv"
        input_context_df.to_csv(context_csv_path, index=False, encoding="utf-8-sig")
        print(f"Saved context to: {context_csv_path}")

        # Build vector store
        print("\nBuilding vector store from additional context...")
        connector = ChromaDBConnector(persist_directory=persist_directory)
        connector.vector_store(context_csv_path)

        print(f"\nProcessing {total_stories} user stories with context...")
        print("=" * 80)

        for idx, row in input_us_df.iterrows():
            print(f"\nProcessing story {idx + 1}/{total_stories}")

            # Extract user story ID
            user_story_id = self._extract_user_story_id(row, idx)

            # Build user story text
            description = str(row.get("Description", "")) if pd.notna(row.get("Description")) else ""
            acceptance_criteria = str(row.get("AcceptanceCriteria", "")) if pd.notna(row.get("AcceptanceCriteria")) else ""
            
            user_story_text = f"User Story Description: {description}\nAcceptance Criteria: \n{acceptance_criteria}"

            # Retrieve context
            print(f"Retrieving context for User Story ID: {user_story_id}")
            combined_context, docs_with_scores, threshold = connector.retrieval_context(
                query=user_story_text,
                k=self.num_context_retrieve
            )

            # Filter by threshold
            filtered_docs = {score: doc for score, doc in docs_with_scores.items() if score < threshold}

            if filtered_docs:
                # Use context
                print(f"Found {len(filtered_docs)} relevant contexts (threshold: {threshold:.3f})")
                filtered_context = "\n\n".join(filtered_docs.values())

                result = self.generator.get_manual_test(
                    test_type=test_type,
                    input_length=total_stories,
                    query=user_story_text,
                    context=filtered_context
                )

                used_context = "Yes"
                final_context = filtered_context
            else:
                # No relevant context found
                print(f"No relevant context found (threshold: {threshold:.3f}), generating without context")
                result = self.generator.get_manual_test_no_context(
                    test_type=test_type,
                    input_length=total_stories,
                    query=user_story_text
                )

                used_context = "No"
                final_context = ""

            # Create output DataFrame
            output_df = pd.DataFrame([{
                "UserStoryID": user_story_id,
                "TestType": test_type,
                "UsedContext": used_context,
                "UserStory": user_story_text,
                "Context": final_context,
                "ManualTest": result[0]
            }])

            # Save to CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_dir, f"manual_tc_{idx}_{timestamp}.csv")
            output_df.to_csv(output_file, index=False, encoding="utf-8-sig")
            print(f"Saved: {output_file}")

        print("\n" + "=" * 80)
        print(f"Completed processing {total_stories} user stories with context!")
